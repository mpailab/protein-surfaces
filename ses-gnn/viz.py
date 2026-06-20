#!/usr/bin/env python3
"""
viz.py - Визуализация предсказаний модели
Автоопределение архитектуры из чекпоинта
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
import plotly.graph_objects as go
import plotly.io as pio
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_cluster import radius_graph

pio.renderers.default = 'browser'

# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ─────────────────────────────────────────────────────────────────────────────
R_CUTOFF = 3.0
MAX_NEIGHBORS = 48
SIGMA = 1.5
PRED_THRESHOLD = 0.4
INTERFACE_THRESHOLD = 4.0
PDB_DIR = "./pdb_files"
PT_DIR = "./data/graphs"

TYPE_NAMES = {0: 'Atom', 1: 'Pair', 2: 'Probe'}
ATOM_COLORS = {'C': '#BDBEBD', 'N': '#5DA5DA', 'O': '#F17CB0', 'S': '#FFD460', 'P': '#FA9F42'}
WATER_NAMES = {'HOH', 'WAT', 'H2O'}
ION_NAMES = {'NA', 'K', 'CA', 'MG', 'CL', 'ZN', 'FE', 'CU', 'MN', 'CO', 'NI', 'HG'}


class BinaryGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=64, num_layers=6, heads=4, dropout=0.0):
        super().__init__()
        head_dim = hidden // heads
        assert head_dim * heads == hidden
        
        self.lin_in = nn.Linear(in_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden, head_dim, heads=heads, edge_dim=3,
                               dropout=dropout, root_weight=True, beta=False))
            self.norms.append(nn.LayerNorm(hidden))
        
        self.lin_out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x = F.gelu(self.norm_in(self.lin_in(data.x)))
        x = self.dropout(x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, data.edge_index, data.edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
        return self.lin_out(x).squeeze(-1)


def load_model(model_path, device='cuda'):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    hidden = checkpoint['lin_in.weight'].shape[0]
    in_dim = checkpoint['lin_in.weight'].shape[1]
    
    num_layers = 0
    for key in checkpoint.keys():
        if key.startswith('convs.'):
            num_layers = max(num_layers, int(key.split('.')[1]) + 1)
    if num_layers == 0:
        num_layers = 6
    
    print(f"Model: {num_layers} layers, hidden={hidden}")
    
    model = BinaryGNN(in_dim, hidden, num_layers, 4, 0.0).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def find_pt_file(pdb_id, pt_dir):
    files = glob.glob(os.path.join(pt_dir, f"{pdb_id}_*.pt"))
    return files[0] if files else None


def load_pt_file(pt_path):
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    basename = os.path.basename(pt_path).replace('.pt', '')
    parts = basename.split('_')
    
    return {
        'pos': data['pos'].numpy(),
        'x': data['x'].numpy(),
        'y': data['y'].numpy(),
        'edge_index': data['edge_index'],
        'edge_attr': data['edge_attr'],
        'pdb_id': parts[0],
        'mode': 'ppi' if len(parts) >= 3 else 'ligand',
        'receptor_chains': list(parts[1]) if len(parts) >= 2 else None,
        'ligand_chains': list(parts[2]) if len(parts) >= 3 else None
    }


def load_pdb_chains(pdb_path, chain_ids):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', pdb_path)
    model = structure[0]
    atoms, elems = [], []
    target = set(chain_ids) if chain_ids else set()
    for chain in model:
        if not chain_ids or chain.id in target:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':
                        atoms.append(atom.get_coord())
                        elems.append(atom.element.upper())
    return np.array(atoms, dtype=np.float32), elems


def load_ligand_atoms(pdb_path, mode, ligand_chains):
    if mode == 'ppi' and ligand_chains:
        return load_pdb_chains(pdb_path, ligand_chains)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', pdb_path)
    model = structure[0]
    coords, elems = [], []
    for chain in model:
        for residue in chain:
            if residue.id[0].strip() == '':
                continue
            resname = residue.get_resname().strip()
            if resname in WATER_NAMES or resname in ION_NAMES:
                continue
            for atom in residue:
                coords.append(atom.get_coord())
                elems.append(atom.element.upper())
    return np.array(coords, dtype=np.float32), elems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--pdb_dir', default='./pdb_files')
    parser.add_argument('--pt_dir', default='./data/graphs')
    parser.add_argument('--interface_threshold', type=float, default=4.0)
    args = parser.parse_args()
    
    global PDB_DIR, PT_DIR, PRED_THRESHOLD, INTERFACE_THRESHOLD
    PDB_DIR, PT_DIR, PRED_THRESHOLD, INTERFACE_THRESHOLD = args.pdb_dir, args.pt_dir, args.threshold, args.interface_threshold
    
    print(f"Model: {args.model} | Complex: {args.name}")
    
    pdb_id = args.name.split('_')[0]
    pt_path = find_pt_file(pdb_id, PT_DIR)
    if pt_path is None:
        print(f"❌ PT file not found")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_data = load_pt_file(pt_path)
    model = load_model(args.model, device)
    
    pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    prot_atoms, prot_elems = load_pdb_chains(pdb_path, pt_data['receptor_chains'])
    lig_atoms, lig_elems = load_ligand_atoms(pdb_path, pt_data['mode'], pt_data['ligand_chains'])
    
    y_true = (pt_data['y'] <= INTERFACE_THRESHOLD).astype(int)
    
    graph = Data(
        x=torch.tensor(pt_data['x'], dtype=torch.float32),
        pos=torch.tensor(pt_data['pos'], dtype=torch.float32),
        edge_index=pt_data['edge_index'],
        edge_attr=pt_data['edge_attr'],
        y=torch.tensor(y_true, dtype=torch.float32)
    ).to(device)
    
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(graph)).cpu().numpy()
    
    y_pred = (y_pred_prob > PRED_THRESHOLD).astype(int)
    
    tp, fp, fn = ((y_true==1)&(y_pred==1)).sum(), ((y_true==0)&(y_pred==1)).sum(), ((y_true==1)&(y_pred==0)).sum()
    prec, rec, f1 = tp/(tp+fp) if tp+fp>0 else 0, tp/(tp+fn) if tp+fn>0 else 0, 2*prec*rec/(prec+rec) if prec+rec>0 else 0
    
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Визуализация
    pts = pt_data['pos']
    edge_index = pt_data['edge_index']
    
    x_lines, y_lines, z_lines = [], [], []
    for i in range(min(edge_index.shape[1], 50000)):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        x_lines.extend([pts[s][0], pts[d][0], None])
        y_lines.extend([pts[s][1], pts[d][1], None])
        z_lines.extend([pts[s][2], pts[d][2], None])
    
    hover = [f"Interface: {'✅' if y_true[i]==1 else '❌'}<br>Pred: {y_pred_prob[i]:.3f}" for i in range(len(pts))]
    
    fig = go.Figure()
    
    if len(x_lines) > 0:
        fig.add_trace(go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(width=1, color='#CCCCCC'), opacity=0.1, name='Edges', hoverinfo='skip'))
    
    if len(prot_atoms) > 0:
        for elem in np.unique(prot_elems):
            mask = np.array([e == elem for e in prot_elems])
            fig.add_trace(go.Scatter3d(x=prot_atoms[mask,0], y=prot_atoms[mask,1], z=prot_atoms[mask,2], mode='markers', marker=dict(size=4, opacity=0.2, color=ATOM_COLORS.get(elem,'#555')), name=f'Protein ({elem})', hoverinfo='skip'))
    
    fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=2, color='#000', opacity=0.3), name='SES', hoverinfo='skip'))
    
    if len(lig_atoms) > 0:
        fig.add_trace(go.Scatter3d(x=lig_atoms[:,0], y=lig_atoms[:,1], z=lig_atoms[:,2], mode='markers', marker=dict(size=5, color='#FF8C00', opacity=0.6), name='Ligand', hoverinfo='skip'))
    
    for mask, color, label in [(y_true==1, '#F1C40F', 'GT'), ((y_true==1)&(y_pred==1), '#2ECC71', f'TP ({tp})'), ((y_true==0)&(y_pred==1), '#E74C3C', f'FP ({fp})'), ((y_true==1)&(y_pred==0), '#F39C12', f'FN ({fn})')]:
        if mask.any():
            fig.add_trace(go.Scatter3d(x=pts[mask,0], y=pts[mask,1], z=pts[mask,2], mode='markers', marker=dict(size=5 if label.startswith('TP') else 3, color=color, opacity=0.9 if label.startswith('TP') else 0.5), text=[hover[i] for i in np.where(mask)[0]], hoverinfo='text', name=label))
    
    fig.update_layout(title=f"{pdb_id} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}", scene=dict(aspectmode='data'), showlegend=True)
    fig.show()


if __name__ == '__main__':
    main()