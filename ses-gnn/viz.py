#!/usr/bin/env python3
"""
viz.py - Визуализация предсказаний модели.

Возможности:
  - 5-цветная схема SES-точек: TN / TP / FP / FN (GT отдельно)
  - Режимы отображения: all, gt_only, pred_only, errors_only, correct_only
  - Рёбра графа с разбивкой по масштабам (локальный / средний / глобальный)
  - Автоопределение архитектуры из чекпоинта
  - Сводные метрики в заголовке графика

Примеры:
    python viz.py --model best_roc.pt --name 3P0C_D --threshold 0.4
    python viz.py --model best_roc.pt --name 3P0C_D --view errors_only
    python viz.py --model best_roc.pt --name 3P0C_D --edge_scale
    python viz.py --model best_roc.pt --name 3P0C_D --view gt_only --save out.html
"""

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser
import plotly.graph_objects as go
import plotly.io as pio
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

pio.renderers.default = 'browser'


# ─────────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PDB_DIR = "./pdb_files"
DEFAULT_PT_DIR = "./data/graphs"
DEFAULT_PRED_THRESHOLD = 0.4
DEFAULT_INTERFACE_THRESHOLD = 4.0

# Границы масштабов для рёбер (в Å)
SCALE_BOUNDS = [(0.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
SCALE_COLORS = ['#888888', '#3B82F6', '#EF4444']  # серый / синий / красный
SCALE_LABELS = ['edges 0-2 Å', 'edges 2-4 Å', 'edges 4-8 Å']
SCALE_OPACITIES = [0.06, 0.12, 0.25]

ATOM_COLORS = {'C': '#BDBEBD', 'N': '#5DA5DA', 'O': '#F17CB0',
               'S': '#FFD460', 'P': '#FA9F42'}
WATER_NAMES = {'HOH', 'WAT', 'H2O'}
ION_NAMES = {'NA', 'K', 'CA', 'MG', 'CL', 'ZN', 'FE', 'CU', 'MN', 'CO', 'NI', 'HG'}

# Цвета SES-точек по категориям
COLOR_TN = '#BBBBBB'
COLOR_TP = '#2ECC71'
COLOR_FP = '#E74C3C'
COLOR_FN = '#F39C12'
COLOR_GT = '#F1C40F'


# ─────────────────────────────────────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────────────────────────────────────
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
            self.convs.append(
                TransformerConv(hidden, head_dim, heads=heads, edge_dim=3,
                               dropout=dropout, root_weight=True, beta=False)
            )
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

    print(f"Model: {num_layers} layers, hidden={hidden}, in_dim={in_dim}")

    model = BinaryGNN(in_dim, hidden, num_layers, 4, 0.0).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
def find_pt_file(full_name, pt_dir):
    """Точное совпадение full_name.pt, иначе fallback на первый pdb_id_*.pt."""
    exact = os.path.join(pt_dir, f"{full_name}.pt")
    if os.path.isfile(exact):
        return exact
    # fallback: первый по алфавиту среди {pdb_id}_*.pt
    pdb_id = full_name.split('_')[0]
    files = sorted(glob.glob(os.path.join(pt_dir, f"{pdb_id}_*.pt")))
    if files:
        print(f"   ⚠️  {full_name}.pt not found, using {os.path.basename(files[0])}")
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
        'ligand_chains': list(parts[2]) if len(parts) >= 3 else None,
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


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ: ГРАФ
# ─────────────────────────────────────────────────────────────────────────────
def add_edges_uniform(fig, pos, edge_index, edge_attr):
    """Одна серия рёбер, все одинаковые."""
    eidx = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
    n_edges = min(eidx.shape[1], 50000)
    if n_edges == 0:
        return

    x_lines, y_lines, z_lines = [], [], []
    for i in range(n_edges):
        s, d = int(eidx[0, i]), int(eidx[1, i])
        x_lines.extend([pos[s][0], pos[d][0], None])
        y_lines.extend([pos[s][1], pos[d][1], None])
        z_lines.extend([pos[s][2], pos[d][2], None])

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(width=1, color='#888888'),
        opacity=0.15,
        name=f'Graph edges ({n_edges})',
        hoverinfo='skip',
    ))


def add_edges_by_scale(fig, pos, edge_index, edge_attr):
    """Три серии рёбер по масштабам: 0-2, 2-4, 4-8 Å."""
    eidx = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
    eattr = edge_attr.numpy() if torch.is_tensor(edge_attr) else edge_attr

    if eidx.shape[1] == 0:
        return

    # Первая компонента edge_attr — расстояние
    dists = eattr[:, 0]
    total = 0

    for (lo, hi), color, label, opac in zip(
            SCALE_BOUNDS, SCALE_COLORS, SCALE_LABELS, SCALE_OPACITIES):

        mask = (dists >= lo) & (dists < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        total += n

        idxs = np.where(mask)[0]
        # Ограничим на случай огромных графов
        if n > 30000:
            idxs = np.random.RandomState(42).choice(idxs, 30000, replace=False)

        x_lines, y_lines, z_lines = [], [], []
        for i in idxs:
            s, d = int(eidx[0, i]), int(eidx[1, i])
            x_lines.extend([pos[s][0], pos[d][0], None])
            y_lines.extend([pos[s][1], pos[d][1], None])
            z_lines.extend([pos[s][2], pos[d][2], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(width=1, color=color),
            opacity=opac,
            name=f'{label} ({n})',
            hoverinfo='skip',
        ))


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ: SES-ТОЧКИ ПО КАТЕГОРИЯМ
# ─────────────────────────────────────────────────────────────────────────────
def build_hover_texts(y_true, y_pred, y_prob):
    return [
        f"<b>Point {i}</b><br>"
        f"GT: {'interface' if y_true[i] == 1 else 'background'}<br>"
        f"Pred: {y_prob[i]:.3f} ({'interface' if y_pred[i] == 1 else 'background'})"
        for i in range(len(y_true))
    ]


def add_ses_points(fig, pos, y_true, y_pred, y_prob, view, n_total):
    """5 trace-ов SES-точек с разными цветами по категориям."""
    size = 2 if n_total > 10000 else 3
    hover = build_hover_texts(y_true, y_pred, y_prob)

    # Маски по категориям
    m_tp = (y_true == 1) & (y_pred == 1)
    m_fn = (y_true == 1) & (y_pred == 0)
    m_fp = (y_true == 0) & (y_pred == 1)
    m_tn = (y_true == 0) & (y_pred == 0)

    # Что показывать в зависимости от режима
    if view == 'all':
        traces = [
            ('TN', m_tn, COLOR_TN, 0.15, size),
            ('FN (missed)', m_fn, COLOR_FN, 0.85, size + 2),
            ('TP (correct)', m_tp, COLOR_TP, 0.95, size + 2),
            ('FP (false alarm)', m_fp, COLOR_FP, 0.85, size + 2),
        ]
    elif view == 'gt_only':
        traces = [
            ('GT interface', m_tp | m_fn, COLOR_GT, 0.9, size + 2),
        ]
    elif view == 'pred_only':
        traces = [
            ('Pred interface', m_tp | m_fp, COLOR_TP, 0.9, size + 2),
        ]
    elif view == 'errors_only':
        traces = [
            ('FN (missed)', m_fn, COLOR_FN, 0.9, size + 3),
            ('FP (false alarm)', m_fp, COLOR_FP, 0.9, size + 3),
        ]
    elif view == 'correct_only':
        traces = [
            ('TN', m_tn, COLOR_TN, 0.15, size),
            ('TP (correct)', m_tp, COLOR_TP, 0.95, size + 2),
        ]
    else:
        raise ValueError(f"Unknown view: {view}")

    for label, mask, color, opacity, sz in traces:
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=pos[mask, 0], y=pos[mask, 1], z=pos[mask, 2],
            mode='markers',
            marker=dict(size=sz, color=color, opacity=opacity),
            text=[hover[i] for i in np.where(mask)[0]],
            hoverinfo='text',
            name=f'{label} ({int(mask.sum())})',
        ))


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ: ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
def visualize(pdb_id, pt_data, prot_atoms, prot_elems, lig_atoms,
              y_true, y_pred, y_prob, metrics,
              view='all', edge_scale=False, threshold=0.4,
              interface_threshold=4.0, save_path=None,
              auto_threshold=False):
    pos = pt_data['pos']
    n_total = len(pos)

    fig = go.Figure()

    # 1. Рёбра графа
    if edge_scale:
        add_edges_by_scale(fig, pos, pt_data['edge_index'], pt_data['edge_attr'])
    else:
        add_edges_uniform(fig, pos, pt_data['edge_index'], pt_data['edge_attr'])

    # 2. Белковые атомы
    if len(prot_atoms) > 0:
        for elem in np.unique(prot_elems):
            mask = np.array([e == elem for e in prot_elems])
            if mask.sum() == 0:
                continue
            color = ATOM_COLORS.get(elem, '#555555')
            fig.add_trace(go.Scatter3d(
                x=prot_atoms[mask, 0], y=prot_atoms[mask, 1], z=prot_atoms[mask, 2],
                mode='markers',
                marker=dict(size=6, opacity=0.15, color=color),
                name=f'Protein ({elem})',
                hoverinfo='skip',
            ))

    # 3. SES-точки по категориям
    add_ses_points(fig, pos, y_true, y_pred, y_prob, view, n_total)

    # 4. Лиганд / партнёр
    if len(lig_atoms) > 0:
        fig.add_trace(go.Scatter3d(
            x=lig_atoms[:, 0], y=lig_atoms[:, 1], z=lig_atoms[:, 2],
            mode='markers',
            marker=dict(size=10, color='#9B59B6', opacity=0.5),
            name=f'Partner ({len(lig_atoms)})',
            hoverinfo='skip',
        ))

    # 5. Заголовок
    thr_label = f"pred thr={threshold:.3f}"
    if auto_threshold:
        thr_label += " ⚠️auto (GT-based, for visualization only)"

    title = (
        f"<b>{pdb_id}</b> | view={view} | {thr_label}, "
        f"GT ≤ {interface_threshold:.1f} Å<br>"
        f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}<br>"
        f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
        f"TN={metrics['tn']}"
    )

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='x', yaxis_title='y', zaxis_title='z',
        ),
        showlegend=True,
        legend=dict(itemsizing='constant'),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"   💾 Saved: {save_path}")
    else:
        try:
            fig.show()
        except Exception as e:
            fallback = f"viz_{pdb_id}.html"
            fig.write_html(fallback)
            print(f"   ⚠️  fig.show() failed ({e})")
            print(f"   💾 Saved to {fallback} instead")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--threshold', type=float, default=DEFAULT_PRED_THRESHOLD,
                        help='Порог предсказания')
    parser.add_argument('--pdb_dir', default=DEFAULT_PDB_DIR)
    parser.add_argument('--pt_dir', default=DEFAULT_PT_DIR)
    parser.add_argument('--interface_threshold', type=float,
                        default=DEFAULT_INTERFACE_THRESHOLD,
                        help='Порог GT интерфейса (Å)')
    parser.add_argument('--view', default='all',
                        choices=['all', 'gt_only', 'pred_only',
                                 'errors_only', 'correct_only'],
                        help='Что показывать')
    parser.add_argument('--auto_threshold', action='store_true',
                        help='Подобрать порог так, чтобы Pred_pos = GT_pos '
                             '(ТОЛЬКО для визуализации, не для метрик)')
    parser.add_argument('--edge_scale', action='store_true',
                        help='Разбить рёбра по масштабам 0-2/2-4/4-8 Å')
    parser.add_argument('--save', type=str, default=None,
                        help='Путь для сохранения HTML (вместо открытия)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"VISUALIZATION")
    print(f"{'='*60}")
    print(f"Model:  {args.model}")
    print(f"Sample: {args.name}")
    print(f"View:   {args.view}")
    print(f"Pred threshold:      {args.threshold}")
    print(f"Interface threshold: {args.interface_threshold} Å")

    # Найти .pt — ищем по полному имени (pdb_chain1_chain2), fallback на pdb_*
    pt_path = find_pt_file(args.name, args.pt_dir)
    pdb_id = args.name.split('_')[0]
    if pt_path is None:
        print(f"❌ PT not found for {pdb_id} in {args.pt_dir}")
        return
    print(f"\nPT file: {pt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Загрузка данных
    print("\n📦 Loading PT...")
    pt_data = load_pt_file(pt_path)
    print(f"   N points: {len(pt_data['pos'])}")
    print(f"   N edges:  {pt_data['edge_index'].shape[1]}")
    print(f"   Receptor chains: {pt_data['receptor_chains']}")
    print(f"   Ligand chains:   {pt_data['ligand_chains']}")

    # PDB
    pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"❌ PDB not found: {pdb_path}")
        return
    print(f"\nPDB file: {pdb_path}")

    # Модель
    print("\n📦 Loading model...")
    model = load_model(args.model, device)

    # Атомы
    print("\n📦 Loading atoms (MODEL 1)...")
    prot_atoms, prot_elems = load_pdb_chains(pdb_path, pt_data['receptor_chains'])
    lig_atoms, _ = load_ligand_atoms(pdb_path, pt_data['mode'], pt_data['ligand_chains'])
    print(f"   Protein atoms: {len(prot_atoms)}")
    print(f"   Ligand atoms:  {len(lig_atoms)}")

    # GT
    y_true = (pt_data['y'] <= args.interface_threshold).astype(int)

    # Inference
    print("\n🔮 Predicting...")
    graph = Data(
        x=torch.tensor(pt_data['x'], dtype=torch.float32),
        pos=torch.tensor(pt_data['pos'], dtype=torch.float32),
        edge_index=pt_data['edge_index'],
        edge_attr=pt_data['edge_attr'],
        y=torch.tensor(y_true, dtype=torch.float32),
    ).to(device)

    with torch.no_grad():
        logits = model(graph)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    if args.auto_threshold:
        # Порог = квантиль y_prob такой, что Pred_pos = GT_pos.
        # Использует GT — только для визуализации, не для метрик!
        target_ratio = y_true.mean()
        # y_prob > threshold → pred_pos ~ target_ratio
        # Значит threshold = квантиль уровня (1 - target_ratio)
        effective_threshold = float(np.quantile(y_prob, 1.0 - target_ratio))
        print(f"\n⚠️  auto_threshold: threshold={effective_threshold:.4f} "
              f"(target pred_pos_ratio={target_ratio:.3f})")
        print(f"   ⚠️  Использует GT — только для визуализации, не для метрик")
    else:
        effective_threshold = args.threshold

    y_pred = (y_prob > effective_threshold).astype(int)

    # Метрики
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = dict(tp=tp, fp=fp, fn=fn, tn=tn,
                   precision=precision, recall=recall, f1=f1)

    print(f"\n📊 Metrics:")
    print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"   y_prob: mean={y_prob.mean():.3f}  "
          f"median={np.median(y_prob):.3f}  "
          f"p90={np.percentile(y_prob, 90):.3f}  "
          f"max={y_prob.max():.3f}")
    print(f"   y_prob на GT-интерфейсе: mean={y_prob[y_true==1].mean():.3f}")
    print(f"   y_prob на фоне:          mean={y_prob[y_true==0].mean():.3f}")
    print(f"   Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"   GT interface:  {y_true.sum()} ({100*y_true.sum()/len(y_true):.2f}%)")
    print(f"   Pred interface: {y_pred.sum()} ({100*y_pred.sum()/len(y_pred):.2f}%)")

    print(f"\n🎨 Rendering (view={args.view})...")
    visualize(pdb_id, pt_data, prot_atoms, prot_elems, lig_atoms,
              y_true, y_pred, y_prob, metrics,
              view=args.view,
              edge_scale=args.edge_scale,
              threshold=effective_threshold,
              interface_threshold=args.interface_threshold,
              save_path=args.save,
              auto_threshold=args.auto_threshold)
    print("✅ Done.")


if __name__ == '__main__':
    main()