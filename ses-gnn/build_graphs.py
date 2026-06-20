#!/usr/bin/env python3
"""
build_graphs.py - Этап 2: Построение графов из сырых SES данных
Быстро, можно запускать с разными параметрами R_CUTOFF
"""

import os
import sys
import gc
import glob
import argparse
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from tqdm import tqdm

R_CUTOFF = 2.0
MAX_NEIGHBORS = 24
SIGMA = 1.5


def build_graph_for_file(raw_path, out_dir, r_cutoff=R_CUTOFF, max_neighbors=MAX_NEIGHBORS):
    """Загружает сырые данные, строит граф, сохраняет"""
    
    raw_data = torch.load(raw_path, map_location='cpu', weights_only=False)
    
    pos = raw_data['pos']
    x = raw_data['x']
    norms = raw_data['norms']
    y = raw_data['y']
    
    # Строим граф
    edge_index = radius_graph(pos, r=r_cutoff, max_num_neighbors=max_neighbors, loop=False)
    
    if edge_index.shape[1] == 0:
        print(f"Warning: No edges for {raw_path}")
        return False
    
    src, dst = edge_index
    edge_dist = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
    n_cos = (norms[src] * norms[dst]).sum(dim=1, keepdim=True)
    edge_w = torch.exp(-edge_dist**2 / (2 * SIGMA**2)) * (1.0 + n_cos) / 2.0
    edge_attr = torch.cat([edge_dist, n_cos, edge_w], dim=-1)
    
    # Сохраняем с графом
    graph_data = {
        'x': x,
        'pos': pos,
        'norms': norms,
        'y': y,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'pdb_id': raw_data.get('pdb_id', ''),
        'mode': raw_data.get('mode', 'ligand'),
        'receptor_chains': raw_data.get('receptor_chains', []),
        'ligand_chains': raw_data.get('ligand_chains', [])
    }
    
    out_name = os.path.basename(raw_path)
    out_path = os.path.join(out_dir, out_name)
    torch.save(graph_data, out_path)
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='./processed/raw',
                        help='Directory with raw SES .pt files')
    parser.add_argument('--out_dir', type=str, default='./processed/graphs',
                        help='Output directory for graphs')
    parser.add_argument('--r_cutoff', type=float, default=2.0,
                        help='Radius for graph edges (Å)')
    parser.add_argument('--max_neighbors', type=int, default=24,
                        help='Maximum neighbors per node')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.pt")))
    print(f"Found {len(raw_paths)} raw files")
    
    saved = 0
    for raw_path in tqdm(raw_paths, desc="Building graphs"):
        try:
            if build_graph_for_file(raw_path, args.out_dir, args.r_cutoff, args.max_neighbors):
                saved += 1
        except Exception as e:
            print(f"Error: {raw_path} - {e}")
    
    print(f"✅ Saved {saved} graphs to {args.out_dir}")


if __name__ == '__main__':
    main()