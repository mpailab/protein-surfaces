#!/usr/bin/env python3
"""
build_graphs.py - Построение мультимасштабных графов с отбором центроидов
Использует Farthest Point Sampling (FPS) для выбора репрезентативных рёбер
на среднем (2-4Å) и глобальном (4-8Å) масштабах.
Локальные рёбра (0-2Å) сохраняются все.
"""

import os
import sys
import glob
import argparse
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np


def farthest_point_sampling(points, k):
    """
    Выбирает k точек из множества points (тензор [M, 3]) методом FPS.
    Возвращает индексы выбранных точек (тензор [k]).
    """
    M = points.shape[0]
    if M <= k:
        return torch.arange(M, device=points.device)
    
    indices = torch.zeros(k, dtype=torch.long, device=points.device)
    indices[0] = 0
    dists = torch.full((M,), float('inf'), device=points.device)
    
    for i in range(1, k):
        cur_point = points[indices[i-1]].unsqueeze(0)
        cur_dists = torch.cdist(points, cur_point).squeeze(1)
        dists = torch.min(dists, cur_dists)
        next_idx = torch.argmax(dists)
        indices[i] = next_idx
    
    return indices


def build_multiscale_graph(pos, norms, r_local, r_mid, r_global, k_mid, k_global, sigma):
    """
    Строит мультимасштабный граф для заданных координат и нормалей.
    Возвращает edge_index, edge_attr.
    """
    device = pos.device
    N = pos.shape[0]
    
    # 1. Локальные рёбра (0-r_local) — все
    e_local = radius_graph(pos, r=r_local, loop=False)
    
    # 2. Средние рёбра (r_local - r_mid) — отбираем центроиды
    e_all_mid = radius_graph(pos, r=r_mid, loop=False)
    src_mid, dst_mid = e_all_mid
    dist_mid = torch.norm(pos[src_mid] - pos[dst_mid], dim=1)
    mask_mid = dist_mid >= r_local
    e_mid_filtered = e_all_mid[:, mask_mid]
    
    selected_mid_src = []
    selected_mid_dst = []
    
    for i in range(N):
        mask_i = (src_mid == i) & mask_mid
        if not mask_i.any():
            continue
        neigh_idx = dst_mid[mask_i]
        if len(neigh_idx) == 0:
            continue
        if len(neigh_idx) <= k_mid:
            selected_mid_src.extend([i] * len(neigh_idx))
            selected_mid_dst.extend(neigh_idx.tolist())
        else:
            neigh_pos = pos[neigh_idx]
            fps_indices = farthest_point_sampling(neigh_pos, k_mid)
            selected_neigh = neigh_idx[fps_indices]
            selected_mid_src.extend([i] * len(selected_neigh))
            selected_mid_dst.extend(selected_neigh.tolist())
    
    # 3. Глобальные рёбра (r_mid - r_global) — аналогично
    e_all_global = radius_graph(pos, r=r_global, loop=False)
    src_global, dst_global = e_all_global
    dist_global = torch.norm(pos[src_global] - pos[dst_global], dim=1)
    mask_global = dist_global >= r_mid
    e_global_filtered = e_all_global[:, mask_global]
    
    selected_global_src = []
    selected_global_dst = []
    
    for i in range(N):
        mask_i = (src_global == i) & mask_global
        if not mask_i.any():
            continue
        neigh_idx = dst_global[mask_i]
        if len(neigh_idx) == 0:
            continue
        if len(neigh_idx) <= k_global:
            selected_global_src.extend([i] * len(neigh_idx))
            selected_global_dst.extend(neigh_idx.tolist())
        else:
            neigh_pos = pos[neigh_idx]
            fps_indices = farthest_point_sampling(neigh_pos, k_global)
            selected_neigh = neigh_idx[fps_indices]
            selected_global_src.extend([i] * len(selected_neigh))
            selected_global_dst.extend(selected_neigh.tolist())
    
    # 4. Объединяем все рёбра
    src_local, dst_local = e_local
    src_all = torch.cat([
        src_local,
        torch.tensor(selected_mid_src, device=device, dtype=torch.long),
        torch.tensor(selected_global_src, device=device, dtype=torch.long)
    ])
    dst_all = torch.cat([
        dst_local,
        torch.tensor(selected_mid_dst, device=device, dtype=torch.long),
        torch.tensor(selected_global_dst, device=device, dtype=torch.long)
    ])
    
    edge_index = torch.stack([src_all, dst_all], dim=0)
    
    # 5. Вычисляем edge_attr для всех рёбер
    src, dst = edge_index
    edge_dist = torch.norm(pos[src] - pos[dst], dim=1, keepdim=True)
    n_cos = (norms[src] * norms[dst]).sum(dim=1, keepdim=True)
    edge_w = torch.exp(-edge_dist**2 / (2 * sigma**2)) * (1.0 + n_cos) / 2.0
    edge_attr = torch.cat([edge_dist, n_cos, edge_w], dim=-1)
    
    return edge_index, edge_attr


def process_file(raw_path, out_dir, params):
    out_name = os.path.basename(raw_path)
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        return True
    """Загружает сырые данные, строит мультимасштабный граф, сохраняет."""
    raw_data = torch.load(raw_path, map_location='cpu', weights_only=False)
    
    pos = raw_data['pos']
    x = raw_data['x']
    norms = raw_data['norms']
    y = raw_data['y']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos = pos.to(device)
    norms = norms.to(device)
    
    edge_index, edge_attr = build_multiscale_graph(
        pos, norms,
        r_local=params['r_local'],
        r_mid=params['r_mid'],
        r_global=params['r_global'],
        k_mid=params['k_mid'],
        k_global=params['k_global'],
        sigma=params['sigma']
    )
    
    graph_data = {
        'x': x,
        'pos': pos.cpu(),
        'norms': norms.cpu(),
        'y': y,
        'edge_index': edge_index.cpu(),
        'edge_attr': edge_attr.cpu(),
        'pdb_id': raw_data.get('pdb_id', ''),
        'mode': raw_data.get('mode', 'ligand'),
        'receptor_chains': raw_data.get('receptor_chains', []),
        'ligand_chains': raw_data.get('ligand_chains', [])
    }
    
    torch.save(graph_data, out_path)
    
    del pos, norms, edge_index, edge_attr
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Построение мультимасштабных графов с FPS'
    )
    parser.add_argument('--raw_dir', type=str, default='./processed/raw',
                        help='Папка с сырыми .pt файлами')
    parser.add_argument('--out_dir', type=str, default='./processed/graphs_multiscale',
                        help='Папка для сохранения графов')
    parser.add_argument('--r_local', type=float, default=2.0,
                        help='Локальный радиус (все рёбра)')
    parser.add_argument('--r_mid', type=float, default=4.0,
                        help='Средний радиус (для отбора центроидов)')
    parser.add_argument('--r_global', type=float, default=8.0,
                        help='Глобальный радиус (для отбора центроидов)')
    parser.add_argument('--k_mid', type=int, default=10,
                        help='Число центроидов на среднем масштабе')
    parser.add_argument('--k_global', type=int, default=10,
                        help='Число центроидов на глобальном масштабе')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Параметр затухания для edge_w')
    parser.add_argument('--n_files', type=int, default=None,
                        help='Число файлов для обработки (для теста)')
    args = parser.parse_args()
    
    params = {
        'r_local': args.r_local,
        'r_mid': args.r_mid,
        'r_global': args.r_global,
        'k_mid': args.k_mid,
        'k_global': args.k_global,
        'sigma': args.sigma,
    }
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.pt")))
    if args.n_files:
        raw_paths = raw_paths[:args.n_files]
    
    print(f"🔧 Параметры графа:")
    print(f"   Локальный радиус: {params['r_local']}Å (все рёбра)")
    print(f"   Средний радиус: {params['r_mid']}Å, центроидов: {params['k_mid']}")
    print(f"   Глобальный радиус: {params['r_global']}Å, центроидов: {params['k_global']}")
    print(f"   SIGMA: {params['sigma']}")
    print(f"📂 Найдено {len(raw_paths)} файлов")
    
    saved = 0
    for raw_path in tqdm(raw_paths, desc="Построение графов"):
        try:
            if process_file(raw_path, args.out_dir, params):
                saved += 1
        except Exception as e:
            print(f"❌ Ошибка в {raw_path}: {e}")
    
    print(f"✅ Сохранено {saved} графов в {args.out_dir}")


if __name__ == '__main__':
    main()