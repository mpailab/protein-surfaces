#!/usr/bin/env python3
"""
analyze_graph_degrees.py

Для готовых мультимасштабных графов (после FPS) считает:
  - реальные степени вершин (суммарно и по масштабам) — из edge_index
  - кандидатов до FPS — строит radius_graph на pos заново

Пример:
    python analyze_graph_degrees.py \
        --data_dir ./processed/graphs_multiscale \
        --n_files 20 \
        --out_json ./graph_degrees.json
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
from torch_cluster import radius_graph
from tqdm import tqdm


def analyze_file(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    pos = d['pos']
    ei = d['edge_index']
    ea = d['edge_attr']
    N = len(pos)

    # Первая компонента edge_attr — расстояние
    dist = ea[:, 0].numpy()
    src = ei[0].numpy()

    # Разбиваем рёбра по масштабам (границы — как при построении)
    loc_mask = dist < 2.0
    mid_mask = (dist >= 2.0) & (dist < 4.0)
    glob_mask = (dist >= 4.0) & (dist < 8.0)

    # Степени ПОСЛЕ FPS (исходящие)
    deg_loc = np.bincount(src[loc_mask], minlength=N)
    deg_mid = np.bincount(src[mid_mask], minlength=N)
    deg_glob = np.bincount(src[glob_mask], minlength=N)
    deg_total = deg_loc + deg_mid + deg_glob

    # Кандидаты ДО FPS — строим radius_graph заново
    def candidates(r_min, r_max):
        ei_all = radius_graph(pos, r=r_max, loop=False, max_num_neighbors=1000)
        #ei_all = radius_graph(pos, r=r_max, loop=False)
        s, t = ei_all
        d_all = (pos[s] - pos[t]).norm(dim=1).numpy()
        mask = (d_all >= r_min) & (d_all < r_max)
        return np.bincount(s.numpy()[mask], minlength=N)

    cand_loc = candidates(0.0, 2.0)
    cand_mid = candidates(2.0, 4.0)
    cand_glob = candidates(4.0, 8.0)

    return {
        'N': int(N),
        # После FPS
        'deg_total_mean': float(deg_total.mean()),
        'deg_total_max': int(deg_total.max()),
        'deg_loc_mean': float(deg_loc.mean()),
        'deg_mid_mean': float(deg_mid.mean()),
        'deg_glob_mean': float(deg_glob.mean()),
        # До FPS (кандидаты)
        'cand_loc_mean': float(cand_loc.mean()),
        'cand_mid_mean': float(cand_mid.mean()),
        'cand_glob_mean': float(cand_glob.mean()),
        'cand_mid_max': int(cand_mid.max()),
        'cand_glob_max': int(cand_glob.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--n_files', type=int, default=20)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.pt')))[:args.n_files]
    print(f'Files to analyze: {len(files)}')

    rows = []
    for f in tqdm(files, desc='Analyzing'):
        try:
            rows.append(analyze_file(f))
        except Exception as e:
            print(f'  skip {os.path.basename(f)}: {e}')

    if not rows:
        print('No data.')
        return

    keys = rows[0].keys()
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}

    print('\n' + '='*64)
    print('СВОДКА по {} файлам'.format(len(rows)))
    print('='*64)
    print(f"N точек в графе (среднее): {summary['N']:.0f}")

    print('\n── ПОСЛЕ FPS (реальные рёбра) ──')
    print(f"  Всего degree:          {summary['deg_total_mean']:5.1f} ± {std['deg_total_mean']:.1f}  (max {summary['deg_total_max']:.0f})")
    print(f"  Локальный (0–2 Å):     {summary['deg_loc_mean']:5.1f} ± {std['deg_loc_mean']:.1f}")
    print(f"  Средний   (2–4 Å):     {summary['deg_mid_mean']:5.1f} ± {std['deg_mid_mean']:.1f}")
    print(f"  Глобальный(4–8 Å):     {summary['deg_glob_mean']:5.1f} ± {std['deg_glob_mean']:.1f}")

    print('\n── ДО FPS (кандидаты в окрестности) ──')
    print(f"  Локальный (0–2 Å):     {summary['cand_loc_mean']:5.1f} ± {std['cand_loc_mean']:.1f}")
    print(f"  Средний   (2–4 Å):     {summary['cand_mid_mean']:5.1f} ± {std['cand_mid_mean']:.1f}  (max {summary['cand_mid_max']:.0f})")
    print(f"  Глобальный(4–8 Å):     {summary['cand_glob_mean']:5.1f} ± {std['cand_glob_mean']:.1f}  (max {summary['cand_glob_max']:.0f})")

    print('\n── СЖАТИЕ FPS ──')
    cm = summary['cand_mid_mean'] / max(summary['deg_mid_mean'], 1e-6)
    cg = summary['cand_glob_mean'] / max(summary['deg_glob_mean'], 1e-6)
    print(f"  Средний:    {summary['cand_mid_mean']:.1f}  →  {summary['deg_mid_mean']:.1f}   (в {cm:.1f}x)")
    print(f"  Глобальный: {summary['cand_glob_mean']:.1f}  →  {summary['deg_glob_mean']:.1f}   (в {cg:.1f}x)")

    if args.out_json:
        with open(args.out_json, 'w') as fh:
            json.dump({'summary': summary, 'std': std, 'per_file': rows}, fh, indent=2)
        print(f'\n✅ Saved: {args.out_json}')


if __name__ == '__main__':
    main()