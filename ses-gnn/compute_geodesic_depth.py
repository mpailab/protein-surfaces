#!/usr/bin/env python3
"""
compute_geodesic_depth.py

Для каждого .pt файла считает топологическую глубину интерфейсных точек:
  depth[i] = число хопов по локальным рёбрам (0–2 Å) от точки i до
             ближайшей ФОНОВОЙ точки.

  depth = 0 для фона
  depth = 1 для границы интерфейса
  depth = k для ядра интерфейса (глубоко внутри)

Это топологический аналог физического расстояния `y` до партнёра.
Используется для WBCE на датасетах без партнёра (dMaSIF, DeepPPISP).

Алгоритм: мультиисточниковый BFS через виртуальный источник.
Виртуальный узел соединяется со всеми фоновыми точками (вес 1).
BFS от виртуального → расстояние до каждой точки. depth = dist - 1.

Пример:
    python compute_geodesic_depth.py \
        --data_dir ./processed/graphs_multiscale_new \
        --out_dir ./processed/graphs_multiscale_new_depth \
        --threshold 4.0 \
        --local_r 2.0 \
        --n_files 500
"""

import os
import glob
import time
import argparse
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


def compute_depth(pos, edge_index, edge_attr, iface_mask, local_r=2.0,
                  max_depth=None):
    """
    pos:         [N, 3] tensor
    edge_index:  [2, E] tensor
    edge_attr:   [E, 3] tensor  (0-я компонента = расстояние)
    iface_mask:  [N] bool numpy — какие точки интерфейсные
    local_r:     радиус локальных рёбер (Å)

    Возвращает depth [N] float32. 0 для фона, >0 для интерфейса.
    """
    N = pos.shape[0]

    # Если нет фона — depth = 0 везде
    bg_mask = ~iface_mask
    if bg_mask.sum() == 0 or iface_mask.sum() == 0:
        return np.zeros(N, dtype=np.float32)

    # Перевести в numpy
    src = edge_index[0].numpy() if torch.is_tensor(edge_index) else edge_index[0]
    dst = edge_index[1].numpy() if torch.is_tensor(edge_index) else edge_index[1]
    dists = edge_attr[:, 0].numpy() if torch.is_tensor(edge_attr) else edge_attr[:, 0]

    # Оставить только локальные рёбра (0 < d < local_r)
    local = (dists < local_r) & (dists > 1e-6)
    src_l, dst_l = src[local], dst[local]

    if len(src_l) == 0:
        return np.zeros(N, dtype=np.float32)

    # Неориентированный граф (симметризация)
    rows = np.concatenate([src_l, dst_l])
    cols = np.concatenate([dst_l, src_l])
    data = np.ones(len(rows), dtype=np.float32)
    adj = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Добавить виртуальный узел N, соединить со всеми фоновыми
    bg_idx = np.where(bg_mask)[0]
    virtual = N
    N_aug = N + 1

    adj_coo = adj.tocoo()
    rows_aug = np.concatenate([adj_coo.row, np.full(len(bg_idx), virtual)])
    cols_aug = np.concatenate([adj_coo.col, bg_idx])
    data_aug = np.concatenate([adj_coo.data, np.ones(len(bg_idx), dtype=np.float32)])

    adj_aug = csr_matrix((data_aug, (rows_aug, cols_aug)),
                         shape=(N_aug, N_aug))

    # BFS от виртуального узла
    try:
        dist = dijkstra(adj_aug, indices=virtual, unweighted=True,
                        directed=False, min_only=True)
    except TypeError:
        # Fallback: dijkstra с indices возвращает [1, N_aug]
        dist = dijkstra(adj_aug, indices=virtual, unweighted=True,
                        directed=False)
        dist = dist.ravel()

    # dist[bg] = 1 (через виртуальный), dist[граница] = 2, ..., ядро = k+1
    # depth = расстояние в хопах до ближайшего фона = dist - 1
    depth = np.zeros(N, dtype=np.float32)
    finite = np.isfinite(dist[:N]) & (dist[:N] > 0)
    depth[finite] = (dist[:N][finite] - 1).astype(np.float32)
    # Фон = 0, недостижимые = 0
    depth[bg_mask] = 0.0

    if max_depth is not None:
        depth = np.minimum(depth, max_depth)

    return depth


def process_file(path, out_dir, threshold=4.0, local_r=2.0, max_depth=None):
    out_path = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(out_path):
        return True, None

    try:
        d = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        return False, f"load: {e}"

    try:
        y_dist = d['y'].numpy()
        iface_mask = y_dist < threshold

        depth = compute_depth(
            pos=d['pos'],
            edge_index=d['edge_index'],
            edge_attr=d['edge_attr'],
            iface_mask=iface_mask,
            local_r=local_r,
            max_depth=max_depth,
        )

        d['y_depth'] = torch.from_numpy(depth)
        d['y_depth_threshold'] = float(threshold)
        d['y_depth_local_r'] = float(local_r)
        torch.save(d, out_path)

        # Статистика для отчёта
        stats = {
            'n': len(depth),
            'n_iface': int(iface_mask.sum()),
            'depth_max': float(depth.max()),
            'depth_median_iface': float(np.median(depth[iface_mask])) if iface_mask.sum() > 0 else 0.0,
        }
        return True, stats
    except Exception as e:
        return False, f"process: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='Папка с .pt (pos, edge_index, edge_attr, y)')
    ap.add_argument('--out_dir', required=True,
                    help='Куда сохранять .pt с добавленным y_depth')
    ap.add_argument('--threshold', type=float, default=4.0,
                    help='Порог интерфейса по y (Å)')
    ap.add_argument('--local_r', type=float, default=2.0,
                    help='Радиус локальных рёбер (Å)')
    ap.add_argument('--max_depth', type=int, default=None,
                    help='Обрезать depth сверху (None = без обрезки)')
    ap.add_argument('--n_files', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.pt')))
    if args.n_files:
        files = files[:args.n_files]

    print(f"Files:      {len(files)}")
    print(f"Threshold:  {args.threshold} Å")
    print(f"Local r:    {args.local_r} Å")
    print(f"Max depth:  {args.max_depth}")
    print(f"Out:        {args.out_dir}")
    print()

    saved = 0
    failed = []
    depth_maxes, depth_medians = [], []

    t0 = time.time()
    for f in tqdm(files, desc='Computing depth'):
        ok, stats = process_file(f, args.out_dir, args.threshold,
                                 args.local_r, args.max_depth)
        if ok:
            saved += 1
            if stats:
                depth_maxes.append(stats['depth_max'])
                depth_medians.append(stats['depth_median_iface'])
        else:
            failed.append((os.path.basename(f), stats))
        if (saved + len(failed)) % 500 == 0:
            rate = (saved + len(failed)) / max(time.time() - t0, 1) * 60
            print(f"  progress: {saved + len(failed)}/{len(files)} "
                  f"| {rate:.0f} files/min", flush=True)

    print()
    print("="*70)
    print(f"  Saved:  {saved}")
    print(f"  Failed: {len(failed)}")
    print("="*70)

    if depth_maxes:
        print(f"\nDepth statistics (по белкам):")
        print(f"  depth_max:           mean={np.mean(depth_maxes):.2f}  "
              f"max={np.max(depth_maxes):.0f}")
        print(f"  depth_median_iface:  mean={np.mean(depth_medians):.2f}")

    if failed[:5]:
        print("\nFirst failures:")
        for name, err in failed[:5]:
            print(f"  {name}: {err}")


if __name__ == '__main__':
    main()