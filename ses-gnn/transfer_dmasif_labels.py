#!/usr/bin/env python3
"""
transfer_dmasif_labels.py

Переносит разметку dMaSIF (iface) на наши графы.

Для каждого нашего графа:
  1. Ищем соответствующий .ply (по имени)
  2. Для каждой НАШЕЙ точки находим ближайшую ИХ точку
  3. Если distance <= max_dist → берём их iface
  4. Если distance > max_dist → label = 0
  5. Сохраняем в новом каталоге с y = 0 (iface) / 99 (background)
     — чтобы встроенная бинаризация (y < 4) работала корректно.

Пример:
    python transfer_dmasif_labels.py \
        --graph_dir ./processed/graphs_multiscale \
        --ply_dir ./01-benchmark_surfaces \
        --out_dir ./processed/graphs_multiscale_dmasif \
        --test_list testing_filtered.txt \
        --max_dist 1.5
"""

import os
import glob
import argparse
import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm


def read_ply_iface(ply_path):
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    fields = {pr.name for pr in v.properties}
    if 'iface' not in fields:
        return None, None
    pts = np.stack([
        np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])
    ], axis=1).astype(np.float32)
    labels = np.asarray(v['iface']).astype(np.int8)
    return pts, labels


def find_ply(ply_dir, graph_basename):
    """
    graph_basename = '1A0G_A_B'   → ply = '1A0G_A.ply'
    graph_basename = '1A0G_A'     → ply = '1A0G_A.ply'
    """
    parts = graph_basename.split('_')
    if len(parts) < 2:
        return None
    pdb = parts[0]
    chain = parts[1]
    cand = os.path.join(ply_dir, f"{pdb}_{chain}.ply")
    if os.path.isfile(cand):
        return cand
    # Fallback
    m = sorted(glob.glob(os.path.join(ply_dir, f"{pdb}_{chain}*.ply")))
    return m[0] if m else None


def transfer_one(graph_path, ply_path, max_dist):
    """Возвращает обновлённый graph dict или None."""
    g = torch.load(graph_path, map_location='cpu', weights_only=False)
    our_pos = g['pos'].numpy().astype(np.float32)

    their_pts, their_labels = read_ply_iface(ply_path)
    if their_pts is None:
        return None, "no iface in ply"

    tree = cKDTree(their_pts)
    dists, idxs = tree.query(our_pos, k=1)

    in_range = dists <= max_dist
    our_labels = np.where(in_range, their_labels[idxs], 0).astype(np.int8)

    # Конвертируем в distance-like формат:
    #   iface -> 0.0  (при binarize y < 4 даст 1)
    #   фон   -> 99.0 (при binarize y < 4 даст 0)
    y_new = np.where(our_labels == 1, 0.0, 99.0).astype(np.float32)

    g['y'] = torch.from_numpy(y_new)
    g['y_dmasif_binary'] = torch.from_numpy(our_labels.astype(np.float32))
    g['label_source'] = 'dmasif_iface'
    g['match_max_dist'] = float(max_dist)

    stats = {
        'n_our': int(len(our_pos)),
        'n_pos': int(our_labels.sum()),
        'pos_ratio': float(our_labels.mean()),
        'in_range_frac': float(in_range.mean()),
        'mean_dist': float(dists.mean()),
        'median_dist': float(np.median(dists)),
    }
    return g, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_dir', required=True,
                    help='Наши графы (обучающие, .pt)')
    ap.add_argument('--ply_dir', required=True,
                    help='dMaSIF .ply файлы')
    ap.add_argument('--out_dir', required=True,
                    help='Куда писать графы с новой разметкой')
    ap.add_argument('--test_list', default=None,
                    help='Опционально: список ID для обработки. '
                         'Если не указан — все .pt из graph_dir.')
    ap.add_argument('--max_dist', type=float, default=1.5)
    ap.add_argument('--exclude_list', default=None,
                    help='Опционально: ID, которые надо исключить '
                         '(например, testing_filtered.txt).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Список графов
    if args.test_list and os.path.isfile(args.test_list):
        with open(args.test_list) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        graph_paths = [os.path.join(args.graph_dir, f"{x}.pt") for x in ids]
    else:
        graph_paths = sorted(glob.glob(
            os.path.join(args.graph_dir, '*.pt')))

    # Исключения
    if args.exclude_list and os.path.isfile(args.exclude_list):
        with open(args.exclude_list) as f:
            exclude = set(ln.strip() for ln in f if ln.strip())
        # graph_paths = [p for p in graph_paths ...]   # ← было неправильно
        filtered = []
        skipped_excluded = []
        for p in graph_paths:
            base = os.path.splitext(os.path.basename(p))[0]   # e.g. "3SN6_A_R"
            parts = base.split('_')
            if len(parts) >= 2:
                prefix = '_'.join(parts[:2])                  # "3SN6_A"
                if prefix in exclude:
                    skipped_excluded.append(base)
                    continue
            filtered.append(p)
        graph_paths = filtered
        print(f"🚫 Excluded {len(skipped_excluded)} graphs "
              f"(matching test prefixes)")
        if skipped_excluded[:5]:
            print(f"   First 5: {skipped_excluded[:5]}")

    print(f"📋 Graphs to process: {len(graph_paths)}")

    saved = 0
    no_ply = []
    no_iface = []
    skipped = []
    all_stats = []

    for gp in tqdm(graph_paths, desc="Transferring"):
        base = os.path.splitext(os.path.basename(gp))[0]
        out_path = os.path.join(args.out_dir, f"{base}.pt")
        if os.path.exists(out_path):
            saved += 1
            continue

        ply_path = find_ply(args.ply_dir, base)
        if ply_path is None:
            no_ply.append(base)
            continue

        try:
            g_new, stats = transfer_one(gp, ply_path, args.max_dist)
        except Exception as e:
            skipped.append((base, str(e)))
            continue

        if g_new is None:
            no_iface.append(base)
            continue

        torch.save(g_new, out_path)
        saved += 1

        stats['id'] = base
        all_stats.append(stats)

    # ── Отчёт ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ✅ Saved:      {saved}")
    print(f"  ⚠️  No .ply:    {len(no_ply)}")
    print(f"  ⚠️  No iface:   {len(no_iface)}")
    print(f"  ❌ Skipped:    {len(skipped)}")
    print(f"{'='*70}")

    if all_stats:
        pos_ratios = np.array([s['pos_ratio'] for s in all_stats])
        print(f"\n  Pos_ratio on our points (transferred from dMaSIF):")
        print(f"     mean   : {pos_ratios.mean():.4f} "
              f"({100 * pos_ratios.mean():.2f}%)")
        print(f"     median : {np.median(pos_ratios):.4f}")
        print(f"     std    : {pos_ratios.std():.4f}")
        print(f"     min    : {pos_ratios.min():.4f}")
        print(f"     max    : {pos_ratios.max():.4f}")

        in_range_fracs = np.array([s['in_range_frac'] for s in all_stats])
        print(f"\n  In-range fraction (our points with their ply within "
              f"{args.max_dist} Å):")
        print(f"     mean   : {in_range_fracs.mean():.4f}")
        print(f"     median : {np.median(in_range_fracs):.4f}")

        # Распределение
        print(f"\n  Распределение pos_ratio:")
        bins = [0, 0.001, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 1.01]
        labels = ['0', '<0.1%', '0.1-1%', '1-3%', '3-5%',
                  '5-10%', '10-20%', '20-50%', '>50%']
        counts, _ = np.histogram(pos_ratios, bins=bins)
        for lab, cnt in zip(labels, counts):
            bar = '█' * int(50 * cnt / max(counts))
            print(f"     {lab:>9s}: {cnt:4d}  {bar}")

        # Sanity — сколько белков с pos_ratio < 0.01 (почти всё фон)
        n_tiny = int((pos_ratios < 0.01).sum())
        if n_tiny > 0:
            print(f"\n  ⚠️  {n_tiny} белков с pos_ratio < 1% — "
                  f"возможно, плохая передача разметки")

    if no_ply[:5]:
        print(f"\n  No .ply (first 5):")
        for x in no_ply[:5]:
            print(f"     {x}")

    # Сохраняем статистику
    if all_stats:
        import json
        with open(os.path.join(args.out_dir, '_transfer_stats.json'), 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n  💾 Per-file stats: "
              f"{os.path.join(args.out_dir, '_transfer_stats.json')}")


if __name__ == '__main__':
    main()