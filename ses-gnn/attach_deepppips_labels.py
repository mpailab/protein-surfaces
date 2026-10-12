#!/usr/bin/env python3
"""
attach_deepppips_labels.py

Для каждого .pt в graphs_deepppips_test_residue:
  1. Загружает residue-level labels из deepppips_protein_labels.pkl.
  2. Сопоставляет residue_id (индекс в map) ↔ индекс в labels.
  3. Сохраняет:
     - residue_gt [R] — residue-level labels (для агрегации)
     - point_gt   [N] — point-level labels (наследуется от residue)

Проверка соответствия: len(labels) == num_residues (в .pt).
"""

import os
import pickle
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_dir', required=True)
    ap.add_argument('--labels_pkl', required=True,
                    help='deepppips_protein_labels.pkl')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.labels_pkl, 'rb') as f:
        protein_to_labels = pickle.load(f)
    print(f"Loaded labels for {len(protein_to_labels)} proteins")

    graph_paths = sorted(glob.glob(os.path.join(args.graph_dir, '*.pt')))
    print(f"Graphs: {len(graph_paths)}")

    saved = 0
    failed = []
    stats = []

    for gp in tqdm(graph_paths, desc="Attaching labels"):
        base = os.path.splitext(os.path.basename(gp))[0]
        out_path = os.path.join(args.out_dir, f"{base}.pt")
        if os.path.exists(out_path):
            saved += 1
            continue

        # Пробуем несколько форматов
        key = base
        if key not in protein_to_labels:
            # Пробуем без '_': 2WD5_A -> 2WD5A
            if '_' in base:
                alt = base.replace('_', '')  # 2WD5_A -> 2WD5A
                if alt in protein_to_labels:
                    key = alt
                else:
                    # Пробуем с '_' перед последней буквой
                    # 2WD5_A -> 2WD5A (PDB + chain)
                    parts = base.split('_')
                    if len(parts) == 2 and len(parts[0]) == 4:
                        # Уже правильный формат
                        pass
            if key not in protein_to_labels:
                failed.append((base, 'no labels'))
                continue

        labels = np.asarray(protein_to_labels[key])
        try:
            d = torch.load(gp, weights_only=False)
            rid = d['residue_id'].numpy()
            n_res = d['num_residues']

            # Проверка соответствия
            if len(labels) != n_res:
                failed.append((base, f'len mismatch: labels={len(labels)}, num_res={n_res}'))
                continue

            # residue_gt: для каждого residue_id его label
            residue_gt = labels.astype(np.int64)  # [R]

            # point_gt: для каждой точки — label её residue
            point_gt = residue_gt[rid]  # [N]

            d['residue_gt'] = torch.from_numpy(residue_gt).long()
            d['point_gt'] = torch.from_numpy(point_gt).long()

            torch.save(d, out_path)
            saved += 1
            stats.append({
                'id': base,
                'n_points': len(d['pos']),
                'n_residues': n_res,
                'pos_ratio_residue': float(labels.mean()),
                'pos_ratio_point': float(point_gt.mean()),
            })
        except Exception as e:
            failed.append((base, str(e)))

    print(f"\n{'='*60}")
    print(f"  ✅ Saved:   {saved}")
    print(f"  ❌ Failed:  {len(failed)}")
    print(f"{'='*60}")

    if failed:
        print("\n  First 10 failures:")
        for name, err in failed[:10]:
            print(f"    {name}: {err}")

    if stats:
        import json
        with open(os.path.join(args.out_dir, '_labels_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        rr = np.array([s['pos_ratio_residue'] for s in stats])
        pr = np.array([s['pos_ratio_point'] for s in stats])
        print(f"\n  Residue pos_ratio: mean={rr.mean():.3f}, median={np.median(rr):.3f}")
        print(f"  Point   pos_ratio: mean={pr.mean():.3f}, median={np.median(pr):.3f}")


if __name__ == '__main__':
    main()