#!/usr/bin/env python3
"""
make_filtered_list.py

Читает testing.txt + ply_stats.json, строит testing_filtered.txt,
исключая вырожденные файлы (pos_ratio < eps_low или > eps_high).

Пример:
    python make_filtered_list.py \
        --list testing.txt \
        --stats ply_stats.json \
        --out testing_filtered.txt \
        --eps_low 0.005 \
        --eps_high 0.95
"""

import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True,
                    help='Исходный testing.txt')
    ap.add_argument('--stats', required=True,
                    help='ply_stats.json')
    ap.add_argument('--out', default='testing_filtered.txt')
    ap.add_argument('--eps_low', type=float, default=0.005,
                    help='Нижний порог pos_ratio (по умолчанию 0.5%%)')
    ap.add_argument('--eps_high', type=float, default=0.95,
                    help='Верхний порог pos_ratio (по умолчанию 95%%)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    # 1. Читаем исходный список
    with open(args.list) as f:
        all_ids = [ln.strip() for ln in f if ln.strip()]

    # 2. Читаем статистику
    with open(args.stats) as f:
        stats = json.load(f)

    per_file = {r['id']: r for r in stats.get('per_file', [])}
    missing = set(stats.get('missing', []))

    kept, dropped_degenerate, dropped_missing = [], [], []

    for cid in all_ids:
        base = os.path.splitext(cid)[0]

        if base in missing:
            dropped_missing.append(base)
            continue

        r = per_file.get(base)
        if r is None:
            dropped_missing.append(base)
            continue

        pr = r['pos_ratio']
        if pr < args.eps_low or pr > args.eps_high:
            dropped_degenerate.append((base, pr))
            continue

        kept.append(cid)

    # 3. Сохраняем
    with open(args.out, 'w') as f:
        for cid in kept:
            f.write(cid + '\n')

    print(f"📋 Input:  {len(all_ids)}")
    print(f"✅ Kept:   {len(kept)}")
    print(f"⚠️  Dropped (degenerate): {len(dropped_degenerate)}")
    print(f"⚠️  Dropped (missing/ply not found): {len(dropped_missing)}")
    print(f"💾 Saved:  {args.out}")

    if dropped_degenerate:
        print(f"\n  Degenerate (pos_ratio вне [{args.eps_low}, "
              f"{args.eps_high}]):")
        for cid, pr in dropped_degenerate:
            print(f"     {cid:25s}  pos_ratio={pr:.4f}")

    if dropped_missing:
        print(f"\n  Missing .ply:")
        for cid in dropped_missing[:20]:
            print(f"     {cid}")
        if len(dropped_missing) > 20:
            print(f"     ... и ещё {len(dropped_missing) - 20}")

    # 4. Sanity check
    total_pos = sum(per_file[c]['n_pos']
                    for c in kept if c in per_file)
    total_pts = sum(per_file[c]['n_points']
                    for c in kept if c in per_file)
    print(f"\n  Global pos_ratio (filtered): "
          f"{total_pos / total_pts:.4f} "
          f"({100 * total_pos / total_pts:.2f}%)")


if __name__ == '__main__':
    main()