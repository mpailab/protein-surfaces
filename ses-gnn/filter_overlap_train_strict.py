#!/usr/bin/env python3
"""
filter_overlap_train_strict.py

Убирает из overlap_train_filtered.txt:
  1. Графы, чей PDB ID встречается в testing_filtered.txt
     (устраняет partner-chain leakage и переиспользование структур).
  2. Графы с pos_ratio < min_pos_ratio (из _transfer_stats.json).

Пример:
    python filter_overlap_train_strict.py \
        --train overlap_train_filtered.txt \
        --test testing_filtered.txt \
        --stats ./processed/graphs_train_dmasif/_transfer_stats.json \
        --out overlap_train_final.txt \
        --min_pos_ratio 0.01
"""

import os
import json
import argparse


def parse_pdb(name):
    return name.split('_')[0].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='overlap_train_filtered.txt')
    ap.add_argument('--test', default='testing_filtered.txt')
    ap.add_argument('--stats', default='./processed/graphs_train_dmasif/_transfer_stats.json')
    ap.add_argument('--out', default='overlap_train_final.txt')
    ap.add_argument('--min_pos_ratio', type=float, default=0.01,
                    help='Минимальный pos_ratio, ниже — исключаем')
    args = ap.parse_args()

    # 1. PDB IDs из теста
    with open(args.test) as f:
        test_names = [os.path.splitext(ln.strip())[0]
                      for ln in f if ln.strip()]
    test_pdbs = {parse_pdb(n) for n in test_names}
    print(f"📖 Test: {len(test_names)} имён, {len(test_pdbs)} уникальных PDB")

    # 2. pos_ratio из статистики (только для диагностики)
    bad_pos_ratio = set()
    if os.path.isfile(args.stats):
        with open(args.stats) as f:
            stats = json.load(f)
        for s in stats:
            if s.get('pos_ratio', 1.0) < args.min_pos_ratio:
                bad_pos_ratio.add(s['id'])
        print(f"📊 Stats: {len(stats)} файлов, "
              f"{len(bad_pos_ratio)} с pos_ratio < {args.min_pos_ratio}")
    else:
        print(f"⚠️  Stats file не найден: {args.stats} — пропускаем фильтр pos_ratio")

    # 3. Читаем train
    with open(args.train) as f:
        train_names = [os.path.splitext(ln.strip())[0]
                       for ln in f if ln.strip()]
    print(f"📖 Train: {len(train_names)} имён")

    # 4. Фильтруем
    keep = []
    dropped_pdb = []
    dropped_pos = []
    for n in train_names:
        pdb = parse_pdb(n)
        if pdb in test_pdbs:
            dropped_pdb.append(n)
            continue
        if n in bad_pos_ratio:
            dropped_pos.append(n)
            continue
        keep.append(n)

    # 5. Сохраняем
    with open(args.out, 'w') as f:
        for x in keep:
            f.write(x + '\n')

    print(f"\n{'='*70}")
    print("FILTER RESULT")
    print(f"{'='*70}")
    print(f"  Input:                       {len(train_names)}")
    print(f"  Dropped (PDB in test):       {len(dropped_pdb)}")
    print(f"  Dropped (pos_ratio < {args.min_pos_ratio}):  {len(dropped_pos)}")
    print(f"  Kept:                        {len(keep)}")
    print(f"  Saved:                       {args.out}")

    if dropped_pdb:
        print(f"\n  Dropped by PDB overlap (first 20):")
        for x in dropped_pdb[:20]:
            print(f"     {x}")
        if len(dropped_pdb) > 20:
            print(f"     ... и ещё {len(dropped_pdb) - 20}")

    if dropped_pos:
        print(f"\n  Dropped by pos_ratio (all):")
        for x in dropped_pos:
            print(f"     {x}")


if __name__ == '__main__':
    main()