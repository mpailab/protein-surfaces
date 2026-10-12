#!/usr/bin/env python3
"""
make_overlap_lists.py

Из наших графов делает два списка:
  - overlap_train.txt: графы, чей (PDB, chain1) в их training.txt
  - overlap_test.txt:  графы, чей (PDB, chain1) в их testing.txt

Также сохраняет JSON с деталями (для отчёта).

Пример:
    python make_overlap_lists.py \
        --our_graph_dir ./processed/graphs_multiscale \
        --their_train_txt ./training.txt \
        --their_test_txt ./testing.txt \
        --out_train overlap_train.txt \
        --out_test overlap_test.txt \
        --out_json overlap_lists.json
"""

import os
import glob
import json
import argparse
from tqdm import tqdm


def parse_our_name(basename):
    parts = basename.split('_')
    if len(parts) == 3:
        return parts[0].upper(), parts[1].upper(), parts[2].upper()
    elif len(parts) == 2:
        return parts[0].upper(), parts[1].upper(), None
    return None


def parse_their_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split('_')
    if len(parts) < 2:
        return None
    return parts[0].upper(), parts[1].upper()


def read_txt(path):
    if not os.path.isfile(path):
        print(f"⚠️  Нет файла: {path}")
        return set()
    with open(path) as f:
        lines = f.readlines()
    out = set()
    for ln in lines:
        parsed = parse_their_line(ln)
        if parsed is not None:
            out.add(parsed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--our_graph_dir', required=True)
    ap.add_argument('--their_train_txt', default='training.txt')
    ap.add_argument('--their_test_txt', default='testing.txt')
    ap.add_argument('--out_train', default='overlap_train.txt')
    ap.add_argument('--out_test', default='overlap_test.txt')
    ap.add_argument('--out_json', default='overlap_lists.json')
    args = ap.parse_args()

    print("📂 Scanning our graphs...")
    paths = sorted(glob.glob(os.path.join(args.our_graph_dir, '*.pt')))
    print(f"   Наших графов: {len(paths)}")

    print("📖 Reading dMaSIF lists...")
    their_train = read_txt(args.their_train_txt)
    their_test = read_txt(args.their_test_txt)
    print(f"   train: {len(their_train)}   test: {len(their_test)}")

    train_files, test_files = [], []
    train_keys, test_keys = set(), set()
    skipped = []

    for p in tqdm(paths, desc="Matching"):
        base = os.path.splitext(os.path.basename(p))[0]
        parsed = parse_our_name(base)
        if parsed is None:
            skipped.append(base)
            continue
        pdb, c1, c2 = parsed
        key = (pdb, c1)

        if key in their_train and key not in their_test:
            train_files.append(base)
            train_keys.add(key)
        elif key in their_test and key not in their_train:
            test_files.append(base)
            test_keys.add(key)
        elif key in their_train and key in their_test:
            # Пересечение — такие исключаем из обоих, чтобы не было утечки
            skipped.append((base, 'in_both'))
        # иначе — не в dMaSIF benchmark, пропускаем

    # ── Сохранение ─────────────────────────────────────────────────
    with open(args.out_train, 'w') as f:
        for x in train_files:
            f.write(x + '\n')
    with open(args.out_test, 'w') as f:
        for x in test_files:
            f.write(x + '\n')

    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ")
    print(f"{'='*70}")
    print(f"  overlap_train.txt:  {len(train_files)} графов "
          f"({len(train_keys)} ключей)")
    print(f"  overlap_test.txt:   {len(test_files)} графов "
          f"({len(test_keys)} ключей)")
    print(f"  Пропущено:          {len(skipped)}")

    # Проверка пересечений
    inter = train_keys & test_keys
    if inter:
        print(f"  ⚠️  Пересечение train/test ключей: {len(inter)} — "
              f"исключены из обоих")
    else:
        print(f"  ✅ Пересечений train/test нет")

    # Сохранение JSON
    out = {
        'summary': {
            'n_train_graphs': len(train_files),
            'n_train_keys': len(train_keys),
            'n_test_graphs': len(test_files),
            'n_test_keys': len(test_keys),
        },
        'train_files': train_files,
        'test_files': test_files,
        'train_keys': [list(k) for k in sorted(train_keys)],
        'test_keys': [list(k) for k in sorted(test_keys)],
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved: {args.out_train}, {args.out_test}, {args.out_json}")

    # Примеры
    print(f"\n  First 5 train: {train_files[:5]}")
    print(f"  First 5 test:  {test_files[:5]}")


if __name__ == '__main__':
    main()