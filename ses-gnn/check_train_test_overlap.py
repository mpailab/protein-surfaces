#!/usr/bin/env python3
"""
check_train_test_overlap.py

Проверяет, что overlap_train_filtered.txt и testing_filtered.txt
не пересекаются по (PDB, chain1).

Пример:
    python check_train_test_overlap.py \
        --train overlap_train_filtered.txt \
        --test testing_filtered.txt \
        --out overlap_check.json
"""

import os
import json
import argparse
from collections import defaultdict


def parse_key(name):
    """
    '1A0G_B_A' -> ('1A0G', 'B')
    '1A0G_B'   -> ('1A0G', 'B')
    """
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0].upper(), parts[1].upper()
    return None


def read_keys(path):
    with open(path) as f:
        names = [os.path.splitext(ln.strip())[0]
                 for ln in f if ln.strip()]
    keys = defaultdict(list)   # key -> [names]
    bad = []
    for n in names:
        k = parse_key(n)
        if k is None:
            bad.append(n)
            continue
        keys[k].append(n)
    return dict(keys), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='overlap_train_filtered.txt')
    ap.add_argument('--test', default='testing_filtered.txt')
    ap.add_argument('--out', default='overlap_check.json')
    args = ap.parse_args()

    train_keys, bad_tr = read_keys(args.train)
    test_keys, bad_te = read_keys(args.test)

    print("=" * 70)
    print("CHECK TRAIN/TEST OVERLAP")
    print("=" * 70)
    print(f"  Train file: {args.train}")
    print(f"    names:  {sum(len(v) for v in train_keys.values())}")
    print(f"    keys:   {len(train_keys)}")
    print(f"  Test file:  {args.test}")
    print(f"    names:  {sum(len(v) for v in test_keys.values())}")
    print(f"    keys:   {len(test_keys)}")

    if bad_tr:
        print(f"  ⚠️  Bad train names: {bad_tr[:5]}")
    if bad_te:
        print(f"  ⚠️  Bad test names:  {bad_te[:5]}")

    # Пересечение по (PDB, chain1)
    common = set(train_keys.keys()) & set(test_keys.keys())

    print()
    print("=" * 70)
    if not common:
        print("✅ Пересечений НЕТ — train и test чистые")
    else:
        print(f"❌ Пересечение: {len(common)} ключей (PDB, chain1)")
    print("=" * 70)

    if common:
        print("\n  Общие ключи (первые 20):")
        for k in sorted(common)[:20]:
            tr = train_keys[k]
            te = test_keys[k]
            print(f"     {k[0]}_{k[1]:<4s}  "
                  f"train={tr}  test={te}")

    # Дополнительно: пересечение по PDB ID (без учёта цепи)
    train_pdbs = {k[0] for k in train_keys.keys()}
    test_pdbs = {k[0] for k in test_keys.keys()}
    pdb_common = train_pdbs & test_pdbs

    print(f"\n  ── По PDB ID (без учёта цепи) ──")
    print(f"    Train unique PDB: {len(train_pdbs)}")
    print(f"    Test unique PDB:  {len(test_pdbs)}")
    if pdb_common:
        print(f"    ⚠️  Общих PDB ID: {len(pdb_common)}")
        for p in sorted(pdb_common)[:10]:
            print(f"       {p}")
    else:
        print(f"    ✅ Общих PDB ID нет")

    # Обратный сценарий: chain1 в train, chain2 в test (что тоже утечка!)
    # Ищем: (pdb, chain1) из train, и (pdb, chain2) из train, где chain2
    # попадает в test_keys[(pdb, chain2)]
    print(f"\n  ── Partner-chain leakage ──")
    print(f"    Проверка: chain2 из train-пары не должен быть chain1 в test")

    partner_leaks = []
    for (pdb, c1), names in train_keys.items():
        for n in names:
            parts = n.split('_')
            if len(parts) == 3:
                c2 = parts[2].upper()
                if (pdb, c2) in test_keys:
                    partner_leaks.append((n, f"{pdb}_{c2}"))

    if partner_leaks:
        print(f"    ⚠️  Partner leakage: {len(partner_leaks)} случаев")
        for tr_name, te_key in partner_leaks[:10]:
            print(f"       train={tr_name}  ↔  test key={te_key}")
    else:
        print(f"    ✅ Partner leakage нет")

    # Сохраняем результат
    out = {
        'train_file': args.train,
        'test_file': args.test,
        'n_train_names': sum(len(v) for v in train_keys.values()),
        'n_test_names': sum(len(v) for v in test_keys.values()),
        'n_train_keys': len(train_keys),
        'n_test_keys': len(test_keys),
        'n_common_keys': len(common),
        'common_keys': [list(k) for k in sorted(common)],
        'n_common_pdbs': len(pdb_common),
        'common_pdbs': sorted(pdb_common),
        'n_partner_leaks': len(partner_leaks),
        'partner_leaks': partner_leaks,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved: {args.out}")


if __name__ == '__main__':
    main()