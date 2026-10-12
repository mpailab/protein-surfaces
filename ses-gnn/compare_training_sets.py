#!/usr/bin/env python3
"""
compare_training_sets.py

Сравнивает наши обучающие графы с dMaSIF training.txt.

Логика сопоставления:
  Наши файлы:  PDB_CHAIN1_CHAIN2.pt  (пары)
  Их файлы:    PDB_CHAIN             (одна цепь)

Совпадением считается PDB_CHAIN1 из нашей пары == PDB_CHAIN из их списка.

Пример:
    python compare_training_sets.py \
        --our_graph_dir ./processed/graphs_multiscale \
        --their_train_txt ./training.txt \
        --their_test_txt ./testing.txt \
        --out compare_train.json
"""

import os
import re
import glob
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


def parse_our_name(basename):
    """
    '1A0G_A_B' -> ('1A0G', 'A', 'B')
    '1A0G_A'   -> ('1A0G', 'A', None)
    '3SN6_HL_A'-> ('3SN6', 'HL', 'A')   # цепь может быть многосимвольной
    Возвращает (pdb, chain1, chain2) или None, если не распарсилось.
    """
    parts = basename.split('_')
    if len(parts) == 3:
        return parts[0].upper(), parts[1].upper(), parts[2].upper()
    elif len(parts) == 2:
        return parts[0].upper(), parts[1].upper(), None
    return None


def parse_their_line(line):
    """
    '1A0G_A' -> ('1A0G', 'A')
    Возвращает (pdb, chain) или None.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split('_')
    if len(parts) < 2:
        return None
    return parts[0].upper(), parts[1].upper()


def find_our_graphs(graph_dir):
    """Возвращает список (basename, pdb, chain1, chain2)."""
    paths = sorted(glob.glob(os.path.join(graph_dir, '*.pt')))
    result = []
    bad = []
    for p in tqdm(paths, desc="Scanning our graphs"):
        base = os.path.splitext(os.path.basename(p))[0]
        parsed = parse_our_name(base)
        if parsed is None:
            bad.append(base)
            continue
        pdb, c1, c2 = parsed
        result.append({
            'basename': base,
            'path': p,
            'pdb': pdb,
            'chain1': c1,
            'chain2': c2,
        })
    if bad:
        print(f"⚠️  Не распарсилось: {len(bad)} (first 5: {bad[:5]})")
    return result


def read_txt(txt_path):
    """Читает .txt со списком PDB_CHAIN → set[(pdb, chain)]."""
    if not os.path.isfile(txt_path):
        print(f"⚠️  Нет файла: {txt_path}")
        return set()
    with open(txt_path) as f:
        lines = f.readlines()
    out = set()
    bad = 0
    for ln in tqdm(lines, desc=f"Reading {os.path.basename(txt_path)}"):
        parsed = parse_their_line(ln)
        if parsed is None:
            bad += 1
            continue
        out.add(parsed)
    if bad:
        print(f"   ⚠️  Пропущено пустых/битых строк: {bad}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--our_graph_dir', required=True)
    ap.add_argument('--their_train_txt', default='training.txt')
    ap.add_argument('--their_test_txt', default='testing.txt')
    ap.add_argument('--out', default='compare_train.json')
    ap.add_argument('--show_examples', type=int, default=10)
    args = ap.parse_args()

    # 1. Наши графы
    print("=" * 70)
    print("Шаг 1. Сканируем наши графы")
    print("=" * 70)
    ours = find_our_graphs(args.our_graph_dir)
    print(f"✅ Наших графов: {len(ours)}")

    # 2. Их списки
    print("\n" + "=" * 70)
    print("Шаг 2. Читаем dMaSIF списки")
    print("=" * 70)
    their_train = read_txt(args.their_train_txt)
    their_test = read_txt(args.their_test_txt)
    their_all = their_train | their_test
    print(f"✅ Их train: {len(their_train)}  |  test: {len(their_test)}  "
          f"|  all: {len(their_all)}")

    # 3. Сопоставление по префиксу (PDB, chain1)
    print("\n" + "=" * 70)
    print("Шаг 3. Сопоставление")
    print("=" * 70)

    # Индексы
    our_by_key = defaultdict(list)   # (pdb, chain1) -> [graph, ...]
    for g in tqdm(ours, desc="Indexing our graphs"):
        key = (g['pdb'], g['chain1'])
        our_by_key[key].append(g)

    matched_train = []
    matched_test = []
    matched_both = []
    unmatched = []

    for key, graphs in tqdm(our_by_key.items(), desc="Matching"):
        in_train = key in their_train
        in_test = key in their_test
        if in_train and in_test:
            matched_both.append((key, graphs))
        elif in_train:
            matched_train.append((key, graphs))
        elif in_test:
            matched_test.append((key, graphs))
        else:
            unmatched.append((key, graphs))

    # Разворачиваем в графы
    n_matched_train = sum(len(gs) for _, gs in matched_train)
    n_matched_test = sum(len(gs) for _, gs in matched_test)
    n_matched_both = sum(len(gs) for _, gs in matched_both)
    n_unmatched = sum(len(gs) for _, gs in unmatched)

    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ")
    print(f"{'='*70}")
    print(f"  Всего наших графов:                {len(ours)}")
    print(f"  Уникальных ключей (PDB, chain1):   {len(our_by_key)}")
    print()
    print(f"  ✅ Матчатся с их TRAIN:            {n_matched_train} графов "
          f"({len(matched_train)} ключей)")
    print(f"  ⚠️  Матчатся с их TEST:            {n_matched_test} графов "
          f"({len(matched_test)} ключей)")
    print(f"  ⚠️  Матчатся и с train, и с test:  {n_matched_both} графов "
          f"({len(matched_both)} ключей)")
    print(f"  ❌ Не матчатся ни с чем:           {n_unmatched} графов "
          f"({len(unmatched)} ключей)")

    # Дополнительно: сколько их белков мы покрываем
    our_train_keys = {(g['pdb'], g['chain1']) for g in ours}
    covered_train = our_train_keys & their_train
    covered_test = our_train_keys & their_test
    print()
    print(f"  Покрытие их TRAIN:  {len(covered_train)} / {len(their_train)} "
          f"({100 * len(covered_train) / max(len(their_train), 1):.1f}%)")
    print(f"  Покрытие их TEST:   {len(covered_test)} / {len(their_test)} "
          f"({100 * len(covered_test) / max(len(their_test), 1):.1f}%)")

    # 4. Примеры
    if args.show_examples > 0:
        print(f"\n  Примеры совпадений с TRAIN "
              f"(first {args.show_examples}):")
        for key, gs in matched_train[:args.show_examples]:
            files = [g['basename'] for g in gs]
            print(f"     {key[0]}_{key[1]}  →  {files}")

        print(f"\n  Примеры совпадений с TEST "
              f"(first {args.show_examples}):")
        for key, gs in matched_test[:args.show_examples]:
            files = [g['basename'] for g in gs]
            print(f"     {key[0]}_{key[1]}  →  {files}")

        print(f"\n  Примеры несовпадений "
              f"(first {args.show_examples}):")
        for key, gs in unmatched[:args.show_examples]:
            files = [g['basename'] for g in gs]
            print(f"     {key[0]}_{key[1]}  →  {files}")

    # 5. Сохранение
    out = {
        'summary': {
            'n_our_graphs': len(ours),
            'n_unique_keys': len(our_by_key),
            'n_matched_train_graphs': n_matched_train,
            'n_matched_test_graphs': n_matched_test,
            'n_matched_both_graphs': n_matched_both,
            'n_unmatched_graphs': n_unmatched,
            'n_matched_train_keys': len(matched_train),
            'n_matched_test_keys': len(matched_test),
            'n_matched_both_keys': len(matched_both),
            'n_unmatched_keys': len(unmatched),
            'coverage_train': len(covered_train),
            'coverage_test': len(covered_test),
            'their_train_size': len(their_train),
            'their_test_size': len(their_test),
        },
        'matched_train_keys': [
            {'key': list(k), 'files': [g['basename'] for g in gs]}
            for k, gs in matched_train
        ],
        'matched_test_keys': [
            {'key': list(k), 'files': [g['basename'] for g in gs]}
            for k, gs in matched_test
        ],
        'matched_both_keys': [
            {'key': list(k), 'files': [g['basename'] for g in gs]}
            for k, gs in matched_both
        ],
        'unmatched_keys': [
            {'key': list(k), 'files': [g['basename'] for g in gs]}
            for k, gs in unmatched
        ],
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Saved: {args.out}")

    # 6. Рекомендация
    print(f"\n{'='*70}")
    print("РЕКОМЕНДАЦИЯ")
    print(f"{'='*70}")
    if n_matched_train >= 500:
        print(f"  ✅ Достаточно графов для обучения на dMaSIF-разметке: "
              f"{n_matched_train}")
        print(f"     Можно строить графы заново на their TRAIN и обучаться.")
    elif n_matched_train >= 100:
        print(f"  ⚠️  Мало данных ({n_matched_train}), обучение возможно, "
              f"но с меньшей надёжностью.")
        print(f"     Лучше достроить SES+графы для остальных белков "
              f"из training.txt.")
    else:
        print(f"  ❌ Очень мало совпадений ({n_matched_train}). "
              f"Обучение на своих графах невозможно.")
        print(f"     Нужно строить SES+графы для всех {len(their_train)} "
              f"белков из training.txt.")
    print()


if __name__ == '__main__':
    main()