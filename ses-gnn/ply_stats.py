#!/usr/bin/env python3
"""
ply_stats.py

Собирает статистику по .ply файлам dMaSIF:
  - распределение iface (бинарность)
  - pos_ratio на файл
  - среднее / медиана / min / max
  - число точек на файл
  - поиск вырожденных случаев (pos_ratio == 0 или == 1)

Флаг --filter позволяет исключить вырожденные файлы из агрегированной
статистики (но они всё равно попадут в per_file и degenerate list).

Пример:
    python ply_stats.py \
        --list testing.txt \
        --ply_dir testing-ply \
        --out ply_stats.json \
        --filter
"""

import os
import glob
import json
import argparse
import numpy as np
from plyfile import PlyData
from tqdm import tqdm


def find_ply(ply_dir, cid):
    """Ищет {cid}.ply, {cid}_*.ply, *{cid}*.ply."""
    p = os.path.join(ply_dir, f"{cid}.ply")
    if os.path.isfile(p):
        return p
    m = sorted(glob.glob(os.path.join(ply_dir, f"{cid}_*.ply")))
    if m:
        return m[0]
    m = sorted(glob.glob(os.path.join(ply_dir, '**', f"{cid}*.ply"),
                            recursive=True))
    if m:
        return m[0]
    return None


def read_iface(ply_path):
    """Читает поле iface из .ply. Возвращает np.array или None."""
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    fields = {p.name for p in v.properties}
    if 'iface' not in fields:
        return None
    return np.asarray(v['iface']).astype(np.float32)


def print_distribution(ratios, title="Pos_ratio"):
    """Печатает гистограмму распределения pos_ratio."""
    print(f"\nРаспределение {title}:")
    bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.01]
    labels = ['0', '<0.1%', '0.1-0.5%', '0.5-1%', '1-2%',
              '2-5%', '5-10%', '10-20%', '20-50%', '>50%']
    counts, _ = np.histogram(ratios, bins=bins)
    maxc = max(counts) if len(counts) else 1
    for lab, cnt in zip(labels, counts):
        bar = '█' * int(50 * cnt / maxc) if maxc > 0 else ''
        print(f"   {lab:>9s}: {cnt:4d}  {bar}")


def print_summary(rows, title):
    """Печатает агрегированную статистику по списку rows."""
    if not rows:
        print(f"\n{title}: нет данных.")
        return
    ratios = np.array([r['pos_ratio'] for r in rows])
    npts = np.array([r['n_points'] for r in rows])
    npos = np.array([r['n_pos'] for r in rows])
    total_pos = int(npos.sum())
    total_pts = int(npts.sum())

    print(f"\n{'='*70}")
    print(f"{title} ({len(rows)} files)")
    print(f"{'='*70}")

    print(f"\nPos_ratio (доля интерфейсных точек):")
    print(f"   mean   : {ratios.mean():.5f}  ({ratios.mean()*100:.3f}%)")
    print(f"   median : {np.median(ratios):.5f}")
    print(f"   std    : {ratios.std():.5f}")
    print(f"   min    : {ratios.min():.5f}  ({ratios.min()*100:.3f}%)")
    print(f"   max    : {ratios.max():.5f}  ({ratios.max()*100:.3f}%)")
    print(f"   Q1/Q3  : {np.percentile(ratios,25):.5f} / "
          f"{np.percentile(ratios,75):.5f}")
    print(f"   total pos: {total_pos:,}")
    print(f"   total pts: {total_pts:,}")
    print(f"   global pos_ratio: {total_pos / total_pts:.5f} "
          f"({100 * total_pos / total_pts:.3f}%)")

    print(f"\nPoints per file:")
    print(f"   mean   : {npts.mean():.0f}")
    print(f"   median : {np.median(npts):.0f}")
    print(f"   min    : {npts.min()}")
    print(f"   max    : {npts.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True,
                    help='Список ID (по одному на строку)')
    ap.add_argument('--ply_dir', required=True,
                    help='Папка с .ply файлами')
    ap.add_argument('--out', default='ply_stats.json')
    ap.add_argument('--filter', action='store_true',
                    help='Исключить из агрегатов файлы с '
                         'pos_ratio == 0 или == 1')
    ap.add_argument('--eps', type=float, default=1e-4,
                    help='Порог для вырожденных (по умолчанию 1e-4)')
    ap.add_argument('--show_worst', type=int, default=10)
    args = ap.parse_args()

    with open(args.list) as f:
        ids = [os.path.splitext(ln.strip())[0]
               for ln in f if ln.strip()]
    print(f"📋 IDs: {len(ids)}")

    rows = []
    missing = []
    non_binary = []

    for cid in tqdm(ids, desc="Scanning"):
        ply_path = find_ply(args.ply_dir, cid)
        if ply_path is None:
            missing.append(cid)
            continue

        iface = read_iface(ply_path)
        if iface is None:
            missing.append((cid, "no iface field"))
            continue

        uniq = np.unique(iface)
        is_binary = set(uniq.tolist()).issubset({0.0, 1.0})
        if not is_binary:
            non_binary.append((cid, uniq.tolist()))

        n = int(len(iface))
        n_pos = int((iface == 1).sum())
        pos_ratio = float(n_pos) / n if n > 0 else 0.0

        rows.append({
            'id': cid,
            'ply': os.path.basename(ply_path),
            'n_points': n,
            'n_pos': n_pos,
            'pos_ratio': pos_ratio,
            'unique_values': uniq.tolist(),
            'is_binary': is_binary,
        })

    if not rows:
        print("❌ No data.")
        return

    # ── Полная статистика ────────────────────────────────────────────
    print_summary(rows, "STATISTICS (ALL FILES)")

    # ── Вырожденные случаи ──────────────────────────────────────────
    zero_pos = [r for r in rows if r['pos_ratio'] <= args.eps]
    all_pos = [r for r in rows if r['pos_ratio'] >= 1.0 - args.eps]
    degenerate = zero_pos + all_pos

    print(f"\nВырожденные случаи "
          f"(eps={args.eps}):")
    print(f"   pos_ratio == 0 : {len(zero_pos)}")
    for r in zero_pos[:10]:
        print(f"     {r['id']} (n={r['n_points']}, pos={r['n_pos']})")
    print(f"   pos_ratio == 1 : {len(all_pos)}")
    for r in all_pos[:10]:
        print(f"     {r['id']} (n={r['n_points']}, pos={r['n_pos']})")

    # ── Распределение по всем ───────────────────────────────────────
    ratios_all = np.array([r['pos_ratio'] for r in rows])
    print_distribution(ratios_all, "Pos_ratio (ALL)")

    # ── Фильтрованная статистика ────────────────────────────────────
    if args.filter:
        rows_clean = [r for r in rows
                      if args.eps < r['pos_ratio'] < 1.0 - args.eps]
        print_summary(rows_clean, "STATISTICS (FILTERED)")
        ratios_clean = np.array([r['pos_ratio'] for r in rows_clean])
        print_distribution(ratios_clean, "Pos_ratio (FILTERED)")

        # Топ-худших и топ-лучших по pos_ratio (без вырожденных)
        rows_sorted = sorted(rows_clean, key=lambda x: x['pos_ratio'])
        print(f"\nСамые НИЗКИЕ pos_ratio ({args.show_worst}):")
        for r in rows_sorted[:args.show_worst]:
            print(f"   {r['id']:25s}  pos={r['n_pos']:5d}/{r['n_points']:6d}  "
                  f"({r['pos_ratio']*100:.3f}%)")
        print(f"\nСамые ВЫСОКИЕ pos_ratio ({args.show_worst}):")
        for r in rows_sorted[-args.show_worst:]:
            print(f"   {r['id']:25s}  pos={r['n_pos']:5d}/{r['n_points']:6d}  "
                  f"({r['pos_ratio']*100:.3f}%)")
    else:
        rows_sorted = sorted(rows, key=lambda x: x['pos_ratio'])
        print(f"\nСамые НИЗКИЕ pos_ratio ({args.show_worst}):")
        for r in rows_sorted[:args.show_worst]:
            print(f"   {r['id']:25s}  pos={r['n_pos']:5d}/{r['n_points']:6d}  "
                  f"({r['pos_ratio']*100:.3f}%)")
        print(f"\nСамые ВЫСОКИЕ pos_ratio ({args.show_worst}):")
        for r in rows_sorted[-args.show_worst:]:
            print(f"   {r['id']:25s}  pos={r['n_pos']:5d}/{r['n_points']:6d}  "
                  f"({r['pos_ratio']*100:.3f}%)")

    # ── Не-бинарные ─────────────────────────────────────────────────
    if non_binary:
        print(f"\n⚠️  НЕ-БИНАРНЫЕ метки: {len(non_binary)}")
        for cid, uniq in non_binary[:10]:
            print(f"   {cid}: {uniq[:10]}...")
    else:
        print(f"\n✅ Все метки бинарные ({{0, 1}})")

    if missing:
        print(f"\n⚠️  Не найдено .ply: {len(missing)}")
        for item in missing[:10]:
            if isinstance(item, tuple):
                print(f"   {item[0]}: {item[1]}")
            else:
                print(f"   {item}")

    # ── Сохранение ──────────────────────────────────────────────────
    ratios_all = np.array([r['pos_ratio'] for r in rows])
    summary_all = {
        'n_files': len(rows),
        'n_missing': len(missing),
        'n_non_binary': len(non_binary),
        'n_zero_pos': len(zero_pos),
        'n_all_pos': len(all_pos),
        'pos_ratio_mean': float(ratios_all.mean()),
        'pos_ratio_median': float(np.median(ratios_all)),
        'pos_ratio_std': float(ratios_all.std()),
        'pos_ratio_min': float(ratios_all.min()),
        'pos_ratio_max': float(ratios_all.max()),
        'global_pos_ratio': float(
            sum(r['n_pos'] for r in rows) /
            sum(r['n_points'] for r in rows)
        ),
        'total_pos': int(sum(r['n_pos'] for r in rows)),
        'total_points': int(sum(r['n_points'] for r in rows)),
    }

    out = {
        'summary_all': summary_all,
        'per_file': rows,
        'missing': [m if isinstance(m, str) else m[0] for m in missing],
        'non_binary': non_binary,
        'degenerate': [r['id'] for r in degenerate],
    }

    if args.filter:
        rows_clean = [r for r in rows
                      if args.eps < r['pos_ratio'] < 1.0 - args.eps]
        ratios_clean = np.array([r['pos_ratio'] for r in rows_clean])
        out['summary_filtered'] = {
            'n_files': len(rows_clean),
            'pos_ratio_mean': float(ratios_clean.mean()),
            'pos_ratio_median': float(np.median(ratios_clean)),
            'pos_ratio_std': float(ratios_clean.std()),
            'pos_ratio_min': float(ratios_clean.min()),
            'pos_ratio_max': float(ratios_clean.max()),
            'global_pos_ratio': float(
                sum(r['n_pos'] for r in rows_clean) /
                sum(r['n_points'] for r in rows_clean)
            ),
            'total_pos': int(sum(r['n_pos'] for r in rows_clean)),
            'total_points': int(sum(r['n_points'] for r in rows_clean)),
        }

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved to {args.out}")


if __name__ == '__main__':
    main()