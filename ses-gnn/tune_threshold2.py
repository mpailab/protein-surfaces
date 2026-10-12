#!/usr/bin/env python3
"""
tune_threshold.py — быстрая версия без sklearn в горячем цикле.

Пример:
    python tune_threshold.py \
        --scores ./eval_dmasif_k1/matched_scores.npz \
        --labels ./eval_dmasif_k1/their_labels.npz \
        --out ./eval_dmasif_k1/threshold_search.json \
        --criterion per_protein_f1
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


def eval_global_fast(labels, scores, th):
    preds = (scores > th).astype(np.int8)
    # ВАЖНО: dtype=np.int64 при sum — иначе int8 переполняется
    tp = int(((preds == 1) & (labels == 1)).sum(dtype=np.int64))
    fp = int(((preds == 1) & (labels == 0)).sum(dtype=np.int64))
    fn = int(((preds == 0) & (labels == 1)).sum(dtype=np.int64))
    tn = int(((preds == 0) & (labels == 0)).sum(dtype=np.int64))

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    # MCC — считаем через float, чтобы избежать переполнения при умножении
    tp_, fp_, fn_, tn_ = float(tp), float(fp), float(fn), float(tn)
    denom = np.sqrt((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
    mcc = ((tp_ * tn_ - fp_ * fn_) / denom) if denom else 0.0

    return {'f1': float(f1), 'precision': float(p), 'recall': float(r),
            'accuracy': float(acc), 'mcc': float(mcc)}


def eval_per_protein_fast(ids, labels, scores, th):
    f1s, mccs = [], []
    for cid in ids:
        y = labels[cid].astype(np.int8)
        p = (scores[cid] > th).astype(np.int8)

        # dtype=np.int64 при sum — обязательно
        tp = int(((p == 1) & (y == 1)).sum(dtype=np.int64))
        fp = int(((p == 1) & (y == 0)).sum(dtype=np.int64))
        fn = int(((p == 0) & (y == 1)).sum(dtype=np.int64))
        tn = int(((p == 0) & (y == 0)).sum(dtype=np.int64))

        if (tp + fn) == 0 and (tn + fp) == 0:
            continue

        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

        tp_, fp_, fn_, tn_ = float(tp), float(fp), float(fn), float(tn)
        denom = np.sqrt((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
        mcc = ((tp_ * tn_ - fp_ * fn_) / denom) if denom else 0.0

        f1s.append(f1)
        mccs.append(mcc)

    return {
        'f1_mean': float(np.mean(f1s)) if f1s else 0.0,
        'mcc_mean': float(np.mean(mccs)) if mccs else 0.0,
        'n_proteins': len(f1s),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--out', default='threshold_search.json')
    ap.add_argument('--criterion', default='per_protein_f1',
                    choices=['per_protein_f1', 'global_f1',
                             'global_mcc', 'per_protein_mcc'])
    ap.add_argument('--th_min', type=float, default=0.01)
    ap.add_argument('--th_max', type=float, default=0.99)
    ap.add_argument('--th_step', type=float, default=0.005)
    ap.add_argument('--save_top', type=int, default=10)
    args = ap.parse_args()

    scores = load_npz(args.scores)
    labels = load_npz(args.labels)
    ids = sorted(set(scores.keys()) & set(labels.keys()))
    print(f"📂 Proteins: {len(ids)}")

    all_lbl = np.concatenate([labels[c] for c in ids]).astype(np.int8)
    all_scr = np.concatenate([scores[c] for c in ids]).astype(np.float32)
    print(f"   total points: {len(all_lbl):,}")
    print(f"   pos_ratio:    {all_lbl.mean():.4f}")
    print(f"   global ROC-AUC: {roc_auc_score(all_lbl, all_scr):.4f}")
    print(f"   global PR-AUC:  {average_precision_score(all_lbl, all_scr):.4f}")

    thresholds = np.arange(args.th_min, args.th_max + 1e-9, args.th_step)
    print(f"\n🔍 Scanning {len(thresholds)} thresholds "
          f"[{args.th_min}, {args.th_max}] step {args.th_step}")

    rows = []
    for th in tqdm(thresholds, desc="Scanning"):
        g = eval_global_fast(all_lbl, all_scr, th)
        pp = eval_per_protein_fast(ids, labels, scores, th)
        rows.append({
            'threshold': float(th),
            **{f'global_{k}': v for k, v in g.items()},
            **{f'pp_{k}': v for k, v in pp.items()},
        })

    key_map = {
        'per_protein_f1': 'pp_f1_mean',
        'global_f1': 'global_f1',
        'global_mcc': 'global_mcc',
        'per_protein_mcc': 'pp_mcc_mean',
    }
    key = key_map[args.criterion]
    rows_sorted = sorted(rows, key=lambda r: r[key], reverse=True)
    best = rows_sorted[0]

    print(f"\n{'='*70}")
    print(f"🏆 BEST by {args.criterion} ({key})")
    print(f"{'='*70}")
    print(f"   threshold : {best['threshold']:.4f}")
    print(f"   global F1 : {best['global_f1']:.4f}")
    print(f"   global P  : {best['global_precision']:.4f}")
    print(f"   global R  : {best['global_recall']:.4f}")
    print(f"   global MCC: {best['global_mcc']:.4f}")
    print(f"   pp F1     : {best['pp_f1_mean']:.4f}")
    print(f"   pp MCC    : {best['pp_mcc_mean']:.4f}")

    print(f"\nTOP {args.save_top} thresholds by {key}:")
    print(f"   {'th':>6s}  {'gF1':>6s}  {'gP':>6s}  {'gR':>6s}  "
          f"{'gMCC':>7s}  {'ppF1':>6s}  {'ppMCC':>7s}")
    for r in rows_sorted[:args.save_top]:
        print(f"   {r['threshold']:6.3f}  "
              f"{r['global_f1']:6.3f}  "
              f"{r['global_precision']:6.3f}  "
              f"{r['global_recall']:6.3f}  "
              f"{r['global_mcc']:7.3f}  "
              f"{r['pp_f1_mean']:6.3f}  "
              f"{r['pp_mcc_mean']:7.3f}")

    print(f"\n(справочно) Альтернативные пороги:")
    for crit in ['global_f1', 'global_mcc', 'pp_f1_mean', 'pp_mcc_mean']:
        best_c = max(rows, key=lambda r: r[crit])
        print(f"   max {crit:18s}: th={best_c['threshold']:.3f}  "
              f"gF1={best_c['global_f1']:.3f}  "
              f"gMCC={best_c['global_mcc']:.3f}  "
              f"ppF1={best_c['pp_f1_mean']:.3f}")

    out = {
        'criterion': args.criterion,
        'best': best,
        'top': rows_sorted[:args.save_top],
        'all': rows,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved to {args.out}")


if __name__ == '__main__':
    main()