#!/usr/bin/env python3
"""
residue_level_eval.py

Zero-shot residue-level оценка point-level модели.

Для каждого .pt:
  1. Инференс → поточечные скоры.
  2. Агрегация к residue_id: mean pooling, max pooling.
  3. Residue-level GT: остаток интерфейсный, если >= 1 точка has y_dmasif_binary=1.

Метрики: ROC-AUC, PR-AUC, F1, MCC (при подобранном пороге).

Пример:
    python residue_level_eval.py \
        --model ./logs/12x32h4_dmasif_v2/checkpoint_epoch_16.pt \
        --data_dir ./processed/graphs_test_dmasif_residue \
        --output_dir ./eval_residue_v2ep16 \
        --layers 12 --hidden 32 --heads 4 --in_dim 14
"""

import os
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, matthews_corrcoef
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─── Модель (копия из evaluate_on_dmasif.py) ────────────────────────
class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=32, num_layers=12, heads=4, dropout=0.1):
        super().__init__()
        head_dim = hidden // heads
        assert head_dim * heads == hidden

        self.lin_in = nn.Linear(in_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(hidden, head_dim, heads=heads, edge_dim=3,
                                dropout=dropout, root_weight=True, beta=False)
            )
            self.norms.append(nn.LayerNorm(hidden))

        self.lin_out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = F.gelu(self.norm_in(self.lin_in(x)))
        x = self.dropout(x)

        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)

        return self.lin_out(x).squeeze(-1)


# ─── Инференс ───────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, g, device):
    data = Data(
        x=g['x'].float().to(device),
        pos=g['pos'].float().to(device),
        edge_index=g['edge_index'].long().to(device),
        edge_attr=g['edge_attr'].float().to(device),
    )
    logits = model(data)
    return torch.sigmoid(logits).cpu().numpy()


# ─── Агрегация к residue-level ──────────────────────────────────────
def aggregate_to_residues(scores, residue_ids, n_res, mode='mean'):
    """
    scores: [N] поточечные скоры
    residue_ids: [N] long
    n_res: общее число остатков
    mode: 'mean' или 'max'

    Возвращает:
      res_scores: [R] float — агрегированный скор
      res_has:    [R] bool — есть ли у остатка хотя бы одна точка
    """
    res_scores = np.zeros(n_res, dtype=np.float32)
    res_counts = np.zeros(n_res, dtype=np.int32)

    if mode == 'mean':
        for r, s in zip(residue_ids, scores):
            res_scores[r] += s
            res_counts[r] += 1
        res_has = res_counts > 0
        res_scores[res_has] /= res_counts[res_has]
    elif mode == 'max':
        res_scores[:] = -1.0
        for r, s in zip(residue_ids, scores):
            if s > res_scores[r]:
                res_scores[r] = s
        res_has = res_scores >= 0.0
        res_scores[~res_has] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return res_scores, res_has


def build_residue_gt(y_binary, residue_ids, n_res):
    """
    Остаток интерфейсный, если хотя бы одна его точка имеет y=1.
    """
    res_gt = np.zeros(n_res, dtype=np.int32)
    for r, y in zip(residue_ids, y_binary):
        if y == 1:
            res_gt[r] = 1
    return res_gt


# ─── Метрики ────────────────────────────────────────────────────────
def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return None

    preds = (scores > threshold).astype(int)
    try:
        roc = float(roc_auc_score(labels, scores))
    except Exception:
        roc = 0.5
    try:
        pr = float(average_precision_score(labels, scores))
    except Exception:
        pr = 0.0
    return {
        'roc_auc': roc,
        'pr_auc': pr,
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'mcc': float(matthews_corrcoef(labels, preds)),
        'n_residues': int(len(labels)),
        'pos_ratio': float(labels.mean()),
    }


def find_best_threshold(labels, scores, metric='f1'):
    best_th = 0.5
    best_val = -1.0
    for th in np.arange(0.01, 0.99, 0.005):
        m = compute_metrics(labels, scores, threshold=th)
        if m is None:
            continue
        v = m[metric]
        if v > best_val:
            best_val = v
            best_th = th
    return best_th, best_val


# ─── Main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--heads', type=int, required=True)
    ap.add_argument('--in_dim', type=int, default=14)
    ap.add_argument('--max_files', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Модель
    print(f"📦 Loading model: {args.model}")
    state = torch.load(args.model, map_location=device, weights_only=False)
    model = SimpleGNN(
        in_dim=args.in_dim, hidden=args.hidden,
        num_layers=args.layers, heads=args.heads, dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # Файлы
    files = sorted(glob.glob(os.path.join(args.data_dir, '*.pt')))
    if args.max_files:
        files = files[:args.max_files]
    print(f"📂 Files: {len(files)}")

    all_res_mean_score = []
    all_res_max_score = []
    all_res_gt = []
    per_protein = []

    for f in tqdm(files, desc="Residue-level eval"):
        d = torch.load(f, map_location='cpu', weights_only=False)

        # Инференс
        scores = run_inference(model, d, device)
        rid = d['residue_id'].numpy()
        yb = d['y_dmasif_binary'].numpy().astype(int)
        n_res = d['num_residues']

        # Агрегация
        res_mean, res_has = aggregate_to_residues(scores, rid, n_res, mode='mean')
        res_max, _ = aggregate_to_residues(scores, rid, n_res, mode='max')

        # GT
        res_gt = build_residue_gt(yb, rid, n_res)

        # Только остатки с точками
        idx = np.where(res_has)[0]
        res_mean = res_mean[idx]
        res_max = res_max[idx]
        res_gt = res_gt[idx]

        if len(np.unique(res_gt)) < 2:
            continue

        all_res_mean_score.append(res_mean)
        all_res_max_score.append(res_max)
        all_res_gt.append(res_gt)

        # Per-protein
        per_protein.append({
            'id': os.path.basename(f).replace('.pt', ''),
            'n_residues': int(len(res_gt)),
            'pos_ratio': float(res_gt.mean()),
        })

    # Склеиваем
    all_mean = np.concatenate(all_res_mean_score)
    all_max = np.concatenate(all_res_max_score)
    all_gt = np.concatenate(all_res_gt)

    print(f"\n{'='*70}")
    print(f"RESIDUE-LEVEL RESULTS (zero-shot)")
    print(f"{'='*70}")
    print(f"  Total residues:  {len(all_gt):,}")
    print(f"  pos_ratio:       {all_gt.mean():.4f}")

    for mode_name, scores in [('MEAN pooling', all_mean),
                               ('MAX pooling',  all_max)]:
        print(f"\n── {mode_name} ──")
        # Threshold-free
        m_05 = compute_metrics(all_gt, scores, threshold=0.5)
        print(f"  threshold=0.5:")
        print(f"    ROC-AUC = {m_05['roc_auc']:.4f}")
        print(f"    PR-AUC  = {m_05['pr_auc']:.4f}")
        print(f"    F1      = {m_05['f1']:.4f}")
        print(f"    MCC     = {m_05['mcc']:.4f}")

        # Best threshold by F1
        best_th, best_f1 = find_best_threshold(all_gt, scores, metric='f1')
        m_best = compute_metrics(all_gt, scores, threshold=best_th)
        print(f"  best threshold = {best_th:.3f}:")
        print(f"    F1        = {m_best['f1']:.4f}")
        print(f"    Precision = {m_best['precision']:.4f}")
        print(f"    Recall    = {m_best['recall']:.4f}")
        print(f"    MCC       = {m_best['mcc']:.4f}")

    # Сохраняем
    out = {
        'config': vars(args),
        'summary': {
            'n_residues_total': int(len(all_gt)),
            'pos_ratio': float(all_gt.mean()),
            'mean_pooling': compute_metrics(all_gt, all_mean),
            'max_pooling': compute_metrics(all_gt, all_max),
            'mean_best_th': find_best_threshold(all_gt, all_mean, 'f1'),
            'max_best_th': find_best_threshold(all_gt, all_max, 'f1'),
        },
        'per_protein': per_protein,
    }
    with open(os.path.join(args.output_dir, 'residue_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()