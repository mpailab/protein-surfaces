#!/usr/bin/env python3
"""
residue_level_eval_deepppips.py

Zero-shot residue-level оценка на DeepPPIPS test.
Использует point_gt и residue_gt из graphs_deepppips_test_labeled.
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


class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=32, num_layers=12, heads=4, dropout=0.1):
        super().__init__()
        head_dim = hidden // heads
        self.lin_in = nn.Linear(in_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(hidden, head_dim, heads=heads, edge_dim=3,
                                dropout=dropout, root_weight=True, beta=False))
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


@torch.no_grad()
def run_inference(model, g, device):
    data = Data(
        x=g['x'].float().to(device),
        pos=g['pos'].float().to(device),
        edge_index=g['edge_index'].long().to(device),
        edge_attr=g['edge_attr'].float().to(device),
    )
    return torch.sigmoid(model(data)).cpu().numpy()


def aggregate_to_residues(scores, residue_ids, n_res, mode='mean'):
    res_scores = np.zeros(n_res, dtype=np.float32)
    res_counts = np.zeros(n_res, dtype=np.int32)
    if mode == 'mean':
        for r, s in zip(residue_ids, scores):
            res_scores[r] += s
            res_counts[r] += 1
        has = res_counts > 0
        res_scores[has] /= res_counts[has]
    elif mode == 'max':
        res_scores[:] = -1.0
        for r, s in zip(residue_ids, scores):
            if s > res_scores[r]:
                res_scores[r] = s
        has = res_scores >= 0.0
        res_scores[~has] = 0.0
    return res_scores, has


def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return None
    preds = (scores > threshold).astype(int)
    try:
        roc = float(roc_auc_score(labels, scores))
    except:
        roc = 0.5
    try:
        pr = float(average_precision_score(labels, scores))
    except:
        pr = 0.0
    return {
        'roc_auc': roc, 'pr_auc': pr,
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'mcc': float(matthews_corrcoef(labels, preds)),
        'n_points': int(len(labels)),
        'pos_ratio': float(labels.mean()),
    }


def find_best_threshold(labels, scores, metric='f1'):
    best_th, best_val = 0.5, -1.0
    for th in np.arange(0.01, 0.99, 0.005):
        m = compute_metrics(labels, scores, threshold=th)
        if m and m[metric] > best_val:
            best_val = m[metric]
            best_th = th
    return best_th, best_val

def compute_metrics_fast(labels, scores, roc_auc, pr_auc, threshold=0.5):
    """Быстрые метрики: ROC/PR переданы снаружи."""
    preds = (scores > threshold).astype(int)
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'mcc': float(matthews_corrcoef(labels, preds)),
        'n_points': int(len(labels)),
        'pos_ratio': float(labels.mean()),
    }


def find_best_threshold_fast(labels, scores, roc_auc, pr_auc, metric='f1'):
    """ROC/PR считаются один раз, перебираем только порог."""
    best_th, best_val = 0.5, -1.0
    for th in np.arange(0.01, 0.99, 0.005):
        m = compute_metrics_fast(labels, scores, roc_auc, pr_auc, threshold=th)
        v = m[metric]
        if v > best_val:
            best_val = v
            best_th = th
    return best_th, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--heads', type=int, required=True)
    ap.add_argument('--in_dim', type=int, default=14)
    ap.add_argument('--subset', default='all',
                    choices=['all', 'clean', 'overlap'],
                    help='all = все, clean = без dMaSIF train overlap, overlap = только с overlap')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Subset
    if args.subset != 'all':
        list_file = f'deepppips_test_{args.subset}_proteins.txt'
        with open(list_file) as f:
            subset_names = set(l.strip() for l in f if l.strip())
        print(f"Using subset '{args.subset}': {len(subset_names)} proteins")
    else:
        subset_names = None

    state = torch.load(args.model, map_location=device, weights_only=False)
    model = SimpleGNN(in_dim=args.in_dim, hidden=args.hidden,
                      num_layers=args.layers, heads=args.heads, dropout=0.0).to(device)
    model.load_state_dict(state)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.data_dir, '*.pt')))
    if subset_names:
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in subset_names]
    print(f"Files: {len(files)}")

    # Point-level и residue-level сбор
    all_point_scores = []
    all_point_gt = []
    all_res_mean = []
    all_res_max = []
    all_res_gt = []

    for f in tqdm(files, desc="Evaluating"):
        d = torch.load(f, map_location='cpu', weights_only=False)
        scores = run_inference(model, d, device)
        rid = d['residue_id'].numpy()
        point_gt = d['point_gt'].numpy()
        residue_gt = d['residue_gt'].numpy()
        n_res = d['num_residues']

        # Point-level
        all_point_scores.append(scores)
        all_point_gt.append(point_gt)

        # Residue-level
        res_mean, has = aggregate_to_residues(scores, rid, n_res, 'mean')
        res_max, _ = aggregate_to_residues(scores, rid, n_res, 'max')

        idx = np.where(has)[0]
        all_res_mean.append(res_mean[idx])
        all_res_max.append(res_max[idx])
        all_res_gt.append(residue_gt[idx])

    # Объединяем
    pt_scores = np.concatenate(all_point_scores)
    pt_gt = np.concatenate(all_point_gt)
    res_mean_s = np.concatenate(all_res_mean)
    res_max_s = np.concatenate(all_res_max)
    res_gt = np.concatenate(all_res_gt)

    print(f"\n{'='*70}")
    print(f"Subset: {args.subset}")
    print(f"Point-level points:     {len(pt_gt):,} (pos_ratio={pt_gt.mean():.4f})")
    print(f"Residue-level residues: {len(res_gt):,} (pos_ratio={res_gt.mean():.4f})")

    # ── Point-level (быстро) ──
    pt_roc = roc_auc_score(pt_gt, pt_scores)
    pt_pr = average_precision_score(pt_gt, pt_scores)
    pt_best_th, _ = find_best_threshold_fast(pt_gt, pt_scores, pt_roc, pt_pr, 'f1')
    pt_best = compute_metrics_fast(pt_gt, pt_scores, pt_roc, pt_pr, pt_best_th)

    print(f"\n── POINT-LEVEL ──")
    print(f"  ROC-AUC: {pt_roc:.4f}")
    print(f"  PR-AUC:  {pt_pr:.4f}")
    print(f"  best threshold = {pt_best_th:.3f}:")
    print(f"    F1  = {pt_best['f1']:.4f}")
    print(f"    MCC = {pt_best['mcc']:.4f}")

    # ── Residue mean ──
    rm_roc = roc_auc_score(res_gt, res_mean_s)
    rm_pr = average_precision_score(res_gt, res_mean_s)
    rm_th, _ = find_best_threshold_fast(res_gt, res_mean_s, rm_roc, rm_pr, 'f1')
    rm_best = compute_metrics_fast(res_gt, res_mean_s, rm_roc, rm_pr, rm_th)

    print(f"\n── RESIDUE MEAN ──")
    print(f"  ROC-AUC: {rm_roc:.4f}")
    print(f"  PR-AUC:  {rm_pr:.4f}")
    print(f"  best threshold = {rm_th:.3f}:")
    print(f"    F1  = {rm_best['f1']:.4f}")
    print(f"    MCC = {rm_best['mcc']:.4f}")

    # ── Residue max ──
    rx_roc = roc_auc_score(res_gt, res_max_s)
    rx_pr = average_precision_score(res_gt, res_max_s)
    rx_th, _ = find_best_threshold_fast(res_gt, res_max_s, rx_roc, rx_pr, 'f1')
    rx_best = compute_metrics_fast(res_gt, res_max_s, rx_roc, rx_pr, rx_th)

    print(f"\n── RESIDUE MAX ──")
    print(f"  ROC-AUC: {rx_roc:.4f}")
    print(f"  PR-AUC:  {rx_pr:.4f}")
    print(f"  best threshold = {rx_th:.3f}:")
    print(f"    F1  = {rx_best['f1']:.4f}")
    print(f"    MCC = {rx_best['mcc']:.4f}")

    # Сохраняем
    out = {
        'config': vars(args),
        'point_level': {'metrics': pt_best, 'best_th': pt_best_th},
        'residue_mean': {'metrics': rm_best, 'best_th': rm_th},
        'residue_max':  {'metrics': rx_best, 'best_th': rx_th},
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()