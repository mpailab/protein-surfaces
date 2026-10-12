#!/usr/bin/env python3
"""
bootstrap_iou.py

Paired bootstrap на белках для ΔAUC(IoU vs rho) между двумя моделями.

Использует ОДНИ И ТЕ ЖЕ resampled proteins для обеих моделей (paired),
потому что для каждого белка обе модели видели одинаковый вход.

Возвращает распределение ΔAUC, 95% CI, p-value.

Пример:
    python bootstrap_iou.py \
        --model1 ./logs/smooth_l0/best_roc.pt \
        --model2 ./logs/wbse_a05/best_roc.pt \
        --data_dir ./processed/graphs_multiscale_new \
        --n_files 500 --n_val 50 \
        --n_bootstrap 1000
"""

import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz


class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=32, num_layers=8, heads=4, dropout=0.0):
        super().__init__()
        head_dim = hidden // heads
        self.lin_in = nn.Linear(in_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransformerConv(hidden, head_dim, heads=heads,
                                              edge_dim=3, dropout=dropout,
                                              root_weight=True, beta=False))
            self.norms.append(nn.LayerNorm(hidden))
        self.lin_out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = F.gelu(self.norm_in(self.lin_in(x)))
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.gelu(x)
        return self.lin_out(x).squeeze(-1)


def load_model(path, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    hidden = state['lin_in.weight'].shape[0]
    in_dim = state['lin_in.weight'].shape[1]
    num_layers = max([int(k.split('.')[1]) + 1
                      for k in state if k.startswith('convs.')] or [8])
    model = SimpleGNN(in_dim, hidden, num_layers, heads=4, dropout=0.0).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_val_files(data_dir, n_files, n_val):
    all_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    random.Random(42).shuffle(all_paths)
    return all_paths[n_files:n_files + n_val]


def iou_core_at_quantile(probs, y_dist, rho, core_eps=1.5):
    """IoU между top-rho точек по prob и ядром (y < core_eps)."""
    t = np.quantile(probs, 1.0 - rho)
    pred = probs > t
    core = y_dist < core_eps
    inter = (pred & core).sum()
    union = (pred | core).sum()
    return float(inter / union) if union > 0 else 0.0


def auc_from_profiles(iou_matrix, rhos, x_min=0.05, x_max=0.70):
    """
    iou_matrix: [N_proteins, N_rhos] — среднее по resampled proteins.
    Возвращает AUC в диапазоне [x_min, x_max].
    """
    mean_iou = iou_matrix.mean(axis=0)   # [N_rhos]
    rhos_arr = np.asarray(rhos)
    mask = (rhos_arr >= x_min) & (rhos_arr <= x_max)
    if mask.sum() < 2:
        return 0.0
    order = np.argsort(rhos_arr[mask])
    return float(trapz(mean_iou[mask][order], rhos_arr[mask][order]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model1', required=True, help='baseline')
    ap.add_argument('--model2', required=True, help='новая модель (например α=0.5)')
    ap.add_argument('--data_dir', default='./processed/graphs_multiscale_new')
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--n_val', type=int, default=50)
    ap.add_argument('--n_bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--core_eps', type=float, default=1.5)
    ap.add_argument('--rho_grid', nargs='+', type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                             0.40, 0.45, 0.50, 0.60, 0.70])
    args = ap.parse_args()

    device = torch.device('cpu')
    val_files = get_val_files(args.data_dir, args.n_files, args.n_val)
    print(f"Val files:     {len(val_files)}")
    print(f"Bootstrap:     {args.n_bootstrap}")
    print(f"Model 1 (base): {args.model1}")
    print(f"Model 2:        {args.model2}")
    print()

    # Загрузка val
    val_raw = []
    for f in val_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        val_raw.append(dict(
            x=d['x'].float(),
            ei=d['edge_index'].long(),
            ea=d['edge_attr'].float(),
            y_dist=d['y'].numpy(),
        ))
    N = len(val_raw)

    # Inference обоих моделей
    def infer(model_path):
        model = load_model(model_path, device)
        probs_list = []
        with torch.no_grad():
            for d in val_raw:
                logits = model(d['x'], d['ei'], d['ea'])
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        return probs_list

    print("Inference model1...")
    probs1 = infer(args.model1)
    print("Inference model2...")
    probs2 = infer(args.model2)

    # Per-protein IoU на каждой rho для каждой модели
    rhos = args.rho_grid
    iou1 = np.zeros((N, len(rhos)), dtype=np.float32)
    iou2 = np.zeros((N, len(rhos)), dtype=np.float32)

    print("\nComputing per-protein IoU...")
    for i in range(N):
        y_dist = val_raw[i]['y_dist']
        for j, rho in enumerate(rhos):
            iou1[i, j] = iou_core_at_quantile(probs1[i], y_dist, rho,
                                              args.core_eps)
            iou2[i, j] = iou_core_at_quantile(probs2[i], y_dist, rho,
                                              args.core_eps)

    # Point estimates
    auc1 = auc_from_profiles(iou1, rhos)
    auc2 = auc_from_profiles(iou2, rhos)
    delta = auc2 - auc1

    print(f"\nТочечные оценки на всех {N} белках:")
    print(f"  AUC(IoU|rho) model1 = {auc1:.4f}")
    print(f"  AUC(IoU|rho) model2 = {auc2:.4f}")
    print(f"  ΔAUC              = {delta:+.4f}")

    # Paired bootstrap
    print(f"\nPaired bootstrap ({args.n_bootstrap} итераций)...")
    rng = np.random.RandomState(args.seed)
    deltas = np.zeros(args.n_bootstrap, dtype=np.float32)

    for b in range(args.n_bootstrap):
        idx = rng.choice(N, N, replace=True)  # paired: те же proteins
        a1 = auc_from_profiles(iou1[idx], rhos)
        a2 = auc_from_profiles(iou2[idx], rhos)
        deltas[b] = a2 - a1

    # Статистика
    mean_d = deltas.mean()
    std_d = deltas.std()
    q_lo, q_hi = np.percentile(deltas, [2.5, 97.5])
    # p-value: доля бутстрап-итераций с ΔAUC <= 0 (для односторонней H1: Δ>0)
    p_onesided = float((deltas <= 0).mean())
    # Двухсторонняя
    p_twosided = 2 * min(p_onesided, 1.0 - p_onesided)

    print()
    print("="*70)
    print("РЕЗУЛЬТАТ BOOTSTRAP")
    print("="*70)
    print(f"  ΔAUC (model2 − model1):")
    print(f"    point estimate: {delta:+.4f}")
    print(f"    bootstrap mean: {mean_d:+.4f}")
    print(f"    bootstrap std:  {std_d:.4f}")
    print(f"    95% CI:         [{q_lo:+.4f}, {q_hi:+.4f}]")
    print()
    print(f"  p-value (H1: Δ > 0):  {p_onesided:.4f}")
    print(f"  p-value (two-sided):  {p_twosided:.4f}")
    print()

    # Интерпретация
    print("Интерпретация:")
    if q_lo > 0:
        print("  ✅ 95% CI исключает 0 — эффект статистически значим")
    elif q_hi < 0:
        print("  ❌ 95% CI целиком ниже 0 — model2 хуже")
    elif p_onesided < 0.05:
        print(f"  ⚠️  95% CI включает 0, но p={p_onesided:.3f} < 0.05 "
              f"(one-sided)")
    else:
        print(f"  ⚠️  Эффект НЕ значим при 95% (p={p_onesided:.3f})")


if __name__ == '__main__':
    main()