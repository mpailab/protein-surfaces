#!/usr/bin/env python3
"""
check_confidence_profile.py

Сравнивает "профиль уверенности" предсказаний разных чекпоинтов.
Не требует GPU (CPU-only), не мешает обучению.

Метрики:
  - mean y_prob по bin расстояния до партнёра (0-1, 1-2, ..., 8+)
  - Spearman correlation между y (dist) и y_prob
  - Contrast: prob(y<2) − prob(3<y<4)
  - Graph smoothness: mean((logit_i − logit_j)^2) на val

Пример:
    python check_confidence_profile.py \
        --models ./logs/smooth_l0/best_roc.pt \
                 ./logs/smooth_logit_l01/best_roc.pt \
        --data_dir ./processed/graphs_multiscale_new \
        --n_files 500 --n_val 50
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
from torch_geometric.data import Data


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
    num_layers = 0
    for k in state:
        if k.startswith('convs.'):
            num_layers = max(num_layers, int(k.split('.')[1]) + 1)
    model = SimpleGNN(in_dim, hidden, num_layers, heads=4, dropout=0.0).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_val_files(data_dir, n_files, n_val):
    all_paths = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    random.Random(42).shuffle(all_paths)
    return all_paths[n_files:n_files + n_val]


def infer_val(model, val_files, device='cpu'):
    """
    Возвращает:
        y_prob_flat:    [P] — вероятности на всех точках val
        y_dist_flat:    [P] — расстояние до партнёра (сырое y)
        smooth_edges:   mean((logit_i − logit_j)^2 * w) по всем рёбрам val
    """
    probs_all, dists_all = [], []
    smooth_sum, smooth_n = 0.0, 0

    with torch.no_grad():
        for f in val_files:
            d = torch.load(f, map_location='cpu', weights_only=False)
            x = d['x'].float().to(device)
            ei = d['edge_index'].long().to(device)
            ea = d['edge_attr'].float().to(device)
            y_dist = d['y'].float().numpy()

            logits = model(x, ei, ea)
            probs = torch.sigmoid(logits).cpu().numpy()

            probs_all.append(probs)
            dists_all.append(y_dist)

            # Smoothness (в logits, как в train)
            src, dst = ei[0], ei[1]
            diff2 = (logits[src] - logits[dst]) ** 2
            w = torch.exp(-ea[:, 0] / 2.0)   # tau=2 (как в train)
            smooth_sum += (w * diff2).mean().item() * len(diff2)
            smooth_n += len(diff2)

    probs_flat = np.concatenate(probs_all)
    dists_flat = np.concatenate(dists_all)
    smooth_mean = smooth_sum / max(smooth_n, 1)
    return probs_flat, dists_flat, smooth_mean


def profile_table(probs, dists, bins=None):
    """mean y_prob по bin дистанции."""
    if bins is None:
        bins = [0, 1, 2, 3, 4, 6, 8, 12, 20, 1e9]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-6', '6-8', '8-12', '12-20', '>20']
    rows = []
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        m = (dists >= lo) & (dists < hi)
        if m.sum() == 0:
            rows.append((lab, 0, np.nan))
            continue
        rows.append((lab, int(m.sum()), float(probs[m].mean())))
    return rows


def spearman_corr(a, b):
    """Быстрая реализация Spearman без scipy для разреженных данных."""
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True,
                    help='Список чекпоинтов для сравнения')
    ap.add_argument('--data_dir', default='./processed/graphs_multiscale_new')
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--n_val', type=int, default=50)
    ap.add_argument('--labels', nargs='*', default=None,
                    help='Опционально: имена для моделей')
    args = ap.parse_args()

    device = torch.device('cpu')
    val_files = get_val_files(args.data_dir, args.n_files, args.n_val)
    print(f"Val files: {len(val_files)}")
    print(f"Models:    {len(args.models)}")

    results = {}
    for i, mpath in enumerate(args.models):
        label = args.labels[i] if args.labels else os.path.basename(os.path.dirname(mpath))
        print(f"\n→ {label}: {mpath}")
        model = load_model(mpath, device)
        probs, dists, smooth = infer_val(model, val_files, device)
        results[label] = dict(probs=probs, dists=dists, smooth=smooth)

    # Профиль по дистанции
    print("\n" + "="*80)
    print("ПРОФИЛЬ y_prob ПО bin РАССТОЯНИЯ ДО ПАРТНЁРА")
    print("="*80)
    header = f"{'bin (Å)':<10} {'N':>8}"
    for lab in results:
        header += f"  {lab:>16}"
    print(header)
    print("-"*len(header))

    ref = next(iter(results.values()))
    rows_per_model = {}
    for lab, r in results.items():
        rows_per_model[lab] = profile_table(r['probs'], r['dists'])
    for i, (bin_lab, n, _) in enumerate(rows_per_model[list(results)[0]]):
        line = f"{bin_lab:<10} {n:>8}"
        for lab in results:
            _, _, v = rows_per_model[lab][i]
            line += f"  {v:>16.4f}"
        print(line)

    # Метрики
    print("\n" + "="*80)
    print("СВОДНЫЕ МЕТРИКИ")
    print("="*80)
    print(f"{'model':<20} {'spearman(dist, prob)':>22} {'contrast(y<2 − 3..4)':>22} {'graph smooth':>14}")
    for lab, r in results.items():
        # Только точки с y < 20 (иначе шум от далёкого фона)
        m = r['dists'] < 20
        rho = spearman_corr(r['dists'][m], r['probs'][m])
        # Contrast внутри интерфейса
        core = r['probs'][(r['dists'] < 2)]
        rim = r['probs'][(r['dists'] >= 3) & (r['dists'] < 4)]
        contrast = core.mean() - rim.mean()
        print(f"{lab:<20} {rho:>22.4f} {contrast:>22.4f} {r['smooth']:>14.6f}")

    print("\nИнтерпретация:")
    print("  spearman < 0  → чем ближе к партнёру, тем выше prob (хорошо)")
    print("  contrast > 0  → центр горячее края (хорошо)")
    print("  graph smooth  → степень согласованности соседей (меньше = глаже)")


if __name__ == '__main__':
    main()