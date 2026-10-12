#!/usr/bin/env python3
"""
check_threshold_dynamics.py

Анализ того, как пятно предсказанного интерфейса меняется с порогом.
Ключевой вопрос: "сжимается ли пятно от краёв к центру при росте порога,
или распадается на куски?"

Метрики для каждого порога t:
  - pred_ratio       = доля точек с prob > t
  - n_components     = число связных компонент в пятне
  - largest_ratio    = (размер крупнейшей) / (всего в пятне)
  - IoU_core         = пересечение с GT-центром (y < 1.5) / объединение

Пример:
    CUDA_VISIBLE_DEVICES='' python -u check_threshold_dynamics.py \
        --models ./logs/smooth_l0/best_roc.pt \
                 ./logs/wbce_a05/best_roc.pt \
        --labels baseline wbce_a05 \
        --data_dir ./processed/graphs_multiscale_new \
        --n_files 500 --n_val 50 \
        --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7
"""

import os
import glob
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.nn import TransformerConv


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


def analyze_threshold(probs, y_dist, edge_index, threshold, core_eps=1.5):
    """
    Для одного белка и одного порога возвращает метрики связности пятна.
    """
    mask = probs > threshold
    n_pred = int(mask.sum())
    if n_pred < 2:
        return dict(pred_ratio=float(mask.mean()), n_comp=0,
                    largest_ratio=0.0, iou_core=0.0, n_pred=n_pred)

    # Подграф на предсказанных точках
    ei = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
    src, dst = ei[0], ei[1]
    keep = mask[src] & mask[dst]
    sub_src, sub_dst = src[keep], dst[keep]

    n = len(probs)
    adj = csr_matrix((np.ones(len(sub_src)), (sub_src, sub_dst)), shape=(n, n))
    adj = adj + adj.T
    n_comp, labels = connected_components(adj[mask][:, mask], directed=False)
    comp_sizes = np.bincount(labels)
    largest = int(comp_sizes.max())

    # IoU с "ядром" интерфейса (y < core_eps)
    core = y_dist < core_eps
    inter = (mask & core).sum()
    union = (mask | core).sum()
    iou_core = float(inter / union) if union > 0 else 0.0

    return dict(
        pred_ratio=float(mask.mean()),
        n_comp=int(n_comp),
        largest_ratio=float(largest / n_pred),
        iou_core=iou_core,
        n_pred=n_pred,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--data_dir', default='./processed/graphs_multiscale_new')
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--n_val', type=int, default=50)
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument('--core_eps', type=float, default=1.5,
                    help='Радиус "ядра" интерфейса (Å)')
    args = ap.parse_args()

    device = torch.device('cpu')
    val_files = get_val_files(args.data_dir, args.n_files, args.n_val)
    print(f"Val files: {len(val_files)}")

    # Собираем статистику: для каждой модели × каждого порога — среднее по белкам
    model_names = args.labels if args.labels else [
        os.path.basename(os.path.dirname(m)) for m in args.models
    ]

    results = {name: {t: [] for t in args.thresholds} for name in model_names}

    for mpath, mname in zip(args.models, model_names):
        print(f"\n→ {mname}: {mpath}")
        model = load_model(mpath, device)

        for f in val_files:
            d = torch.load(f, map_location='cpu', weights_only=False)
            with torch.no_grad():
                logits = model(
                    d['x'].float(), d['edge_index'].long(), d['edge_attr'].float()
                )
                probs = torch.sigmoid(logits).cpu().numpy()
            y_dist = d['y'].numpy()
            ei_raw = d['edge_index']
            ei = ei_raw.numpy() if torch.is_tensor(ei_raw) else ei_raw

            for t in args.thresholds:
                m = analyze_threshold(probs, y_dist, ei, t, args.core_eps)
                results[mname][t].append(m)

    # Агрегация
    def agg(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals))

    print("\n" + "="*94)
    print("ДИНАМИКА ПЯТНА ПО ПОРОГУ")
    print("="*94)
    header = f"{'model':<14} {'thr':>5} {'pred%':>7} {'n_comp':>7} {'largest':>8} {'IoU_core':>10}"
    print(header)
    print("-"*len(header))
    for mname in model_names:
        for t in args.thresholds:
            rows = results[mname][t]
            pred = agg(rows, 'pred_ratio')
            ncomp = agg(rows, 'n_comp')
            largest = agg(rows, 'largest_ratio')
            iou = agg(rows, 'iou_core')
            print(f"{mname:<14} {t:>5.2f} {pred*100:>6.2f}% {ncomp:>7.1f} "
                  f"{largest:>8.3f} {iou:>10.4f}")
        print("-"*len(header))

    print("\nИнтерпретация:")
    print("  largest → 1.0   пятно цельное (хорошо)")
    print("  largest < 0.5   пятно фрагментировано (плохо)")
    print("  IoU_core растёт при росте thr — пятно сжимается к ядру (хорошо)")
    print("  n_comp растёт при росте thr — пятно распадается (плохо)")


if __name__ == '__main__':
    main()