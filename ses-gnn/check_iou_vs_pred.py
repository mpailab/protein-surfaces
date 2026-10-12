#!/usr/bin/env python3
"""
check_iou_vs_pred.py

Отвечает на вопрос: "улучшение IoU_core у WBCE — это реальное улучшение
ранжирования, или просто сдвиг абсолютных скоров?"

Ключевая идея: сравнивать модели при ОДИНАКОВОМ operating point (pred_ratio),
а не при одинаковом пороге.

Два способа:
  1. Global threshold sweep. Один порог t для всех белков. Для каждого t
     смотрим средний (pred_ratio, IoU_core). Учитывает и сдвиг калибровки,
     и качество ранжирования.

  2. Per-protein quantile. Для каждого белка порог = квантиль уровня (1-ρ),
     так что pred_ratio ≈ ρ. Усредняем IoU_core по белкам. Это
     калибровочно-инвариантный тест — учитывает ТОЛЬКО ранжирование
     внутри белка.

Если AUC(IoU vs pred%) в способе 2 у WBCE > baseline — улучшение реальное.
Если одинаково — эффект объясняется сдвигом калибровки.

Пример:
    python check_iou_vs_pred.py \
        --models ./logs/smooth_l0/best_roc.pt \
                 ./logs/wbce_a05/best_roc.pt \
                 ./logs/wbse_a10/best_roc.pt \
        --labels baseline wbce_a05 wbce_a10 \
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


def iou_core_at_threshold(probs, y_dist, threshold, core_eps=1.5):
    """IoU между точками с prob > threshold и ядром (y < core_eps)."""
    pred = probs > threshold
    core = y_dist < core_eps
    inter = (pred & core).sum()
    union = (pred | core).sum()
    return float(inter / union) if union > 0 else 0.0


def curve_global_threshold(val_data, thresholds):
    """Global threshold sweep: один порог для всех белков."""
    curve = []
    for t in thresholds:
        pr, ious = [], []
        for probs, y_dist in val_data:
            m = probs > t
            pr.append(m.mean())
            ious.append(iou_core_at_threshold(probs, y_dist, t))
        curve.append({
            'thr': t,
            'pred_ratio': float(np.mean(pr)),
            'iou_core': float(np.mean(ious)),
        })
    return curve


def curve_quantile(val_data, rhos):
    """Per-protein quantile: порог = квантиль, дающий pred_ratio ≈ rho."""
    curve = []
    for rho in rhos:
        ious = []
        for probs, y_dist in val_data:
            t = np.quantile(probs, 1.0 - rho)
            ious.append(iou_core_at_threshold(probs, y_dist, t))
        curve.append({
            'rho': rho,
            'iou_core': float(np.mean(ious)),
        })
    return curve


def auc_on_grid(xs, ys, x_min=0.05, x_max=0.70):
    """AUC(IoU vs pred_ratio) в диапазоне [x_min, x_max]."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    mask = (xs >= x_min) & (xs <= x_max)
    if mask.sum() < 2:
        return 0.0
    order = np.argsort(xs[mask])
    return float(trapz(ys[mask][order], xs[mask][order]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--data_dir', default='./processed/graphs_multiscale_new')
    ap.add_argument('--n_files', type=int, default=500)
    ap.add_argument('--n_val', type=int, default=50)
    ap.add_argument('--core_eps', type=float, default=1.5,
                    help='Радиус ядра интерфейса (y < core_eps, Å)')
    ap.add_argument('--thr_grid', nargs='+', type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                             0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80])
    ap.add_argument('--rho_grid', nargs='+', type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                             0.40, 0.45, 0.50, 0.60, 0.70])
    args = ap.parse_args()

    device = torch.device('cpu')
    val_files = get_val_files(args.data_dir, args.n_files, args.n_val)
    print(f"Val files:  {len(val_files)}")
    print(f"Core eps:   y < {args.core_eps} Å")

    model_names = args.labels if args.labels else [
        os.path.basename(os.path.dirname(m)) for m in args.models
    ]

    # Загрузить val данные один раз
    print("\n📦 Loading val data...")
    val_raw = []
    for f in val_files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        val_raw.append(dict(
            x=d['x'].float(),
            ei=d['edge_index'].long(),
            ea=d['edge_attr'].float(),
            y_dist=d['y'].numpy(),
        ))
    print(f"   loaded {len(val_raw)} proteins")

    # Инференс каждой модели
    all_probs = {}
    for mpath, mname in zip(args.models, model_names):
        print(f"\n📦 Inference: {mname}")
        model = load_model(mpath, device)
        probs_list = []
        with torch.no_grad():
            for d in val_raw:
                logits = model(d['x'], d['ei'], d['ea'])
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        all_probs[mname] = probs_list
        print(f"   done")

    def get_val_data(mname):
        return [(all_probs[mname][i], val_raw[i]['y_dist'])
                for i in range(len(val_raw))]

    # ── СПОСОБ 1: Global threshold ─────────────────────────────────
    print("\n" + "="*100)
    print("СПОСОБ 1: ГЛОБАЛЬНЫЙ ПОРОГ (учитывает и сдвиг калибровки, и ранжирование)")
    print("="*100)

    curves_global = {m: curve_global_threshold(get_val_data(m), args.thr_grid)
                     for m in model_names}

    # Header
    header = f"{'thr':>6}"
    for m in model_names:
        header += f"  {m[:10]+' pred%':>16}  {m[:10]+' IoU':>14}"
    print("\n" + header)
    print("-" * len(header))

    for i, t in enumerate(args.thr_grid):
        line = f"{t:>6.2f}"
        for m in model_names:
            c = curves_global[m][i]
            line += f"  {c['pred_ratio']*100:>15.2f}%  {c['iou_core']:>14.4f}"
        print(line)

    print(f"\nAUC(IoU vs pred%) на [5%, 70%]:")
    for m in model_names:
        pr = [c['pred_ratio'] for c in curves_global[m]]
        iou = [c['iou_core'] for c in curves_global[m]]
        auc = auc_on_grid(pr, iou)
        print(f"   {m:>16}: {auc:.4f}")

    # ── СПОСОБ 2: Per-protein quantile ─────────────────────────────
    print("\n" + "="*100)
    print("СПОСОБ 2: PER-PROTEIN QUANTILE (только ранжирование — сдвиг убирается)")
    print("="*100)

    curves_q = {m: curve_quantile(get_val_data(m), args.rho_grid)
                for m in model_names}

    header = f"{'rho':>6}"
    for m in model_names:
        header += f"  {m[:12]+' IoU':>16}"
    print("\n" + header)
    print("-" * len(header))

    for i, rho in enumerate(args.rho_grid):
        line = f"{rho:>6.2f}"
        for m in model_names:
            line += f"  {curves_q[m][i]['iou_core']:>16.4f}"
        print(line)

    print(f"\nAUC(IoU vs rho) на [0.05, 0.70]:")
    aucs_q = {}
    for m in model_names:
        rho = [c['rho'] for c in curves_q[m]]
        iou = [c['iou_core'] for c in curves_q[m]]
        auc = auc_on_grid(rho, iou)
        aucs_q[m] = auc
        print(f"   {m:>16}: {auc:.4f}")

    # ── Сводка ─────────────────────────────────────────────────────
    print("\n" + "="*100)
    print("СВОДКА")
    print("="*100)
    print("Способ 1: AUC(pred%) — учитывает и калибровку, и ранжирование")
    print("Способ 2: AUC(rho)    — учитывает ТОЛЬКО ранжирование внутри белка")
    print()

    if len(model_names) >= 2:
        base = model_names[0]
        print(f"Сравнение с baseline = {base}:")
        for m in model_names[1:]:
            d1 = aucs_q[m] - aucs_q[base]
            verdict = ("РЕАЛЬНОЕ улучшение" if d1 > 0.005
                       else "статистически значимо (умеренное)" if d1 > 0.002
                       else "только сдвиг калибровки" if abs(d1) < 0.002
                       else "хуже baseline")
            print(f"   {m}: ΔAUC(quantile) = {d1:+.4f}   → {verdict}")
        print()
        print("Интерпретация:")
        print("  ΔAUC(quantile) > +0.005  →  WBCE реально улучшает ранжирование")
        print("  |ΔAUC(quantile)| < 0.003 →  эффект объясним сдвигом скоров")
        print("  ΔAUC(quantile) < −0.005  →  WBCE ухудшает ранжирование")


if __name__ == '__main__':
    main()