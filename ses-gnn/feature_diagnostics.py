#!/usr/bin/env python3
"""
feature_diagnostics.py

Сравнение трёх моделей на ОДНИХ И ТЕХ ЖЕ признаках вершины (14D):
  1. LogisticRegression  — линейная разделимость
  2. MLP (1 hidden layer) — нелинейная разделимость БЕЗ графа
  3. GNN                  — нелинейность + топология (референс)

Цель: разложить вклад GNN на "нелинейность признаков" и "топология графа".

Пример:
    python feature_diagnostics.py \
        --graph_dir ./processed/graphs_multiscale \
        --out_json ./diag_own_ppi.json \
        --n_files 50 \
        --label_key y --label_threshold 4.0

    python feature_diagnostics.py \
        --graph_dir ./processed/graphs_test_dmasif_residue \
        --out_json ./diag_dmasif.json \
        --n_files 50 \
        --label_key y_dmasif_binary --label_threshold 0.5

    python feature_diagnostics.py \
        --graph_dir ./processed/graphs_deepppips_train_labeled \
        --out_json ./diag_deepppips.json \
        --n_files 50 \
        --label_key point_gt --label_threshold 0.5
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def is_ppi(d):
    return d.get('mode', 'ppi') == 'ppi'

# ───────────────────────────────────────────────────────────────────────
# Загрузка признаков и меток
# ───────────────────────────────────────────────────────────────────────
def load_features(graph_dir, n_files, label_key='y',
                  label_threshold=4.0, only_ppi=False):
    """
    Возвращает X [N, 14], y [N] (int), список файлов.
    Если only_ppi=True — пропускаем графы с mode != 'ppi'.
    """
    files = sorted(glob.glob(os.path.join(graph_dir, '*.pt')))
    if only_ppi:
        files = [f for f in files
                 if torch.load(f, map_location='cpu',
                               weights_only=False).get('mode', 'ppi') == 'ppi']
    files = files[:n_files]

    Xs, ys = [], []
    n_skipped = 0

    for f in tqdm(files, desc="Loading"):
        d = torch.load(f, map_location='cpu', weights_only=False)

        if only_ppi and d.get('mode', 'ppi') != 'ppi':
            n_skipped += 1
            continue

        if label_key not in d:
            print(f"⚠️  {os.path.basename(f)}: no key '{label_key}'")
            continue
        x = d['x'].numpy().astype(np.float32)
        y_raw = d[label_key].numpy().astype(np.float32)

        uniq = np.unique(y_raw)
        if set(uniq.tolist()).issubset({0.0, 1.0}):
            y = y_raw.astype(np.int64)
        else:
            y = (y_raw < label_threshold).astype(np.int64)

        Xs.append(x)
        ys.append(y)

    if n_skipped:
        print(f"   Skipped {n_skipped} non-PPI graphs")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, files


class MLP(nn.Module):
    def __init__(self, in_dim=14, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for i in range(num_layers):
            # Сжимаем ширину: 64 → 32 → 16 ...
            h = max(16, hidden // (2 ** i))
            layers += [
                nn.Linear(prev, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.constant_(self.net[-1].bias, -1.5)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, hidden=64, num_layers=1,
              epochs=50, lr=1e-3, batch_size=8192, device='cuda'):
    model = MLP(in_dim=X_train.shape[1], hidden=hidden,
                num_layers=num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).to(device)

    n = len(Xt)
    best_val_roc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(Xv)).cpu().numpy()
            if len(np.unique(y_val)) > 1:
                roc = roc_auc_score(y_val, probs)
                if roc > best_val_roc:
                    best_val_roc = roc
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_roc


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_dir', required=True)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--n_files', type=int, default=50,
                    help='Число графов для выборки')
    ap.add_argument('--val_split', type=float, default=0.2,
                    help='Доля файлов на валидацию (split по файлам!)')
    ap.add_argument('--label_key', default='y')
    ap.add_argument('--label_threshold', type=float, default=4.0)
    ap.add_argument('--mlp_hidden', type=int, default=64)
    ap.add_argument('--mlp_layers', type=int, default=1,
                    help='Число скрытых слоёв в MLP')
    ap.add_argument('--mlp_epochs', type=int, default=50)
    ap.add_argument('--only_ppi', action='store_true',
                    help='Использовать только графы с mode=ppi '
                         '(для Own PPI-датасета)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 1. Загрузка с разбиением по ФАЙЛАМ (важно!) ──────────────────
    files = sorted(glob.glob(os.path.join(args.graph_dir, '*.pt')))

    if args.only_ppi:
        print(f"📂 Filtering PPI (scanning {len(files)} files)...")
        files_ppi = []
        for f in tqdm(files, desc="Scanning mode"):
            try:
                # Грузим ТОЛЬКО метаданные: mode и ничего больше.
                # torch.load возвращает весь объект, но мы его сразу выбрасываем.
                d = torch.load(f, map_location='cpu', weights_only=False)
                is_ppi = d.get('mode', 'ppi') == 'ppi'
                del d                            # <-- критично: сразу освобождаем
            except Exception:
                continue
            if is_ppi:
                files_ppi.append(f)
        print(f"   → {len(files_ppi)} / {len(files)} are PPI")
        files = files_ppi

    files = files[:args.n_files]
    n_val = max(1, int(len(files) * args.val_split))
    train_files = files[:-n_val]
    val_files = files[-n_val:]

    print(f"Files: {len(files)} (train={len(train_files)}, val={len(val_files)})")

    def _load_one(f):
        d = torch.load(f, map_location='cpu', weights_only=False)
        x = d['x'].numpy().astype(np.float32)
        y_raw = d[args.label_key].numpy().astype(np.float32)
        if set(np.unique(y_raw).tolist()).issubset({0.0, 1.0}):
            y = y_raw.astype(np.int64)
        else:
            y = (y_raw < args.label_threshold).astype(np.int64)
        del d
        return x, y

    Xs_tr, ys_tr = [], []
    for f in train_files:
        x, y = _load_one(f)
        Xs_tr.append(x); ys_tr.append(y)
    X_tr = np.concatenate(Xs_tr, axis=0)
    y_tr = np.concatenate(ys_tr, axis=0)

    Xs_va, ys_va = [], []
    for f in val_files:
        x, y = _load_one(f)
        Xs_va.append(x); ys_va.append(y)
    X_va = np.concatenate(Xs_va, axis=0)
    y_va = np.concatenate(ys_va, axis=0)

    print(f"Train points: {len(X_tr):,}  (pos={y_tr.mean():.3f})")
    print(f"Val points:   {len(X_va):,}  (pos={y_va.mean():.3f})")

    # ── 2. LogReg ────────────────────────────────────────────────────
    print("\n[1/3] LogisticRegression...")
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_va_s = scaler.transform(X_va)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(X_tr_s, y_tr)
    p_lr = clf.predict_proba(X_va_s)[:, 1]
    roc_lr = roc_auc_score(y_va, p_lr)
    pr_lr = average_precision_score(y_va, p_lr)
    print(f"   ROC={roc_lr:.4f}  PR={pr_lr:.4f}")

    # ── 3. MLP ───────────────────────────────────────────────────────
    print(f"\n[2/3] MLP (hidden={args.mlp_hidden})...")
    model, best_val_roc_during = train_mlp(
        X_tr, y_tr, X_va, y_va,
        hidden=args.mlp_hidden, num_layers=args.mlp_layers,
        epochs=args.mlp_epochs, device=device,
    )
    model.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(
            model(torch.from_numpy(X_va).to(device))
        ).cpu().numpy()
    roc_mlp = roc_auc_score(y_va, p_mlp)
    pr_mlp = average_precision_score(y_va, p_mlp)
    print(f"   ROC={roc_mlp:.4f}  PR={pr_mlp:.4f}")

    # ── 4. Сохранение ────────────────────────────────────────────────
    out = {
        'config': vars(args),
        'n_train_points': int(len(X_tr)),
        'n_val_points': int(len(X_va)),
        'pos_ratio_train': float(y_tr.mean()),
        'pos_ratio_val': float(y_va.mean()),
        'logreg': {'roc_auc': float(roc_lr), 'pr_auc': float(pr_lr)},
        'mlp':    {'roc_auc': float(roc_mlp), 'pr_auc': float(pr_mlp),
                   'hidden': args.mlp_hidden,
                   'num_layers': args.mlp_layers},    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved: {args.out_json}")

    # ── 5. Сводка ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY — {os.path.basename(args.graph_dir)}")
    print(f"{'='*60}")
    print(f"  LogReg:  ROC = {roc_lr:.4f}")
    print(f"  MLP-1:   ROC = {roc_mlp:.4f}   (Δ vs LogReg = {roc_mlp-roc_lr:+.4f})")
    print(f"  GNN:     (см. отдельные логи)")


if __name__ == '__main__':
    main()