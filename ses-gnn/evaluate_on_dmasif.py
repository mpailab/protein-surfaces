#!/usr/bin/env python3
"""
evaluate_on_dmasif.py

Оценка нашей модели на данных dMaSIF с ограничением длины матчинга.

Логика:
  1. Читаем testing_list.txt → ищем .ply и .pt
  2. Инференс нашей модели → per-point scores на наших точках
  3. Для каждой ИХ точки находим ближайшую НАШУ через KD-tree
  4. Если distance > match_max_dist → скор = 0 (unmatched)
  5. Метрики по ИХ разметке и нашим скорам

Пример:
    python evaluate_on_dmasif.py \
        --model best_f1_12x32h4_scratch.pt \
        --test_list testing_filtered.txt \
        --dmasif_dir testing-ply \
        --our_graph_dir ./processed/graphs_test_single \
        --output_dir ./eval_dmasif_k1_d15 \
        --label_field iface \
        --layers 12 --hidden 32 --heads 4 --in_dim 14 \
        --threshold 0.15 \
        --k 1 --match_max_dist 1.5 \
        --save_matched
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from scipy.spatial import cKDTree
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════
# МОДЕЛЬ
# ═══════════════════════════════════════════════════════════════════════
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.lin_out.bias, -0.85)

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


# ═══════════════════════════════════════════════════════════════════════
# FILE IO
# ═══════════════════════════════════════════════════════════════════════
def read_list(path):
    with open(path) as f:
        ids = [os.path.splitext(ln.strip())[0] for ln in f if ln.strip()]
    return ids


def find_file(search_dir, cid, ext, extra_templates=None):
    p = os.path.join(search_dir, f"{cid}.{ext}")
    if os.path.isfile(p):
        return p
    m = sorted(glob.glob(os.path.join(search_dir, f"{cid}_*.{ext}")))
    if m:
        return m[0]
    m = sorted(glob.glob(os.path.join(search_dir, '**', f"{cid}*.{ext}"),
                            recursive=True))
    if m:
        return m[0]
    if extra_templates:
        for tmpl in extra_templates:
            m = sorted(glob.glob(os.path.join(search_dir,
                                               tmpl.format(id=cid, ext=ext)),
                                  recursive=True))
            if m:
                return m[0]
    return None


def read_ply(ply_path, label_field='iface'):
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    v = ply['vertex']
    fields = {pr.name for pr in v.properties}

    pts = np.stack([
        np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])
    ], axis=1).astype(np.float32)

    labels = None
    if label_field in fields and label_field not in ('x', 'y', 'z'):
        labels = np.asarray(v[label_field]).astype(np.float32)
    else:
        for name in ['iface', 'label', 'labels', 'gt', 'y_label']:
            if name in fields:
                labels = np.asarray(v[name]).astype(np.float32)
                break
    if labels is not None and np.unique(labels).size > 2:
        labels = (labels > 0.5).astype(np.float32)
    return pts, labels


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(model, g, device, smooth_iters=2, smooth_alpha=0.5):
    """
    Запуск модели на графе с опциональным сглаживанием скоров на графе.
    
    Сглаживание усредняет скоры вершины с её соседями по графу:
      probs_new[v] = (1-alpha) * probs[v] + alpha * mean(probs[u] for u in neighbors(v))
    
    Это:
      - Гасит одиночные выбросы (isolated FP)
      - Усиливает связные области (правильный интерфейс — contiguous)
      - Стабилизирует калибровку
    
    Args:
        model: обученная SimpleGNN
        g: dict с ключами 'x', 'pos', 'edge_index', 'edge_attr'
        device: torch.device
        smooth_iters: число итераций сглаживания (0 = отключено)
        smooth_alpha: вес соседей, 0..1 (0 = без сглаживания, 1 = только соседи)
    
    Returns:
        np.ndarray [N] — сглаженные вероятности
    """
    data = Data(
        x=g['x'].float().to(device),
        pos=g['pos'].float().to(device),
        edge_index=g['edge_index'].long().to(device),
        edge_attr=g['edge_attr'].float().to(device),
    )
    
    # Forward pass
    logits = model(data)
    probs = torch.sigmoid(logits)
    
    # Опциональное сглаживание на графе
    if smooth_iters > 0:
        edge_index = data.edge_index
        src, dst = edge_index
        
        for _ in range(smooth_iters):
            agg = torch.zeros_like(probs)
            cnt = torch.zeros_like(probs)
            
            # Агрегация соседей: для каждой вершины dst суммируем probs её соседей src
            agg.index_add_(0, dst, probs[src])
            cnt.index_add_(0, dst, torch.ones_like(probs[src]))
            
            # Среднее по соседям (защита от деления на ноль для изолированных вершин)
            has_neighbors = cnt > 0
            neighbor_mean = torch.where(
                has_neighbors,
                agg / cnt.clamp(min=1),
                probs,  # для изолированных вершин оставляем исходное значение
            )
            
            # Интерполяция: собственный скор + среднее соседей
            probs = (1 - smooth_alpha) * probs + smooth_alpha * neighbor_mean
    
    return probs.cpu().numpy()
# ═══════════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════════
def match_scores(our_points, our_scores, their_points,
                 k=1, weighting='inverse_distance', eps=1e-6,
                 max_dist=None):
    """
    Для каждой их точки находим k ближайших наших.
    Если max_dist задан и distance > max_dist → скор = 0 (unmatched).
    """
    tree = cKDTree(our_points)
    dists, idxs = tree.query(their_points, k=k)

    if k == 1:
        matched = our_scores[idxs].copy()
        nearest_d = dists.copy()
    else:
        if weighting == 'inverse_distance':
            w = 1.0 / (dists + eps)
        elif weighting == 'softmax':
            w = np.exp(-dists / 0.5)
        else:
            w = np.ones_like(dists)
        w = w / w.sum(axis=1, keepdims=True)
        matched = (our_scores[idxs] * w).sum(axis=1)
        nearest_d = dists[:, 0]

    # Отсечение: unmatched → 0
    if max_dist is not None:
        unmatched_mask = nearest_d > max_dist
        matched = matched.copy()
        matched[unmatched_mask] = 0.0
    else:
        unmatched_mask = np.zeros(len(matched), dtype=bool)

    return matched, nearest_d, unmatched_mask

# ═══════════════════════════════════════════════════════════════════════
# METRICS (без sklearn в горячем цикле)
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(np.int8)
    scores = np.asarray(scores).astype(np.float32)
    if len(np.unique(labels)) < 2:
        return None

    preds = (scores > threshold).astype(np.int8)
    tp = int(((preds == 1) & (labels == 1)).sum(dtype=np.int64))
    fp = int(((preds == 1) & (labels == 0)).sum(dtype=np.int64))
    fn = int(((preds == 0) & (labels == 1)).sum(dtype=np.int64))
    tn = int(((preds == 0) & (labels == 0)).sum(dtype=np.int64))

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    try:
        roc = float(roc_auc_score(labels, scores))
    except Exception:
        roc = 0.5
    try:
        pr = float(average_precision_score(labels, scores))
    except Exception:
        pr = 0.0

    return {
        'roc_auc': roc, 'pr_auc': pr, 'f1': float(f1),
        'precision': float(p), 'recall': float(r), 'accuracy': float(acc),
        'n_points': int(len(labels)),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'pos_ratio': float(labels.mean()),
        'pred_pos_ratio': float(preds.mean()),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--test_list', required=True)
    ap.add_argument('--dmasif_dir', required=True)
    ap.add_argument('--our_graph_dir', required=True)
    ap.add_argument('--output_dir', default='./eval_dmasif')

    ap.add_argument('--label_field', default='iface')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--weighting', default='inverse_distance',
                    choices=['inverse_distance', 'softmax', 'uniform'])
    ap.add_argument('--match_max_dist', type=float, default=1.5,
                    help='Максимальная длина матчинга (Å). '
                         'Дальше — unmatched (скор=0).')
    ap.add_argument('--degenerate_eps', type=float, default=1e-4,
                    help='Исключать файлы с pos_ratio < eps или > 1-eps')

    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--heads', type=int, required=True)
    ap.add_argument('--in_dim', type=int, default=14)

    ap.add_argument('--ply_template', action='append', default=[])
    ap.add_argument('--graph_template', action='append', default=[])
    ap.add_argument('--max_files', type=int, default=None)
    ap.add_argument('--save_matched', action='store_true')
    ap.add_argument('--smooth_iters', type=int, default=2,
                    help='Число итераций сглаживания скоров (0 = отключено)')
    ap.add_argument('--smooth_alpha', type=float, default=0.5,
                    help='Вес соседей при сглаживании (0..1)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"match_max_dist = {args.match_max_dist} Å")
    print(f"threshold      = {args.threshold}")
    print(f"k              = {args.k}  (weighting={args.weighting})")
    print(f"degenerate_eps = {args.degenerate_eps}")

    # Список
    ids = read_list(args.test_list)
    if args.max_files:
        ids = ids[:args.max_files]
    print(f"\n📋 test list: {len(ids)} entries")

    # Модель
    print(f"\n📦 Loading model: {args.model}")
    state = torch.load(args.model, map_location=device, weights_only=False)
    model = SimpleGNN(
        in_dim=args.in_dim, hidden=args.hidden,
        num_layers=args.layers, heads=args.heads, dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"   in_dim={args.in_dim} hidden={args.hidden} "
          f"layers={args.layers} heads={args.heads}")

    per_protein = []
    matched_store = {}
    missing = []
    skipped = []
    degenerate = []

    for cid in tqdm(ids, desc="Evaluating"):
        # .ply
        ply_path = find_file(args.dmasif_dir, cid, 'ply', args.ply_template)
        if ply_path is None:
            missing.append((cid, 'ply'))
            continue

        # .pt
        graph_path = find_file(args.our_graph_dir, cid, 'pt',
                                args.graph_template)
        if graph_path is None:
            missing.append((cid, 'graph'))
            continue

        # Их данные
        try:
            their_pts, their_labels = read_ply(ply_path, args.label_field)
        except Exception as e:
            skipped.append((cid, f"read_ply: {e}"))
            continue
        if their_labels is None:
            skipped.append((cid, "no labels"))
            continue

        # Проверка вырожденных
        pos_ratio = float(their_labels.mean())
        if pos_ratio < args.degenerate_eps or pos_ratio > 1 - args.degenerate_eps:
            degenerate.append((cid, pos_ratio))
            continue

        # Наш граф
        try:
            g = torch.load(graph_path, map_location='cpu', weights_only=False)
        except Exception as e:
            skipped.append((cid, f"load graph: {e}"))
            continue

        our_pts = g['pos'].numpy().astype(np.float32)

        # Инференс
        try:
            our_scores = run_inference(
                model, g, device,
                smooth_iters=args.smooth_iters,
                smooth_alpha=args.smooth_alpha,
            )
        except Exception as e:
            skipped.append((cid, f"inference: {e}"))
            continue

        # Матчинг с отсечением
        try:
            matched, nearest_d, unmatched_mask = match_scores(
                our_pts, our_scores, their_pts,
                k=args.k, weighting=args.weighting,
                max_dist=args.match_max_dist,
            )
        except Exception as e:
            skipped.append((cid, f"matching: {e}"))
            continue

        # Метрики
        m = compute_metrics(their_labels, matched, args.threshold)
        if m is None:
            skipped.append((cid, "single class"))
            continue

        # Диагностика unmatched
        pos_theirs = their_labels == 1
        neg_theirs = their_labels == 0
        m['id'] = cid
        m['ply'] = os.path.basename(ply_path)
        m['graph'] = os.path.basename(graph_path)
        m['n_their_points'] = int(len(their_pts))
        m['n_our_points'] = int(len(our_pts))
        m['unmatched_total'] = int(unmatched_mask.sum())
        m['unmatched_frac'] = float(unmatched_mask.mean())
        m['unmatched_pos'] = int((unmatched_mask & pos_theirs).sum())
        m['unmatched_neg'] = int((unmatched_mask & neg_theirs).sum())
        m['unmatched_pos_frac'] = (
            float((unmatched_mask & pos_theirs).sum() /
                  max(pos_theirs.sum(), 1))
        )
        m['mean_match_dist'] = float(nearest_d.mean())
        m['median_match_dist'] = float(np.median(nearest_d))

        per_protein.append(m)

        if args.save_matched:
            matched_store[cid] = {
                'their_labels': their_labels,
                'our_scores_on_their_points': matched,
                'match_dists': nearest_d,
                'unmatched_mask': unmatched_mask,
            }

    # Итоги
    print(f"\n✅ Processed:   {len(per_protein)}")
    print(f"⚠️  Missing:    {len(missing)}")
    print(f"⚠️  Skipped:    {len(skipped)}")
    print(f"⚠️  Degenerate: {len(degenerate)}")
    if missing:
        print("   Missing (first 10):")
        for x in missing[:10]:
            print(f"     {x}")
    if skipped:
        print("   Skipped (first 10):")
        for x in skipped[:10]:
            print(f"     {x}")
    if degenerate:
        print("   Degenerate (first 10):")
        for cid, pr in degenerate[:10]:
            print(f"     {cid}: pos_ratio={pr:.4f}")

    if not per_protein:
        print("❌ No valid results.")
        return

    keys = ['roc_auc', 'pr_auc', 'f1', 'precision',
            'recall', 'accuracy', 'pos_ratio', 'pred_pos_ratio']
    avg = {k: float(np.mean([p[k] for p in per_protein])) for k in keys}

    # Global
    all_lbl = np.concatenate([matched_store[c]['their_labels']
                              for c in matched_store])
    all_scr = np.concatenate(
        [matched_store[c]['our_scores_on_their_points']
         for c in matched_store]
    )
    global_m = compute_metrics(all_lbl, all_scr, args.threshold)

    # Unmatched агрегат
    unm_total = sum(p['unmatched_total'] for p in per_protein)
    unm_pos = sum(p['unmatched_pos'] for p in per_protein)
    unm_neg = sum(p['unmatched_neg'] for p in per_protein)
    total_pts = sum(p['n_their_points'] for p in per_protein)
    total_pos = sum(int(p['pos_ratio'] * p['n_their_points'])
                    for p in per_protein)

    def _show(title, d):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)
        for k in ['f1', 'precision', 'recall', 'accuracy', 'roc_auc', 'pr_auc']:
            print(f"   {k:10s}: {d[k]:.4f}")
        if 'n_points' in d:
            print(f"   n_points : {d['n_points']:,}")
            print(f"   TP={d['tp']:,}  FP={d['fp']:,}  "
                  f"FN={d['fn']:,}  TN={d['tn']:,}")

    _show("PER-PROTEIN (averaged):", avg)
    _show("GLOBAL (all points):", global_m)

    print("\n" + "=" * 78)
    print("UNMATCHED summary")
    print("=" * 78)
    print(f"  match_max_dist = {args.match_max_dist} Å")
    print(f"  Total their points:     {total_pts:,}")
    print(f"  Unmatched:              {unm_total:,} "
          f"({100 * unm_total / max(total_pts, 1):.1f}%)")
    print(f"    iface=1 (missed):     {unm_pos:,} "
          f"({100 * unm_pos / max(total_pos, 1):.1f}% от их интерфейса)")
    print(f"    iface=0 (background): {unm_neg:,}")

    # Список
    sorted_p = sorted(per_protein, key=lambda x: x['f1'])
    print("\nWORST 10 (by F1):")
    for p in sorted_p[:10]:
        print(f"   {p['id']:20s}  F1={p['f1']:.3f}  P={p['precision']:.3f}  "
              f"R={p['recall']:.3f}  ROC={p['roc_auc']:.3f}  "
              f"unm_pos={p['unmatched_pos']}  n={p['n_their_points']}")
    print("\nBEST 10 (by F1):")
    for p in sorted_p[-10:]:
        print(f"   {p['id']:20s}  F1={p['f1']:.3f}  P={p['precision']:.3f}  "
              f"R={p['recall']:.3f}  ROC={p['roc_auc']:.3f}  "
              f"unm_pos={p['unmatched_pos']}  n={p['n_their_points']}")

    # Сохранение
    out = {
        'config': vars(args),
        'per_protein_avg': avg,
        'global': global_m,
        'per_protein': per_protein,
        'missing': missing,
        'skipped': skipped,
        'degenerate': degenerate,
        'unmatched_summary': {
            'total_points': total_pts,
            'total_positives_est': total_pos,
            'unmatched_total': unm_total,
            'unmatched_pos': unm_pos,
            'unmatched_neg': unm_neg,
        },
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    if args.save_matched:
        np.savez_compressed(
            os.path.join(args.output_dir, 'matched_scores.npz'),
            **{c: v['our_scores_on_their_points']
               for c, v in matched_store.items()}
        )
        np.savez_compressed(
            os.path.join(args.output_dir, 'their_labels.npz'),
            **{c: v['their_labels'] for c, v in matched_store.items()}
        )
        # Дополнительно: расстояния и маска unmatched
        np.savez_compressed(
            os.path.join(args.output_dir, 'match_dists.npz'),
            **{c: v['match_dists'] for c, v in matched_store.items()}
        )

    print(f"\n✅ Saved to {args.output_dir}")


if __name__ == '__main__':
    main()