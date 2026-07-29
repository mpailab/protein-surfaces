#!/usr/bin/env python3
"""
evaluate.py - Оценка модели на новых данных
- Использует последние N файлов из директории (не пересекается с train/val)
- Подбирает оптимальный порог методом бисекции (если не указан фиксированный)
- Считает per-protein и global метрики
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                            precision_score, recall_score, accuracy_score,
                            confusion_matrix)
from tqdm import tqdm
import json
import random
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONTACT_THRESHOLD = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# МОДЕЛЬ
# ─────────────────────────────────────────────────────────────────────────────
class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=128, num_layers=6, heads=4, dropout=0.1):
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


def load_data(file_path):
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        return data
    elif hasattr(data, 'x'):
        return {
            'x': data.x,
            'pos': data.pos,
            'y': data.y,
            'edge_index': data.edge_index if hasattr(data, 'edge_index') else None,
            'edge_attr': data.edge_attr if hasattr(data, 'edge_attr') else None,
            'pdb_id': getattr(data, 'pdb_id', 'unknown'),
        }
    else:
        raise ValueError(f"Unknown data format")


def create_graph(data):
    """Создаёт граф из загруженных данных (использует готовые edge_index и edge_attr)"""
    if 'edge_index' in data and data['edge_index'] is not None:
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
    else:
        raise ValueError(f"No precomputed graph found in {data.get('pdb_id', 'unknown')}")
    
    return Data(
        x=data['x'].float(),
        pos=data['pos'].float(),
        edge_index=edge_index,
        edge_attr=edge_attr.float() if edge_attr is not None else None,
        y=data['y'].float()
    )


def find_optimal_threshold_bisection(probs_list, labels_list, n_iter=30):
    """
    Поиск оптимального порога методом бисекции (золотое сечение)
    Максимизирует средний F1 по белкам
    """
    
    def compute_avg_f1(thresh):
        """Вычисляет средний F1 по всем белкам при заданном пороге"""
        f1s = []
        for probs, labels in zip(probs_list, labels_list):
            preds = (probs > thresh).astype(int)
            y_true = (labels < CONTACT_THRESHOLD).astype(int)
            f1s.append(f1_score(y_true, preds, zero_division=0))
        return np.mean(f1s)
    
    # 1. Грубый поиск для определения диапазона
    print("   Stage 1: Coarse search...")
    coarse_thresholds = np.arange(0.01, 0.99, 0.02)
    f1s = []
    for t in tqdm(coarse_thresholds, desc="   Coarse", leave=False):
        f1s.append(compute_avg_f1(t))
    
    best_idx = np.argmax(f1s)
    best_coarse = coarse_thresholds[best_idx]
    print(f"   Coarse best: {best_coarse:.2f} (F1={f1s[best_idx]:.4f})")
    
    # 2. Уточняем бисекцией (золотое сечение) в окрестности
    print(f"   Stage 2: Fine search around {best_coarse:.2f}...")
    
    # Определяем диапазон для уточнения
    low = max(0.01, best_coarse - 0.10)
    high = min(0.99, best_coarse + 0.10)
    
    # Золотое сечение
    golden_ratio = (5**0.5 - 1) / 2
    
    for i in range(n_iter):
        mid1 = high - golden_ratio * (high - low)
        mid2 = low + golden_ratio * (high - low)
        
        f1_1 = compute_avg_f1(mid1)
        f1_2 = compute_avg_f1(mid2)
        
        if f1_1 > f1_2:
            high = mid2
        else:
            low = mid1
        
        if (i + 1) % 5 == 0:
            current_best = (low + high) / 2
            current_f1 = compute_avg_f1(current_best)
            print(f"      Iter {i+1}/{n_iter}: threshold={current_best:.4f}, F1={current_f1:.4f}")
    
    best_thresh = (low + high) / 2
    best_f1 = compute_avg_f1(best_thresh)
    
    # 3. Дополнительная проверка: мелкая сетка вокруг найденного порога
    print(f"   Stage 3: Final refinement...")
    fine_thresholds = np.arange(max(0.01, best_thresh - 0.05), 
                                min(0.99, best_thresh + 0.05), 
                                0.002)
    
    best_fine = best_thresh
    best_f1_fine = best_f1
    
    for t in fine_thresholds:
        f1 = compute_avg_f1(t)
        if f1 > best_f1_fine:
            best_f1_fine = f1
            best_fine = t
    
    print(f"   Final: threshold={best_fine:.4f}, F1={best_f1_fine:.4f}")
    
    return best_fine, best_f1_fine


def evaluate_with_threshold(model, loader, device, fixed_threshold=None):
    """Оценка модели с подбором порога (или фиксированным)"""
    model.eval()
    
    # Собираем предсказания для всех белков
    all_probs = []
    all_labels = []
    all_pdb_ids = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting"):
            if data is None:
                continue
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            
            y_dist = data.y.cpu()
            pdb_id = getattr(data, 'pdb_id', 'unknown')
            if isinstance(pdb_id, list):
                pdb_id = '_'.join(str(x) for x in pdb_id)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_dist.numpy())
            all_pdb_ids.append(str(pdb_id))
    
    n_proteins = len(all_probs)
    total_points = sum(len(p) for p in all_probs)
    print(f"\n📊 Proteins: {n_proteins}, total points: {total_points:,}")
    
    # Определяем порог
    if fixed_threshold is not None:
        best_threshold = fixed_threshold
        print(f"\n🔍 Using fixed threshold: {best_threshold:.4f}")
        # Для информации посчитаем средний F1 при этом пороге
        f1s = []
        for probs, labels in zip(all_probs, all_labels):
            preds = (probs > best_threshold).astype(int)
            y_true = (labels < CONTACT_THRESHOLD).astype(int)
            f1s.append(f1_score(y_true, preds, zero_division=0))
        best_avg_f1 = np.mean(f1s)
        print(f"   Average F1 at this threshold: {best_avg_f1:.4f}")
    else:
        print("\n🔍 Searching for optimal threshold (bisection)...")
        best_threshold, best_avg_f1 = find_optimal_threshold_bisection(all_probs, all_labels)
        print(f"\n✅ Optimal threshold: {best_threshold:.4f}")
        print(f"   Best average F1: {best_avg_f1:.4f}")
    
    # Считаем все метрики при оптимальном пороге
    print("\n📊 Computing final metrics...")
    per_protein_metrics = []
    
    for probs, labels, pdb_id in zip(all_probs, all_labels, all_pdb_ids):
        preds = (probs > best_threshold).astype(int)
        y_true = (labels < CONTACT_THRESHOLD).astype(int)
        
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        has_both = len(np.unique(y_true)) == 2
        if has_both:
            try:
                roc_auc = roc_auc_score(y_true, probs)
                pr_auc = average_precision_score(y_true, probs)
            except:
                roc_auc = 0.5
                pr_auc = 0.0
        else:
            roc_auc = 0.5
            pr_auc = 0.0
        
        per_protein_metrics.append({
            'pdb_id': pdb_id,
            'n_points': len(probs),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
        })
    
    # Усредняем по белкам
    avg_metrics = {}
    for key in ['precision', 'recall', 'f1', 'accuracy', 'roc_auc', 'pr_auc']:
        values = [m[key] for m in per_protein_metrics]
        avg_metrics[key] = np.mean(values)
    
    # Суммируем матрицу ошибок
    total_tp = sum(m['tp'] for m in per_protein_metrics)
    total_fp = sum(m['fp'] for m in per_protein_metrics)
    total_fn = sum(m['fn'] for m in per_protein_metrics)
    total_tn = sum(m['tn'] for m in per_protein_metrics)
    
    # Глобальные метрики (все точки вместе)
    all_probs_flat = np.concatenate(all_probs)
    all_labels_flat = np.concatenate(all_labels)
    all_y_true = (all_labels_flat < CONTACT_THRESHOLD).astype(int)
    preds_flat = (all_probs_flat > best_threshold).astype(int)
    
    global_metrics = {
        'f1': f1_score(all_y_true, preds_flat, zero_division=0),
        'precision': precision_score(all_y_true, preds_flat, zero_division=0),
        'recall': recall_score(all_y_true, preds_flat, zero_division=0),
        'accuracy': accuracy_score(all_y_true, preds_flat),
        'roc_auc': roc_auc_score(all_y_true, all_probs_flat),
        'pr_auc': average_precision_score(all_y_true, all_probs_flat),
        'mean_prob': float(all_probs_flat.mean()),
        'pos_ratio': float(all_y_true.mean()),
        'pred_pos_ratio': float(preds_flat.mean()),
        'tp': int(total_tp),
        'fp': int(total_fp),
        'fn': int(total_fn),
        'tn': int(total_tn),
    }
    
    # Вывод
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n  Threshold used: {best_threshold:.4f}")
    
    print("\n📊 PER-PROTEIN (averaged):")
    print(f"   F1:        {avg_metrics['f1']:.4f}")
    print(f"   Precision: {avg_metrics['precision']:.4f}")
    print(f"   Recall:    {avg_metrics['recall']:.4f}")
    print(f"   Accuracy:  {avg_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:   {avg_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {avg_metrics['pr_auc']:.4f}")
    
    print("\n📊 GLOBAL (all points together):")
    print(f"   F1:        {global_metrics['f1']:.4f}")
    print(f"   Precision: {global_metrics['precision']:.4f}")
    print(f"   Recall:    {global_metrics['recall']:.4f}")
    print(f"   Accuracy:  {global_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:   {global_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {global_metrics['pr_auc']:.4f}")
    print(f"   Mean prob: {global_metrics['mean_prob']:.4f}")
    print(f"   Pos ratio: {global_metrics['pos_ratio']:.4f}")
    print(f"   Pred pos:  {global_metrics['pred_pos_ratio']:.4f}")
    
    print(f"\n  Confusion matrix (total):")
    print(f"    TP: {total_tp:,}  FP: {total_fp:,}")
    print(f"    FN: {total_fn:,}  TN: {total_tn:,}")
    
    # Худшие и лучшие белки
    sorted_metrics = sorted(per_protein_metrics, key=lambda x: x['f1'])
    
    print("\n" + "="*80)
    print("WORST 10 PROTEINS (by F1)")
    print("="*80)
    for m in sorted_metrics[:10]:
        print(f"  {m['pdb_id']:<20} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}  n={m['n_points']}")
    
    print("\n" + "="*80)
    print("BEST 10 PROTEINS (by F1)")
    print("="*80)
    for m in sorted_metrics[-10:]:
        print(f"  {m['pdb_id']:<20} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}  n={m['n_points']}")
    
    return {
        'threshold': float(best_threshold),
        'per_protein': avg_metrics,
        'global': global_metrics,
        'per_protein_details': per_protein_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./processed/graphs')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--n_files', type=int, default=100)
    parser.add_argument('--threshold_contact', type=float, default=4.0)
    parser.add_argument('--train_val_size', type=int, default=3020,
                        help='Number of files used for training + validation (to skip)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold for predictions (if not set, optimal will be searched)')
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--shuffle_seed', type=int, default=None,
                    help='Seed for shuffling files (use 42 to match training)')
    args = parser.parse_args()
    
    global CONTACT_THRESHOLD
    CONTACT_THRESHOLD = args.threshold_contact
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем модель
    print("\n📦 Loading model...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    hidden = args.hidden if args.hidden is not None else checkpoint['lin_in.weight'].shape[0]
    num_layers = args.layers if args.layers is not None else 0
    
    if num_layers == 0:
        for key in checkpoint.keys():
            if key.startswith('convs.'):
                num_layers = max(num_layers, int(key.split('.')[1]) + 1)
        if num_layers == 0:
            num_layers = 6
    
    heads = args.heads
    
    in_dim = 14
    model = SimpleGNN(in_dim, hidden=hidden, num_layers=num_layers, heads=heads, dropout=0.0).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"   hidden={hidden}, layers={num_layers}, heads={heads}")
    
    # Загружаем данные
    print(f"\n📂 Loading data from {args.data_dir}...")
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(all_files)
    test_files = all_files[args.train_val_size:args.train_val_size + args.n_files]
    print(f"   First 10 test files: {[os.path.basename(f) for f in test_files[:10]]}")
    print(f"   Skipping first {args.train_val_size} files (train+val split)")
    print(f"   Found {len(all_files)} total, using {len(test_files)} test files")
    # ─────────────────────────────────────────────────────────────────────────────
    # Загрузка сохранённых списков train/val (если есть)
    # ─────────────────────────────────────────────────────────────────────────────
    train_list_path = 'train_files.txt'
    if os.path.exists(train_list_path):
        with open(train_list_path, 'r') as f:
            train_names = set(line.strip() for line in f if line.strip())
        test_names = set(os.path.basename(f) for f in test_files)
        overlap = test_names & train_names
        if overlap:
            print(f"   ⚠️ WARNING: {len(overlap)} test files overlap with train!")
            print(f"   Overlapping files: {list(overlap)[:10]}...")
        else:
            print(f"   ✅ No overlap between test and train files.")
        print(f"   Train: {len(train_names)}, Test: {len(test_names)}")
    
    dataset = []
    for f in tqdm(test_files, desc="Loading"):
        try:
            data = load_data(f)
            graph = create_graph(data)
            pdb_id = data.get('pdb_id', os.path.basename(f).replace('.pt', ''))
            if isinstance(pdb_id, list):
                pdb_id = '_'.join(str(x) for x in pdb_id)
            graph.pdb_id = str(pdb_id)
            dataset.append(graph)
        except Exception as e:
            print(f"   Error: {f} - {e}")
    
    print(f"   Loaded {len(dataset)} proteins")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print("\n🔮 Evaluating...")
    result = evaluate_with_threshold(model, loader, device, fixed_threshold=args.threshold)
    
    output_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n✅ Results saved to {output_path}")


if __name__ == '__main__':
    main()