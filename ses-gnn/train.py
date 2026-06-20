#!/usr/bin/env python3
"""
train.py - Максимально простая версия, без контекстов, без сложностей
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import glob
import os
import random
import gc
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# ─────────────────────────────────────────────────────────────────────────────
# Простой импорт autocast
# ─────────────────────────────────────────────────────────────────────────────
try:
    from torch.amp import autocast, GradScaler
except:
    from torch.cuda.amp import autocast, GradScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG - всё просто
# ─────────────────────────────────────────────────────────────────────────────
LR = 1e-3
EPOCHS = 300
CONTACT_THRESHOLD = 4.0
HARD_NEGATIVE_RATIO = 3
NUM_LAYERS = 6
HIDDEN = 128
HEADS = 4


class SimpleDataset(Dataset):
    def __init__(self, file_paths, threshold=CONTACT_THRESHOLD):
        self.file_paths = []
        for p in file_paths:
            try:
                data = torch.load(p, map_location='cpu', weights_only=False)
                if (data['y'] < threshold).sum() > 0:
                    self.file_paths.append(p)
            except:
                continue
        print(f"  Loaded {len(self.file_paths)} files")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location='cpu', weights_only=False)
        
        # Бинаризация
        y = (data['y'].float() < CONTACT_THRESHOLD).float()
        
        return Data(
            x=data['x'],
            pos=data['pos'],
            edge_index=data['edge_index'],
            edge_attr=data['edge_attr'],
            y=y
        )


def collate_fn(batch):
    return batch[0] if batch[0] is not None else None


class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=128, num_layers=6, heads=4, dropout=0.1):
        super().__init__()
        
        head_dim = hidden // heads
        assert head_dim * heads == hidden
        
        # Вход
        self.lin_in = nn.Linear(in_dim, hidden)
        self.norm_in = nn.LayerNorm(hidden)
        
        # Слои
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(hidden, head_dim, heads=heads, edge_dim=3,
                               dropout=dropout, root_weight=True, beta=False)
            )
            self.norms.append(nn.LayerNorm(hidden))
        
        # Выход
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
        
        # bias = logit(0.3) ≈ -0.85
        nn.init.constant_(self.lin_out.bias, -0.85)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        x = F.gelu(self.norm_in(self.lin_in(x)))
        x = self.dropout(x)
        
        for conv, norm in zip(self.convs, self.norms):
            x = x + conv(x, edge_index, edge_attr=edge_attr)  # residual
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
        
        return self.lin_out(x).squeeze(-1)  # логиты


def compute_loss(logits, targets, hard_ratio=3):
    """Hard negative mining"""
    with torch.no_grad():
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]
        
        n_pos = len(pos_idx)
        if n_pos == 0:
            return None, 0, 0
        
        # Все положительные
        sampled_idx = pos_idx
        
        # Top-K отрицательных
        if len(neg_idx) > 0:
            neg_logits = logits[neg_idx]
            sorted_idx = torch.argsort(neg_logits, descending=True)
            n_hard = min(len(neg_idx), n_pos * hard_ratio)
            sampled_idx = torch.cat([sampled_idx, neg_idx[sorted_idx[:n_hard]]])
    
    loss = F.binary_cross_entropy_with_logits(logits[sampled_idx], targets[sampled_idx])
    return loss, n_pos, len(sampled_idx) - n_pos


def compute_metrics(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            if data is None:
                continue
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
    
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs > 0.5
    
    return {
        'roc_auc': roc_auc_score(labels, probs),
        'pr_auc': average_precision_score(labels, probs),
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'mean_prob': probs.mean(),
        'pos_ratio': labels.mean(),
        'pred_pos_ratio': preds.mean(),
    }


def print_stats(files, title="Statistics"):
    print(f"\n{title}:")
    print("-" * 70)
    print(f"{'File':<40} {'Points':<10} {'Contacts':<12} {'Ratio':<8}")
    print("-" * 70)
    
    ratios = []
    for p in files[:20]:
        data = torch.load(p, map_location='cpu')
        y = data['y'].float()
        n = len(y)
        c = (y < CONTACT_THRESHOLD).sum().item()
        r = c / n
        ratios.append(r)
        print(f"  {os.path.basename(p):<40} {n:<10} {c:<12} {r:.2%}")
    
    if ratios:
        print("-" * 70)
        print(f"  {'AVERAGE':<40} {'':<10} {'':<12} {np.mean(ratios):.2%}")
    print("-" * 70)


def train(data_dir, ckpt_path, n_files=2):
    all_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    random.Random(42).shuffle(all_paths)
    
    n_files = min(n_files, len(all_paths))
    train_files = all_paths[:n_files]
    val_files = all_paths[n_files:min(n_files+20, len(all_paths))]
    
    print(f"\n{'='*80}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Data: {data_dir}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print(f"Layers: {NUM_LAYERS}, Hidden: {HIDDEN}, Heads: {HEADS}")
    print(f"Threshold: {CONTACT_THRESHOLD}Å")
    print(f"Hard ratio: {HARD_NEGATIVE_RATIO}")
    print(f"LR: {LR}")
    
    print_stats(train_files, "📊 TRAIN")
    print_stats(val_files, "📊 VAL")
    
    # Датасеты
    train_dataset = SimpleDataset(train_files)
    val_dataset = SimpleDataset(val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Проверяем данные
    sample = None
    for data in train_loader:
        if data is not None:
            sample = data
            break
    
    if sample is None:
        raise RuntimeError("No valid data!")
    
    in_dim = sample.x.shape[1]
    print(f"\nInput dim: {in_dim}")
    print(f"Sample: {len(sample.x)} points, Positive: {sample.y.sum().item()}")
    
    # Модель
    model = SimpleGNN(in_dim, HIDDEN, NUM_LAYERS, HEADS).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    
    best_f1 = 0.0
    
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        total_pos = 0
        total_neg = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            if data is None:
                continue
            
            data = data.to(device)
            
            with autocast(enabled=True):
                logits = model(data)
                loss, n_pos, n_neg = compute_loss(logits, data.y, HARD_NEGATIVE_RATIO)
            
            if loss is None:
                continue
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            n_batches += 1
            total_pos += n_pos
            total_neg += n_neg
            epoch_loss += loss.item()
        
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        
        if epoch % 5 == 0 or epoch <= 10:
            train_metrics = compute_metrics(model, train_loader, device)
            val_metrics = compute_metrics(model, val_loader, device)
            
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}")
            print(f"{'='*80}")
            print(f"  Loss: {epoch_loss/n_batches:.4f}")
            print(f"  Sampling: pos={total_pos/n_batches:.0f}, neg={total_neg/n_batches:.0f} (ratio={total_neg/total_pos:.1f})")
            
            print(f"\n  📊 TRAIN:")
            print(f"     ROC-AUC: {train_metrics['roc_auc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}")
            print(f"     F1: {train_metrics['f1']:.4f} | P: {train_metrics['precision']:.4f} | R: {train_metrics['recall']:.4f}")
            print(f"     Pos: {train_metrics['pos_ratio']:.2%} | Pred: {train_metrics['pred_pos_ratio']:.2%}")
            print(f"     Mean prob: {train_metrics['mean_prob']:.4f}")
            
            print(f"\n  📊 VAL:")
            print(f"     ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
            print(f"     F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")
            print(f"     Pos: {val_metrics['pos_ratio']:.2%} | Pred: {val_metrics['pred_pos_ratio']:.2%}")
            print(f"     Mean prob: {val_metrics['mean_prob']:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), ckpt_path)
                print(f"\n  ✅ New best F1: {best_f1:.4f}")
            
            torch.save(model.state_dict(), "last.pt")
            print(f"{'='*80}\n")
    
    print(f"\n✅ Best VAL F1: {best_f1:.4f}")
    # После обучения
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"Best F1={best_f1:.4f} @ threshold={best_thresh:.2f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed/graphs')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--n_files', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    global CONTACT_THRESHOLD, NUM_LAYERS, HIDDEN, LR
    CONTACT_THRESHOLD = args.threshold
    NUM_LAYERS = args.layers
    HIDDEN = args.hidden
    LR = args.lr
    
    train(args.data_dir, args.ckpt, n_files=args.n_files)


if __name__ == '__main__':
    main()