#!/usr/bin/env python3
"""
train_clean_ddp.py - Обучение с DistributedDataParallel на нескольких GPU
С поддержкой gradient checkpointing для экономии памяти
"""

import os
import sys
import gc
import glob
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data, Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONTACT_THRESHOLD = 4.0
HARD_NEGATIVE_RATIO = 3
NUM_LAYERS = 6
HIDDEN = 128
HEADS = 4
LR = 1e-3
EPOCHS = 300
USE_CHECKPOINT = True  # gradient checkpointing

try:
    from torch.amp import autocast, GradScaler
except:
    from torch.cuda.amp import autocast, GradScaler


# ─────────────────────────────────────────────────────────────────────────────
# МОДЕЛЬ С GRADIENT CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
class SimpleGNN(nn.Module):
    def __init__(self, in_dim=14, hidden=128, num_layers=6, heads=4, dropout=0.1, use_checkpoint=True):
        super().__init__()
        
        head_dim = hidden // heads
        assert head_dim * heads == hidden
        
        self.use_checkpoint = use_checkpoint
        
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
    
    def _forward_layer(self, conv, norm, x, edge_index, edge_attr):
        """Один слой с возможностью checkpointing"""
        if self.use_checkpoint and self.training:
            # Для checkpointing нужно передавать все аргументы
            x = torch.utils.checkpoint.checkpoint(
                self._conv_forward, conv, norm, x, edge_index, edge_attr,
                use_reentrant=False
            )
        else:
            x = self._conv_forward(conv, norm, x, edge_index, edge_attr)
        return x
    
    def _conv_forward(self, conv, norm, x, edge_index, edge_attr):
        """Внутренняя функция для checkpointing"""
        x = x + conv(x, edge_index, edge_attr=edge_attr)
        x = norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        x = F.gelu(self.norm_in(self.lin_in(x)))
        x = self.dropout(x)
        
        for conv, norm in zip(self.convs, self.norms):
            x = self._forward_layer(conv, norm, x, edge_index, edge_attr)
        
        return self.lin_out(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# ДАТАСЕТ
# ─────────────────────────────────────────────────────────────────────────────
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
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"  Loaded {len(self.file_paths)} files")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location='cpu', weights_only=False)
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


# ─────────────────────────────────────────────────────────────────────────────
# ФУНКЦИИ ПОТЕРИ И МЕТРИК
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(logits, targets, hard_ratio=3):
    with torch.no_grad():
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]
        
        n_pos = len(pos_idx)
        if n_pos == 0:
            return None, 0, 0
        
        sampled_idx = pos_idx
        
        if len(neg_idx) > 0:
            neg_logits = logits[neg_idx]
            sorted_idx = torch.argsort(neg_logits, descending=True)
            n_hard = min(len(neg_idx), n_pos * hard_ratio)
            sampled_idx = torch.cat([sampled_idx, neg_idx[sorted_idx[:n_hard]]])
    
    loss = F.binary_cross_entropy_with_logits(logits[sampled_idx], targets[sampled_idx])
    return loss, n_pos, len(sampled_idx) - n_pos


def compute_metrics(model, loader, device, rank=0):
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


# ─────────────────────────────────────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────
def train_worker(rank, world_size, args):
    """Функция, запускаемая на каждом GPU через mp.spawn"""
    
    # 1. Инициализация DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 2. Установка устройства
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # 3. Загрузка списка файлов
    all_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    random.Random(42).shuffle(all_paths)
    
    n_files = min(args.n_files, len(all_paths))
    train_files = all_paths[:n_files]
    val_files = all_paths[n_files:min(n_files+20, len(all_paths))]
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"TRAINING CONFIGURATION (DDP on {world_size} GPUs)")
        print(f"{'='*80}")
        print(f"Data: {args.data_dir}")
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")
        print(f"Layers: {args.layers}, Hidden: {args.hidden}, Heads: {args.heads}")
        print(f"Threshold: {args.threshold}Å")
        print(f"Hard ratio: {HARD_NEGATIVE_RATIO}")
        print(f"LR: {args.lr}")
        print(f"Gradient checkpointing: {args.checkpoint}")
    
    # 4. Создание датасетов
    train_dataset = SimpleDataset(train_files)
    val_dataset = SimpleDataset(val_files)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, 
                              sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            sampler=val_sampler, collate_fn=collate_fn, num_workers=0)
    
    # 5. Проверка размерности
    sample = None
    for data in train_loader:
        if data is not None:
            sample = data
            break
    
    if sample is None:
        raise RuntimeError("No valid data!")
    
    in_dim = sample.x.shape[1]
    if rank == 0:
        print(f"\nInput dim: {in_dim}")
        print(f"Sample: {len(sample.x)} points, Positive: {sample.y.sum().item()}")
    
    # 6. СОЗДАНИЕ МОДЕЛИ (ДО DDP) - ИСПОЛЬЗУЕМ args.hidden!
    model = SimpleGNN(
        in_dim=in_dim, 
        hidden=args.hidden, 
        num_layers=args.layers, 
        heads=args.heads,
        use_checkpoint=args.checkpoint
    ).to(device)
    
    # 7. RESUME - ЗАГРУЗКА ВЕСОВ ДО DDP
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            # Проверяем совместимость размерностей
            model_dict = model.state_dict()
            pretrained_dict = {}
            skipped_keys = []
            
            for k, v in checkpoint.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    skipped_keys.append(k)
            
            if skipped_keys and rank == 0:
                print(f"⚠️ Skipped {len(skipped_keys)} layers due to shape mismatch")
                print(f"   First 5 skipped: {skipped_keys[:5]}")
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            if rank == 0:
                print(f"✅ Loaded {len(pretrained_dict)} layers from {args.resume}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ Failed to load resume: {e}")
    
    # 8. ОБЁРТКА В DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: {total_params:,} total, {trainable_params:,} trainable")
    
    # 9. Оптимизатор и scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    
    best_f1 = 0.0
    
    # 10. Цикл обучения
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"STARTING TRAINING")
        print(f"{'='*80}\n")
    
    for epoch in range(1, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        total_pos = 0
        total_neg = 0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False) if rank == 0 else train_loader
        
        for data in iterator:
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
            
            # Очищаем память от графа
            if data is not None:
                del data
                if n_batches % 100 == 0:
                    torch.cuda.empty_cache()
        
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        
        if rank == 0 and (epoch % 5 == 0 or epoch <= 10 or epoch == EPOCHS):
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
                torch.save(model.module.state_dict(), args.ckpt)
                print(f"\n  ✅ New best F1: {best_f1:.4f}")
            
            torch.save(model.module.state_dict(), "last.pt")
            print(f"{'='*80}\n")
    
    # 11. Финальный поиск порога
    if rank == 0:
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                if data is None: continue
                data = data.to(device)
                logits = model(data)
                all_probs.append(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
        
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        
        best_f1 = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs > thresh).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"\n🎯 Optimal threshold search:")
        print(f"   Best F1={best_f1:.4f} @ threshold={best_thresh:.2f}")
        print(f"   (Default 0.5 gave F1={f1_score(labels, (probs>0.5).astype(int), zero_division=0):.4f})")
    
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed/graphs')
    parser.add_argument('--ckpt', default='best_model_ddp.pt')
    parser.add_argument('--n_files', type=int, default=3000)
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=64)  # ← changed default to 64
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint', action='store_true', default=True,
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--no_checkpoint', action='store_false', dest='checkpoint',
                        help='Disable gradient checkpointing')
    args = parser.parse_args()
    
    # Обновляем глобальные константы для обратной совместимости
    global CONTACT_THRESHOLD
    CONTACT_THRESHOLD = args.threshold
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"⚠️ Only {world_size} GPU found. For DDP need at least 2.")
        print("Please use train_clean.py for single GPU training.")
        sys.exit(1)
    
    print(f"🚀 Launching DDP training on {world_size} GPUs")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()