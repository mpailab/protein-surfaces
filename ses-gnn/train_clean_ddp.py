#!/usr/bin/env python3
"""
train_clean_ddp.py - Обучение с DistributedDataParallel на нескольких GPU
- Тренировка: DDP на всех GPU
- Валидация: только на rank 0 (без DDP, чтобы избежать зависаний)
"""

import os
import sys
import gc
import glob
import argparse
import time
import random
import csv
from datetime import datetime
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

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not installed. Install with: pip install tensorboard")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONTACT_THRESHOLD = 4.0
NUM_LAYERS = 6
HIDDEN = 32
HEADS = 4
LR = 1e-3
EPOCHS = 300
USE_CHECKPOINT = True
N_VAL = 50

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
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self._conv_forward, conv, norm, x, edge_index, edge_attr,
                use_reentrant=False
            )
        else:
            x = self._conv_forward(conv, norm, x, edge_index, edge_attr)
        return x
    
    def _conv_forward(self, conv, norm, x, edge_index, edge_attr):
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
def compute_loss(logits, targets, hard_ratio=2.0):
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
            n_hard = min(len(neg_idx), int(n_pos * hard_ratio))
            sampled_idx = torch.cat([sampled_idx, neg_idx[sorted_idx[:n_hard]]])
    
    loss = F.binary_cross_entropy_with_logits(logits[sampled_idx], targets[sampled_idx])
    return loss, n_pos, len(sampled_idx) - n_pos


def compute_metrics_ddp(model, loader, device, rank, world_size, desc="Evaluating"):
    """Train eval: каждый ранк считает свою часть данных"""
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()
    
    all_probs, all_labels = [], []
    
    # tqdm только на rank 0
    iterator = tqdm(loader, desc=desc, leave=False) if rank == 0 else loader
    
    with torch.no_grad():
        for data in iterator:
            if data is None:
                continue
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            del data
    
    local_probs = np.concatenate(all_probs) if all_probs else np.array([])
    local_labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    if len(local_probs) == 0:
        return None
    
    preds = local_probs > 0.5
    unique_labels = np.unique(local_labels)
    
    return {
        'roc_auc': roc_auc_score(local_labels, local_probs) if len(unique_labels) > 1 else 0.5,
        'pr_auc': average_precision_score(local_labels, local_probs) if len(unique_labels) > 1 else 0.0,
        'f1': f1_score(local_labels, preds, zero_division=0),
        'precision': precision_score(local_labels, preds, zero_division=0),
        'recall': recall_score(local_labels, preds, zero_division=0),
        'mean_prob': float(local_probs.mean()),
        'pos_ratio': float(local_labels.mean()),
        'pred_pos_ratio': float(preds.mean()),
        'n_points': len(local_probs),
    }


def compute_metrics_single_gpu(model, loader, device, desc="Evaluating"):
    """Val eval: только на rank 0, без DDP"""
    if loader is None:
        return None
    
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()
    
    all_probs, all_labels = [], []
    
    # tqdm для валидации
    iterator = tqdm(loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for data in iterator:
            if data is None:
                continue
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())
            del data
    
    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    
    if len(probs) == 0:
        return None
    
    preds = probs > 0.5
    unique_labels = np.unique(labels)
    
    return {
        'roc_auc': roc_auc_score(labels, probs) if len(unique_labels) > 1 else 0.5,
        'pr_auc': average_precision_score(labels, probs) if len(unique_labels) > 1 else 0.0,
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'mean_prob': float(probs.mean()),
        'pos_ratio': float(labels.mean()),
        'pred_pos_ratio': float(preds.mean()),
        'n_points': len(probs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV ЛОГГЕР
# ─────────────────────────────────────────────────────────────────────────────
class CSVLogger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{experiment_name}.csv")
        self.fieldnames = [
            'epoch', 'train_loss', 'train_roc_auc', 'train_pr_auc', 'train_f1', 
            'train_precision', 'train_recall', 'train_pos_ratio', 'train_pred_pos_ratio',
            'val_roc_auc', 'val_pr_auc', 'val_f1', 'val_precision', 'val_recall',
            'val_pos_ratio', 'val_pred_pos_ratio', 'val_n_points',
            'sampling_pos', 'sampling_neg', 'sampling_ratio', 
            'best_f1', 'best_roc_auc', 'best_pr_auc'
        ]
        self.file = open(self.csv_path, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()
    
    def log(self, data):
        self.writer.writerow(data)
        self.file.flush()
    
    def close(self):
        self.file.close()


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
    if rank == 0:
        with open('train_files.txt', 'w') as f:
            for p in train_files:
                f.write(os.path.basename(p) + '\n')
        print(f"✅ Saved train files list to train_files.txt")
    val_files = all_paths[n_files:n_files + args.n_val]
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"TRAINING CONFIGURATION (DDP on {world_size} GPUs)")
        print(f"{'='*80}")
        print(f"Data: {args.data_dir}")
        print(f"Train: {len(train_files)}, Val: {len(val_files)}")
        print(f"Layers: {args.layers}, Hidden: {args.hidden}, Heads: {args.heads}")
        print(f"Threshold: {args.threshold}Å")
        print(f"Hard ratio: {args.hard_ratio}")
        print(f"LR: {args.lr}")
        print(f"Gradient checkpointing: {args.checkpoint}")
        print(f"Log dir: {args.log_dir}")
        print(f"Checkpoints: {args.ckpt_f1} (F1), {args.ckpt_roc} (ROC), {args.ckpt_pr} (PR)")
    
    # 4. Создание датасетов
    train_dataset = SimpleDataset(train_files)
    val_dataset = SimpleDataset(val_files)
    
    # 4a. Train loader с DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        sampler=train_sampler, collate_fn=collate_fn, num_workers=0
    )
    
    # 4b. Val loader без DDP (только на rank 0)
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
    else:
        val_loader = None
    
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
    
    # 6. Создание модели
    model = SimpleGNN(
        in_dim=in_dim, 
        hidden=args.hidden, 
        num_layers=args.layers, 
        heads=args.heads,
        use_checkpoint=args.checkpoint
    ).to(device)
    
    # 7. Resume
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
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
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            if rank == 0:
                print(f"✅ Loaded {len(pretrained_dict)} layers from {args.resume}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ Failed to load resume: {e}")
    
    # 8. Обёртка в DDP
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Params: {total_params:,} total, {trainable_params:,} trainable")
    
    # 9. Оптимизатор и scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()
    
    # 10. Лучшие метрики
    best_f1 = 0.0
    best_roc_auc = 0.0
    best_pr_auc = 0.0
    best_f1_epoch = 0
    best_roc_epoch = 0
    best_pr_epoch = 0
    
    # 11. TensorBoard и CSV (только на rank 0)
    if rank == 0:
        writer = None
        if HAS_TENSORBOARD:
            log_dir = os.path.join(args.log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            writer = SummaryWriter(log_dir)
            print(f"📊 TensorBoard logs: {log_dir}")
        
        csv_logger = CSVLogger(args.log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 12. Цикл обучения
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
                loss, n_pos, n_neg = compute_loss(logits, data.y, args.hard_ratio)
            
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
            
            if data is not None:
                del data
                if n_batches % 100 == 0:
                    torch.cuda.empty_cache()
        
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Метрики (только на rank 0)
        if rank == 0 and (epoch % 5 == 0 or epoch <= 10 or epoch == EPOCHS):
            print(f"\n📊 Computing metrics for epoch {epoch}...")
            
            # 12a. Train eval — DDP режим
            train_metrics = compute_metrics_ddp(
                model, train_loader, device, rank, world_size, desc="Train eval"
            )
            
            # 12b. Val eval — одна GPU, без DDP
            val_metrics = compute_metrics_single_gpu(
                model, val_loader, device, desc="Val eval"
            )
            
            # Обновляем лучшие метрики и сохраняем модели
            if val_metrics:
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_f1_epoch = epoch
                    torch.save(model.module.state_dict(), args.ckpt_f1)
                    print(f"\n  ✅ New best F1: {best_f1:.4f} (epoch {epoch})")
                
                if val_metrics['roc_auc'] > best_roc_auc:
                    best_roc_auc = val_metrics['roc_auc']
                    best_roc_epoch = epoch
                    torch.save(model.module.state_dict(), args.ckpt_roc)
                    print(f"  ✅ New best ROC-AUC: {best_roc_auc:.4f} (epoch {epoch})")
                
                if val_metrics['pr_auc'] > best_pr_auc:
                    best_pr_auc = val_metrics['pr_auc']
                    best_pr_epoch = epoch
                    torch.save(model.module.state_dict(), args.ckpt_pr)
                    print(f"  ✅ New best PR-AUC: {best_pr_auc:.4f} (epoch {epoch})")
            
            # Сохраняем периодические чекпоинты
            if epoch % 25 == 0:
                ckpt_name = f"checkpoint_epoch_{epoch}.pt"
                torch.save(model.module.state_dict(), os.path.join(args.log_dir, ckpt_name))
                print(f"  💾 Saved periodic checkpoint: {ckpt_name}")
            
            torch.save(model.module.state_dict(), "last.pt")
            
            # Вывод
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch}")
            print(f"{'='*80}")
            print(f"  Loss: {epoch_loss/n_batches:.4f}")
            print(f"  Sampling: pos={total_pos/n_batches:.0f}, neg={total_neg/n_batches:.0f} (ratio={total_neg/total_pos:.1f})")
            
            if train_metrics:
                print(f"\n  📊 TRAIN (DDP, rank 0 part):")
                print(f"     ROC-AUC: {train_metrics['roc_auc']:.4f} | PR-AUC: {train_metrics['pr_auc']:.4f}")
                print(f"     F1: {train_metrics['f1']:.4f} | P: {train_metrics['precision']:.4f} | R: {train_metrics['recall']:.4f}")
                print(f"     Pos: {train_metrics['pos_ratio']:.2%} | Pred: {train_metrics['pred_pos_ratio']:.2%}")
                print(f"     Mean prob: {train_metrics['mean_prob']:.4f}")
                print(f"     N points: {train_metrics['n_points']:,}")
            
            if val_metrics:
                print(f"\n  📊 VAL (single GPU):")
                print(f"     ROC-AUC: {val_metrics['roc_auc']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f}")
                print(f"     F1: {val_metrics['f1']:.4f} | P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f}")
                print(f"     Pos: {val_metrics['pos_ratio']:.2%} | Pred: {val_metrics['pred_pos_ratio']:.2%}")
                print(f"     Mean prob: {val_metrics['mean_prob']:.4f}")
                print(f"     N points: {val_metrics['n_points']:,}")
            
            print(f"\n  🏆 Best so far:")
            print(f"     F1: {best_f1:.4f} (epoch {best_f1_epoch})")
            print(f"     ROC-AUC: {best_roc_auc:.4f} (epoch {best_roc_epoch})")
            print(f"     PR-AUC: {best_pr_auc:.4f} (epoch {best_pr_epoch})")
            print(f"{'='*80}\n")
            
            # TensorBoard
            if writer:
                if train_metrics:
                    for k, v in train_metrics.items():
                        writer.add_scalar(f'train/{k}', v, epoch)
                if val_metrics:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f'val/{k}', v, epoch)
                writer.add_scalar('train/loss', epoch_loss/n_batches, epoch)
                writer.add_scalar('train/sampling_ratio', total_neg/total_pos, epoch)
                writer.add_scalar('best/f1', best_f1, epoch)
                writer.add_scalar('best/roc_auc', best_roc_auc, epoch)
                writer.add_scalar('best/pr_auc', best_pr_auc, epoch)
                writer.flush()
            
            # CSV
            if train_metrics and val_metrics:
                csv_logger.log({
                    'epoch': epoch,
                    'train_loss': epoch_loss/n_batches,
                    'train_roc_auc': train_metrics['roc_auc'],
                    'train_pr_auc': train_metrics['pr_auc'],
                    'train_f1': train_metrics['f1'],
                    'train_precision': train_metrics['precision'],
                    'train_recall': train_metrics['recall'],
                    'train_pos_ratio': train_metrics['pos_ratio'],
                    'train_pred_pos_ratio': train_metrics['pred_pos_ratio'],
                    'val_roc_auc': val_metrics['roc_auc'],
                    'val_pr_auc': val_metrics['pr_auc'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_pos_ratio': val_metrics['pos_ratio'],
                    'val_pred_pos_ratio': val_metrics['pred_pos_ratio'],
                    'val_n_points': val_metrics['n_points'],
                    'sampling_pos': total_pos/n_batches,
                    'sampling_neg': total_neg/n_batches,
                    'sampling_ratio': total_neg/total_pos,
                    'best_f1': best_f1,
                    'best_roc_auc': best_roc_auc,
                    'best_pr_auc': best_pr_auc,
                })
    
    # 13. Финальный поиск порога
    if rank == 0:
        if os.path.exists(args.ckpt_f1):
            model.module.load_state_dict(torch.load(args.ckpt_f1, map_location=device))
        else:
            model.module.load_state_dict(torch.load(args.ckpt, map_location=device))
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
        y_true = (labels < CONTACT_THRESHOLD).astype(int)
        
        best_f1_thresh = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.05, 0.95, 0.05):
            preds = (probs > thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1_thresh:
                best_f1_thresh = f1
                best_thresh = thresh
        
        print(f"\n🎯 Optimal threshold search (on best F1 model):")
        print(f"   Best F1={best_f1_thresh:.4f} @ threshold={best_thresh:.2f}")
        print(f"   (Default 0.5 gave F1={f1_score(y_true, (probs>0.5).astype(int), zero_division=0):.4f})")
        
        with open(os.path.join(args.log_dir, 'final_results.txt'), 'w') as f:
            f.write(f"Best F1: {best_f1:.4f} (epoch {best_f1_epoch}) -> {args.ckpt_f1}\n")
            f.write(f"Best ROC-AUC: {best_roc_auc:.4f} (epoch {best_roc_epoch}) -> {args.ckpt_roc}\n")
            f.write(f"Best PR-AUC: {best_pr_auc:.4f} (epoch {best_pr_epoch}) -> {args.ckpt_pr}\n")
            f.write(f"Optimal threshold: {best_thresh:.2f} (F1={best_f1_thresh:.4f})\n")
        
        csv_logger.close()
        if writer:
            writer.close()
        print(f"\n✅ Logs saved to {args.log_dir}")
        print(f"   - F1 best: {args.ckpt_f1}")
        print(f"   - ROC best: {args.ckpt_roc}")
        print(f"   - PR best: {args.ckpt_pr}")
    
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed/graphs_exp3')
    parser.add_argument('--ckpt', default='best_model.pt')
    parser.add_argument('--ckpt_f1', default='best_f1.pt')
    parser.add_argument('--ckpt_roc', default='best_roc.pt')
    parser.add_argument('--ckpt_pr', default='best_pr.pt')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--n_files', type=int, default=3000)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=4.0)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint', action='store_true', default=True)
    parser.add_argument('--no_checkpoint', action='store_false', dest='checkpoint')
    parser.add_argument('--hard_ratio', type=float, default=2.0,
                    help='Hard negative mining ratio (default: 2.0)')
    args = parser.parse_args()
    
    global CONTACT_THRESHOLD, N_VAL
    CONTACT_THRESHOLD = args.threshold
    N_VAL = args.n_val
    
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"⚠️ Only {world_size} GPU found. For DDP need at least 2.")
        print("Please use train_clean.py for single GPU training.")
        sys.exit(1)
    
    print(f"🚀 Launching DDP training on {world_size} GPUs")
    mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()