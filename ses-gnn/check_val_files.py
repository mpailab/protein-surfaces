#!/usr/bin/env python3
"""
check_val_files.py - Проверка файлов, которые попадают в валидацию
Показывает размеры и интерфейсную долю для файлов 5000-5100
"""

import os
import glob
import random
import numpy as np
import torch
from collections import defaultdict

CONTACT_THRESHOLD = 4.0
RANDOM_SEED = 42

def analyze_files(data_dir, start_idx=5000, end_idx=5100):
    """Анализирует файлы в указанном диапазоне"""
    
    # Загружаем все файлы с тем же порядком, что в train_clean_ddp.py
    all_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    random.Random(RANDOM_SEED).shuffle(all_paths)
    
    # Берём нужный диапазон
    val_files = all_paths[start_idx:end_idx]
    
    print("="*80)
    print(f"FILES {start_idx}-{end_idx} (total: {len(val_files)})")
    print("="*80)
    
    stats = []
    
    for i, f in enumerate(val_files):
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            if isinstance(data, dict):
                y = data['y'].numpy() if torch.is_tensor(data['y']) else data['y']
                pdb_id = data.get('pdb_id', os.path.basename(f).replace('.pt', ''))
                mode = data.get('mode', 'unknown')
            else:
                y = data.y.numpy()
                pdb_id = getattr(data, 'pdb_id', os.path.basename(f).replace('.pt', ''))
                mode = getattr(data, 'mode', 'unknown')
            
            n_points = len(y)
            interface_mask = y < CONTACT_THRESHOLD
            n_interface = interface_mask.sum()
            interface_ratio = n_interface / n_points if n_points > 0 else 0
            
            stats.append({
                'index': start_idx + i,
                'pdb_id': pdb_id,
                'mode': mode,
                'n_points': n_points,
                'n_interface': n_interface,
                'interface_ratio': interface_ratio,
                'y_min': y.min(),
                'y_max': y.max(),
                'y_mean': y.mean(),
                'file': os.path.basename(f),
            })
            
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")
    
    # Сортируем по размеру (убывание)
    stats_sorted = sorted(stats, key=lambda x: x['n_points'], reverse=True)
    
    # Вывод
    print(f"\n📊 Total files: {len(stats)}")
    print(f"   Total points: {sum(s['n_points'] for s in stats):,}")
    print(f"   Total interface points: {sum(s['n_interface'] for s in stats):,}")
    print(f"   Avg interface ratio: {np.mean([s['interface_ratio'] for s in stats]):.2%}")
    print(f"   Max points: {max(s['n_points'] for s in stats):,}")
    print(f"   Min points: {min(s['n_points'] for s in stats):,}")
    
    # Крупные белки (>30000 точек)
    large = [s for s in stats if s['n_points'] > 30000]
    if large:
        print(f"\n🔴 LARGE PROTEINS (>30k points): {len(large)}")
        for s in large:
            print(f"   {s['pdb_id']:<20} {s['n_points']:,} points, interface: {s['interface_ratio']:.2%}")
    
    # Белки с малым интерфейсом (<5%)
    small_interface = [s for s in stats if s['interface_ratio'] < 0.05]
    if small_interface:
        print(f"\n🟡 SMALL INTERFACE (<5%): {len(small_interface)}")
        for s in small_interface[:10]:
            print(f"   {s['pdb_id']:<20} {s['n_points']:,} points, interface: {s['interface_ratio']:.2%}")
    
    # Топ-10 самых больших
    print("\n📊 TOP 10 LARGEST PROTEINS:")
    print("-"*60)
    for s in stats_sorted[:10]:
        print(f"  {s['index']}: {s['pdb_id']:<20} {s['n_points']:,} points, interface: {s['interface_ratio']:.2%}, mode: {s['mode']}")
    
    # Топ-10 самых маленьких
    print("\n📊 TOP 10 SMALLEST PROTEINS:")
    print("-"*60)
    for s in stats_sorted[-10:]:
        print(f"  {s['index']}: {s['pdb_id']:<20} {s['n_points']:,} points, interface: {s['interface_ratio']:.2%}, mode: {s['mode']}")
    
    # Распределение интерфейсной доли
    ratios = [s['interface_ratio'] for s in stats]
    print(f"\n📊 Interface ratio distribution:")
    print(f"   Mean: {np.mean(ratios):.2%}")
    print(f"   Median: {np.median(ratios):.2%}")
    print(f"   Std: {np.std(ratios):.2%}")
    print(f"   Min: {np.min(ratios):.2%}")
    print(f"   Max: {np.max(ratios):.2%}")
    
    return stats

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed/graphs_exp3')
    parser.add_argument('--start', type=int, default=5000)
    parser.add_argument('--end', type=int, default=5050)
    args = parser.parse_args()
    
    analyze_files(args.data_dir, args.start, args.end)