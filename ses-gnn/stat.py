#!/usr/bin/env python3
"""
analyze_graph_density.py - Анализ плотности графа
Показывает, сколько соседей у каждой точки в зависимости от радиуса
"""

import os
import glob
import torch
import numpy as np
from torch_cluster import radius_graph
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_graph_density(data_dir, radii=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0], n_files=10):
    """Анализирует, сколько соседей попадает в радиус для разных белков"""
    
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))[:n_files]
    
    results = {r: [] for r in radii}
    
    for f in tqdm(files, desc="Analyzing"):
        data = torch.load(f, map_location='cpu', weights_only=False)
        
        if isinstance(data, dict):
            pos = data['pos']
        else:
            pos = data.pos
        
        n_points = len(pos)
        
        for r in radii:
            # Строим граф без ограничения max_num_neighbors
            edge_index = radius_graph(pos, r=r, max_num_neighbors=1000, loop=False)
            
            # Считаем степень каждой вершины
            degrees = torch.zeros(n_points, dtype=torch.long)
            for i in range(edge_index.shape[1]):
                src = edge_index[0, i].item()
                degrees[src] += 1
            
            # Статистика
            results[r].append({
                'file': os.path.basename(f),
                'n_points': n_points,
                'mean_degree': degrees.float().mean().item(),
                'max_degree': degrees.max().item(),
                'min_degree': degrees.min().item(),
                'pct_isolated': (degrees == 0).float().mean().item(),
                'pct_above_48': (degrees > 48).float().mean().item(),
                'pct_above_24': (degrees > 24).float().mean().item(),
            })
    
    # Вывод статистики
    print("\n" + "="*80)
    print("GRAPH DENSITY ANALYSIS")
    print("="*80)
    
    for r in radii:
        print(f"\n📊 Radius = {r}Å")
        print("-"*60)
        
        mean_degrees = [res['mean_degree'] for res in results[r]]
        pct_above_24 = [res['pct_above_24'] for res in results[r]]
        pct_above_48 = [res['pct_above_48'] for res in results[r]]
        
        print(f"  Mean degree: {np.mean(mean_degrees):.1f} ± {np.std(mean_degrees):.1f}")
        print(f"  Max degree: {max(res['max_degree'] for res in results[r])}")
        print(f"  % points with >24 neighbors: {np.mean(pct_above_24)*100:.1f}%")
        print(f"  % points with >48 neighbors: {np.mean(pct_above_48)*100:.1f}%")
        print(f"  % isolated points: {np.mean([res['pct_isolated'] for res in results[r]])*100:.1f}%")
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Средняя степень vs радиус
    ax = axes[0]
    for r in radii:
        mean_degrees = [res['mean_degree'] for res in results[r]]
        ax.scatter([r] * len(mean_degrees), mean_degrees, alpha=0.3, s=10)
    ax.boxplot([[res['mean_degree'] for res in results[r]] for r in radii], 
               labels=[f"{r}Å" for r in radii])
    ax.set_xlabel('Radius')
    ax.set_ylabel('Mean degree')
    ax.set_title('Mean number of neighbors vs radius')
    ax.axhline(y=24, color='r', linestyle='--', label='Current MAX_NEIGHBORS=24')
    ax.axhline(y=48, color='g', linestyle='--', label='Proposed MAX_NEIGHBORS=48')
    ax.legend()
    
    # 2. Доля точек с >48 соседей
    ax = axes[1]
    pct_above_48 = []
    for r in radii:
        pct_above_48.append(np.mean([res['pct_above_48'] for res in results[r]]) * 100)
    ax.bar([f"{r}Å" for r in radii], pct_above_48)
    ax.set_xlabel('Radius')
    ax.set_ylabel('% points with >48 neighbors')
    ax.set_title('Points exceeding max_neighbors limit')
    ax.axhline(y=10, color='r', linestyle='--', label='10% threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('graph_density_analysis.png', dpi=150)
    print(f"\n✅ Saved graph_density_analysis.png")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./processed/raw')
    parser.add_argument('--n_files', type=int, default=10)
    args = parser.parse_args()
    
    analyze_graph_density(args.data_dir, n_files=args.n_files)