#!/usr/bin/env python3
"""
preprocess_ses.py - Универсальная предобработка для SES поверхности
Поддерживает:
- Small molecule лиганды: формат PDBID_CHAIN (расстояние до атомов лиганда)
- PPI: формат PDBID_CHAIN1_CHAIN2 (расстояние до SES поверхности другого белка)

Для ВСЕХ режимов сохраняет y = расстояние до ближайшей точки (атома или SES)
Бинаризация выполняется при обучении.
"""

import os
import sys
import gc
import glob
import argparse
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'H': 1.20}
PROBE_RADIUS = 1.4
POINT_AREA = 1.0
ATOM_FILTER_SAMPLES = 256
PAIR_FILTER_SAMPLES = 24
MAX_PROBE_TRIPLES = 3_000_000

WATER_NAMES = {'HOH', 'WAT', 'H2O'}
ION_NAMES = {'NA', 'K', 'CA', 'MG', 'CL', 'ZN', 'FE', 'CU', 'MN', 'CO', 'NI', 'HG'}

ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
NUM_ELEM_TYPES = 6


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_coords(coords: torch.Tensor, tol=1e-3):
    mask = torch.isfinite(coords).all(dim=1)
    coords = coords[mask]
    if coords.shape[0] < 2:
        return coords
    
    keys = torch.round(coords / tol)
    unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
    first_idx = torch.full((unique_keys.shape[0],), coords.shape[0], dtype=torch.long, device=coords.device)
    first_idx.scatter_reduce_(0, inverse, torch.arange(coords.shape[0], device=coords.device), reduce='amin')
    
    return coords[first_idx]


def load_chain_coords_with_elements(pdb_path: str, chain_ids: list):
    """Загружает атомы указанных цепей с элементами (только MODEL 1)"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', pdb_path)
    model = structure[0]
    
    coords, radii, elem_onehot = [], [], []
    target_chains = set(chain_ids)
    
    for chain in model:
        if chain.id in target_chains:
            for atom in chain.get_atoms():
                if atom.element == 'H':
                    continue
                coords.append(atom.get_coord())
                radii.append(VDW_RADII.get(atom.element, 1.70))
                vec = torch.zeros(NUM_ELEM_TYPES, dtype=torch.float32)
                vec[ELE2NUM.get(atom.element.upper(), 0)] = 1.0
                elem_onehot.append(vec)
    
    if not coords:
        raise RuntimeError(f"No heavy atoms in chains {chain_ids}")
    
    coords = torch.tensor(np.array(coords, dtype=np.float32))
    radii = torch.tensor(np.array(radii, dtype=np.float32))
    elem_onehot = torch.stack(elem_onehot)
    
    return coords, radii, elem_onehot


def extract_ligand_coords_small_molecule(pdb_path: str):
    """Извлекает координаты лиганда для small molecule (HETATM)"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', pdb_path)
    model = structure[0]
    lig_coords = []
    
    for chain in model:
        for residue in chain:
            if residue.id[0].strip() == '':
                continue
            resname = residue.get_resname().strip()
            if resname in WATER_NAMES or resname in ION_NAMES:
                continue
            for atom in residue:
                lig_coords.append(atom.get_coord())
    
    if lig_coords:
        return torch.from_numpy(np.array(lig_coords, dtype=np.float32))
    return torch.empty(0, 3)


def parse_dataset_entry(line: str):
    parts = line.strip().split('_')
    if len(parts) == 2:
        return {
            'pdb_id': parts[0],
            'mode': 'ligand',
            'receptor_chains': list(parts[1]),
            'ligand_chains': []
        }
    elif len(parts) == 3:
        return {
            'pdb_id': parts[0],
            'mode': 'ppi',
            'receptor_chains': list(parts[1]),
            'ligand_chains': list(parts[2])
        }
    raise ValueError(f"Unknown format: {line}")


# ─────────────────────────────────────────────────────────────────────────────
# SES SURFACE BUILDING
# ─────────────────────────────────────────────────────────────────────────────
def build_ses_surface(coords, radii, elements, device):
    """Строит SES поверхность и возвращает точки, нормали, признаки"""
    from ses.analytic import _sample_analytic_samples
    
    samples = _sample_analytic_samples(
        coords, radii,
        probe_radius=PROBE_RADIUS,
        point_area=POINT_AREA,
        include_normals=True,
        include_atom_weights=False,
        atom_filter_samples=ATOM_FILTER_SAMPLES,
        pair_filter_samples=PAIR_FILTER_SAMPLES,
        max_probe_triples=MAX_PROBE_TRIPLES
    )
    
    pts = samples.points
    norms = samples.normals
    block_types = samples.block_types
    
    if pts.numel() == 0:
        raise ValueError("SES sampling failed")
    if norms is None:
        norms = torch.zeros_like(pts)
    
    patch_onehot = F.one_hot((block_types - 1).clamp(0, 2), num_classes=3).float()
    
    support_idx = samples.support_indices.clamp_min(0)
    support_mask = samples.support_mask
    elem_per_support = elements[support_idx] * support_mask.unsqueeze(-1)
    elem_multihot = elem_per_support.sum(dim=1).clamp(0.0, 1.0)
    
    chem_weights = torch.tensor([
        [ 0.0,  1.0],  # C
        [ 0.0,  0.0],  # H
        [-0.5,  0.0],  # O
        [-0.3,  0.0],  # N
        [ 0.0,  0.5],  # S
        [ 0.0,  0.5]   # SE
    ], dtype=elements.dtype, device=elements.device)
    
    atom_chem = elements @ chem_weights
    chem_per_support = atom_chem[support_idx] * support_mask.unsqueeze(-1)
    chem_sum = chem_per_support.sum(dim=1)
    chem_count = support_mask.sum(dim=1, keepdim=True).clamp_min(1)
    chem_props = chem_sum / chem_count
    
    x = torch.cat([norms, patch_onehot, elem_multihot, chem_props], dim=-1)
    
    return pts, norms, x


def build_ses_raw_small_molecule(receptor_coords, receptor_radii, atom_elements, 
                                  target_coords, pdb_id, mode):
    """Для small molecule: расстояние до атомов лиганда"""
    from ses.analytic import _sample_analytic_samples
    
    samples = _sample_analytic_samples(
        receptor_coords, receptor_radii,
        probe_radius=PROBE_RADIUS,
        point_area=POINT_AREA,
        include_normals=True,
        include_atom_weights=False,
        atom_filter_samples=ATOM_FILTER_SAMPLES,
        pair_filter_samples=PAIR_FILTER_SAMPLES,
        max_probe_triples=MAX_PROBE_TRIPLES
    )
    
    pts = samples.points
    norms = samples.normals
    block_types = samples.block_types
    
    if pts.numel() == 0:
        raise ValueError("SES sampling failed")
    if norms is None:
        norms = torch.zeros_like(pts)
    
    patch_onehot = F.one_hot((block_types - 1).clamp(0, 2), num_classes=3).float()
    
    support_idx = samples.support_indices.clamp_min(0)
    support_mask = samples.support_mask
    elem_per_support = atom_elements[support_idx] * support_mask.unsqueeze(-1)
    elem_multihot = elem_per_support.sum(dim=1).clamp(0.0, 1.0)
    
    chem_weights = torch.tensor([
        [ 0.0,  1.0], [ 0.0,  0.0], [-0.5,  0.0],
        [-0.3,  0.0], [ 0.0,  0.5], [ 0.0,  0.5]
    ], dtype=atom_elements.dtype, device=atom_elements.device)
    
    atom_chem = atom_elements @ chem_weights
    chem_per_support = atom_chem[support_idx] * support_mask.unsqueeze(-1)
    chem_sum = chem_per_support.sum(dim=1)
    chem_count = support_mask.sum(dim=1, keepdim=True).clamp_min(1)
    chem_props = chem_sum / chem_count
    
    x = torch.cat([norms, patch_onehot, elem_multihot, chem_props], dim=-1)
    
    if len(target_coords) > 0:
        dists = torch.cdist(pts, target_coords)
        y = dists.min(dim=1)[0]
    else:
        y = torch.full((len(pts),), 99.0, dtype=torch.float32)
    
    return {
        'pos': pts.cpu(),
        'x': x.cpu(),
        'norms': norms.cpu(),
        'y': y.cpu(),
        'pdb_id': pdb_id,
        'mode': mode,
        'num_points': len(pts)
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING MAIN
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(dataset_file, pdb_dir, out_dir, max_time_per_pdb=180):
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(dataset_file, 'r') as f:
        entries = [parse_dataset_entry(line) for line in f if line.strip()]
    
    print(f"📋 Загружено {len(entries)} записей из {dataset_file}")
    
    saved, skipped, failed = 0, 0, 0
    total_start = time.time()
    
    for entry in tqdm(entries, desc="Preprocessing"):
        pdb_id = entry['pdb_id']
        mode = entry['mode']
        receptor_chains = entry['receptor_chains']
        ligand_chains = entry['ligand_chains']
        
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            skipped += 1
            continue
        
        step_start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            if mode == 'ppi':
                # 1. Загружаем атомы обоих белков
                rec_coords, rec_radii, rec_elements = load_chain_coords_with_elements(pdb_path, receptor_chains)
                lig_coords, lig_radii, lig_elements = load_chain_coords_with_elements(pdb_path, ligand_chains)
                
                rec_coords = rec_coords.to(device)
                rec_radii = rec_radii.to(device)
                rec_elements = rec_elements.to(device)
                lig_coords = lig_coords.to(device)
                lig_radii = lig_radii.to(device)
                lig_elements = lig_elements.to(device)
                
                # 2. Строим SES поверхности ДЛЯ ОБОИХ белков
                pts_A, norms_A, x_A = build_ses_surface(rec_coords, rec_radii, rec_elements, device)
                pts_B, norms_B, x_B = build_ses_surface(lig_coords, lig_radii, lig_elements, device)
                
                # 3. Вычисляем расстояние ОТ SES ДО SES!
                dists_A_to_B = torch.cdist(pts_A, pts_B).min(dim=1)[0]
                dists_B_to_A = torch.cdist(pts_B, pts_A).min(dim=1)[0]
                
                # 4. Сохраняем для белка A
                out_name_A = f"{pdb_id}_{''.join(receptor_chains)}_{''.join(ligand_chains)}.pt"
                data_A = {
                    'pos': pts_A.cpu(),
                    'x': x_A.cpu(),
                    'norms': norms_A.cpu(),
                    'y': dists_A_to_B.cpu(),
                    'pdb_id': f"{pdb_id}_{''.join(receptor_chains)}",
                    'mode': 'ppi',
                    'receptor_chains': receptor_chains,
                    'ligand_chains': ligand_chains,
                    'num_points': len(pts_A)
                }
                torch.save(data_A, os.path.join(out_dir, out_name_A))
                saved += 1
                
                # 5. Сохраняем для белка B
                out_name_B = f"{pdb_id}_{''.join(ligand_chains)}_{''.join(receptor_chains)}.pt"
                data_B = {
                    'pos': pts_B.cpu(),
                    'x': x_B.cpu(),
                    'norms': norms_B.cpu(),
                    'y': dists_B_to_A.cpu(),
                    'pdb_id': f"{pdb_id}_{''.join(ligand_chains)}",
                    'mode': 'ppi',
                    'receptor_chains': ligand_chains,
                    'ligand_chains': receptor_chains,
                    'num_points': len(pts_B)
                }
                torch.save(data_B, os.path.join(out_dir, out_name_B))
                saved += 1
                
            else:  # ligand mode
                rec_coords, rec_radii, rec_elements = load_chain_coords_with_elements(pdb_path, receptor_chains)
                rec_coords = rec_coords.to(device)
                rec_radii = rec_radii.to(device)
                rec_elements = rec_elements.to(device)
                
                lig_coords = extract_ligand_coords_small_molecule(pdb_path)
                if len(lig_coords) == 0:
                    skipped += 1
                    continue
                lig_coords = lig_coords.to(device)
                
                data = build_ses_raw_small_molecule(
                    rec_coords, rec_radii, rec_elements, lig_coords,
                    f"{pdb_id}_{''.join(receptor_chains)}", mode
                )
                data['receptor_chains'] = receptor_chains
                data['ligand_chains'] = []
                
                out_name = f"{pdb_id}_{''.join(receptor_chains)}.pt"
                torch.save(data, os.path.join(out_dir, out_name))
                saved += 1
            
            elapsed = time.time() - step_start
            if elapsed > max_time_per_pdb:
                tqdm.write(f"⚠️ {pdb_id} took {elapsed:.1f}s")
            
        except Exception as e:
            failed += 1
            tqdm.write(f"\n❌ {pdb_id} FAILED: {e}")
            traceback.print_exc(file=sys.stderr)
        
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"📊 ПРЕПРОЦЕССИНГ ЗАВЕРШЁН")
    print(f"{'='*60}")
    print(f"  Сохранено:   {saved}")
    print(f"  Пропущено:   {skipped}")
    print(f"  Ошибок:      {failed}")
    print(f"  Время:       {total_elapsed:.1f} сек")
    print(f"  Выходная папка: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--pdb_dir', type=str, default='./pdb_files')
    parser.add_argument('--out_dir', type=str, default='./processed')
    args = parser.parse_args()
    
    preprocess(args.dataset, args.pdb_dir, args.out_dir)


if __name__ == '__main__':
    main()