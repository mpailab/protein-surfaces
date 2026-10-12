#!/usr/bin/env python3
"""
preprocess_single_chain.py

Построение SES-поверхности для ОДИНОЧНОЙ цепи (или нескольких цепей,
рассматриваемых как единый рецептор, например 'BG').

Используется для задачи binding site identification (как в dMaSIF):
на вход подаётся одна цепь, партнёр на этапе инференса неизвестен.

Формат входного списка (по одной строке):
    1EZ1_B
    1F06_A
    1GP2_BG     # гетеродимер B+G как единый рецептор
    ...

Отличие от preprocess.py:
  - Вход: PDBID_CHAIN (без второго партнёра)
  - y НЕ вычисляется (ставится 0). Метрики считаются по разметке dMaSIF
    через матчинг точек (см. evaluate_on_dmasif.py).
  - Формат сохранения совместим с build_graphs.py и evaluate_on_dmasif.py.

Пример:
    python preprocess_single_chain.py \
        --list testing.txt \
        --pdb_dir ./pdb_files \
        --out_dir ./processed/raw_test_single
"""

import os
import sys
import gc
import time
import argparse
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (идентично preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80, 'H': 1.20}
PROBE_RADIUS = 1.4
POINT_AREA = 1.0
ATOM_FILTER_SAMPLES = 256
PAIR_FILTER_SAMPLES = 24
MAX_PROBE_TRIPLES = 3_000_000

ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
NUM_ELEM_TYPES = 6


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES (идентично preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────
def load_chain_coords_with_elements(pdb_path: str, chain_ids: list):
    """Загружает тяжёлые атомы указанных цепей с элементами (MODEL 1)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# SES SURFACE BUILDING (идентично preprocess.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_ses_surface(coords, radii, elements, device):
    """Строит SES-поверхность; возвращает pts, norms, x (14-D)."""
    from ses.analytic import _sample_analytic_samples

    samples = _sample_analytic_samples(
        coords, radii,
        probe_radius=PROBE_RADIUS,
        point_area=POINT_AREA,
        include_normals=True,
        include_atom_weights=False,
        atom_filter_samples=ATOM_FILTER_SAMPLES,
        pair_filter_samples=PAIR_FILTER_SAMPLES,
        max_probe_triples=MAX_PROBE_TRIPLES,
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
        [ 0.0,  0.5],  # SE
    ], dtype=elements.dtype, device=elements.device)

    atom_chem = elements @ chem_weights
    chem_per_support = atom_chem[support_idx] * support_mask.unsqueeze(-1)
    chem_sum = chem_per_support.sum(dim=1)
    chem_count = support_mask.sum(dim=1, keepdim=True).clamp_min(1)
    chem_props = chem_sum / chem_count

    x = torch.cat([norms, patch_onehot, elem_multihot, chem_props], dim=-1)

    return pts, norms, x


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_single_chain_entry(line):
    """
    '1acb_I'   -> ('1acb', 'I')
    '2WD5A'    -> ('2WD5', 'A')     # DeepPPIPS dset164
    '1k5d_AB'  -> ('1k5d', 'AB')
    """
    line = line.strip()
    if '_' in line:
        parts = line.split('_', 1)
        return parts[0], parts[1]
    else:
        # Формат PDB_CHAIN без '_', например 2WD5A
        return line[:4], line[4:]

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def process_one(pdb_id: str, chain_str: str, pdb_dir: str, out_dir: str,
                force_cpu: bool = False):
    pdb_path = None
    for candidate in [f"{pdb_id}.pdb",
                      f"{pdb_id.upper()}.pdb",
                      f"{pdb_id.lower()}.pdb"]:
        cand = os.path.join(pdb_dir, candidate)
        if os.path.isfile(cand):
            pdb_path = cand
            break
    if pdb_path is None:
        raise FileNotFoundError(f"Missing PDB: {pdb_dir}/{pdb_id}.pdb")

    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 'BG' -> ['B','G'], 'B' -> ['B']
    chain_list = list(chain_str)

    coords, radii, elems = load_chain_coords_with_elements(pdb_path, chain_list)
    coords = coords.to(device)
    radii = radii.to(device)
    elems = elems.to(device)

    pts, norms, x = build_ses_surface(coords, radii, elems, device)

    # y — placeholder. Для инференса не используется.
    # Метрики считаются по разметке dMaSIF через матчинг точек.
    y = torch.zeros(len(pts), dtype=torch.float32)

    data = {
        'pos': pts.cpu(),
        'x': x.cpu(),
        'norms': norms.cpu(),
        'y': y,
        'pdb_id': f"{pdb_id}_{chain_str}",
        'mode': 'ppi',                          # для консистентности с preprocess.py
        'receptor_chains': chain_list,          # ['B'] или ['B','G']
        'ligand_chains': [],                    # партнёра нет
        'num_points': int(len(pts)),
    }

    out_name = f"{pdb_id}_{chain_str}.pt"
    torch.save(data, os.path.join(out_dir, out_name))
    return out_name


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(list_file: str, pdb_dir: str, out_dir: str, force_cpu: bool = False):
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(list_file, 'r') as f:
        entries = [ln.strip() for ln in f if ln.strip()]

    print(f"📋 Загружено {len(entries)} записей из {list_file}")

    saved, skipped, failed = 0, 0, 0
    failed_details = []
    total_start = time.time()

    for line in tqdm(entries, desc="Preprocessing"):
        try:
            pdb_id, chain_str = parse_single_chain_entry(line)
        except Exception as e:
            failed += 1
            failed_details.append((line, f"parse: {e}"))
            tqdm.write(f"❌ {line}: parse error: {e}")
            continue

        # Ищем PDB в разных регистрах
        pdb_path = None
        for candidate in [f"{pdb_id}.pdb",
                          f"{pdb_id.upper()}.pdb",
                          f"{pdb_id.lower()}.pdb"]:
            cand = os.path.join(pdb_dir, candidate)
            if os.path.isfile(cand):
                pdb_path = cand
                break

        if pdb_path is None:
            skipped += 1
            tqdm.write(f"⚠️  {line}: missing PDB file {pdb_dir}/{pdb_id}.pdb")
            continue

        try:
            process_one(pdb_id, chain_str, pdb_dir, out_dir, force_cpu=force_cpu)
            saved += 1
        except Exception as e:
            failed += 1
            failed_details.append((line, str(e)))
            tqdm.write(f"❌ {line} FAILED: {e}")
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

    if failed_details:
        print(f"\n  Первые 10 ошибок:")
        for name, why in failed_details[:10]:
            print(f"     {name}: {why}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, required=True,
                        help='Список PDBID_CHAIN (по одному на строку)')
    parser.add_argument('--pdb_dir', type=str, default='./pdb_files')
    parser.add_argument('--out_dir', type=str, default='./processed/raw_single')
    parser.add_argument('--cpu', action='store_true',
                        help='Принудительно использовать CPU (медленнее, но без OOM)')
    args = parser.parse_args()

    preprocess(args.list, args.pdb_dir, args.out_dir, force_cpu=args.cpu)


if __name__ == '__main__':
    main()