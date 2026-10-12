#!/usr/bin/env python3
"""
assign_residue_ids.py

Для каждого .pt файла (с pos точек SES) восстанавливает residue_id
для каждой точки через top-K radius-corrected KD-tree по тяжёлым атомам PDB.

Логика:
  1. Строим KD-tree по тяжёлым атомам PDB.
  2. Для каждой точки SES находим top-K ближайших атомов.
  3. Для каждого кандидата считаем deviation = |dist - vdw_radius(atom)|
  4. Выбираем атом с минимальным deviation.
  5. Присваиваем точке residue_id этого атома.

Сохраняем в новом каталоге, НЕ трогая исходные .pt.

Пример:
    python assign_residue_ids.py \
        --graph_dir ./processed/graphs_test_single \
        --pdb_dir ./pdb_files \
        --out_dir ./processed/graphs_test_residue \
        --top_k 5
"""

import os
import re
import glob
import argparse
import numpy as np
import torch
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from tqdm import tqdm


# Ван-дер-ваальсовы радиусы (Å) — те же, что в preprocess.py
VDW_RADII = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
             'P': 1.80, 'H': 1.20, 'SE': 1.90}


def parse_pdb_atoms(pdb_path, chain_ids):
    """
    Возвращает список атомов с полями:
      coords  [N, 3] float32
      radius  [N]    float32
      res_id  [N]    int (уникальный residue_id в формате (chain, resnum, icode))
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('mol', pdb_path)
    model = structure[0]

    coords, radii, res_ids = [], [], []
    target = set(chain_ids)

    for chain in model:
        if chain.id not in target:
            continue
        for residue in chain:
            # Пропускаем HETATM (воды, ионы, лиганды) — только standard amino acids
            if residue.id[0].strip() != '':
                continue
            # residue.id = (hetflag, resseq, icode)
            res_key = (chain.id, residue.id[1], residue.id[2].strip())
            for atom in residue:
                if atom.element == 'H':
                    continue
                coords.append(atom.get_coord())
                r = VDW_RADII.get(atom.element.upper(), 1.70)
                radii.append(r)
                res_ids.append(res_key)

    if not coords:
        raise RuntimeError(f"No heavy atoms in {pdb_path} chains {chain_ids}")

    coords = np.array(coords, dtype=np.float32)
    radii = np.array(radii, dtype=np.float32)
    # Уникальные residue_id через словарь
    unique_res = {k: i for i, k in enumerate(sorted(set(res_ids)))}
    res_ids_int = np.array([unique_res[k] for k in res_ids], dtype=np.int64)

    # Карта: int → (chain, resnum, icode)
    id_to_res = {v: k for k, v in unique_res.items()}

    return coords, radii, res_ids_int, id_to_res


def extract_chain_ids(pdb_id, graph_basename):
    """
    Граф называется, например, '3LIZ_HL_A.pt'.
    Из имени берём цепь 'HL' — но PDB должен содержать и A, и HL.
    В graphs_test_single имена — PDB_CHAIN1_CHAIN2 (для PPI) или PDB_CHAIN.
    Точки в графе — это SES цепи A (первая). Значит, для residue ID
    нужны цепи, которые «создают» эту SES — receptor_chains.
    """
    parts = graph_basename.split('_')
    if len(parts) >= 2:
        # Например 3LIZ_HL_A → receptor = 'HL'
        # или 1A0G_A_B → receptor = 'A'
        return list(parts[1])
    return None


def find_pdb(pdb_dir, pdb_id):
    """Ищет PDB в разных регистрах."""
    for name in [f"{pdb_id}.pdb",
                 f"{pdb_id.upper()}.pdb",
                 f"{pdb_id.lower()}.pdb"]:
        cand = os.path.join(pdb_dir, name)
        if os.path.isfile(cand):
            return cand
    return None

def assign_for_one(graph_path, pdb_dir, out_path, top_k=5):
    """
    Читает graph .pt, PDB, присваивает residue_id, сохраняет.
    """
    g = torch.load(graph_path, map_location='cpu', weights_only=False)
    pos = g['pos'].numpy().astype(np.float32)  # [N, 3]
    N = len(pos)

    graph_basename = os.path.splitext(os.path.basename(graph_path))[0]
    pdb_id = graph_basename.split('_')[0]
    chain_ids = extract_chain_ids(pdb_id, graph_basename)

    pdb_path = find_pdb(pdb_dir, pdb_id)
    if pdb_path is None:
        raise FileNotFoundError(f"PDB not found for {graph_basename}")
    if chain_ids is None:
        raise ValueError(f"Cannot parse chain_ids from {graph_basename}")

    atom_coords, atom_radii, atom_res_ids, id_to_res = parse_pdb_atoms(
        pdb_path, chain_ids
    )

    # KD-tree по атомам
    tree = cKDTree(atom_coords)
    k = min(top_k, len(atom_coords))
    dists, idxs = tree.query(pos, k=k)   # [N, k]

    # deviation = |dist - vdw_radius(atom)|
    # atom_radii[idxs] — [N, k]
    deviations = np.abs(dists - atom_radii[idxs])

    # Индекс лучшего кандидата в каждой строке
    best_k = np.argmin(deviations, axis=1)  # [N]
    best_atom_idx = idxs[np.arange(N), best_k]  # [N]
    residue_ids = atom_res_ids[best_atom_idx]  # [N]

    # Сохраняем в g
    g['residue_id'] = torch.from_numpy(residue_ids).long()
    g['num_residues'] = len(id_to_res)
    g['residue_id_map'] = id_to_res  # dict: int → (chain, resnum, icode)

    torch.save(g, out_path)
    return N, len(np.unique(residue_ids)), len(id_to_res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph_dir', required=True,
                    help='Папка с нашими .pt (pos, x, y)')
    ap.add_argument('--pdb_dir', required=True,
                    help='Папка с PDB-файлами')
    ap.add_argument('--out_dir', required=True,
                    help='Куда сохранять .pt с residue_id')
    ap.add_argument('--top_k', type=int, default=5,
                    help='Top-K ближайших атомов для radius-corrected выбора')
    ap.add_argument('--max_files', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    graph_paths = sorted(glob.glob(os.path.join(args.graph_dir, '*.pt')))
    if args.max_files:
        graph_paths = graph_paths[:args.max_files]
    print(f"📂 Graphs: {len(graph_paths)}")

    saved = 0
    failed = []
    stats = []

    for gp in tqdm(graph_paths, desc="Assigning residue_id"):
        base = os.path.splitext(os.path.basename(gp))[0]
        out_path = os.path.join(args.out_dir, f"{base}.pt")
        if os.path.exists(out_path):
            saved += 1
            continue
        try:
            n_pts, n_used_res, n_total_res = assign_for_one(
                gp, args.pdb_dir, out_path, top_k=args.top_k
            )
            saved += 1
            stats.append({
                'id': base,
                'n_points': n_pts,
                'n_used_residues': n_used_res,
                'n_total_residues': n_total_res,
            })
        except Exception as e:
            failed.append((base, str(e)))

    print(f"\n{'='*60}")
    print(f"  ✅ Saved:   {saved}")
    print(f"  ❌ Failed:  {len(failed)}")
    print(f"{'='*60}")

    if failed:
        print("\n  First 10 failures:")
        for name, err in failed[:10]:
            print(f"    {name}: {err}")

    # Статистика
    if stats:
        import json
        with open(os.path.join(args.out_dir, '_residue_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)

        n_pts_arr = np.array([s['n_points'] for s in stats])
        n_res_arr = np.array([s['n_used_residues'] for s in stats])
        print(f"\n  Средне точек на белок:   {n_pts_arr.mean():.0f}")
        print(f"  Средне остатков (used):  {n_res_arr.mean():.0f}")
        print(f"  Минимум остатков:        {n_res_arr.min()}")
        print(f"  Максимум остатков:       {n_res_arr.max()}")


if __name__ == '__main__':
    main()