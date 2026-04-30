from pathlib import Path

import pytest
import torch

from data import read_pdb_tensors
from molecule.examples import get_molecule
from ses_blocks import (
    ATOM_BLOCK_TYPE,
    PAIR_BLOCK_TYPE,
    PROBE_BLOCK_TYPE,
    build_analytic_ses_blocks,
    sample_analytic_ses_points,
    sample_atom_contact_blocks,
    sample_pair_torus_blocks,
    sample_probe_sphere_blocks,
)


def _nearest_neighbor_median(points: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(points, points)
    distances[distances == 0] = float("inf")
    return distances.min(dim=1).values.median()


def _three_atom_cavity(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.3, 0.0],
        ],
        dtype=dtype,
    )
    radii = torch.full((3,), 1.5, dtype=dtype)
    return coords, radii


def _water_atoms(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=dtype,
    )
    radii = torch.tensor([1.52, 1.20, 1.20], dtype=dtype)
    return coords, radii


def test_sample_analytic_ses_points_returns_support_metadata() -> None:
    coords, radii = _three_atom_cavity()

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=5.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        include_atom_weights=True,
        max_grid_points=100_000,
    )

    assert samples.points.ndim == 2
    assert samples.points.shape[1] == 3
    assert samples.points.shape[0] > 0
    assert torch.isfinite(samples.points).all()
    assert samples.support_indices.shape == samples.support_weights.shape
    assert samples.support_mask.shape == samples.support_indices.shape
    assert samples.support_indices.shape[0] == samples.points.shape[0]
    assert samples.atom_weights is not None
    assert samples.atom_weights.shape == (samples.points.shape[0], coords.shape[0])
    assert torch.allclose(
        samples.atom_weights.sum(dim=-1),
        torch.ones(samples.points.shape[0], dtype=samples.points.dtype),
    )
    assert (samples.support_indices[samples.support_mask] >= 0).all()
    assert set(samples.block_types.tolist()).issubset(
        {ATOM_BLOCK_TYPE, PAIR_BLOCK_TYPE, PROBE_BLOCK_TYPE},
    )


def test_sample_analytic_ses_points_preserves_gradient_path() -> None:
    coords, radii = _three_atom_cavity(dtype=torch.float32)
    coords.requires_grad_(True)
    radii.requires_grad_(True)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=6.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        max_grid_points=100_000,
    )

    assert samples.points.shape[0] > 0
    samples.points.square().sum().backward()

    assert coords.grad is not None
    assert radii.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert torch.isfinite(radii.grad).all()
    assert float(coords.grad.abs().sum()) > 0
    assert float(radii.grad.abs().sum()) > 0


def test_analytic_blocks_can_be_sampled_independently() -> None:
    coords, radii = _three_atom_cavity()

    blocks, context = build_analytic_ses_blocks(
        coords,
        radii,
        1.4,
        atom_filter_samples=16,
        pair_filter_samples=6,
        max_grid_points=100_000,
    )
    atom_samples = sample_atom_contact_blocks(
        context,
        blocks.atom_indices,
        point_area=8.0,
    )
    pair_samples = sample_pair_torus_blocks(
        context,
        blocks.pair_indices,
        point_area=8.0,
    )
    probe_samples = sample_probe_sphere_blocks(
        context,
        blocks,
        point_area=8.0,
    )

    assert blocks.num_blocks > 0
    assert atom_samples.support_indices.shape[1] == 1
    assert pair_samples.support_indices.shape[1] == 2
    assert probe_samples.support_indices.shape[1] == blocks.probe_support_indices.shape[1]
    assert atom_samples.points.shape[0] + pair_samples.points.shape[0] + probe_samples.points.shape[0] > 0


def test_water_probe_sphere_samples_are_mirrored_across_molecular_plane() -> None:
    coords, radii = _water_atoms()

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.01,
        atom_filter_samples=128,
        pair_filter_samples=32,
        max_grid_points=200_000,
    )
    probe_mask = samples.block_types == PROBE_BLOCK_TYPE
    probe_block_ids, probe_counts = torch.unique(
        samples.block_indices[probe_mask],
        return_counts=True,
    )

    assert probe_block_ids.numel() == 2
    assert probe_counts[0] == probe_counts[1]

    first_points = samples.points[probe_mask & (samples.block_indices == probe_block_ids[0])]
    second_points = samples.points[probe_mask & (samples.block_indices == probe_block_ids[1])]
    assert torch.allclose(first_points[:, :2], second_points[:, :2], atol=1e-12)
    assert torch.allclose(first_points[:, 2], -second_points[:, 2], atol=1e-12)

    hydrogen_exchange_axis = coords[1] / torch.linalg.norm(coords[1]) - coords[2] / torch.linalg.norm(coords[2])
    hydrogen_exchange_axis = hydrogen_exchange_axis / torch.linalg.norm(hydrogen_exchange_axis)
    first_signed_offsets = first_points @ hydrogen_exchange_axis
    second_signed_offsets = second_points @ hydrogen_exchange_axis
    assert abs(float(first_signed_offsets.mean())) < 1e-5
    assert abs(float(second_signed_offsets.mean())) < 1e-5


def test_ethanol_default_filters_keep_narrow_reentrant_blocks() -> None:
    molecule = get_molecule("Ethanol")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    blocks, _ = build_analytic_ses_blocks(
        coords,
        radii,
        1.4,
        max_grid_points=200_000,
    )

    assert blocks.atom_indices.shape[0] == len(molecule["elements"])
    assert blocks.pair_indices.shape[0] >= 22
    assert blocks.probe_seed_indices.shape[0] >= 24


def test_benzene_uses_unfiltered_pair_graph_for_probe_blocks() -> None:
    molecule = get_molecule("Benzene")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.1,
        max_grid_points=200_000,
    )

    assert samples.blocks is not None
    assert samples.blocks.pair_indices.shape[0] > 0
    assert samples.blocks.probe_seed_indices.shape[0] > 0
    assert bool((samples.block_types == PROBE_BLOCK_TYPE).any().item())

    probe_mask = samples.block_types == PROBE_BLOCK_TYPE
    support_counts = samples.support_mask[probe_mask].sum(dim=-1)
    positive_weight_counts = (
        (samples.support_weights[probe_mask] > 1e-12) & samples.support_mask[probe_mask]
    ).sum(dim=-1)
    assert bool((support_counts > 3).any().item())
    assert bool((positive_weight_counts > 0).all().item())

    probe_points = samples.points[probe_mask]
    rounded_probe_points = torch.round(probe_points * 1e8).to(torch.long)
    assert torch.unique(rounded_probe_points, dim=0).shape[0] == probe_points.shape[0]


def test_probe_density_scale_increases_probe_patch_samples() -> None:
    molecule = get_molecule("Benzene")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    base_samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=200_000,
    )
    denser_samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.5,
        max_grid_points=200_000,
    )

    base_probe_count = int((base_samples.block_types == PROBE_BLOCK_TYPE).sum().item())
    denser_probe_count = int((denser_samples.block_types == PROBE_BLOCK_TYPE).sum().item())
    assert denser_probe_count > base_probe_count


def test_cyclohexane_probe_triangles_include_interior_samples() -> None:
    molecule = get_molecule("Cyclohexane chair")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=200_000,
    )

    probe_mask = samples.block_types == PROBE_BLOCK_TYPE
    assert bool(probe_mask.any().item())

    positive_weight_counts = (
        (samples.support_weights > 1e-12) & samples.support_mask
    ).sum(dim=-1)
    for block_id in torch.unique(samples.block_indices[probe_mask]):
        block_mask = probe_mask & (samples.block_indices == block_id)
        assert bool((positive_weight_counts[block_mask] >= 3).any().item())


def test_cyclohexane_default_filter_keeps_small_atom_contact_patches() -> None:
    molecule = get_molecule("Cyclohexane chair")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=200_000,
    )

    assert samples.blocks is not None
    assert samples.blocks.atom_indices.shape[0] == len(molecule["elements"])

    atom_mask = samples.block_types == ATOM_BLOCK_TYPE
    owner_counts = torch.bincount(
        samples.support_indices[atom_mask, 0],
        minlength=len(molecule["elements"]),
    )
    assert bool((owner_counts > 0).all().item())


def test_cyclohexane_probe_density_matches_atom_and_pair_defaults() -> None:
    molecule = get_molecule("Cyclohexane chair")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=200_000,
    )

    atom_median = _nearest_neighbor_median(samples.points[samples.block_types == ATOM_BLOCK_TYPE])
    pair_median = _nearest_neighbor_median(samples.points[samples.block_types == PAIR_BLOCK_TYPE])
    probe_median = _nearest_neighbor_median(samples.points[samples.block_types == PROBE_BLOCK_TYPE])

    assert abs(float(probe_median - atom_median)) < 0.02
    assert abs(float(probe_median - pair_median)) < 0.02


def test_hexane_multi_support_probe_patches_cover_full_hull() -> None:
    molecule = get_molecule("C6H14")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=500_000,
    )

    assert samples.blocks is not None
    support_counts = samples.blocks.probe_support_mask.sum(dim=-1)
    large_block_ids = (support_counts == 6).nonzero(as_tuple=False).reshape(-1)
    assert large_block_ids.numel() == 2

    for block_id in large_block_ids:
        block_mask = (
            (samples.block_types == PROBE_BLOCK_TYPE)
            & (samples.block_indices == block_id)
        )
        points = samples.points[block_mask]
        assert points.shape[0] >= 80

        center = samples.blocks.probe_center_hints[block_id].to(
            dtype=points.dtype,
            device=points.device,
        )
        support_indices = samples.blocks.probe_support_indices[
            block_id,
            samples.blocks.probe_support_mask[block_id],
        ]
        normals = (center.unsqueeze(0) - coords[support_indices]) / (
            radii[support_indices] + 1.4
        ).unsqueeze(-1)
        anchor = normals.mean(dim=0)
        anchor = anchor / torch.linalg.norm(anchor)
        directions = center.unsqueeze(0) - points
        directions = directions / torch.linalg.norm(
            directions,
            dim=-1,
            keepdim=True,
        )
        anchor_angles = torch.acos((directions @ anchor).clamp(-1, 1))
        assert float(anchor_angles.max()) > 0.65


def test_hexane_coarse_probe_sampling_handles_interior_supports() -> None:
    molecule = get_molecule("C6H14")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.5,
        probe_density_scale=1.0,
        max_grid_points=500_000,
    )

    assert torch.isfinite(samples.points).all()
    assert int((samples.block_types == PROBE_BLOCK_TYPE).sum().item()) > 0
    assert samples.blocks is not None

    support_counts = samples.blocks.probe_support_mask.sum(dim=-1)
    large_block_ids = (support_counts == 6).nonzero(as_tuple=False).reshape(-1)
    assert large_block_ids.numel() == 2
    for block_id in large_block_ids:
        block_mask = (
            (samples.block_types == PROBE_BLOCK_TYPE)
            & (samples.block_indices == block_id)
        )
        assert int(block_mask.sum().item()) >= 12


def test_cryptand_large_probe_patches_are_not_edge_overconcentrated() -> None:
    molecule = get_molecule("Cryptand cage")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=0.03,
        probe_density_scale=1.0,
        max_grid_points=500_000,
    )

    assert samples.blocks is not None
    support_counts = samples.blocks.probe_support_mask.sum(dim=-1)
    large_block_ids = (support_counts == support_counts.max()).nonzero(
        as_tuple=False,
    ).reshape(-1)
    assert int(support_counts.max().item()) >= 8

    for block_id in large_block_ids:
        block_mask = (
            (samples.block_types == PROBE_BLOCK_TYPE)
            & (samples.block_indices == block_id)
        )
        points = samples.points[block_mask]
        assert points.shape[0] > 100

        center = samples.blocks.probe_center_hints[block_id].to(
            dtype=points.dtype,
            device=points.device,
        )
        support_indices = samples.blocks.probe_support_indices[
            block_id,
            samples.blocks.probe_support_mask[block_id],
        ]
        normals = (center.unsqueeze(0) - coords[support_indices]) / (
            radii[support_indices] + 1.4
        ).unsqueeze(-1)
        anchor = normals.mean(dim=0)
        anchor = anchor / torch.linalg.norm(anchor)
        directions = center.unsqueeze(0) - points
        directions = directions / torch.linalg.norm(
            directions,
            dim=-1,
            keepdim=True,
        )
        anchor_angles = torch.acos((directions @ anchor).clamp(-1, 1))
        central_cap = anchor_angles < 0.35
        assert int(central_cap.sum().item()) >= 20

        low, high = torch.quantile(
            anchor_angles,
            torch.tensor([0.33, 0.67], dtype=points.dtype),
        )
        distances = torch.cdist(points, points)
        distances[distances == 0] = float("inf")
        nearest = distances.min(dim=1).values
        inner_median = nearest[anchor_angles <= low].median()
        middle_median = nearest[(anchor_angles > low) & (anchor_angles < high)].median()
        outer_median = nearest[anchor_angles >= high].median()

        rounded_points = torch.round(points * 1e8).to(torch.long)
        assert torch.unique(rounded_points, dim=0).shape[0] == points.shape[0]
        assert float(middle_median / inner_median) > 0.7
        assert float(outer_median / inner_median) > 0.7


def test_double_chamber_cage_probe_blocks_cover_both_chambers() -> None:
    molecule = get_molecule("Double chamber cage")
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    blocks, _ = build_analytic_ses_blocks(
        coords,
        radii,
        1.4,
        max_grid_points=500_000,
    )

    centers = blocks.probe_center_hints
    assert blocks.probe_seed_indices.shape[0] >= 70
    assert bool((centers[:, 0] < -1.0).any().item())
    assert bool((centers[:, 0] > 1.0).any().item())


def test_sample_analytic_ses_points_smoke_on_real_benchmark_pdb() -> None:
    pdb_path = Path("Data/01-benchmark_pdbs/2DS2_A.pdb")
    if not pdb_path.exists():
        pytest.skip("benchmark PDB fixture is not available")

    atoms = read_pdb_tensors(
        pdb_path,
        include_hydrogens=False,
        unknown_elements="skip",
        dtype=torch.float32,
    )
    coords = atoms.atom_coords[:80]
    radii = atoms.atom_radii[:80]

    samples = sample_analytic_ses_points(
        coords,
        radii,
        1.4,
        point_area=30.0,
        atom_filter_samples=12,
        pair_filter_samples=4,
        max_probe_triples=5_000,
        max_grid_points=200_000,
    )

    assert samples.points.shape[0] > 0
    assert torch.isfinite(samples.points).all()
    assert samples.support_indices.shape[0] == samples.points.shape[0]
    assert samples.blocks is not None
    assert samples.blocks.num_blocks > 0
