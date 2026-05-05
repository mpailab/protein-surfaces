import math

import pytest
import torch

from ses.projection import (
    _ProjectionInputs,
    _build_local_atom_indices,
    _centers_feasible_against_all_atoms,
    _centers_feasible_against_local_atoms,
    _compute_active_set_probe_centers,
    _compute_pair_probe_centers,
    _compute_triple_probe_centers,
    _external_reachable_grid,
    _grid_free_mask,
    _grid_free_mask_with_torch,
    _max_pairwise_rows,
    _pair_patch_membership,
    _replace_better_probe_centers,
    _segment_clearance_mask,
    _squared_feasibility_mask,
    _tensor_cache_token,
    _triple_patch_membership,
    _valid_points_against_all_atoms,
    _valid_points_against_local_atoms,
)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.norm(vector).clamp_min(
        torch.finfo(vector.dtype).eps,
    )


def test_projection_input_container_and_tensor_cache_token_contract() -> None:
    points = torch.zeros((1, 2, 3), dtype=torch.float64)
    atom_coords = torch.zeros((2, 3), dtype=torch.float64)
    atom_radii = torch.ones(2, dtype=torch.float64)

    prepared = _ProjectionInputs(points, atom_coords, atom_radii)
    first_token = _tensor_cache_token(atom_coords)
    second_token = _tensor_cache_token(atom_coords)

    assert prepared.points is points
    assert prepared.atom_coords is atom_coords
    assert prepared.atom_radii is atom_radii
    assert first_token == second_token
    assert str(atom_coords.dtype) in first_token


def test_projection_local_atom_and_feasibility_helpers() -> None:
    atom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=torch.float64,
    )
    atom_radii = torch.ones(3, dtype=torch.float64)

    local_indices = _build_local_atom_indices(atom_coords, max_neighbors=2)
    cached_local_indices = _build_local_atom_indices(atom_coords, max_neighbors=2)

    assert local_indices.shape == (3, 2)
    assert torch.equal(local_indices[:, 0], torch.arange(3))
    assert torch.equal(cached_local_indices, local_indices)

    centers = torch.tensor(
        [[[2.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    local_atom_coords = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    local_ext_radii = torch.ones((2, 1), dtype=torch.float64)

    assert torch.equal(
        _squared_feasibility_mask(centers, local_atom_coords, local_ext_radii),
        torch.tensor([[True], [False]]),
    )

    points = torch.tensor(
        [[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    paired_local_coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    paired_local_radii = torch.ones((2, 2), dtype=torch.float64)

    assert torch.equal(
        _valid_points_against_local_atoms(
            points,
            paired_local_coords,
            paired_local_radii,
        ),
        torch.tensor([True, False]),
    )
    assert torch.equal(
        _valid_points_against_all_atoms(
            points,
            paired_local_coords[0],
            paired_local_radii[0],
        ),
        torch.tensor([True, False]),
    )

    candidate_centers = points.view(1, 2, 3).expand(2, 2, 3)
    assert torch.equal(
        _centers_feasible_against_local_atoms(
            candidate_centers,
            paired_local_coords,
            paired_local_radii,
        ),
        torch.tensor([[True, False], [True, False]]),
    )
    assert torch.equal(
        _centers_feasible_against_all_atoms(
            points,
            paired_local_coords[0],
            paired_local_radii[0].square(),
        ),
        torch.tensor([True, False]),
    )
    assert _max_pairwise_rows(0, 10) == 1
    assert _max_pairwise_rows(5, 0) == 5
    assert 1 <= _max_pairwise_rows(10, 4) <= 10


def test_projection_pair_and_triple_probe_helpers_on_known_geometry() -> None:
    atom_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, math.sqrt(3.0), 0.0],
        ],
        dtype=torch.float64,
    )
    expanded_radii = torch.full((3,), 2.0, dtype=torch.float64)
    first_indices = torch.tensor([0], dtype=torch.long)
    second_indices = torch.tensor([1], dtype=torch.long)
    third_indices = torch.tensor([2], dtype=torch.long)

    pair_point = torch.tensor(
        [[1.0, math.sqrt(3.0) - 1.0, 0.0]],
        dtype=torch.float64,
    )
    pair_center = _compute_pair_probe_centers(
        pair_point,
        atom_coords,
        expanded_radii,
        first_indices,
        second_indices,
    )

    assert torch.allclose(
        torch.linalg.norm(pair_center - atom_coords[:2], dim=-1),
        expanded_radii[:2],
        atol=1e-12,
        rtol=0,
    )
    assert bool(
        _pair_patch_membership(
            pair_point,
            pair_center,
            atom_coords,
            expanded_radii,
            first_indices,
            second_indices,
        ).squeeze().item()
    )

    triple_centers, triple_valid = _compute_triple_probe_centers(
        atom_coords,
        expanded_radii,
        first_indices,
        second_indices,
        third_indices,
    )
    top_center = triple_centers[0, 0]
    top_normals = (top_center.unsqueeze(0) - atom_coords) / expanded_radii[:, None]
    top_point = top_center - _normalize(top_normals.mean(dim=0))

    assert triple_centers.shape == (1, 2, 3)
    assert torch.equal(triple_valid, torch.tensor([[True, True]]))
    assert torch.allclose(
        torch.linalg.norm(triple_centers[:, :, None, :] - atom_coords, dim=-1),
        expanded_radii.view(1, 1, 3),
        atol=1e-12,
        rtol=0,
    )
    triple_membership = _triple_patch_membership(
        top_point.unsqueeze(0),
        triple_centers.unsqueeze(1),
        atom_coords,
        expanded_radii,
        first_indices.view(1, 1),
        second_indices.view(1, 1),
        third_indices.view(1, 1),
    )
    assert bool(triple_membership[0, 0, 0].item())


def test_projection_center_selection_and_segment_helpers() -> None:
    best_centers = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    best_scores = torch.tensor([5.0, 1.0], dtype=torch.float64)
    candidate_centers = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    candidate_scores = torch.tensor(
        [[4.0, 6.0], [0.5, 0.2]],
        dtype=torch.float64,
    )
    candidate_mask = torch.tensor([[True, True], [False, False]])

    replaced_centers, replaced_scores = _replace_better_probe_centers(
        best_centers,
        best_scores,
        candidate_centers,
        candidate_scores,
        candidate_mask,
    )

    assert torch.allclose(
        replaced_centers,
        torch.tensor([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        replaced_scores,
        torch.tensor([4.0, 1.0], dtype=torch.float64),
    )

    atom_coords = torch.zeros((1, 3), dtype=torch.float64)
    atom_radii = torch.ones(1, dtype=torch.float64)
    points = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    probe_centers, valid_mask = _compute_active_set_probe_centers(
        points,
        atom_coords,
        atom_radii,
        probe_radius=1.0,
    )

    assert torch.equal(valid_mask, torch.tensor([[True]]))
    assert torch.allclose(
        probe_centers,
        torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float64),
    )

    starts = torch.tensor(
        [[-2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    ends = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.equal(
        _segment_clearance_mask(
            starts,
            ends,
            atom_coords,
            atom_radii.square(),
        ),
        torch.tensor([False, True]),
    )


def test_projection_grid_occupancy_and_reachability_helpers() -> None:
    atom_coords = torch.zeros((1, 3), dtype=torch.float64)
    expanded_radii = torch.tensor([0.75], dtype=torch.float64)
    expanded_radii_sq = expanded_radii.square()
    bbox_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)
    dims = torch.tensor([3, 3, 3], dtype=torch.long)
    grid_spacing = 1.0
    grid_points = int(dims.prod().item())

    torch_mask = _grid_free_mask_with_torch(
        atom_coords,
        expanded_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )
    dispatch_mask = _grid_free_mask(
        atom_coords,
        expanded_radii,
        expanded_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )

    assert torch.equal(dispatch_mask, torch_mask)
    assert not bool(torch_mask[13].item())

    reached_grid, reached_bbox_min, reached_spacing, reached_dims = (
        _external_reachable_grid(
            atom_coords,
            expanded_radii,
            torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64),
            grid_spacing=1.0,
            max_grid_points=10_000,
            cache_key=("projection-internals-grid",),
        )
    )
    cached_reached_grid, cached_bbox_min, cached_spacing, cached_dims = (
        _external_reachable_grid(
            atom_coords,
            expanded_radii,
            torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float64),
            grid_spacing=1.0,
            max_grid_points=10_000,
            cache_key=("projection-internals-grid",),
        )
    )

    assert reached_grid.dtype == torch.bool
    assert reached_grid.shape == tuple(int(value.item()) for value in reached_dims)
    assert bool(reached_grid.any().item())
    assert torch.equal(cached_reached_grid, reached_grid)
    assert torch.equal(cached_bbox_min, reached_bbox_min)
    assert cached_spacing == reached_spacing
    assert torch.equal(cached_dims, reached_dims)
