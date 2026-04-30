import math

import pytest
import torch

from ses.analytic import (
    ATOM_BLOCK_TYPE,
    PAIR_BLOCK_TYPE,
    AnalyticBlocks,
    AnalyticSamples,
    ExteriorContext,
    _ProbeBlocks,
    _allocate_counts_by_weights,
    _build_exterior_context,
    _candidate_pair_indices,
    _candidate_pair_indices_kdtree,
    _candidate_pair_indices_torch,
    _candidate_triple_indices,
    _centers_feasible_against_atoms,
    _concat_samples,
    _convex_hull_normal_order,
    _counts_from_areas,
    _deduplicate_centers,
    _deduplicate_weight_rows,
    _dense_atom_features,
    _dense_atom_weights,
    _empty_probe_blocks,
    _empty_samples,
    _extract_atom_blocks,
    _extract_pair_blocks,
    _extract_probe_blocks,
    _fibonacci_unit_vectors,
    _invert_pair_arc_cdf,
    _low_discrepancy_cap_directions,
    _orthonormal_tangent_basis,
    _packed_fibonacci_directions,
    _packed_pair_parameters,
    _packed_probe_patch_weights,
    _packed_simplex_weights,
    _pad_columns,
    _pair_arc_cdfs,
    _pair_circle_parameters,
    _pair_patch_area_estimates,
    _pair_points_from_params,
    _pair_sample_counts,
    _polygon_direction_weights,
    _polygon_probe_patch_weights,
    _prepare_atom_inputs,
    _probe_center_supports,
    _probe_patch_area_estimates,
    _rejection_spherical_polygon_weight_rows,
    _rejection_spherical_triangle_weight_rows,
    _simplex_weight_rows,
    _slerp_unit_vectors,
    _sort_normals_around_mean,
    _spherical_triangle_area,
    _spherical_triangle_weight_rows,
    _sqrt_eps,
    _triangle_cone_coefficients,
    _triangle_weight_rows,
    _validate_point_area,
    _well_spaced_candidate_count,
    _well_spaced_direction_indices,
)


def _equilateral_atoms() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, math.sqrt(3.0), 0.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.ones(3, dtype=torch.float64)
    return coords, radii


def _unit_polygon_normals() -> torch.Tensor:
    normals = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return normals / torch.linalg.norm(normals, dim=-1, keepdim=True)


def _row_set(rows: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(int(value) for value in row.tolist()) for row in rows}


def test_analytic_context_block_extraction_and_containers() -> None:
    coords, radii = _equilateral_atoms()
    context = _build_exterior_context(
        coords,
        radii,
        1.0,
        max_grid_points=100_000,
    )
    centers = torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    assert isinstance(context, ExteriorContext)
    assert context.device == coords.device
    assert context.dtype == coords.dtype
    assert context.num_atoms == 3
    assert torch.equal(context.centers_feasible(centers), torch.tensor([True, False]))
    assert torch.equal(
        context.centers_accessible(centers, torch.tensor([True, False])),
        torch.tensor([True, False]),
    )
    assert torch.equal(context.centers_exterior(centers), torch.tensor([True, False]))

    atom_indices = _extract_atom_blocks(context, filter_samples=8)
    pair_indices = _extract_pair_blocks(context, filter_samples=4)
    probe_blocks = _extract_probe_blocks(
        context,
        pair_indices=_candidate_pair_indices(context),
        max_support_atoms=4,
        max_triples=10,
    )
    empty_probe_blocks = _empty_probe_blocks(context.device, max_support_atoms=5)

    assert atom_indices.dtype == torch.long
    assert atom_indices.ndim == 1
    assert pair_indices.ndim == 2
    assert pair_indices.shape[1] == 2
    assert isinstance(probe_blocks, _ProbeBlocks)
    assert probe_blocks.probe_support_indices.shape[1] == 4
    assert isinstance(empty_probe_blocks, _ProbeBlocks)
    assert empty_probe_blocks.probe_seed_indices.shape == (0, 3)
    assert empty_probe_blocks.probe_support_indices.shape == (0, 5)

    blocks = AnalyticBlocks(
        atom_indices=atom_indices,
        pair_indices=pair_indices,
        probe_seed_indices=probe_blocks.probe_seed_indices,
        probe_center_signs=probe_blocks.probe_center_signs,
        probe_support_indices=probe_blocks.probe_support_indices,
        probe_support_mask=probe_blocks.probe_support_mask,
        probe_center_hints=probe_blocks.probe_center_hints,
    )

    assert blocks.num_blocks == (
        atom_indices.shape[0]
        + pair_indices.shape[0]
        + probe_blocks.probe_seed_indices.shape[0]
    )


def test_analytic_input_density_and_unit_direction_helpers() -> None:
    coords, radii = _prepare_atom_inputs(
        torch.tensor([[0, 0, 0]], dtype=torch.int64),
        torch.tensor([[1.5]], dtype=torch.float32),
        probe_radius=1.0,
    )

    assert coords.dtype == torch.float32
    assert radii.shape == (1,)

    _validate_point_area(point_area=1.0, oversample_factor=1.0)
    with pytest.raises(ValueError, match="point_area"):
        _validate_point_area(point_area=0.0, oversample_factor=1.0)
    with pytest.raises(ValueError, match="oversample_factor"):
        _validate_point_area(point_area=1.0, oversample_factor=0.0)

    assert _sqrt_eps(torch.float64) > 0

    directions = _fibonacci_unit_vectors(
        5,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    packed_directions = _packed_fibonacci_directions(
        torch.tensor([2, 0, 3]),
        torch.float64,
        torch.device("cpu"),
    )
    counts = _counts_from_areas(
        torch.tensor([0.0, 2.1], dtype=torch.float64),
        point_area=1.0,
        oversample_factor=1.0,
    )
    feasible = _centers_feasible_against_atoms(
        torch.tensor([[3.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        pair_budget=1,
    )

    assert directions.shape == (5, 3)
    assert torch.allclose(
        torch.linalg.norm(directions, dim=-1),
        torch.ones(5, dtype=torch.float64),
    )
    assert _fibonacci_unit_vectors(
        0,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ).shape == (0, 3)
    assert packed_directions.shape == (5, 3)
    assert torch.equal(counts, torch.tensor([1, 3]))
    assert torch.equal(feasible, torch.tensor([True, False]))


def test_analytic_pair_graph_and_pair_sampling_helpers() -> None:
    coords, radii = _equilateral_atoms()
    context = _build_exterior_context(coords, radii, 1.0)
    expected_pairs = {(0, 1), (0, 2), (1, 2)}

    assert _row_set(_candidate_pair_indices_torch(context)) == expected_pairs
    assert _row_set(_candidate_pair_indices_kdtree(context)) == expected_pairs
    assert _row_set(_candidate_pair_indices(context)) == expected_pairs

    pair_indices = torch.tensor([[0, 1]], dtype=torch.long)
    centers, circle_radii, basis_u, basis_v, valid = _pair_circle_parameters(
        context.atom_coords,
        context.expanded_radii,
        pair_indices,
    )
    areas = _pair_patch_area_estimates(context, pair_indices, arc_quadrature=8)
    counts = _pair_sample_counts(
        context,
        pair_indices,
        point_area=10.0,
        oversample_factor=1.0,
    )
    cdfs = _pair_arc_cdfs(context, pair_indices, arc_quadrature=8)
    inverted = _invert_pair_arc_cdf(
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        cdfs[0],
    )
    rows, theta_values, arc_fracs = _packed_pair_parameters(
        torch.tensor([3]),
        context,
        pair_indices,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    slerp = _slerp_unit_vectors(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
        torch.tensor([math.pi / 2], dtype=torch.float64),
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
    )
    probe_centers, points, first_weights, second_weights = _pair_points_from_params(
        context.atom_coords,
        context.expanded_radii,
        context.probe_radius,
        pair_indices.expand(theta_values.shape[0], -1),
        theta_values,
        arc_fracs,
    )

    assert torch.allclose(
        centers,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        circle_radii,
        torch.tensor([math.sqrt(3.0)], dtype=torch.float64),
    )
    assert torch.equal(valid, torch.tensor([True]))
    assert torch.allclose(
        (basis_u * basis_v).sum(dim=-1),
        torch.zeros(1, dtype=torch.float64),
    )
    assert bool((areas > 0).all().item())
    assert bool((counts >= 1).all().item())
    assert cdfs.shape == (1, 9)
    assert torch.allclose(cdfs[:, 0], torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(cdfs[:, -1], torch.ones(1, dtype=torch.float64))
    assert bool((inverted >= 0).all().item())
    assert bool((inverted <= 1).all().item())
    assert torch.equal(rows, torch.zeros(3, dtype=torch.long))
    assert theta_values.shape == arc_fracs.shape == (3,)
    assert torch.allclose(
        slerp[0, 0],
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        slerp[0, -1],
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
    )
    assert probe_centers.shape == points.shape == (3, 3)
    assert torch.isfinite(points).all()
    assert torch.isfinite(first_weights).all()
    assert torch.isfinite(second_weights).all()


def test_analytic_triple_probe_support_helpers() -> None:
    coords, radii = _equilateral_atoms()
    expanded_radii = radii + 1.0
    pair_indices = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.long)
    triple_indices = _candidate_triple_indices(
        num_atoms=3,
        pair_indices=pair_indices,
        max_triples=None,
    )
    center_z = math.sqrt(8.0 / 3.0)
    centers = torch.tensor(
        [
            [1.0, math.sqrt(3.0) / 3.0, center_z],
            [1.0, math.sqrt(3.0) / 3.0, -center_z],
        ],
        dtype=torch.float64,
    )
    support_indices, support_mask = _probe_center_supports(
        centers,
        triple_indices.repeat(2, 1),
        coords,
        expanded_radii,
        max_support_atoms=4,
        tolerance=1e-8,
    )
    keep_indices = _deduplicate_centers(
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        tolerance=0.01,
    )
    areas = _probe_patch_area_estimates(
        centers[:1],
        support_indices[:1],
        support_mask[:1],
        coords,
        expanded_radii,
        probe_radius=1.0,
    )

    assert torch.equal(triple_indices, torch.tensor([[0, 1, 2]]))
    assert support_indices.shape == (2, 4)
    assert torch.equal(support_mask.sum(dim=-1), torch.tensor([3, 3]))
    assert _row_set(support_indices[:, :3]) == {(0, 1, 2)}
    assert torch.equal(keep_indices, torch.tensor([0, 2]))
    assert areas.shape == (1,)
    assert float(areas[0]) > 0


def test_analytic_spherical_weight_and_direction_helpers() -> None:
    dtype = torch.float64
    device = torch.device("cpu")
    triangle_vertices = torch.eye(3, dtype=dtype, device=device)
    triangle_area = _spherical_triangle_area(
        triangle_vertices[0],
        triangle_vertices[1],
        triangle_vertices[2],
    )
    polygon_normals = _unit_polygon_normals()
    polygon_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    polygon_slots = torch.arange(4, dtype=torch.long, device=device)
    anchor_norm = torch.linalg.norm(polygon_normals.mean(dim=0))
    anchor_weights = torch.full(
        (4,),
        1.0 / (4.0 * float(anchor_norm)),
        dtype=dtype,
        device=device,
    )

    sorted_order = _sort_normals_around_mean(triangle_vertices)
    hull_order = _convex_hull_normal_order(polygon_normals, polygon_axis)
    simplex_rows, simplex_weights = _packed_simplex_weights(
        torch.tensor([2, 1]),
        torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
        ),
        dtype=dtype,
        device=device,
    )
    packed_rows, packed_weights = _packed_probe_patch_weights(
        torch.zeros((2, 3), dtype=dtype, device=device),
        torch.tensor([[0, 1, 2, -1], [0, 1, 2, 3]], dtype=torch.long),
        torch.tensor(
            [[True, True, True, False], [True, True, True, True]],
        ),
        -polygon_normals,
        torch.ones(4, dtype=dtype, device=device),
        torch.tensor([4, 8]),
        dtype=dtype,
        device=device,
    )
    polygon_weights = _polygon_probe_patch_weights(
        8,
        polygon_normals,
        polygon_slots,
        max_support=4,
        dtype=dtype,
        device=device,
    )
    rejection_polygon_weights = _rejection_spherical_polygon_weight_rows(
        8,
        polygon_normals[hull_order],
        polygon_slots[hull_order],
        polygon_axis,
        anchor_weights,
        max_support=4,
        dtype=dtype,
        device=device,
    )
    assigned_weights, assigned_dirs = _polygon_direction_weights(
        polygon_axis.unsqueeze(0),
        polygon_normals[hull_order],
        polygon_slots[hull_order],
        polygon_axis,
        anchor_weights,
        max_support=4,
        dtype=dtype,
        device=device,
    )
    deduplicated_weights = _deduplicate_weight_rows(
        torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0]],
            dtype=dtype,
            device=device,
        )
    )
    allocated_counts = _allocate_counts_by_weights(
        5,
        torch.tensor([0.2, 0.8, 0.0], dtype=dtype, device=device),
    )
    simplex_weights_direct = _simplex_weight_rows(
        3,
        support_count=2,
        max_support=4,
        dtype=dtype,
        device=device,
    )
    spherical_triangle_weights = _spherical_triangle_weight_rows(
        4,
        triangle_vertices,
        dtype,
        device,
    )
    rejection_triangle_weights = _rejection_spherical_triangle_weight_rows(
        4,
        triangle_vertices,
        triangle_area,
        dtype,
        device,
    )
    well_spaced_count = _well_spaced_candidate_count(3)
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    well_spaced_indices = _well_spaced_direction_indices(
        directions,
        count=2,
        preferred_axis=polygon_axis,
    )
    cap_directions = _low_discrepancy_cap_directions(
        5,
        polygon_axis,
        cap_cos=0.5,
        offset=0,
        dtype=dtype,
        device=device,
    )
    tangent_u, tangent_v = _orthonormal_tangent_basis(polygon_axis)
    cone_coefficients, cone_inside = _triangle_cone_coefficients(
        torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=dtype,
            device=device,
        )
        / torch.tensor([[1.0], [math.sqrt(3.0)]], dtype=dtype, device=device),
        triangle_vertices,
    )
    triangle_weights = _triangle_weight_rows(
        4,
        max_support=5,
        dtype=dtype,
        device=device,
    )

    assert torch.allclose(triangle_area, torch.tensor(math.pi / 2, dtype=dtype))
    assert set(sorted_order.tolist()) == {0, 1, 2}
    assert set(hull_order.tolist()) == {0, 1, 2, 3}
    assert simplex_rows.shape[0] == simplex_weights.shape[0]
    assert simplex_weights.shape[1] == 4
    assert packed_rows.shape[0] == packed_weights.shape[0]
    assert packed_weights.shape[1] == 4
    assert polygon_weights.shape[1] == 4
    assert polygon_weights.shape[0] > 0
    assert torch.isfinite(polygon_weights).all()
    assert rejection_polygon_weights.shape[1] == 4
    assert assigned_weights.shape == (1, 4)
    assert torch.allclose(assigned_dirs, polygon_axis.unsqueeze(0))
    assert torch.equal(deduplicated_weights, torch.tensor([[0.5, 0.5], [1.0, 0.0]]))
    assert int(allocated_counts.sum().item()) == 5
    assert bool((allocated_counts[:2] >= 1).all().item())
    assert simplex_weights_direct.shape == (3, 4)
    assert torch.allclose(
        simplex_weights_direct.sum(dim=-1),
        torch.ones(3, dtype=dtype),
    )
    assert spherical_triangle_weights.shape[1] == 3
    assert rejection_triangle_weights.shape[1] == 3
    assert rejection_triangle_weights.shape[0] > 0
    assert well_spaced_count >= 64
    assert _well_spaced_candidate_count(0) == 0
    assert well_spaced_indices.shape == (2,)
    assert len(set(well_spaced_indices.tolist())) == 2
    assert cap_directions.shape == (5, 3)
    assert torch.allclose(
        torch.linalg.norm(cap_directions, dim=-1),
        torch.ones(5, dtype=dtype),
        atol=1e-12,
        rtol=0,
    )
    assert bool((cap_directions @ polygon_axis >= 0.5 - 1e-12).all().item())
    assert abs(float(torch.dot(tangent_u, polygon_axis))) < 1e-12
    assert abs(float(torch.dot(tangent_v, polygon_axis))) < 1e-12
    assert torch.equal(cone_inside, torch.tensor([True, True]))
    assert torch.allclose(
        cone_coefficients.sum(dim=-1),
        torch.ones(2, dtype=dtype),
    )
    assert triangle_weights.shape[1] == 5
    assert triangle_weights.shape[0] >= 4
    assert torch.allclose(
        triangle_weights[:, :3].sum(dim=-1),
        torch.ones(triangle_weights.shape[0], dtype=dtype),
    )


def test_analytic_sample_concatenation_and_dense_atom_helpers() -> None:
    context = _build_exterior_context(
        torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        1.0,
    )
    empty_samples = _empty_samples(context, max_support=2)
    sample_a = AnalyticSamples(
        points=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        block_types=torch.tensor([ATOM_BLOCK_TYPE], dtype=torch.long),
        block_indices=torch.tensor([0], dtype=torch.long),
        support_indices=torch.tensor([[0]], dtype=torch.long),
        support_mask=torch.tensor([[True]]),
        support_weights=torch.tensor([[1.0]], dtype=torch.float64),
    )
    sample_b = AnalyticSamples(
        points=torch.tensor([[1.5, 0.5, 0.0]], dtype=torch.float64),
        block_types=torch.tensor([PAIR_BLOCK_TYPE], dtype=torch.long),
        block_indices=torch.tensor([0], dtype=torch.long),
        support_indices=torch.tensor([[0, 1]], dtype=torch.long),
        support_mask=torch.tensor([[True, True]]),
        support_weights=torch.tensor([[0.25, 0.75]], dtype=torch.float64),
    )
    blocks = AnalyticBlocks(
        atom_indices=torch.tensor([0], dtype=torch.long),
        pair_indices=torch.tensor([[0, 1]], dtype=torch.long),
        probe_seed_indices=torch.empty((0, 3), dtype=torch.long),
        probe_center_signs=torch.empty((0,), dtype=torch.long),
        probe_support_indices=torch.empty((0, 3), dtype=torch.long),
        probe_support_mask=torch.empty((0, 3), dtype=torch.bool),
        probe_center_hints=torch.empty((0, 3), dtype=torch.float64),
    )

    concatenated = _concat_samples(
        (sample_a, sample_b),
        context=context,
        include_atom_weights=True,
        blocks=blocks,
    )
    padded_indices = _pad_columns(
        torch.tensor([[0], [1]], dtype=torch.long),
        columns=3,
        fill_value=-1,
    )
    dense_weights = _dense_atom_weights(
        torch.tensor([[0, -1], [0, 1]], dtype=torch.long),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64),
        num_atoms=2,
    )
    dense_features = _dense_atom_features(
        torch.tensor([[0, -1], [0, 1]], dtype=torch.long),
        torch.tensor([[True, False], [True, True]]),
        num_atoms=2,
        dtype=torch.float64,
    )

    assert isinstance(empty_samples, AnalyticSamples)
    assert empty_samples.points.shape == (0, 3)
    assert concatenated.points.shape == (2, 3)
    assert concatenated.support_indices.shape == (2, 2)
    assert concatenated.atom_weights is not None
    assert torch.allclose(concatenated.atom_weights, dense_weights)
    assert concatenated.blocks is blocks
    assert torch.equal(
        padded_indices,
        torch.tensor([[0, -1, -1], [1, -1, -1]], dtype=torch.long),
    )
    assert torch.allclose(
        dense_weights,
        torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64),
    )
    assert torch.allclose(
        dense_features,
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
    )
