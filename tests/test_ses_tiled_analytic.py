import pytest
import torch

from ses import sample_analytic_points, sample_tiled_analytic_points
from ses.analytic import _dense_atom_features
from ses.tiled_analytic import (
    _build_tile_grid,
    _packed_pair_parameters_uniform_arc,
    _sample_tiled_analytic_samples,
    _sample_tile_local_pair_candidates,
    _sample_tile_local_probe_candidates,
    _segment_clearance_mask_line_grid,
    _tile_atom_intersection_mask,
)
from ses.projection import _segment_clearance_mask


def _two_atom_molecule(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.5, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    radii = torch.tensor([1.4, 1.2], dtype=dtype)
    return coords, radii


def _compact_molecule(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.3, 0.0],
            [1.4, 0.8, 2.1],
        ],
        dtype=dtype,
    )
    radii = torch.full((4,), 1.3, dtype=dtype)
    return coords, radii


def _water_molecule(dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
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


def test_tiled_analytic_points_are_finite_and_outside_atoms() -> None:
    coords, radii = _compact_molecule()

    points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=4.0,
        tile_overlap=2.0,
        max_grid_points=100_000,
    )

    assert points.shape[0] > 0
    assert points.shape[1] == 3
    assert torch.isfinite(points).all()
    dists = torch.cdist(points, coords)
    assert bool((dists >= radii.unsqueeze(0) - 1e-8).all().item())


def test_tiled_analytic_sampler_is_deterministic() -> None:
    coords, radii = _compact_molecule()

    first = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=4.0,
        tile_overlap=2.0,
        max_grid_points=100_000,
    )
    second = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=4.0,
        tile_overlap=2.0,
        max_grid_points=100_000,
    )

    assert torch.equal(first, second)


def test_tiled_analytic_single_tile_matches_analytic_pipeline() -> None:
    coords, radii = _compact_molecule()
    density_scale = 1.55

    tiled_points, tiled_features, tiled_normals = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=512.0,
        tile_overlap=0.0,
        atom_density_scale=density_scale,
        pair_density_scale=density_scale,
        probe_density_scale=density_scale,
        dedup_tolerance=0.05,
        include_atom_features=True,
        include_normals=True,
        max_grid_points=100_000,
    )
    analytic_points, analytic_features, analytic_normals = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        oversample_factor=density_scale,
        probe_density_scale=1.0,
        dedup_tolerance=0.05,
        include_atom_features=True,
        include_normals=True,
        max_grid_points=100_000,
    )

    assert torch.allclose(tiled_points, analytic_points)
    assert torch.equal(tiled_features, analytic_features)
    assert torch.allclose(tiled_normals, analytic_normals)


def test_tiled_pair_parameter_packing_preserves_counts() -> None:
    counts = torch.tensor([0, 3, 1, 4], dtype=torch.long)

    pair_rows, theta_values, arc_fracs = _packed_pair_parameters_uniform_arc(
        counts,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert torch.equal(pair_rows, torch.tensor([1, 1, 1, 2, 3, 3, 3, 3]))
    assert theta_values.shape == (int(counts.sum().item()),)
    assert arc_fracs.shape == theta_values.shape
    assert bool(((arc_fracs >= 0) & (arc_fracs < 1)).all().item())


def test_tiled_analytic_features_are_binary_atom_supports() -> None:
    coords, radii = _compact_molecule()

    points, atom_features = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=4.0,
        tile_overlap=2.0,
        include_atom_features=True,
        max_grid_points=100_000,
    )

    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.all((atom_features == 0) | (atom_features == 1))
    assert torch.all(atom_features.sum(dim=-1) >= 1)


def test_tiled_analytic_water_features_match_support_metadata_and_weights() -> None:
    coords, radii = _water_molecule()

    samples = _sample_tiled_analytic_samples(
        coords,
        radii,
        1.4,
        point_area=0.5,
        include_atom_weights=True,
        max_grid_points=500_000,
    )
    points, atom_features = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=0.5,
        include_atom_features=True,
        max_grid_points=500_000,
    )
    metadata_features = _dense_atom_features(
        samples.support_indices,
        samples.support_mask,
        num_atoms=coords.shape[0],
        dtype=samples.points.dtype,
    )

    assert samples.atom_weights is not None
    assert samples.normals is None
    assert torch.equal(points, samples.points)
    assert torch.equal(atom_features, metadata_features)
    assert samples.atom_weights.shape == atom_features.shape
    assert torch.allclose(
        samples.atom_weights.sum(dim=1),
        torch.ones(samples.points.shape[0], dtype=samples.points.dtype),
    )
    assert torch.all(samples.atom_weights[atom_features == 0] == 0)

    multi_support = atom_features.sum(dim=1) > 1
    first_active_atoms = atom_features[multi_support].argmax(dim=1)
    dominant_atoms = samples.atom_weights[multi_support].argmax(dim=1)
    assert bool((first_active_atoms != dominant_atoms).any().item())


def test_tiled_analytic_can_strip_metadata_for_geometry_only_outputs() -> None:
    coords, radii = _compact_molecule()

    samples = _sample_tiled_analytic_samples(
        coords,
        radii,
        1.4,
        point_area=3.0,
        tile_size=4.0,
        tile_overlap=2.0,
        include_normals=True,
        keep_metadata=False,
        max_grid_points=100_000,
    )

    assert samples.points.shape[0] > 0
    assert samples.normals is not None
    assert samples.normals.shape == samples.points.shape
    assert samples.support_indices.shape == (samples.points.shape[0], 0)
    assert samples.support_mask.shape == (samples.points.shape[0], 0)
    assert samples.support_weights.shape == (samples.points.shape[0], 0)


def test_tile_local_pair_candidates_use_global_pair_block_ids() -> None:
    coords, radii = _compact_molecule()
    probe_radius = 1.4
    grid = _build_tile_grid(
        coords,
        radii,
        probe_radius,
        tile_size=4.0,
        tile_overlap=2.0,
    )
    atom_tile_mask = _tile_atom_intersection_mask(coords, radii, probe_radius, grid)

    samples = _sample_tile_local_pair_candidates(
        coords,
        radii,
        atom_tile_mask,
        grid,
        probe_radius,
        point_area=3.0,
        pair_density_scale=1.0,
        tile_size=4.0,
        include_normals=True,
    )

    assert samples.points.shape[0] > 0
    assert samples.normals is not None
    expected_ids = (
        torch.minimum(samples.support_indices[:, 0], samples.support_indices[:, 1])
        * coords.shape[0]
        + torch.maximum(samples.support_indices[:, 0], samples.support_indices[:, 1])
    )
    assert torch.equal(samples.block_indices, expected_ids)


def test_tile_local_probe_candidates_use_global_seed_block_ids() -> None:
    coords, radii = _compact_molecule()
    probe_radius = 1.4
    grid = _build_tile_grid(
        coords,
        radii,
        probe_radius,
        tile_size=4.0,
        tile_overlap=2.0,
    )
    atom_tile_mask = _tile_atom_intersection_mask(coords, radii, probe_radius, grid)

    samples = _sample_tile_local_probe_candidates(
        coords,
        radii,
        atom_tile_mask,
        grid,
        probe_radius,
        point_area=3.0,
        probe_density_scale=1.0,
        tile_size=4.0,
        max_grid_points=100_000,
        max_probe_triples=100_000,
        include_normals=True,
    )

    assert samples.points.shape[0] > 0
    assert samples.normals is not None
    assert samples.block_indices.min() >= 0
    assert samples.block_indices.max() < 2 * coords.shape[0] ** 3
    assert (samples.support_mask.sum(dim=-1) >= 3).all()


def test_radial_line_grid_clearance_matches_exact_segment_clearance() -> None:
    generator = torch.Generator().manual_seed(3)
    atom_coords = (torch.rand((10_000, 3), generator=generator, dtype=torch.float64) - 0.5) * 140
    atom_radii = torch.rand((10_000,), generator=generator, dtype=torch.float64) * 0.5 + 1.0
    starts = (torch.rand((1_200, 3), generator=generator, dtype=torch.float64) - 0.5) * 40
    ends = starts + torch.randn((1_200, 3), generator=generator, dtype=torch.float64) * 15

    exact = _segment_clearance_mask(starts, ends, atom_coords, atom_radii.square())
    line_grid = _segment_clearance_mask_line_grid(
        starts,
        ends,
        atom_coords,
        atom_radii.square(),
        pairwise_element_budget=8_000_000,
    )

    assert line_grid is not None
    assert torch.equal(line_grid, exact)


def test_tiled_analytic_can_use_exact_accessibility_filter() -> None:
    coords, radii = _compact_molecule()

    points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        exact_accessibility=True,
        max_grid_points=100_000,
    )

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert torch.isfinite(points).all()


def test_tiled_analytic_can_use_auto_tile_parameters() -> None:
    coords, radii = _compact_molecule()

    points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size="auto",
        tile_overlap="auto",
        max_grid_points=100_000,
    )

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert torch.isfinite(points).all()


def test_tiled_analytic_is_close_to_analytic_for_separated_atoms() -> None:
    coords, radii = _two_atom_molecule()

    tiled_points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=2.0,
        tile_size=4.0,
        tile_overlap=2.0,
        atom_density_scale=1.0,
        max_grid_points=100_000,
    )
    analytic_points = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=2.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        max_grid_points=100_000,
    )

    assert tiled_points.shape[0] > 0
    assert analytic_points.shape[0] > 0
    distances = torch.cdist(tiled_points, analytic_points).min(dim=1).values
    assert float(distances.mean()) < 0.35


@pytest.mark.parametrize(
    "params,error",
    [
        ({"point_area": 0.0}, ValueError),
        ({"tile_size": 0.0}, ValueError),
        ({"tile_overlap": -1.0}, ValueError),
        ({"atom_density_scale": 0.0}, ValueError),
        ({"pair_density_scale": -0.1}, ValueError),
        ({"dedup_tolerance": 0.0}, ValueError),
        ({"max_probe_triples": 0}, ValueError),
    ],
)
def test_tiled_analytic_rejects_invalid_inputs(params, error) -> None:
    coords, radii = _two_atom_molecule()

    with pytest.raises(error):
        sample_tiled_analytic_points(coords, radii, 1.4, **params)
