import torch

import ses
from ses import sample_analytic_points, sample_projected_points
from ses.example import analytic_example, first_bindings, projection_example, water_atoms


def _two_separated_atoms() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.5], dtype=torch.float64)
    return coords, radii


def _three_atom_cavity() -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.3, 0.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.full((3,), 1.5, dtype=torch.float64)
    return coords, radii


def test_package_exports_only_high_level_samplers() -> None:
    assert set(ses.__all__) == {
        "sample_analytic_points",
        "sample_projected_points",
    }
    assert ses.sample_analytic_points is sample_analytic_points
    assert ses.sample_projected_points is sample_projected_points
    assert not hasattr(ses, "build_analytic_blocks")
    assert not hasattr(ses, "project_points")


def test_projected_sampler_returns_flat_points_and_one_hot_atom_features() -> None:
    coords, radii = _two_separated_atoms()

    points = sample_projected_points(coords, radii, m=8, probe_radius=0.8)
    feature_points, atom_features = sample_projected_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        include_atom_features=True,
    )

    assert torch.equal(points, feature_points)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.all((atom_features == 0) | (atom_features == 1))
    assert torch.all(atom_features.sum(dim=1) == 1)
    assert torch.all(atom_features.sum(dim=0) > 0)


def test_analytic_sampler_returns_flat_points_and_multi_hot_atom_features() -> None:
    coords, radii = _three_atom_cavity()

    points = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=5.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        max_grid_points=100_000,
    )
    feature_points, atom_features = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=5.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        include_atom_features=True,
        max_grid_points=100_000,
    )

    assert torch.equal(points, feature_points)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.all((atom_features == 0) | (atom_features == 1))
    assert torch.all(atom_features.sum(dim=1) >= 1)
    assert bool((atom_features.sum(dim=1) > 1).any().item())


def test_public_samplers_return_empty_feature_matrices_for_empty_atoms() -> None:
    coords = torch.empty((0, 3), dtype=torch.float64)
    radii = torch.empty((0,), dtype=torch.float64)

    projected_points, projected_features = sample_projected_points(
        coords,
        radii,
        m=4,
        probe_radius=1.4,
        include_atom_features=True,
    )
    analytic_points, analytic_features = sample_analytic_points(
        coords,
        radii,
        1.4,
        include_atom_features=True,
    )

    assert projected_points.shape == (0, 3)
    assert projected_features.shape == (0, 0)
    assert analytic_points.shape == (0, 3)
    assert analytic_features.shape == (0, 0)


def test_examples_decode_atom_feature_bindings() -> None:
    atom_types, _, _ = water_atoms()
    projected_points, projected_features = projection_example()
    analytic_points, analytic_features = analytic_example()

    projected_bindings = first_bindings(projected_features, atom_types)
    analytic_bindings = first_bindings(analytic_features, atom_types)

    assert projected_points.shape[0] == projected_features.shape[0]
    assert analytic_points.shape[0] == analytic_features.shape[0]
    assert projected_features.shape[1] == len(atom_types)
    assert analytic_features.shape[1] == len(atom_types)
    assert projected_bindings
    assert analytic_bindings
    assert all(len(binding) == 1 for binding in projected_bindings)
    assert all(set(binding).issubset(atom_types) for binding in analytic_bindings)
