import torch
import pytest

import ses
from ses import (
    sample_analytic_points,
    sample_projected_points,
    sample_sdf_points,
    sample_tiled_analytic_points,
)
from ses.example import (
    analytic_example,
    first_bindings,
    projection_example,
    sdf_example,
    water_atoms,
)


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


def _assert_points_backprop_to_atom_inputs(
    points: torch.Tensor,
    coords: torch.Tensor,
    radii: torch.Tensor,
) -> None:
    assert points.shape[0] > 0
    assert points.requires_grad

    points.square().sum().backward()

    assert coords.grad is not None
    assert radii.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert torch.isfinite(radii.grad).all()
    assert float(coords.grad.abs().sum()) > 0
    assert float(radii.grad.abs().sum()) > 0


def _assert_unit_normals(points: torch.Tensor, normals: torch.Tensor) -> None:
    assert normals.shape == points.shape
    assert normals.dtype == points.dtype
    assert normals.device == points.device
    assert torch.isfinite(normals).all()
    if normals.shape[0] > 0:
        assert torch.allclose(
            torch.linalg.norm(normals, dim=-1),
            torch.ones(normals.shape[0], dtype=normals.dtype, device=normals.device),
            atol=1e-5,
            rtol=1e-5,
        )


def _assert_public_samplers_preserve_device(device: torch.device) -> None:
    projected_coords, projected_radii = (
        tensor.to(device=device) for tensor in _two_separated_atoms()
    )
    cavity_coords, cavity_radii = (
        tensor.to(device=device) for tensor in _three_atom_cavity()
    )

    sampler_outputs = [
        sample_projected_points(
            projected_coords,
            projected_radii,
            m=4,
            probe_radius=0.8,
            include_atom_features=True,
        ),
        sample_analytic_points(
            cavity_coords,
            cavity_radii,
            1.4,
            point_area=6.0,
            atom_filter_samples=16,
            pair_filter_samples=6,
            include_atom_features=True,
            max_grid_points=100_000,
        ),
        sample_sdf_points(
            projected_coords,
            projected_radii,
            m=8,
            probe_radius=0.8,
            smoothness=0.05,
            level_tolerance=1e-6,
            include_atom_features=True,
            max_grid_points=100_000,
        ),
        sample_tiled_analytic_points(
            cavity_coords,
            cavity_radii,
            1.4,
            point_area=4.0,
            tile_size=4.0,
            tile_overlap=2.0,
            include_atom_features=True,
            max_grid_points=100_000,
        ),
    ]

    for points, atom_features in sampler_outputs:
        assert points.device == device
        assert atom_features.device == device


def _assert_public_samplers_backprop_on_device(device: torch.device) -> None:
    sampler_cases = [
        (
            _two_separated_atoms,
            lambda coords, radii: sample_projected_points(
                coords,
                radii,
                m=8,
                probe_radius=0.8,
            ),
        ),
        (
            _three_atom_cavity,
            lambda coords, radii: sample_analytic_points(
                coords,
                radii,
                1.4,
                point_area=6.0,
                atom_filter_samples=16,
                pair_filter_samples=6,
                max_grid_points=100_000,
            ),
        ),
        (
            _two_separated_atoms,
            lambda coords, radii: sample_sdf_points(
                coords,
                radii,
                m=8,
                probe_radius=0.8,
                smoothness=0.05,
                level_tolerance=1e-6,
                max_grid_points=100_000,
            ),
        ),
        (
            _three_atom_cavity,
            lambda coords, radii: sample_tiled_analytic_points(
                coords,
                radii,
                1.4,
                point_area=4.0,
                tile_size=4.0,
                tile_overlap=2.0,
                max_grid_points=100_000,
            ),
        ),
    ]

    for molecule_factory, sampler in sampler_cases:
        coords, radii = molecule_factory()
        coords = coords.to(device=device).requires_grad_(True)
        radii = radii.to(device=device).requires_grad_(True)
        _assert_points_backprop_to_atom_inputs(sampler(coords, radii), coords, radii)


def test_package_exports_only_high_level_samplers() -> None:
    assert set(ses.__all__) == {
        "sample_analytic_points",
        "sample_projected_points",
        "sample_sdf_points",
        "sample_tiled_analytic_points",
    }
    assert ses.sample_analytic_points is sample_analytic_points
    assert ses.sample_projected_points is sample_projected_points
    assert ses.sample_sdf_points is sample_sdf_points
    assert ses.sample_tiled_analytic_points is sample_tiled_analytic_points
    assert not hasattr(ses, "build_analytic_blocks")
    assert not hasattr(ses, "project_points")


def test_public_samplers_preserve_cpu_device() -> None:
    _assert_public_samplers_preserve_device(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_public_samplers_preserve_cuda_device() -> None:
    _assert_public_samplers_preserve_device(torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_public_samplers_backprop_on_cuda() -> None:
    _assert_public_samplers_backprop_on_device(torch.device("cuda"))


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


def test_projected_sampler_can_return_normals() -> None:
    coords, radii = _two_separated_atoms()

    points, normals = sample_projected_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        include_normals=True,
    )
    feature_points, atom_features, feature_normals = sample_projected_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        include_atom_features=True,
        include_normals=True,
    )

    assert torch.equal(points, feature_points)
    assert torch.equal(normals, feature_normals)
    _assert_unit_normals(points, normals)
    owner_indices = atom_features.argmax(dim=1)
    expected = points - coords[owner_indices]
    expected = expected / torch.linalg.norm(expected, dim=-1, keepdim=True)
    assert torch.allclose(normals, expected, atol=1e-12, rtol=1e-12)


def test_projected_sampler_preserves_gradients_to_atom_inputs() -> None:
    coords, radii = _two_separated_atoms()
    coords.requires_grad_(True)
    radii.requires_grad_(True)

    points = sample_projected_points(coords, radii, m=8, probe_radius=0.8)

    _assert_points_backprop_to_atom_inputs(points, coords, radii)


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


def test_analytic_sampler_can_return_normals() -> None:
    coords, radii = _three_atom_cavity()

    points, normals = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=5.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        include_normals=True,
        max_grid_points=100_000,
    )
    feature_points, atom_features, feature_normals = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=5.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        include_atom_features=True,
        include_normals=True,
        max_grid_points=100_000,
    )

    assert torch.equal(points, feature_points)
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.equal(normals, feature_normals)
    _assert_unit_normals(points, normals)


def test_analytic_sampler_preserves_gradients_to_atom_inputs() -> None:
    coords, radii = _three_atom_cavity()
    coords.requires_grad_(True)
    radii.requires_grad_(True)

    points = sample_analytic_points(
        coords,
        radii,
        1.4,
        point_area=6.0,
        atom_filter_samples=16,
        pair_filter_samples=6,
        max_grid_points=100_000,
    )

    _assert_points_backprop_to_atom_inputs(points, coords, radii)


def test_sdf_sampler_returns_flat_points_and_atom_features() -> None:
    coords, radii = _three_atom_cavity()

    points = sample_sdf_points(
        coords,
        radii,
        m=32,
        probe_radius=1.4,
        smoothness=0.2,
        max_grid_points=100_000,
    )
    feature_points, atom_features = sample_sdf_points(
        coords,
        radii,
        m=32,
        probe_radius=1.4,
        smoothness=0.2,
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


def test_sdf_sampler_can_return_normals() -> None:
    coords, radii = _two_separated_atoms()

    points, normals = sample_sdf_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        smoothness=0.05,
        level_tolerance=1e-6,
        include_normals=True,
        max_grid_points=100_000,
    )
    feature_points, atom_features, feature_normals = sample_sdf_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        smoothness=0.05,
        level_tolerance=1e-6,
        include_atom_features=True,
        include_normals=True,
        max_grid_points=100_000,
    )

    assert torch.equal(points, feature_points)
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.equal(normals, feature_normals)
    _assert_unit_normals(points, normals)


def test_sdf_sampler_preserves_gradients_to_atom_inputs() -> None:
    coords, radii = _two_separated_atoms()
    coords.requires_grad_(True)
    radii.requires_grad_(True)

    points = sample_sdf_points(
        coords,
        radii,
        m=8,
        probe_radius=0.8,
        smoothness=0.05,
        level_tolerance=1e-6,
        max_grid_points=100_000,
    )

    _assert_points_backprop_to_atom_inputs(points, coords, radii)


def test_tiled_analytic_sampler_returns_flat_points_and_atom_features() -> None:
    coords, radii = _three_atom_cavity()

    points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        max_grid_points=100_000,
    )
    feature_points, atom_features = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        include_atom_features=True,
        max_grid_points=100_000,
    )

    assert torch.equal(points, feature_points)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.all((atom_features == 0) | (atom_features == 1))
    assert torch.all(atom_features.sum(dim=1) >= 1)


def test_tiled_analytic_sampler_can_return_normals() -> None:
    coords, radii = _three_atom_cavity()

    points, normals = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        include_normals=True,
        max_grid_points=100_000,
    )
    feature_points, atom_features, feature_normals = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        include_atom_features=True,
        include_normals=True,
        max_grid_points=100_000,
    )

    assert torch.equal(points, feature_points)
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.equal(normals, feature_normals)
    _assert_unit_normals(points, normals)


def test_tiled_analytic_sampler_preserves_gradients_to_atom_inputs() -> None:
    coords, radii = _three_atom_cavity()
    coords.requires_grad_(True)
    radii.requires_grad_(True)

    points = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        point_area=4.0,
        tile_size=4.0,
        tile_overlap=2.0,
        max_grid_points=100_000,
    )

    _assert_points_backprop_to_atom_inputs(points, coords, radii)


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
    sdf_points, sdf_features = sample_sdf_points(
        coords,
        radii,
        m=4,
        probe_radius=1.4,
        include_atom_features=True,
    )
    tiled_points, tiled_features = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        include_atom_features=True,
    )
    (
        projected_normal_points,
        projected_normal_features,
        projected_normals,
    ) = sample_projected_points(
        coords,
        radii,
        m=4,
        probe_radius=1.4,
        include_atom_features=True,
        include_normals=True,
    )
    analytic_normal_points, analytic_normal_features, analytic_normals = sample_analytic_points(
        coords,
        radii,
        1.4,
        include_atom_features=True,
        include_normals=True,
    )
    sdf_normal_points, sdf_normal_features, sdf_normals = sample_sdf_points(
        coords,
        radii,
        m=4,
        probe_radius=1.4,
        include_atom_features=True,
        include_normals=True,
    )
    tiled_normal_points, tiled_normal_features, tiled_normals = sample_tiled_analytic_points(
        coords,
        radii,
        1.4,
        include_atom_features=True,
        include_normals=True,
    )

    assert projected_points.shape == (0, 3)
    assert projected_features.shape == (0, 0)
    assert analytic_points.shape == (0, 3)
    assert analytic_features.shape == (0, 0)
    assert sdf_points.shape == (0, 3)
    assert sdf_features.shape == (0, 0)
    assert tiled_points.shape == (0, 3)
    assert tiled_features.shape == (0, 0)
    assert projected_normal_points.shape == (0, 3)
    assert projected_normal_features.shape == (0, 0)
    assert projected_normals.shape == (0, 3)
    assert analytic_normal_points.shape == (0, 3)
    assert analytic_normal_features.shape == (0, 0)
    assert analytic_normals.shape == (0, 3)
    assert sdf_normal_points.shape == (0, 3)
    assert sdf_normal_features.shape == (0, 0)
    assert sdf_normals.shape == (0, 3)
    assert tiled_normal_points.shape == (0, 3)
    assert tiled_normal_features.shape == (0, 0)
    assert tiled_normals.shape == (0, 3)


def test_examples_decode_atom_feature_bindings() -> None:
    atom_types, _, _ = water_atoms()
    projected_points, projected_features = projection_example()
    analytic_points, analytic_features = analytic_example()
    sdf_points, sdf_features = sdf_example()

    projected_bindings = first_bindings(projected_features, atom_types)
    analytic_bindings = first_bindings(analytic_features, atom_types)
    sdf_bindings = first_bindings(sdf_features, atom_types)

    assert projected_points.shape[0] == projected_features.shape[0]
    assert analytic_points.shape[0] == analytic_features.shape[0]
    assert sdf_points.shape[0] == sdf_features.shape[0]
    assert projected_features.shape[1] == len(atom_types)
    assert analytic_features.shape[1] == len(atom_types)
    assert sdf_features.shape[1] == len(atom_types)
    assert projected_bindings
    assert analytic_bindings
    assert sdf_bindings
    assert all(len(binding) == 1 for binding in projected_bindings)
    assert all(set(binding).issubset(atom_types) for binding in analytic_bindings)
    assert all(set(binding).issubset(atom_types) for binding in sdf_bindings)
