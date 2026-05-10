from pathlib import Path

import numpy as np
import pytest
import torch

import ses.sdf as sdf_module
from ses import sample_analytic_points, sample_projected_points, sample_sdf_points
from ses.sdf import _sdf_values_and_normals


NPY_DATA_DIR = Path(__file__).resolve().parent / "data" / "npy"


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


def _load_fixture_atoms(
    pdb_id: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    atom_coords = torch.as_tensor(
        np.load(NPY_DATA_DIR / f"{pdb_id}_atomxyz.npy"),
        dtype=dtype,
    )
    atom_radii = torch.as_tensor(
        np.load(NPY_DATA_DIR / f"{pdb_id}_atomradii.npy"),
        dtype=dtype,
    )
    return atom_coords, atom_radii


def test_sample_sdf_points_matches_projection_for_separated_atoms() -> None:
    coords, radii = _two_separated_atoms()

    sdf_points, sdf_features = sample_sdf_points(
        coords,
        radii,
        m=24,
        probe_radius=0.8,
        smoothness=0.05,
        level_tolerance=1e-6,
        include_atom_features=True,
    )
    projected_points, projected_features = sample_projected_points(
        coords,
        radii,
        m=24,
        probe_radius=0.8,
        include_atom_features=True,
    )

    assert sdf_points.shape == projected_points.shape
    assert torch.allclose(sdf_points, projected_points, atol=1e-5, rtol=1e-5)
    assert torch.equal(sdf_features, projected_features)


def test_sample_sdf_points_is_comparable_to_analytic_sampler() -> None:
    coords, radii = _three_atom_cavity()

    sdf_points, sdf_features = sample_sdf_points(
        coords,
        radii,
        m=64,
        probe_radius=1.4,
        smoothness=0.2,
        include_atom_features=True,
        max_grid_points=200_000,
    )
    analytic_points = sample_analytic_points(
        coords,
        radii,
        probe_radius=1.4,
        point_area=0.5,
        atom_filter_samples=32,
        pair_filter_samples=12,
        max_grid_points=200_000,
    )

    assert sdf_points.shape[0] > 0
    assert analytic_points.shape[0] > 0
    assert sdf_features.shape == (sdf_points.shape[0], coords.shape[0])
    assert torch.all((sdf_features == 0) | (sdf_features == 1))
    assert torch.all(sdf_features.sum(dim=1) >= 1)

    sdf_to_analytic = torch.cdist(sdf_points, analytic_points).min(dim=1).values
    analytic_to_sdf = torch.cdist(analytic_points, sdf_points).min(dim=1).values
    assert float(sdf_to_analytic.mean()) < 0.35
    assert float(analytic_to_sdf.mean()) < 0.35


def test_sample_sdf_points_stays_outside_atom_interiors() -> None:
    coords, radii = _three_atom_cavity()

    points = sample_sdf_points(
        coords,
        radii,
        m=48,
        probe_radius=1.4,
        smoothness=0.2,
        max_grid_points=100_000,
    )

    assert points.shape[0] > 0
    atom_clearances = torch.cdist(points, coords) - radii.unsqueeze(0)
    assert bool((atom_clearances >= -1e-8).all().item())


def test_sample_sdf_points_runs_on_real_fixture() -> None:
    coords, radii = _load_fixture_atoms("2PQ2_B")

    points, atom_features = sample_sdf_points(
        coords,
        radii,
        m=2,
        probe_radius=1.4,
        smoothness=0.2,
        include_atom_features=True,
        max_grid_points=200_000,
    )

    assert points.shape[0] > 0
    assert points.shape[1] == 3
    assert torch.isfinite(points).all()
    assert atom_features.shape == (points.shape[0], coords.shape[0])
    assert torch.all(atom_features.sum(dim=1) >= 1)


def test_sample_sdf_points_preserves_gradients_to_atom_inputs() -> None:
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
    )

    assert points.requires_grad
    loss = points.square().sum()
    loss.backward()

    assert coords.grad is not None
    assert radii.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert torch.isfinite(radii.grad).all()
    assert float(coords.grad.abs().sum()) > 0
    assert float(radii.grad.abs().sum()) > 0


def test_sdf_adjacency_without_atom_features_skips_dense_features(monkeypatch) -> None:
    coords, radii = _two_separated_atoms()

    def _unexpected_dense_features(*args, **kwargs):
        raise AssertionError("dense SDF atom features should not be built")

    monkeypatch.setattr(sdf_module, "_sdf_atom_features", _unexpected_dense_features)
    points, adjacency = sdf_module.sample_sdf_points(
        coords,
        radii,
        m=12,
        probe_radius=0.8,
        smoothness=0.05,
        level_tolerance=1e-6,
        include_adjacency=True,
        adjacency_neighbors=2,
        adjacency_candidate_neighbors=4,
    )

    adjacency = adjacency.coalesce()
    assert points.shape[0] > 0
    assert adjacency.layout == torch.sparse_coo
    assert adjacency.shape == (points.shape[0], points.shape[0])
    assert torch.allclose(adjacency.to_dense(), adjacency.to_dense().transpose(0, 1))


def test_sdf_normals_preserve_gradients_to_atom_inputs() -> None:
    coords, radii = _three_atom_cavity()
    coords.requires_grad_(True)
    radii.requires_grad_(True)
    centers = coords + torch.tensor(
        [
            [0.0, 0.0, 3.0],
            [0.4, 0.2, 3.2],
            [-0.2, 0.5, 3.1],
        ],
        dtype=coords.dtype,
    )
    expanded_radii = radii + 1.4

    _, normals = _sdf_values_and_normals(
        centers,
        coords,
        expanded_radii,
        smoothness=0.2,
    )
    normal_weights = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [-0.5, 0.3, 0.1],
            [0.6, 0.2, -0.3],
        ],
        dtype=normals.dtype,
    )
    loss = (normals * normal_weights).sum()
    loss.backward()

    assert normals.requires_grad
    assert coords.grad is not None
    assert radii.grad is not None
    assert torch.isfinite(coords.grad).all()
    assert torch.isfinite(radii.grad).all()
    assert float(coords.grad.abs().sum()) > 0
    assert float(radii.grad.abs().sum()) > 0


def test_sample_sdf_points_subsample_spacing_reduces_density() -> None:
    coords, radii = _three_atom_cavity()

    dense_points = sample_sdf_points(
        coords,
        radii,
        m=64,
        probe_radius=1.4,
        smoothness=0.2,
        max_grid_points=100_000,
    )
    coarse_points = sample_sdf_points(
        coords,
        radii,
        m=64,
        probe_radius=1.4,
        smoothness=0.2,
        subsample_spacing=0.75,
        max_grid_points=100_000,
    )

    assert 0 < coarse_points.shape[0] < dense_points.shape[0]


@pytest.mark.parametrize(
    "kwargs, error_type, message",
    [
        ({"m": -1}, ValueError, "m"),
        ({"m": 1.5}, TypeError, "m"),
        ({"probe_radius": 0.0}, ValueError, "probe_radius"),
        ({"smoothness": 0.0}, ValueError, "smoothness"),
        ({"iterations": -1}, ValueError, "iterations"),
        ({"level_tolerance": -0.1}, ValueError, "level_tolerance"),
        ({"subsample_spacing": 0.0}, ValueError, "subsample_spacing"),
        ({"feature_threshold": 0.0}, ValueError, "feature_threshold"),
    ],
)
def test_sample_sdf_points_rejects_invalid_inputs(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    coords, radii = _two_separated_atoms()
    params = {
        "m": 4,
        "probe_radius": 0.8,
    }
    params.update(kwargs)

    with pytest.raises(error_type, match=message):
        sample_sdf_points(coords, radii, **params)
