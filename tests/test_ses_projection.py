import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ses.projection import (
    _PairGeometry,
    _PointAtomGeometry,
    _TangencyGeometry,
    _apply_affine_projector,
    _build_affine_projector,
    _compute_geodesic_distances,
    _compute_pair_geometry,
    _compute_point_atom_geometry,
    _compute_probe_centers,
    _compute_suitable_atom_indices,
    _compute_tangency_geometry,
    _probe_centers_accessible_from_exterior,
    _prefer_any_pair_only_probe_centers,
    _prefer_pair_only_probe_centers,
    _prepare_projection_inputs,
    _recover_probe_centers,
    _select_suitable_atom_indices,
    project_points,
    sample_atom_points,
    sample_projected_points,
    _sample_projected_grid,
)


NPY_DATA_DIR = Path(__file__).resolve().parent / "data" / "npy"
ALL_SES_FIXTURE_IDS = ("2PQ2_B", "4FT4_Q", "4Q6I_J")
SES_FIXTURE_IDS = ("2PQ2_B", "4Q6I_J")


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.norm(vector)


def _load_fixture_atoms(
    pdb_id: str,
    dtype: torch.dtype = torch.float64,
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


def _outward_atom_surface_points(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
) -> torch.Tensor:
    molecule_center = atom_coords.mean(dim=0, keepdim=True)
    directions = atom_coords - molecule_center
    directions = directions / torch.linalg.norm(
        directions,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    return (atom_coords + atom_radii.unsqueeze(-1) * directions).unsqueeze(0)


def _methane_atoms() -> tuple[torch.Tensor, torch.Tensor]:
    tetrahedron = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    tetrahedron = tetrahedron / torch.linalg.norm(
        tetrahedron,
        dim=-1,
        keepdim=True,
    )
    atom_coords = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float64),
            1.09 * tetrahedron,
        ),
        dim=0,
    )
    atom_radii = torch.tensor([1.70, 1.20, 1.20, 1.20, 1.20], dtype=torch.float64)
    return atom_coords, atom_radii


def _octahedral_probe_cage_atoms() -> tuple[torch.Tensor, torch.Tensor]:
    atom_coords = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ],
        dtype=torch.float64,
    )
    atom_radii = torch.full((6,), 1.1, dtype=torch.float64)
    return atom_coords, atom_radii


def _glucose_chair_atoms() -> tuple[torch.Tensor, torch.Tensor]:
    atom_coords = torch.tensor(
        [
            [1.25, 0.0, 0.35],
            [0.62, 1.08, -0.35],
            [-0.62, 1.08, 0.35],
            [-1.25, 0.0, -0.35],
            [-0.62, -1.08, 0.35],
            [0.62, -1.08, -0.35],
            [1.291980477382, 2.250546638021, 0.122401107361],
            [-1.291980477382, 2.250546638021, -0.122401107361],
            [-2.599717449603, 0.0, 0.122401107361],
            [-1.291980477382, -2.250546638021, -0.122401107361],
            [1.937946559, -3.375777876967, 0.282522511546],
            [1.368386117775, -2.383640334188, -0.575477488454],
            [0.37730809587, 0.657246360548, 0.62492563819],
            [-0.37730809587, 0.657246360548, -0.62492563819],
            [-0.762537180905, 0.0, 0.62492563819],
            [-0.37730809587, -0.657246360548, -0.62492563819],
            [0.37730809587, -0.657246360548, -1.32492563819],
            [2.230430874657, -1.888762788571, -0.128179990166],
            [0.506341360892, -2.878517879806, -1.022774986742],
            [1.743100238422, 3.036368157251, 0.4395375151],
            [-1.743100238422, 3.036368157251, -0.4395375151],
            [-3.505821471715, 0.0, 0.4395375151],
            [-1.743100238422, -3.036368157251, -0.4395375151],
            [2.320308813249, -4.041828255336, 0.858522511546],
        ],
        dtype=torch.float64,
    )
    atom_radii = torch.tensor(
        [
            1.52,
            1.70,
            1.70,
            1.70,
            1.70,
            1.70,
            1.52,
            1.52,
            1.52,
            1.52,
            1.52,
            1.70,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
            1.20,
        ],
        dtype=torch.float64,
    )
    return atom_coords, atom_radii


def test_sample_atom_points_returns_uniform_points_on_atom_spheres() -> None:
    atom_coords = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]],
        dtype=torch.float64,
    )
    atom_radii = torch.tensor([[1.5], [2.0]], dtype=torch.float64)
    num_points = 10

    points = sample_atom_points(atom_coords, atom_radii, num_points)

    assert points.shape == (num_points, 2, 3)

    point_dists = torch.linalg.norm(points - atom_coords.unsqueeze(0), dim=-1)
    assert torch.allclose(
        point_dists,
        atom_radii.reshape(1, -1),
        atol=1e-12,
        rtol=0,
    )

    first_atom_directions = (points[:, 0] - atom_coords[0]) / atom_radii[0, 0]
    second_atom_directions = (points[:, 1] - atom_coords[1]) / atom_radii[1, 0]
    expected_z = 1 - 2 * (
        torch.arange(num_points, dtype=torch.float64) + 0.5
    ) / num_points

    assert torch.allclose(first_atom_directions, second_atom_directions, atol=1e-12)
    assert torch.allclose(
        torch.linalg.norm(first_atom_directions, dim=-1),
        torch.ones(num_points, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert torch.allclose(first_atom_directions[:, 2], expected_z, atol=1e-12)


def test_sample_atom_points_promotes_integer_inputs_to_float() -> None:
    atom_coords = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.int64)
    atom_radii = torch.tensor([1.0, 2.0], dtype=torch.float64)

    points = sample_atom_points(atom_coords, atom_radii, 4)

    assert points.dtype == torch.float64
    assert points.shape == (4, 2, 3)


def test_sample_atom_points_handles_empty_axes() -> None:
    empty_atom_points = sample_atom_points(
        atom_coords=torch.empty((0, 3), dtype=torch.float32),
        atom_radii=torch.empty((0, 1), dtype=torch.float32),
        m=5,
    )
    empty_sample_points = sample_atom_points(
        atom_coords=torch.zeros((2, 3), dtype=torch.float32),
        atom_radii=torch.ones((2, 1), dtype=torch.float32),
        m=0,
    )

    assert empty_atom_points.shape == (5, 0, 3)
    assert empty_sample_points.shape == (0, 2, 3)


@pytest.mark.parametrize(
    ("atom_coords", "atom_radii", "m", "exception", "match"),
    [
        (torch.zeros(3), torch.ones(1), 1, ValueError, "atom_coords"),
        (torch.zeros((2, 2)), torch.ones(2), 1, ValueError, "atom_coords"),
        (torch.zeros((2, 3)), torch.ones(3), 1, ValueError, "one radius"),
        (
            torch.zeros((2, 3)),
            torch.tensor([1.0, -1.0]),
            1,
            ValueError,
            "non-negative",
        ),
        (torch.zeros((2, 3)), torch.ones(2), -1, ValueError, "non-negative"),
        (torch.zeros((2, 3)), torch.ones(2), 1.5, TypeError, "integer"),
        (torch.zeros((2, 3)), torch.ones(2), True, TypeError, "integer"),
    ],
)
def test_sample_atom_points_rejects_invalid_inputs(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    m: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        sample_atom_points(atom_coords, atom_radii, m)


def test_sample_projected_points_returns_flat_points_and_atom_features() -> None:
    atom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    atom_radii = torch.tensor([1.0, 1.5], dtype=torch.float64)

    points, atom_features = sample_projected_points(
        atom_coords,
        atom_radii,
        m=8,
        probe_radius=0.8,
        include_atom_features=True,
    )
    points_only = sample_projected_points(
        atom_coords,
        atom_radii,
        m=8,
        probe_radius=0.8,
    )

    assert points.ndim == 2
    assert points.shape[1] == 3
    assert atom_features.shape == (points.shape[0], atom_coords.shape[0])
    assert torch.equal(points, points_only)
    assert torch.all((atom_features == 0) | (atom_features == 1))
    assert torch.all(atom_features.sum(dim=1) == 1)


def test_sample_projected_points_matches_explicit_sampling_and_projection() -> None:
    atom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    atom_radii = torch.tensor([[1.0], [1.5]], dtype=torch.float64)
    num_points = 6
    probe_radius = 0.8

    expected_sampled_points = sample_atom_points(
        atom_coords,
        atom_radii,
        num_points,
    )
    expected_ses_points, expected_valid_mask = project_points(
        points=expected_sampled_points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=probe_radius,
    )

    ses_points, valid_mask = _sample_projected_grid(
        atom_coords,
        atom_radii,
        num_points,
        probe_radius,
    )

    assert torch.allclose(ses_points, expected_ses_points, atol=1e-12, rtol=0)
    assert torch.equal(valid_mask, expected_valid_mask)


@pytest.mark.parametrize("pdb_id", ALL_SES_FIXTURE_IDS)
def test_sample_projected_points_runs_on_all_fixture_npy_molecules(pdb_id: str) -> None:
    atom_coords, atom_radii = _load_fixture_atoms(pdb_id)
    num_points = 2

    ses_points, valid_mask = _sample_projected_grid(
        atom_coords,
        atom_radii,
        num_points,
        probe_radius=1.4,
    )

    assert ses_points.shape == (num_points, atom_coords.shape[0], 3)
    assert valid_mask.shape == ses_points.shape[:2]
    assert valid_mask.any()
    assert torch.isfinite(ses_points).all()

    valid_ses_points = ses_points[valid_mask]
    atom_distances = torch.cdist(valid_ses_points, atom_coords)
    assert bool((atom_distances >= atom_radii.unsqueeze(0) - 1e-8).all().item())


def test_sample_projected_points_handles_empty_sample_axis() -> None:
    atom_coords = torch.zeros((2, 3), dtype=torch.float32)
    atom_radii = torch.ones((2, 1), dtype=torch.float32)

    ses_points, valid_mask = _sample_projected_grid(atom_coords, atom_radii, 0, 1.4)

    assert ses_points.shape == (0, 2, 3)
    assert valid_mask.shape == (0, 2)


def test_sample_projected_points_rejects_negative_probe_radius() -> None:
    with pytest.raises(ValueError, match="probe_radius"):
        _sample_projected_grid(
            atom_coords=torch.zeros((1, 3)),
            atom_radii=torch.ones((1, 1)),
            m=1,
            probe_radius=-0.1,
        )


def test_project_points_keeps_atom_surface_point_without_neighbors() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([1.0], dtype=torch.float64)
    points = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    probe_radius = 0.5

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    assert torch.equal(valid_point_mask, torch.tensor([[True]]))
    assert torch.allclose(projected_points, points)


def test_project_points_keeps_off_axis_single_atom_surface_point() -> None:
    coords = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    radii = torch.tensor([2.0], dtype=torch.float64)
    direction = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
    points = (coords + radii[0] * direction).unsqueeze(0)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=0.75,
    )

    assert torch.equal(valid_point_mask, torch.tensor([[True]]))
    assert torch.allclose(projected_points, points, atol=1e-12, rtol=0)


def test_project_points_uses_single_plane_when_only_one_neighbor_exists() -> None:
    coords = torch.tensor(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    theta = 0.1
    owner_point = coords[0] + torch.tensor(
        [math.cos(theta), math.sin(theta), 0.0],
        dtype=torch.float64,
    )
    points = torch.stack(
        [
            torch.stack(
                [
                    owner_point,
                    coords[1] + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                ]
            )
        ]
    )
    probe_radius = 1.0

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    projected_probe_center = torch.tensor(
        [0.0, math.sqrt((1.0 + probe_radius) ** 2 - 1.5**2), 0.0],
        dtype=torch.float64,
    )
    expected_point = projected_probe_center + _normalize(
        owner_point - projected_probe_center
    )

    assert valid_point_mask[0, 0]
    assert torch.linalg.norm(projected_points[0, 0] - owner_point) > 1e-3
    assert torch.allclose(projected_points[0, 0], expected_point, atol=1e-12, rtol=0)


def test_project_points_projects_two_atom_owner_points_to_shared_probe() -> None:
    coords = torch.tensor(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    probe_radius = 1.0
    theta = 0.25
    points = torch.tensor(
        [
            [
                [coords[0, 0] + math.cos(theta), math.sin(theta), 0.0],
                [coords[1, 0] - math.cos(theta), math.sin(theta), 0.0],
            ]
        ],
        dtype=torch.float64,
    )
    probe_center = torch.tensor(
        [0.0, math.sqrt((1.0 + probe_radius) ** 2 - 1.5**2), 0.0],
        dtype=torch.float64,
    )

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    expected_points = torch.stack(
        [
            probe_center + _normalize(points[0, 0] - probe_center) * probe_radius,
            probe_center + _normalize(points[0, 1] - probe_center) * probe_radius,
        ]
    ).unsqueeze(0)
    assert torch.equal(valid_point_mask, torch.tensor([[True, True]]))
    assert torch.allclose(projected_points, expected_points, atol=1e-12, rtol=0)


def test_project_points_uses_two_planes_for_three_atom_probe_position() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 1.5 * math.sqrt(3.0), 0.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    probe_radius = 1.0

    stable_probe_center = torch.tensor(
        [1.5, math.sqrt(3.0) / 2.0, 1.0],
        dtype=torch.float64,
    )
    owner_direction = _normalize(stable_probe_center - coords[0])
    tangent_direction = _normalize(
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        - owner_direction[2] * owner_direction
    )
    owner_point = _normalize(owner_direction - 0.3 * tangent_direction)

    points = torch.stack(
        [
            torch.stack(
                [
                    owner_point,
                    coords[1] + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
                    coords[2] + torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
                ]
            )
        ]
    )

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    expected_point = stable_probe_center + _normalize(owner_point - stable_probe_center)

    assert valid_point_mask[0, 0]
    assert torch.linalg.norm(projected_points[0, 0] - owner_point) > 1e-3
    assert torch.allclose(projected_points[0, 0], expected_point, atol=1e-12, rtol=0)


def test_project_points_projects_three_atom_owner_points_to_shared_probe() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 1.5 * math.sqrt(3.0), 0.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    probe_radius = 1.0
    probe_center = torch.tensor(
        [1.5, math.sqrt(3.0) / 2.0, 1.0],
        dtype=torch.float64,
    )
    tangent_seed = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    owner_points = []
    for atom_coord in coords:
        owner_direction = _normalize(probe_center - atom_coord)
        tangent_direction = _normalize(
            tangent_seed - torch.dot(tangent_seed, owner_direction) * owner_direction
        )
        owner_points.append(
            atom_coord + _normalize(owner_direction - 0.3 * tangent_direction)
        )
    points = torch.stack([torch.stack(owner_points)])

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    expected_points = torch.stack(
        [
            probe_center + _normalize(points[0, atom_idx] - probe_center) * probe_radius
            for atom_idx in range(3)
        ]
    ).unsqueeze(0)
    assert torch.equal(valid_point_mask, torch.tensor([[True, True, True]]))
    assert torch.allclose(projected_points, expected_points, atol=1e-12, rtol=0)


def test_compute_probe_centers_keeps_water_hydrogen_pair_probe_visible() -> None:
    coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.52, 1.20, 1.20], dtype=torch.float64)
    points = torch.tensor(
        [
            [
                [0.0, -1.52, 0.0],
                [1.0114137140764754, 1.116675813835884, 0.436],
                [-0.2390, 2.1270, 0.0],
            ]
        ],
        dtype=torch.float64,
    )

    probe_centers, valid_point_mask = _compute_probe_centers(
        points,
        coords,
        radii,
        probe_radius=1.4,
    )

    center_atom_dists = torch.linalg.norm(probe_centers[0, 1] - coords, dim=-1)
    assert valid_point_mask[0, 1]
    assert center_atom_dists[0] > radii[0] + 1.4
    assert torch.allclose(
        center_atom_dists[1:],
        radii[1:] + 1.4,
        atol=1e-12,
        rtol=0,
    )


def test_compute_probe_centers_allows_water_oxygen_point_to_hydrogen_pair() -> None:
    coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.52, 1.20, 1.20], dtype=torch.float64)
    points = torch.tensor(
        [
            [
                [0.8208533023094201, 1.1618823052054148, 0.5353777777777777],
                [2.1572, 0.0, 0.0],
                [-0.2390, 2.1270, 0.0],
            ]
        ],
        dtype=torch.float64,
    )

    probe_centers, valid_point_mask = _compute_probe_centers(
        points,
        coords,
        radii,
        probe_radius=1.4,
    )

    center_atom_dists = torch.linalg.norm(probe_centers[0, 0] - coords, dim=-1)
    assert valid_point_mask[0, 0]
    assert center_atom_dists[0] > radii[0] + 1.4
    assert torch.allclose(
        center_atom_dists[1:],
        radii[1:] + 1.4,
        atol=1e-12,
        rtol=0,
    )


def test_compute_probe_centers_allows_ethanol_carbon_points_to_hydrogen_pairs() -> None:
    coords = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [1.520, 0.000, 0.000],
            [2.120, 1.210, 0.000],
            [-0.540, 0.930, 0.000],
            [-0.540, -0.465, 0.805],
            [-0.540, -0.465, -0.805],
            [1.910, -0.520, 0.890],
            [1.910, -0.520, -0.890],
            [2.950, 1.100, 0.000],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor(
        [1.70, 1.70, 1.52, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20],
        dtype=torch.float64,
    )
    probe_radius = 1.4
    points = sample_atom_points(coords, radii, 900)[[39, 60, 67, 288]]

    probe_centers, valid_point_mask = _compute_probe_centers(
        points,
        coords,
        radii,
        probe_radius=probe_radius,
    )

    expectations = (
        (0, 0, (4, 6)),
        (1, 0, (4, 6)),
        (2, 1, (4, 6)),
        (3, 1, (6, 8)),
    )
    expanded_radii = radii + probe_radius
    for sample_idx, owner_idx, hydrogen_pair in expectations:
        center_atom_dists = torch.linalg.norm(
            probe_centers[sample_idx, owner_idx] - coords,
            dim=-1,
        )

        assert valid_point_mask[sample_idx, owner_idx]
        assert center_atom_dists[owner_idx] > expanded_radii[owner_idx]
        assert torch.allclose(
            center_atom_dists[list(hydrogen_pair)],
            expanded_radii[list(hydrogen_pair)],
            atol=1e-12,
            rtol=0,
        )


def test_project_points_keeps_ch4_outward_hydrogen_surface_points() -> None:
    coords, radii = _methane_atoms()
    points = sample_atom_points(coords, radii, 900)
    num_samples = points.shape[0]

    _, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=1.4,
    )

    hydrogen_points = points[:, 1:]
    hydrogen_coords = coords[1:]
    hydrogen_directions = hydrogen_coords / torch.linalg.norm(
        hydrogen_coords,
        dim=-1,
        keepdim=True,
    )
    outward_cosines = (
        (hydrogen_points - hydrogen_coords.unsqueeze(0))
        * hydrogen_directions.unsqueeze(0)
    ).sum(dim=-1) / radii[1:].view(1, -1)

    hydrogen_atom_dists = torch.linalg.norm(
        hydrogen_points.unsqueeze(2) - coords.view(1, 1, -1, 3),
        dim=-1,
    )
    non_owner_mask = torch.ones((4, 5), dtype=torch.bool)
    non_owner_mask[torch.arange(4), torch.arange(1, 5)] = False
    non_owner_radii = radii.view(1, 5).expand(4, 5)[non_owner_mask].view(1, 4, 4)
    outside_non_owner_atoms = (
        hydrogen_atom_dists[:, non_owner_mask].view(num_samples, 4, 4)
        >= non_owner_radii + 1e-8
    ).all(dim=-1)
    outward_hydrogen_ses_mask = (outward_cosines > 0.9) & outside_non_owner_atoms

    assert outward_hydrogen_ses_mask.any()
    assert bool(valid_point_mask[:, 1:][outward_hydrogen_ses_mask].all().item())


def test_probe_centers_accessible_from_exterior_rejects_enclosed_cavity() -> None:
    coords, radii = _octahedral_probe_cage_atoms()
    probe_centers = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]]],
        dtype=torch.float64,
    )

    accessible_mask = _probe_centers_accessible_from_exterior(
        probe_centers=probe_centers,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=1.4,
        grid_spacing=0.2,
    )

    assert torch.equal(accessible_mask, torch.tensor([[False, True]]))


def test_project_points_rejects_interior_probe_center() -> None:
    coords, radii = _octahedral_probe_cage_atoms()
    directions = coords / torch.linalg.norm(coords, dim=-1, keepdim=True)
    points = coords + radii.unsqueeze(-1) * directions
    points[0] = coords[0] - radii[0] * directions[0]

    probe_centers, local_valid_mask = _compute_probe_centers(
        points.unsqueeze(0),
        coords,
        radii,
        probe_radius=1.4,
    )
    projected_points, valid_point_mask = project_points(
        points=points.unsqueeze(0),
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=1.4,
    )

    assert local_valid_mask[0, 0]
    assert torch.allclose(
        probe_centers[0, 0],
        torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
    )
    assert not valid_point_mask[0, 0]
    assert bool(valid_point_mask[0, 1:].all().item())
    assert torch.isfinite(projected_points).all()


def test_project_points_rejects_glucose_chair_infeasible_patch_outlier() -> None:
    coords, radii = _glucose_chair_atoms()
    points = sample_atom_points(coords, radii, 350)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=1.4,
    )

    projection_distances = torch.linalg.norm(projected_points - points, dim=-1)
    assert not valid_point_mask[27, 17]
    assert projection_distances[valid_point_mask].max() < 0.7


def test_any_pair_repair_ignores_irrelevant_ethanol_oxygen_point() -> None:
    coords = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [1.520, 0.000, 0.000],
            [2.120, 1.210, 0.000],
            [-0.540, 0.930, 0.000],
            [-0.540, -0.465, 0.805],
            [-0.540, -0.465, -0.805],
            [1.910, -0.520, 0.890],
            [1.910, -0.520, -0.890],
            [2.950, 1.100, 0.000],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor(
        [1.70, 1.70, 1.52, 1.20, 1.20, 1.20, 1.20, 1.20, 1.20],
        dtype=torch.float64,
    )
    probe_radius = 1.4
    points = sample_atom_points(coords, radii, 900)[[86]]
    point_geometry = _compute_point_atom_geometry(points, coords, radii)
    pair_geometry = _compute_pair_geometry(coords, radii, probe_radius)
    tangency_geometry = _compute_tangency_geometry(
        point_geometry,
        pair_geometry,
        radii,
    )
    geodesic_distances = _compute_geodesic_distances(
        points,
        coords,
        radii,
        pair_geometry,
        tangency_geometry,
    )
    suitable_atom_indices = _compute_suitable_atom_indices(
        point_geometry,
        pair_geometry,
        tangency_geometry,
        geodesic_distances,
    )
    affine_projection, affine_shift = _build_affine_projector(
        points.shape[0],
        point_geometry,
        pair_geometry,
        suitable_atom_indices,
    )
    probe_centers = _recover_probe_centers(
        points,
        coords,
        pair_geometry.atom_ext_radii_sq,
        affine_projection,
        affine_shift,
    )
    probe_centers = _prefer_pair_only_probe_centers(
        points,
        coords,
        pair_geometry,
        suitable_atom_indices,
        probe_radius,
        probe_centers,
    )

    repaired_probe_centers = _prefer_any_pair_only_probe_centers(
        points,
        coords,
        radii,
        pair_geometry,
        tangency_geometry,
        probe_radius,
        probe_centers,
    )

    assert point_geometry.valid_point_mask[0, 2]
    assert torch.allclose(
        repaired_probe_centers[0, 2],
        probe_centers[0, 2],
        atol=1e-12,
        rtol=0,
    )


def test_project_points_promotes_integer_points_to_float() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([1.0], dtype=torch.float64)
    points = torch.tensor([[[1, 0, 0]]], dtype=torch.int64)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=0.5,
    )

    assert projected_points.dtype == torch.float64
    assert torch.equal(valid_point_mask, torch.tensor([[True]]))
    expected_points = coords.unsqueeze(0) + torch.tensor(
        [[[1.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(projected_points, expected_points)


def test_project_points_handles_empty_atom_axis() -> None:
    points = torch.empty((2, 0, 3), dtype=torch.float32)
    coords = torch.empty((0, 3), dtype=torch.float32)
    radii = torch.empty((0,), dtype=torch.float32)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=0.5,
    )

    assert projected_points.shape == (2, 0, 3)
    assert valid_point_mask.shape == (2, 0)


def test_project_points_rejects_negative_probe_radius() -> None:
    points = torch.zeros((1, 1, 3))
    coords = torch.zeros((1, 3))
    radii = torch.ones(1)

    with pytest.raises(ValueError, match="probe_radius"):
        project_points(
            points=points,
            atom_coords=coords,
            atom_radii=radii,
            probe_radius=-0.1,
        )


def test_project_points_rejects_negative_atom_radii() -> None:
    points = torch.zeros((1, 1, 3))
    coords = torch.zeros((1, 3))
    radii = torch.tensor([-1.0])

    with pytest.raises(ValueError, match="atom_radii"):
        project_points(
            points=points,
            atom_coords=coords,
            atom_radii=radii,
            probe_radius=0.5,
        )


@pytest.mark.parametrize("pdb_id", SES_FIXTURE_IDS)
def test_project_points_projects_fixture_npy_atom_surface_points(
    pdb_id: str,
) -> None:
    atom_coords, atom_radii = _load_fixture_atoms(pdb_id)
    points = _outward_atom_surface_points(atom_coords, atom_radii)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=1.4,
    )

    assert projected_points.shape == points.shape
    assert valid_point_mask.shape == points.shape[:2]
    assert valid_point_mask.any()
    assert torch.isfinite(projected_points).all()

    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    atom_sq_norms = atom_coords.square().sum(dim=-1).view(1, 1, -1)
    point_atom_sq_dists = point_sq_norms + atom_sq_norms - 2 * (points @ atom_coords.T)
    atom_radii_sq = atom_radii.square().view(1, 1, -1)
    sq_dist_tol = (
        100
        * torch.finfo(point_atom_sq_dists.dtype).eps
        * torch.maximum(point_atom_sq_dists.abs(), atom_radii_sq).clamp_min(1)
    )
    expected_valid_mask = (
        point_atom_sq_dists >= atom_radii_sq - sq_dist_tol
    ).all(dim=-1)
    assert torch.equal(valid_point_mask, expected_valid_mask)

    valid_projected_points = projected_points[valid_point_mask]
    projected_atom_dists = torch.cdist(valid_projected_points, atom_coords)
    assert bool(
        (projected_atom_dists >= atom_radii.unsqueeze(0) - 1e-8).all().item()
    )


def test_project_points_keeps_fixture_dtype_for_float32_npy_inputs() -> None:
    atom_coords, atom_radii = _load_fixture_atoms("4Q6I_J", dtype=torch.float32)
    points = _outward_atom_surface_points(atom_coords, atom_radii)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=1.4,
    )

    assert projected_points.dtype == torch.float32
    assert valid_point_mask.dtype == torch.bool
    assert torch.isfinite(projected_points).all()


def test_project_points_is_atom_order_equivariant_on_fixture_npy_atoms() -> None:
    atom_coords, atom_radii = _load_fixture_atoms("2PQ2_B")
    points = _outward_atom_surface_points(atom_coords, atom_radii)
    atom_order = torch.arange(atom_coords.shape[0] - 1, -1, -1)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=1.4,
    )
    reordered_projected_points, reordered_valid_point_mask = project_points(
        points=points[:, atom_order],
        atom_coords=atom_coords[atom_order],
        atom_radii=atom_radii[atom_order],
        probe_radius=1.4,
    )

    assert torch.equal(reordered_valid_point_mask, valid_point_mask[:, atom_order])
    assert torch.allclose(
        reordered_projected_points,
        projected_points[:, atom_order],
        atol=1e-8,
        rtol=1e-8,
    )


def test_project_points_keeps_4ft4_fixture_projection_finite() -> None:
    atom_coords, atom_radii = _load_fixture_atoms("4FT4_Q")
    points = sample_atom_points(atom_coords, atom_radii, 2)

    projected_points, valid_point_mask = project_points(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=1.4,
    )

    assert valid_point_mask.any()
    assert torch.isfinite(projected_points).all()


def test_prepare_projection_inputs_validates_and_promotes_tensors() -> None:
    points = torch.tensor([[[1, 0, 0], [2, 0, 0]]], dtype=torch.int64)
    atom_coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    prepared = _prepare_projection_inputs(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=1.0,
    )

    assert prepared.points.shape == (1, 2, 3)
    assert prepared.atom_coords.shape == (2, 3)
    assert prepared.atom_radii.shape == (2,)
    assert prepared.points.dtype == torch.float64
    assert prepared.atom_coords.dtype == torch.float64
    assert prepared.atom_radii.dtype == torch.float64
    assert torch.equal(prepared.atom_radii, torch.tensor([1.0, 2.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("points", "atom_coords", "atom_radii", "probe_radius", "match"),
    [
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((2, 3)),
            torch.ones(2),
            0.0,
            "probe_radius",
        ),
        (
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            torch.ones(2),
            1.0,
            "points",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((2, 2)),
            torch.ones(2),
            1.0,
            "atom_coords",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((3, 3)),
            torch.ones(2),
            1.0,
            "points.shape",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((2, 3)),
            torch.ones(3),
            1.0,
            "radii",
        ),
        (
            torch.zeros((1, 2, 3)),
            torch.zeros((2, 3)),
            torch.tensor([1.0, -1.0]),
            1.0,
            "atom_radii",
        ),
    ],
)
def test_prepare_projection_inputs_rejects_invalid_inputs(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare_projection_inputs(points, atom_coords, atom_radii, probe_radius)


def test_select_suitable_atom_indices_ranks_by_geodesic_distance() -> None:
    suitable_atom_mask = torch.tensor(
        [
            [
                [False, True, True],
                [True, False, False],
                [False, False, False],
            ]
        ]
    )
    geodesic_distances = torch.tensor(
        [
            [
                [0.0, 1.0, 3.0],
                [2.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ]
    )
    pair_normals = torch.zeros((3, 3, 3))

    selected = _select_suitable_atom_indices(
        suitable_atom_mask,
        geodesic_distances,
        pair_normals,
    )

    expected = torch.tensor([[[2, 1], [0, 1], [2, 2]]], dtype=torch.int)
    assert torch.equal(selected, expected)


def test_select_suitable_atom_indices_uses_next_unblocked_neighbor() -> None:
    suitable_atom_mask = torch.tensor(
        [
            [
                [False, True, True, True],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ]
    )
    geodesic_distances = torch.tensor(
        [
            [
                [0.0, 10.0, 9.0, 8.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    pair_normals = torch.zeros((4, 4, 3))
    pair_normals[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    pair_normals[0, 2] = torch.tensor([1.0, 0.0, 0.0])
    pair_normals[0, 3] = torch.tensor([0.0, 1.0, 0.0])

    selected = _select_suitable_atom_indices(
        suitable_atom_mask,
        geodesic_distances,
        pair_normals,
    )

    expected = torch.tensor([[[1, 3], [1, 1], [2, 2], [3, 3]]], dtype=torch.int)
    assert torch.equal(selected, expected)


def test_select_suitable_atom_indices_rejects_distance_shape_mismatch() -> None:
    suitable_atom_mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    geodesic_distances = torch.zeros((1, 2, 3))
    pair_normals = torch.zeros((2, 2, 3))

    with pytest.raises(ValueError, match="geodesic_distances"):
        _select_suitable_atom_indices(
            suitable_atom_mask,
            geodesic_distances,
            pair_normals,
        )


def test_compute_point_atom_geometry_returns_reusable_products() -> None:
    points = torch.tensor([[[0.5, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float64)
    atom_coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)

    geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)

    assert isinstance(geometry, _PointAtomGeometry)
    assert torch.equal(geometry.valid_point_mask, torch.tensor([[False, True]]))
    assert torch.allclose(geometry.coord_sq_norms, torch.tensor([0.0, 4.0], dtype=torch.float64))
    assert torch.allclose(
        geometry.point_coord_dots,
        torch.tensor([[[0.0, 1.0], [0.0, 6.0]]], dtype=torch.float64),
    )


def test_compute_pair_geometry_marks_non_degenerate_expanded_intersections() -> None:
    atom_coords = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    atom_radii = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius=1.0)

    assert isinstance(geometry, _PairGeometry)
    assert torch.allclose(geometry.atom_ext_radii, torch.full((3,), 2.0, dtype=torch.float64))
    assert torch.allclose(geometry.atom_ext_radii_sq, torch.full((3,), 4.0, dtype=torch.float64))
    assert torch.allclose(
        geometry.pair_coord_diffs[0, 1],
        torch.tensor([-3.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(
        geometry.pair_coord_diffs_sq,
        torch.tensor(
            [
                [0.0, 9.0, 100.0],
                [9.0, 0.0, 49.0],
                [100.0, 49.0, 0.0],
            ],
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        geometry.valid_atom_pair_mask,
        torch.tensor(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
    )


def test_compute_tangency_geometry_returns_plane_biases_and_masks() -> None:
    points = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    atom_coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    point_geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)
    pair_geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius=1.0)

    geometry = _compute_tangency_geometry(point_geometry, pair_geometry, atom_radii)

    assert isinstance(geometry, _TangencyGeometry)
    assert torch.allclose(
        geometry.point_coord_diffs_dots,
        torch.tensor([[[0.0, -3.0], [6.0, 0.0]]], dtype=torch.float64),
    )
    assert torch.allclose(
        geometry.tangency_plane_bias,
        torch.tensor([[0.0, -2.25], [6.75, 0.0]], dtype=torch.float64),
    )
    assert torch.equal(
        geometry.tangency_plane_mask,
        torch.tensor([[[True, True], [True, True]]]),
    )


def test_compute_geodesic_distances_uses_owner_sphere_tangency_circle() -> None:
    points = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float64)
    atom_coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    point_geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)
    pair_geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius=1.0)
    tangency_geometry = _compute_tangency_geometry(point_geometry, pair_geometry, atom_radii)

    distances = _compute_geodesic_distances(
        points,
        atom_coords,
        atom_radii,
        pair_geometry,
        tangency_geometry,
    )

    expected_distance = math.acos(0.75)
    assert distances.shape == (1, 2, 2)
    assert torch.allclose(distances[0, 0, 1], torch.tensor(expected_distance, dtype=torch.float64))
    assert torch.allclose(distances[0, 1, 0], torch.tensor(expected_distance, dtype=torch.float64))
    assert torch.equal(distances[0].diagonal(), torch.zeros(2, dtype=torch.float64))


def test_compute_suitable_atom_indices_combines_masks_before_selection() -> None:
    point_geometry = _PointAtomGeometry(
        valid_point_mask=torch.tensor([[True, True, False]]),
        coord_sq_norms=torch.zeros(3),
        point_coord_dots=torch.zeros((1, 3, 3)),
    )
    pair_geometry = _PairGeometry(
        atom_ext_radii=torch.ones(3),
        atom_ext_radii_sq=torch.ones(3),
        pair_coord_diffs=torch.zeros((3, 3, 3)),
        pair_coord_diffs_sq=torch.zeros((3, 3)),
        valid_atom_pair_mask=torch.tensor(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ]
        ),
    )
    tangency_geometry = _TangencyGeometry(
        point_coord_diffs_dots=torch.zeros((1, 3, 3)),
        tangency_plane_bias=torch.zeros((3, 3)),
        tangency_plane_mask=torch.tensor(
            [
                [
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                ]
            ]
        ),
    )
    geodesic_distances = torch.tensor(
        [[[0.0, 1.0, 3.0], [4.0, 0.0, 2.0], [5.0, 6.0, 0.0]]]
    )

    selected = _compute_suitable_atom_indices(
        point_geometry,
        pair_geometry,
        tangency_geometry,
        geodesic_distances,
    )

    expected = torch.tensor([[[2, 1], [0, 2], [2, 2]]], dtype=torch.int)
    assert torch.equal(selected, expected)


def test_build_and_apply_affine_projector_projects_to_selected_pair_planes() -> None:
    atom_coords = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    points = torch.tensor([[[-0.5, 0.1, 0.0], [0.5, 0.1, 0.0]]], dtype=torch.float64)
    point_geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)
    pair_geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius=1.0)
    suitable_atom_indices = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int)

    affine_projection, affine_shift = _build_affine_projector(
        num_samples=1,
        point_geometry=point_geometry,
        pair_geometry=pair_geometry,
        suitable_atom_indices=suitable_atom_indices,
    )
    projected = _apply_affine_projector(points, affine_projection, affine_shift)

    assert affine_projection.shape == (1, 2, 3, 3)
    assert affine_shift.shape == (1, 2, 3)
    assert torch.allclose(projected[..., 0], torch.zeros((1, 2), dtype=torch.float64))
    assert torch.allclose(projected[..., 1:], points[..., 1:])


def test_recover_probe_centers_uses_projected_points_and_atoms() -> None:
    atom_coords = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    points = torch.tensor([[[-0.5, 0.1, 0.0], [0.5, 0.1, 0.0]]], dtype=torch.float64)
    point_geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)
    pair_geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius=1.0)
    suitable_atom_indices = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.int)
    affine_projection, affine_shift = _build_affine_projector(
        num_samples=1,
        point_geometry=point_geometry,
        pair_geometry=pair_geometry,
        suitable_atom_indices=suitable_atom_indices,
    )

    probe_centers = _recover_probe_centers(
        points,
        atom_coords,
        pair_geometry.atom_ext_radii_sq,
        affine_projection,
        affine_shift,
    )

    expected = torch.tensor(
        [[[0.0, math.sqrt(1.75), 0.0], [0.0, math.sqrt(1.75), 0.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(probe_centers, expected, atol=1e-12, rtol=0)


def test_compute_probe_centers_returns_centers_and_validity_mask() -> None:
    atom_coords = torch.tensor([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
    atom_radii = torch.tensor([1.0, 1.0], dtype=torch.float64)
    points = torch.tensor([[[-0.5, 0.1, 0.0], [0.5, 0.1, 0.0]]], dtype=torch.float64)

    probe_centers, valid_point_mask = _compute_probe_centers(
        points,
        atom_coords,
        atom_radii,
        probe_radius=1.0,
    )

    expected_probe_centers = torch.tensor(
        [[[0.0, math.sqrt(1.75), 0.0], [0.0, math.sqrt(1.75), 0.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(valid_point_mask, torch.tensor([[True, True]]))
    assert torch.allclose(probe_centers, expected_probe_centers, atol=1e-12, rtol=0)
