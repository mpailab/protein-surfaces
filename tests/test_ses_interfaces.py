import math

import pytest
import torch

from ses import (
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
    _prepare_projection_inputs,
    _recover_probe_centers,
    _select_suitable_atom_indices,
    project_points_to_ses,
)


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.norm(vector)


def test_project_points_to_ses_keeps_atom_surface_point_without_neighbors() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([1.0], dtype=torch.float64)
    points = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    probe_radius = 0.5

    projected_points, valid_point_mask = project_points_to_ses(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    assert torch.equal(valid_point_mask, torch.tensor([[True]]))
    assert torch.allclose(projected_points, points)


def test_project_points_to_ses_keeps_off_axis_single_atom_surface_point() -> None:
    coords = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    radii = torch.tensor([2.0], dtype=torch.float64)
    direction = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
    points = (coords + radii[0] * direction).unsqueeze(0)

    projected_points, valid_point_mask = project_points_to_ses(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=0.75,
    )

    assert torch.equal(valid_point_mask, torch.tensor([[True]]))
    assert torch.allclose(projected_points, points, atol=1e-12, rtol=0)


def test_project_points_to_ses_uses_single_plane_when_only_one_neighbor_exists() -> None:
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

    projected_points, valid_point_mask = project_points_to_ses(
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


def test_project_points_to_ses_projects_two_atom_owner_points_to_shared_probe() -> None:
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

    projected_points, valid_point_mask = project_points_to_ses(
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


def test_project_points_to_ses_uses_two_planes_for_three_atom_probe_position() -> None:
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

    projected_points, valid_point_mask = project_points_to_ses(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=probe_radius,
    )

    expected_point = stable_probe_center + _normalize(owner_point - stable_probe_center)

    assert valid_point_mask[0, 0]
    assert torch.linalg.norm(projected_points[0, 0] - owner_point) > 1e-3
    assert torch.allclose(projected_points[0, 0], expected_point, atol=1e-12, rtol=0)


def test_project_points_to_ses_projects_three_atom_owner_points_to_shared_probe() -> None:
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

    projected_points, valid_point_mask = project_points_to_ses(
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


def test_project_points_to_ses_promotes_integer_points_to_float() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([1.0], dtype=torch.float64)
    points = torch.tensor([[[1, 0, 0]]], dtype=torch.int64)

    projected_points, valid_point_mask = project_points_to_ses(
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


def test_project_points_to_ses_handles_empty_atom_axis() -> None:
    points = torch.empty((2, 0, 3), dtype=torch.float32)
    coords = torch.empty((0, 3), dtype=torch.float32)
    radii = torch.empty((0,), dtype=torch.float32)

    projected_points, valid_point_mask = project_points_to_ses(
        points=points,
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=0.5,
    )

    assert projected_points.shape == (2, 0, 3)
    assert valid_point_mask.shape == (2, 0)


def test_project_points_to_ses_rejects_negative_probe_radius() -> None:
    points = torch.zeros((1, 1, 3))
    coords = torch.zeros((1, 3))
    radii = torch.ones(1)

    with pytest.raises(ValueError, match="probe_radius"):
        project_points_to_ses(
            points=points,
            atom_coords=coords,
            atom_radii=radii,
            probe_radius=-0.1,
        )


def test_project_points_to_ses_rejects_negative_atom_radii() -> None:
    points = torch.zeros((1, 1, 3))
    coords = torch.zeros((1, 3))
    radii = torch.tensor([-1.0])

    with pytest.raises(ValueError, match="atom_radii"):
        project_points_to_ses(
            points=points,
            atom_coords=coords,
            atom_radii=radii,
            probe_radius=0.5,
        )


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

    selected = _select_suitable_atom_indices(suitable_atom_mask, geodesic_distances)

    expected = torch.tensor([[[2, 1], [0, 1], [2, 2]]], dtype=torch.int)
    assert torch.equal(selected, expected)


def test_select_suitable_atom_indices_rejects_distance_shape_mismatch() -> None:
    suitable_atom_mask = torch.zeros((1, 2, 2), dtype=torch.bool)
    geodesic_distances = torch.zeros((1, 2, 3))

    with pytest.raises(ValueError, match="geodesic_distances"):
        _select_suitable_atom_indices(suitable_atom_mask, geodesic_distances)


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
