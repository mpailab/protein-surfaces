import math

import pytest
import torch

from ses.graph import build_surface_adjacency


def _assert_sparse_symmetric(adjacency: torch.Tensor, num_points: int) -> torch.Tensor:
    adjacency = adjacency.coalesce()
    assert adjacency.layout == torch.sparse_coo
    assert adjacency.shape == (num_points, num_points)
    indices = adjacency.indices()
    values = adjacency.values()
    assert indices.shape[0] == 2
    assert torch.all(indices[0] != indices[1])
    if values.numel() > 0:
        assert torch.isfinite(values).all()
        assert bool((values > 0).all().item())
    dense = adjacency.to_dense()
    assert torch.allclose(dense, dense.transpose(0, 1))
    return dense


def test_surface_adjacency_connects_local_tangent_neighbors() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)

    adjacency = build_surface_adjacency(
        points,
        normals,
        neighbors=2,
        candidate_neighbors=3,
        max_tangent_component=0.1,
    )
    dense = _assert_sparse_symmetric(adjacency, points.shape[0])

    assert torch.allclose(dense[0, 1], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(dense[0, 2], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(dense[1, 3], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(dense[2, 3], torch.tensor(1.0, dtype=torch.float64))
    assert dense[0, 3] == 0
    assert dense[1, 2] == 0


def test_surface_adjacency_geodesic_weights_use_normal_angle() -> None:
    points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = points.clone()

    euclidean = build_surface_adjacency(
        points,
        normals,
        weight_mode="euclidean",
        neighbors=1,
        candidate_neighbors=1,
        min_normal_cosine=0.0,
        max_tangent_component=0.8,
    )
    geodesic = build_surface_adjacency(
        points,
        normals,
        weight_mode="geodesic",
        neighbors=1,
        candidate_neighbors=1,
        min_normal_cosine=0.0,
        max_tangent_component=0.8,
    )

    assert torch.allclose(
        euclidean.coalesce().values(),
        torch.full((2,), math.sqrt(2), dtype=torch.float64),
    )
    assert torch.allclose(
        geodesic.coalesce().values(),
        torch.full((2,), math.pi / 2, dtype=torch.float64),
    )


def test_surface_adjacency_respects_patch_adjacency_metadata() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.06, 0.00, 0.0],
            [0.03, 0.04, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, -1, -1],
            [0, 1, 2],
            [0, 1, -1],
        ],
        dtype=torch.long,
    )
    support_mask = support_indices >= 0
    block_types = torch.tensor([1, 3, 2], dtype=torch.long)
    block_indices = torch.tensor([0, 0, 0], dtype=torch.long)

    adjacency = build_surface_adjacency(
        points,
        normals,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        neighbors=2,
        candidate_neighbors=2,
        max_tangent_component=0.1,
    )
    dense = _assert_sparse_symmetric(adjacency, points.shape[0])

    assert dense[0, 1] == 0
    assert dense[0, 2] > 0
    assert dense[1, 2] > 0


def test_surface_adjacency_handles_empty_point_clouds() -> None:
    points = torch.empty((0, 3), dtype=torch.float32)
    normals = torch.empty((0, 3), dtype=torch.float32)

    adjacency = build_surface_adjacency(points, normals)

    assert adjacency.layout == torch.sparse_coo
    assert adjacency.shape == (0, 0)
    assert adjacency.coalesce().values().shape == (0,)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"weight_mode": "bad"}, ValueError, "weight_mode"),
        ({"neighbors": 0}, ValueError, "neighbors"),
        ({"neighbors": True}, TypeError, "neighbors"),
        ({"candidate_neighbors": 1, "neighbors": 2}, ValueError, "candidate_neighbors"),
        ({"min_normal_cosine": 2.0}, ValueError, "min_normal_cosine"),
        ({"max_tangent_component": 2.0}, ValueError, "max_tangent_component"),
        ({"max_distance": 0.0}, ValueError, "max_distance"),
    ],
)
def test_surface_adjacency_rejects_invalid_options(kwargs, error, match) -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(error, match=match):
        build_surface_adjacency(points, normals, **kwargs)
