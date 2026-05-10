import math

import pytest
import torch

from ses.graph import (
    _candidate_edges_from_keep,
    _disjoint_single_support_relationships,
    _edge_weights,
    _edge_weights_for_edges,
    _effective_pairwise_element_budget,
    _empty_adjacency,
    _masked_topk_candidates,
    _nearest_neighbor_candidates,
    _nearest_topology_candidates,
    _normalize_vectors,
    _pairwise_metric_distances,
    _prepare_block_inputs,
    _prepare_graph_inputs,
    _prepare_point_inputs,
    _prepare_support_inputs,
    _select_angular_neighbors,
    _select_nearest_valid_candidates,
    _sparse_adjacency_from_edges,
    _support_relationships_for_pairs,
    _tangent_bases,
    _topology_allowed_matrix,
    _unique_undirected_edges,
    _validate_positive_int,
    build_surface_adjacency,
    dense_features_to_supports,
)


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


def test_graph_input_preparation_helpers() -> None:
    raw_points = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    prepared_points = _prepare_point_inputs(raw_points)
    assert prepared_points.dtype == torch.float32

    with pytest.raises(ValueError, match="points"):
        _prepare_point_inputs(torch.tensor([0.0, 1.0, 2.0]))

    points, normals = _prepare_graph_inputs(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
    )
    assert points.dtype == torch.float64
    assert torch.allclose(normals, torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))

    with pytest.raises(ValueError, match="normals"):
        _prepare_graph_inputs(torch.zeros((1, 3)), torch.zeros((2, 3)))

    support_indices, support_mask = _prepare_support_inputs(
        torch.tensor([[0, -1], [1, 2]]),
        None,
        2,
        torch.device("cpu"),
    )
    assert support_indices.dtype == torch.long
    assert torch.equal(support_mask, torch.tensor([[True, False], [True, True]]))

    with pytest.raises(ValueError, match="support_mask"):
        _prepare_support_inputs(None, torch.ones((1, 1), dtype=torch.bool), 1, torch.device("cpu"))

    block_types, block_indices = _prepare_block_inputs(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        2,
        torch.device("cpu"),
    )
    assert block_types.dtype == torch.long
    assert block_indices.dtype == torch.long

    with pytest.raises(ValueError, match="block_types"):
        _prepare_block_inputs(torch.tensor([1]), None, 1, torch.device("cpu"))

    assert _validate_positive_int("neighbors", 3) == 3
    with pytest.raises(TypeError, match="neighbors"):
        _validate_positive_int("neighbors", True)
    with pytest.raises(ValueError, match="neighbors"):
        _validate_positive_int("neighbors", 0)


def test_support_relationship_helpers() -> None:
    row_indices = torch.tensor([[0, -1], [0, -1], [0, 1]])
    row_mask = row_indices >= 0
    col_indices = torch.tensor([[0, -1], [0, 1], [1, 0]])
    col_mask = col_indices >= 0

    same, adjacent, boundary = _support_relationships_for_pairs(
        row_indices,
        row_mask,
        col_indices,
        col_mask,
    )
    assert torch.equal(same, torch.tensor([True, False, True]))
    assert torch.equal(adjacent, torch.tensor([False, True, False]))
    assert torch.equal(boundary, torch.ones_like(boundary))

    wide_row_indices = torch.full((1, 1, 16), -1, dtype=torch.long)
    wide_row_indices[..., :3] = torch.tensor([0, 1, 2])
    wide_row_mask = wide_row_indices >= 0
    wide_col_indices = torch.full((1, 3, 16), -1, dtype=torch.long)
    wide_col_indices[0, 0, :3] = torch.tensor([0, 1, 2])
    wide_col_indices[0, 1, :2] = torch.tensor([0, 1])
    wide_col_indices[0, 2, 0] = 4
    wide_col_mask = wide_col_indices >= 0
    wide_same, wide_adjacent, _ = _support_relationships_for_pairs(
        wide_row_indices,
        wide_row_mask,
        wide_col_indices,
        wide_col_mask,
    )
    assert torch.equal(wide_same, torch.tensor([[True, False, False]]))
    assert torch.equal(wide_adjacent, torch.tensor([[False, True, False]]))

    disjoint = _disjoint_single_support_relationships(
        row_indices,
        row_mask,
        torch.tensor([[1, -1], [0, -1], [2, -1]]),
        torch.tensor([[True, False], [True, False], [True, False]]),
    )
    assert torch.equal(disjoint, torch.tensor([True, False, False]))

    support_indices = torch.tensor([[0, -1], [0, 1], [2, -1]])
    support_mask = support_indices >= 0
    block_types = torch.tensor([1, 2, 1])
    block_indices = torch.tensor([0, 0, 0])
    allowed = _topology_allowed_matrix(
        torch.tensor([0, 1]),
        3,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        allow_disjoint_single_support_edges=False,
    )
    expected = torch.tensor([[True, True, True], [True, True, False]])
    assert torch.equal(allowed, expected)

    disjoint_allowed = _topology_allowed_matrix(
        torch.tensor([0]),
        3,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=None,
        block_indices=None,
        allow_disjoint_single_support_edges=True,
    )
    assert torch.equal(disjoint_allowed, torch.tensor([[True, True, True]]))


def test_dense_features_to_supports_helper() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    support_indices, support_mask = dense_features_to_supports(features)

    assert torch.equal(support_indices, torch.tensor([[0, 2], [-1, -1], [1, -1]]))
    assert torch.equal(
        support_mask,
        torch.tensor([[True, True], [False, False], [True, False]]),
    )

    empty_indices, empty_mask = dense_features_to_supports(torch.zeros((2, 3)))
    assert empty_indices.shape == (2, 0)
    assert empty_mask.shape == (2, 0)

    with pytest.raises(ValueError, match="atom_features"):
        dense_features_to_supports(torch.zeros((3,)))


def test_vector_geometry_and_weight_helpers() -> None:
    vectors = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    normalized = _normalize_vectors(vectors)
    assert torch.allclose(normalized[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.allclose(normalized[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    basis_u, basis_v = _tangent_bases(normals)
    assert torch.allclose((basis_u * normals).sum(dim=-1), torch.zeros(2, dtype=torch.float64))
    assert torch.allclose((basis_v * normals).sum(dim=-1), torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(torch.linalg.norm(basis_u, dim=-1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(torch.linalg.norm(basis_v, dim=-1), torch.ones(2, dtype=torch.float64))

    distances = torch.tensor([2.0], dtype=torch.float64)
    normal_cosines = torch.tensor([0.0], dtype=torch.float64)
    assert torch.equal(_edge_weights(distances, normal_cosines, "euclidean"), distances)
    assert torch.allclose(
        _edge_weights(distances, normal_cosines, "geodesic"),
        torch.tensor([math.pi / 2**0.5], dtype=torch.float64),
    )

    euclidean_distances = torch.tensor([[2.0]], dtype=torch.float64)
    row_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    col_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(
        _pairwise_metric_distances(euclidean_distances, row_normals, col_normals, "geodesic"),
        torch.tensor([[math.pi / 2**0.5]], dtype=torch.float64),
    )

    edge_points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_normals = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    edge_weights = _edge_weights_for_edges(
        edge_points,
        edge_normals,
        torch.tensor([0]),
        torch.tensor([1]),
        "geodesic",
    )
    assert edge_weights.requires_grad
    assert torch.allclose(edge_weights, torch.tensor([math.pi / 2**0.5], dtype=torch.float64))


def test_candidate_selection_helpers() -> None:
    metric_distances = torch.tensor([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    euclidean_distances = metric_distances + 10
    valid = torch.tensor([[True, False, True], [False, True, False]])

    indices, distances, euclidean = _masked_topk_candidates(
        2,
        metric_distances,
        euclidean_distances,
        valid,
    )
    assert torch.equal(indices[0], torch.tensor([2, 0]))
    assert torch.allclose(distances[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(euclidean[0], torch.tensor([12.0, 13.0]))
    assert torch.isinf(distances[1, 1])

    empty_indices, empty_distances, empty_euclidean = _masked_topk_candidates(
        0,
        metric_distances,
        euclidean_distances,
        valid,
    )
    assert empty_indices.shape == (2, 0)
    assert empty_distances.shape == (2, 0)
    assert empty_euclidean.shape == (2, 0)

    keep = _select_nearest_valid_candidates(metric_distances, valid, max_neighbors=1)
    assert torch.equal(keep, torch.tensor([[False, False, True], [False, True, False]]))

    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    angles = torch.arange(6, dtype=torch.float64) * (2 * math.pi / 6)
    directions = torch.stack(
        (
            -torch.sin(angles),
            torch.cos(angles),
            torch.zeros_like(angles),
        ),
        dim=-1,
    ).unsqueeze(0)
    angular_keep = _select_angular_neighbors(
        normals,
        directions,
        torch.ones((1, 6), dtype=torch.float64),
        torch.ones((1, 6), dtype=torch.bool),
        max_neighbors=6,
    )
    assert 0 < int(angular_keep.sum().item()) <= 6


def test_pairwise_candidate_builders() -> None:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)

    indices, distances, euclidean = _nearest_neighbor_candidates(
        points,
        normals,
        1,
        weight_mode="euclidean",
        max_distance=1.5,
        pairwise_element_budget=3,
    )
    assert torch.equal(indices[:2, 0], torch.tensor([1, 0]))
    assert torch.allclose(distances[:2, 0], torch.ones(2, dtype=torch.float64))
    assert torch.isinf(distances[2, 0])
    assert torch.isfinite(euclidean[2, 0])

    support_indices = torch.tensor([[0, 1], [0, 1], [2, -1]])
    support_mask = support_indices >= 0
    topo_indices, topo_distances, _ = _nearest_topology_candidates(
        torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.01, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        normals,
        1,
        weight_mode="euclidean",
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=torch.tensor([2, 2, 1]),
        block_indices=torch.tensor([0, 0, 0]),
        max_distance=None,
        allow_disjoint_single_support_edges=False,
        pairwise_element_budget=3,
    )
    assert topo_indices[0, 0] == 1
    assert torch.isfinite(topo_distances[0, 0])


def test_edge_assembly_helpers() -> None:
    row_indices = torch.arange(2).view(2, 1)
    neighbor_indices = torch.tensor([[1, 0], [0, 1]])
    neighbor_distances = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    normal_cosines = torch.ones_like(neighbor_distances)
    keep = torch.ones_like(neighbor_indices, dtype=torch.bool)

    first, second, weights = _candidate_edges_from_keep(
        row_indices,
        neighbor_indices,
        neighbor_distances,
        normal_cosines,
        keep,
        weight_mode="euclidean",
        like=neighbor_distances,
    )
    assert torch.equal(first, torch.tensor([0, 0]))
    assert torch.equal(second, torch.tensor([1, 1]))
    assert torch.allclose(weights, torch.ones(2, dtype=torch.float64))

    unique_first, unique_second, unique_weights = _unique_undirected_edges(
        first,
        second,
        weights,
        num_points=2,
    )
    assert torch.equal(unique_first, torch.tensor([0]))
    assert torch.equal(unique_second, torch.tensor([1]))
    assert torch.allclose(unique_weights, torch.ones(1, dtype=torch.float64))

    adjacency = _sparse_adjacency_from_edges(
        unique_first,
        unique_second,
        unique_weights,
        2,
        neighbor_distances,
    )
    assert torch.allclose(adjacency.to_dense(), torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64))

    empty = _empty_adjacency(3, neighbor_distances)
    assert empty.shape == (3, 3)
    assert empty.coalesce().values().numel() == 0

    assert _effective_pairwise_element_budget(torch.device("cpu"), 5) == 5
    assert _effective_pairwise_element_budget(torch.device("cuda"), 5) >= 5


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


def test_surface_adjacency_prunes_local_diagonals() -> None:
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

    dense = _assert_sparse_symmetric(
        build_surface_adjacency(
            points,
            normals,
            neighbors=3,
            candidate_neighbors=3,
            max_tangent_component=0.1,
            prune_redundant_edges=True,
        ),
        points.shape[0],
    )
    unpruned = _assert_sparse_symmetric(
        build_surface_adjacency(
            points,
            normals,
            neighbors=3,
            candidate_neighbors=3,
            max_tangent_component=0.1,
            prune_redundant_edges=False,
        ),
        points.shape[0],
    )

    assert dense[0, 1] > 0
    assert dense[0, 2] > 0
    assert dense[1, 3] > 0
    assert dense[2, 3] > 0
    assert dense[0, 3] == 0
    assert dense[1, 2] == 0
    assert unpruned[0, 3] > 0
    assert unpruned[1, 2] > 0


def test_surface_adjacency_keeps_angular_ring_and_candidate_diagonals() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)

    dense = _assert_sparse_symmetric(
        build_surface_adjacency(
            points,
            normals,
            neighbors=4,
            candidate_neighbors=4,
            max_tangent_component=0.1,
        ),
        points.shape[0],
    )

    assert dense[0, 1] > 0
    assert dense[0, 2] > 0
    assert dense[0, 3] > 0
    assert dense[0, 4] > 0
    assert dense[1, 2] > 0
    assert dense[2, 3] > 0
    assert dense[3, 4] > 0
    assert dense[1, 4] > 0
    assert dense[1, 3] > 0
    assert dense[2, 4] > 0


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

    assert dense[0, 1] > 0
    assert dense[0, 2] > 0
    assert dense[1, 2] > 0


def test_surface_adjacency_keeps_pair_probe_links_with_weights() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.00, 0.04, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, 1, -1],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    support_mask = support_indices >= 0
    block_types = torch.tensor([2, 3, 3], dtype=torch.long)
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

    assert dense[0, 1] > 0
    assert dense[0, 2] > 0
    assert dense[1, 2] > 0


def test_surface_adjacency_keeps_same_support_edges_across_block_indices() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
        ],
        dtype=torch.long,
    )
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)
    block_types = torch.tensor([2, 2], dtype=torch.long)
    block_indices = torch.tensor([0, 1], dtype=torch.long)

    adjacency = build_surface_adjacency(
        points,
        normals,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        neighbors=1,
        candidate_neighbors=1,
        max_tangent_component=0.1,
    )
    dense = _assert_sparse_symmetric(adjacency, points.shape[0])

    assert dense[0, 1] > 0


def test_surface_adjacency_finds_same_support_edges_when_nearer_points_are_disallowed() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.20, 0.00, 0.0],
            [0.01, 0.00, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [2, -1],
        ],
        dtype=torch.long,
    )
    support_mask = support_indices >= 0
    block_types = torch.tensor([2, 2, 1], dtype=torch.long)
    block_indices = torch.tensor([0, 0, 0], dtype=torch.long)

    adjacency = build_surface_adjacency(
        points,
        normals,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        neighbors=1,
        candidate_neighbors=1,
        max_tangent_component=0.1,
    )
    dense = _assert_sparse_symmetric(adjacency, points.shape[0])

    assert dense[0, 1] > 0
    assert dense[0, 2] == 0


def test_surface_adjacency_uses_single_pool_when_adjacent_patch_is_nearer() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.20, 0.00, 0.0],
            [0.01, 0.00, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [0, -1],
        ],
        dtype=torch.long,
    )
    support_mask = support_indices >= 0
    block_types = torch.tensor([2, 2, 1], dtype=torch.long)
    block_indices = torch.tensor([0, 0, 0], dtype=torch.long)

    adjacency = build_surface_adjacency(
        points,
        normals,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        neighbors=1,
        candidate_neighbors=1,
        max_tangent_component=0.1,
    )
    dense = _assert_sparse_symmetric(adjacency, points.shape[0])

    assert dense[0, 1] == 0
    assert dense[0, 2] > 0


def test_surface_adjacency_can_allow_disjoint_single_support_edges() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor([[0], [1]], dtype=torch.long)
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)

    blocked = _assert_sparse_symmetric(
        build_surface_adjacency(
            points,
            normals,
            support_indices=support_indices,
            support_mask=support_mask,
            neighbors=1,
            candidate_neighbors=1,
            max_tangent_component=0.1,
        ),
        points.shape[0],
    )
    allowed = _assert_sparse_symmetric(
        build_surface_adjacency(
            points,
            normals,
            support_indices=support_indices,
            support_mask=support_mask,
            neighbors=1,
            candidate_neighbors=1,
            max_tangent_component=0.1,
            allow_disjoint_single_support_edges=True,
        ),
        points.shape[0],
    )

    assert blocked[0, 1] == 0
    assert allowed[0, 1] > 0


def test_surface_adjacency_single_support_unrestricted_matches_plain_knn() -> None:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.arange(points.shape[0], dtype=torch.long).unsqueeze(-1)
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)

    plain = build_surface_adjacency(
        points,
        normals,
        neighbors=2,
        candidate_neighbors=2,
    ).coalesce()
    unrestricted = build_surface_adjacency(
        points,
        normals,
        support_indices=support_indices,
        support_mask=support_mask,
        neighbors=2,
        candidate_neighbors=2,
        allow_disjoint_single_support_edges=True,
    ).coalesce()

    assert torch.equal(unrestricted.indices(), plain.indices())
    assert torch.allclose(unrestricted.values(), plain.values())


def test_surface_adjacency_keeps_atom_probe_links_with_weights() -> None:
    points = torch.tensor(
        [
            [0.00, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.00, 0.04, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).expand_as(points)
    support_indices = torch.tensor(
        [
            [0, -1, -1],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    support_mask = support_indices >= 0
    block_types = torch.tensor([1, 3, 3], dtype=torch.long)
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

    assert dense[0, 1] > 0
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
        ({"prune_redundant_edges": 1}, TypeError, "prune_redundant_edges"),
        (
            {"allow_disjoint_single_support_edges": 1},
            TypeError,
            "allow_disjoint_single_support_edges",
        ),
    ],
)
def test_surface_adjacency_rejects_invalid_options(kwargs, error, match) -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(error, match=match):
        build_surface_adjacency(points, normals, **kwargs)
