"""Surface graph construction for sampled SES point clouds."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch


AdjacencyWeightMode = Literal["euclidean", "geodesic"]

_DEFAULT_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET = 64_000_000


def build_surface_adjacency(
    points: torch.Tensor,
    normals: torch.Tensor,
    *,
    support_indices: Optional[torch.Tensor] = None,
    support_mask: Optional[torch.Tensor] = None,
    block_types: Optional[torch.Tensor] = None,
    block_indices: Optional[torch.Tensor] = None,
    weight_mode: AdjacencyWeightMode = "euclidean",
    neighbors: int = 6,
    candidate_neighbors: Optional[int] = None,
    min_normal_cosine: float = -0.25,
    max_tangent_component: float = 0.8,
    max_distance: Optional[float] = None,
    prune_redundant_edges: bool = False,
    allow_disjoint_single_support_edges: bool = False,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Build a sparse undirected adjacency matrix over SES surface samples.

    Edges are selected from a compact GPU-friendly candidate pool.  If SES
    block/support metadata is supplied, the pool is constrained up front to
    points on the same analytic patch or on adjacent patch families: atom-pair,
    atom-probe, and pair-probe.  There are no separate reserve passes; all
    edges compete in the same compact top-k candidate set before local surface
    tests.  Endpoint normals must be reasonably compatible, and the chord
    between points must lie close to both tangent planes.

    Args:
        points: SES point coordinates with shape ``(num_points, 3)``.
        normals: Outward SES normals aligned with ``points``.
        support_indices: Optional padded atom-support indices for each point.
            Padding values are ignored according to ``support_mask``.
        support_mask: Optional boolean validity mask for ``support_indices``.
        block_types: Optional analytic block-family tags for each point.
        block_indices: Optional source block row indices paired with
            ``block_types``.
        weight_mode: ``"euclidean"`` for chord lengths, or ``"geodesic"`` for a
            local normal-angle geodesic approximation.
        neighbors: Maximum number of outgoing surface neighbors kept per point
            before symmetrization.
        candidate_neighbors: Number of nearest-neighbor candidates tested per
            point in the selected adjacency metric.
        min_normal_cosine: Minimum dot product between endpoint normals.
        max_tangent_component: Maximum absolute projection of the chord direction
            onto either endpoint normal.
        max_distance: Optional hard distance cutoff in the selected adjacency
            metric for candidate edges.
        prune_redundant_edges: If true, balance kept neighbors across angular
            sectors in each point's local tangent plane.
        allow_disjoint_single_support_edges: If true, allow one-support points
            from different supports to connect when the geometric surface tests
            accept them. This is useful for samplers that only know each point's
            owner atom and do not emit explicit pair/probe support metadata.
        pairwise_element_budget: Approximate maximum temporary distance-matrix
            size.  GPU runs automatically use a larger floor.

    Returns:
        A coalesced sparse COO tensor with shape ``(num_points, num_points)``.
        The matrix is symmetric and has no diagonal entries.
    """

    points, normals = _prepare_graph_inputs(points, normals)
    num_points = int(points.shape[0])
    if num_points == 0:
        return _empty_adjacency(0, points)
    if num_points == 1:
        return _empty_adjacency(1, points)

    if weight_mode not in ("euclidean", "geodesic"):
        raise ValueError("weight_mode must be 'euclidean' or 'geodesic'")
    neighbors = _validate_positive_int("neighbors", neighbors)
    candidate_neighbors_was_default = candidate_neighbors is None
    if candidate_neighbors is not None:
        candidate_neighbors = _validate_positive_int(
            "candidate_neighbors",
            candidate_neighbors,
        )
        if candidate_neighbors < neighbors:
            raise ValueError("candidate_neighbors must be at least neighbors")
    if not -1.0 <= float(min_normal_cosine) <= 1.0:
        raise ValueError("min_normal_cosine must be in the interval [-1, 1]")
    if not 0.0 <= float(max_tangent_component) <= 1.0:
        raise ValueError("max_tangent_component must be in the interval [0, 1]")
    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be positive")
    if not isinstance(prune_redundant_edges, bool):
        raise TypeError("prune_redundant_edges must be a boolean")
    if not isinstance(allow_disjoint_single_support_edges, bool):
        raise TypeError("allow_disjoint_single_support_edges must be a boolean")
    if pairwise_element_budget <= 0:
        raise ValueError("pairwise_element_budget must be positive")

    support_indices, support_mask = _prepare_support_inputs(
        support_indices,
        support_mask,
        num_points,
        points.device,
    )
    block_types, block_indices = _prepare_block_inputs(
        block_types,
        block_indices,
        num_points,
        points.device,
    )

    has_multi_support_slots = (
        support_indices is not None
        and support_mask is not None
        and support_indices.shape[1] > 1
    )
    if candidate_neighbors_was_default:
        if has_multi_support_slots:
            candidate_neighbors = neighbors + 2
        else:
            candidate_neighbors = max(2 * neighbors, neighbors + 4)

    kept_neighbors = min(neighbors, num_points - 1)
    candidate_neighbors = min(candidate_neighbors, num_points - 1)
    (
        neighbor_indices,
        neighbor_distances,
        neighbor_euclidean_distances,
    ) = _nearest_topology_candidates(
        points,
        normals,
        candidate_neighbors,
        weight_mode=weight_mode,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        max_distance=max_distance,
        allow_disjoint_single_support_edges=allow_disjoint_single_support_edges,
        pairwise_element_budget=pairwise_element_budget,
    )

    row_indices = torch.arange(num_points, dtype=torch.long, device=points.device).view(
        -1,
        1,
    )
    deltas = points[neighbor_indices] - points.unsqueeze(1)
    eps = torch.finfo(points.dtype).eps
    safe_distances = neighbor_euclidean_distances.clamp_min(eps)
    directions = deltas / safe_distances.unsqueeze(-1)

    row_normals = normals.unsqueeze(1)
    col_normals = normals[neighbor_indices]
    normal_cosines = (row_normals * col_normals).sum(dim=-1).clamp(-1, 1)
    row_tangent_components = (directions * row_normals).sum(dim=-1).abs()
    col_tangent_components = (directions * col_normals).sum(dim=-1).abs()
    valid = (
        torch.isfinite(neighbor_distances)
        & torch.isfinite(neighbor_euclidean_distances)
        & (neighbor_euclidean_distances > eps)
        & (normal_cosines >= float(min_normal_cosine))
        & (row_tangent_components <= float(max_tangent_component))
        & (col_tangent_components <= float(max_tangent_component))
    )
    if max_distance is not None:
        valid = valid & (neighbor_distances <= float(max_distance))

    if prune_redundant_edges and kept_neighbors >= 3:
        keep = _select_angular_neighbors(
            normals,
            directions,
            neighbor_distances,
            valid,
            max_neighbors=kept_neighbors,
        )
    else:
        keep = _select_nearest_valid_candidates(
            neighbor_distances,
            valid,
            max_neighbors=kept_neighbors,
        )
    unique_first, unique_second, unique_weights = _unique_undirected_edges(
        *_candidate_edges_from_keep(
            row_indices,
            neighbor_indices,
            neighbor_euclidean_distances,
            normal_cosines,
            keep,
            weight_mode=weight_mode,
            like=points,
        ),
        num_points,
    )

    return _sparse_adjacency_from_edges(
        unique_first,
        unique_second,
        unique_weights,
        num_points,
        points,
    )


def _candidate_edges_from_keep(
    row_indices: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    normal_cosines: torch.Tensor,
    keep: torch.Tensor,
    *,
    weight_mode: AdjacencyWeightMode,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a per-row neighbor keep mask into undirected edge arrays.

    Args:
        row_indices: Column vector with source point indices, shape
            ``(num_points, 1)``.
        neighbor_indices: Candidate destination indices for each source point.
        neighbor_distances: Euclidean candidate distances aligned with
            ``neighbor_indices``.
        normal_cosines: Endpoint normal dot products aligned with
            ``neighbor_indices``.
        keep: Boolean mask selecting accepted candidate slots.
        weight_mode: Edge-weight metric used to convert distances to values.
        like: Tensor that supplies dtype and device for empty outputs.

    Returns:
        ``(first, second, weights)`` arrays for non-self undirected edges, where
        ``first <= second`` for every edge.
    """

    rows = row_indices.expand_as(neighbor_indices)[keep]
    cols = neighbor_indices[keep]
    first = torch.minimum(rows, cols)
    second = torch.maximum(rows, cols)
    non_self = first != second

    distances = neighbor_distances[keep][non_self]
    weights = _edge_weights(distances, normal_cosines[keep][non_self], weight_mode)
    return first[non_self], second[non_self], weights


def _sparse_adjacency_from_edges(
    first: torch.Tensor,
    second: torch.Tensor,
    weights: torch.Tensor,
    num_points: int,
    like: torch.Tensor,
) -> torch.Tensor:
    """Build a symmetric sparse adjacency matrix from undirected edge arrays.

    Args:
        first: First endpoint of each undirected edge.
        second: Second endpoint of each undirected edge.
        weights: Edge weights aligned with ``first`` and ``second``.
        num_points: Number of graph vertices.
        like: Tensor that supplies dtype and device for the sparse matrix.

    Returns:
        A coalesced sparse COO adjacency matrix with mirrored edge entries.
    """

    if weights.numel() == 0:
        return _empty_adjacency(num_points, like)
    indices = torch.cat(
        (
            torch.stack((first, second), dim=0),
            torch.stack((second, first), dim=0),
        ),
        dim=1,
    )
    values = torch.cat((weights, weights), dim=0)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_points, num_points),
        dtype=like.dtype,
        device=like.device,
    ).coalesce()


def _prepare_point_inputs(points: torch.Tensor) -> torch.Tensor:
    """Validate point coordinates and cast them to a floating dtype.

    Args:
        points: Candidate point coordinates with expected shape
            ``(num_points, 3)``.

    Returns:
        ``points`` as a floating-point tensor.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (num_points, 3)")
    point_dtype = points.dtype if points.is_floating_point() else torch.float32
    return points.to(dtype=point_dtype)


def _prepare_graph_inputs(
    points: torch.Tensor,
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate points/normals, align dtype/device, and normalize normals.

    Args:
        points: SES point coordinates with shape ``(num_points, 3)``.
        normals: Surface normals aligned with ``points``.

    Returns:
        ``(points, normals)`` with a common floating dtype on the point device.
    """

    points = _prepare_point_inputs(points)
    if normals.shape != points.shape:
        raise ValueError("normals must have the same shape as points")

    normal_dtype = normals.dtype if normals.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(points.dtype, normal_dtype)
    points = points.to(dtype=common_dtype)
    normals = normals.to(dtype=common_dtype, device=points.device)
    normals = _normalize_vectors(normals)
    return points, normals


def _prepare_support_inputs(
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    num_points: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate sparse support indices and masks.

    Args:
        support_indices: Optional padded atom-support indices, shape
            ``(num_points, max_support)``.
        support_mask: Optional validity mask aligned with ``support_indices``.
        num_points: Expected number of point rows.
        device: Device to which support tensors are moved.

    Returns:
        ``(support_indices, support_mask)`` on ``device``, or ``(None, None)``
        when no support metadata is supplied.
    """

    if support_indices is None:
        if support_mask is not None:
            raise ValueError("support_mask requires support_indices")
        return None, None
    if support_indices.ndim != 2 or support_indices.shape[0] != num_points:
        raise ValueError("support_indices must have shape (num_points, max_support)")
    support_indices = support_indices.to(device=device, dtype=torch.long)
    if support_mask is None:
        support_mask = support_indices >= 0
    elif support_mask.shape != support_indices.shape:
        raise ValueError("support_mask must have the same shape as support_indices")
    else:
        support_mask = support_mask.to(device=device, dtype=torch.bool)
    return support_indices, support_mask


def _prepare_block_inputs(
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
    num_points: int,
    device: torch.device,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Validate analytic block-family metadata.

    Args:
        block_types: Optional integer block-family tags per point.
        block_indices: Optional source block row ids per point.
        num_points: Expected number of point rows.
        device: Device to which block tensors are moved.

    Returns:
        ``(block_types, block_indices)`` on ``device``, or ``(None, None)`` when
        block metadata is not supplied.
    """

    if block_types is None and block_indices is None:
        return None, None
    if block_types is None or block_indices is None:
        raise ValueError("block_types and block_indices must be supplied together")
    if block_types.shape != (num_points,) or block_indices.shape != (num_points,):
        raise ValueError("block_types and block_indices must have shape (num_points,)")
    return (
        block_types.to(device=device, dtype=torch.long),
        block_indices.to(device=device, dtype=torch.long),
    )


def _disjoint_single_support_relationships(
    row_indices: torch.Tensor,
    row_mask: torch.Tensor,
    col_indices: torch.Tensor,
    col_mask: torch.Tensor,
) -> torch.Tensor:
    """Detect pairs of one-support points with different owner atoms.

    Args:
        row_indices: Row-side padded support indices.
        row_mask: Row-side validity mask.
        col_indices: Column-side padded support indices.
        col_mask: Column-side validity mask.

    Returns:
        Boolean tensor whose entries are true for disjoint one-support pairs.
    """

    row_counts = row_mask.sum(dim=-1)
    col_counts = col_mask.sum(dim=-1)
    row_values = torch.where(row_mask, row_indices, torch.full_like(row_indices, -1))
    col_values = torch.where(col_mask, col_indices, torch.full_like(col_indices, -1))
    return (
        (row_counts == 1)
        & (col_counts == 1)
        & (row_values.max(dim=-1).values != col_values.max(dim=-1).values)
    )


def _support_relationships_for_pairs(
    row_indices: torch.Tensor,
    row_mask: torch.Tensor,
    col_indices: torch.Tensor,
    col_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify support-set relationships for broadcasted point pairs.

    Args:
        row_indices: Row-side padded support indices.
        row_mask: Row-side validity mask.
        col_indices: Column-side padded support indices.
        col_mask: Column-side validity mask.

    Returns:
        ``(same_support, adjacent_support, boundary_support)`` boolean tensors.
    """

    row_counts = row_mask.sum(dim=-1)
    col_counts = col_mask.sum(dim=-1)

    row_counts, col_counts = torch.broadcast_tensors(row_counts, col_counts)
    row_subset_col = torch.ones_like(row_counts, dtype=torch.bool)
    for slot in range(int(row_indices.shape[-1])):
        row_present = row_mask[..., slot]
        row_value = row_indices[..., slot]
        row_in_col = (
            (col_indices == row_value.unsqueeze(-1)) & col_mask
        ).any(dim=-1)
        row_subset_col = row_subset_col & (~row_present | row_in_col)

    col_subset_row = torch.ones_like(col_counts, dtype=torch.bool)
    for slot in range(int(col_indices.shape[-1])):
        col_present = col_mask[..., slot]
        col_value = col_indices[..., slot]
        col_in_row = (
            (row_indices == col_value.unsqueeze(-1)) & row_mask
        ).any(dim=-1)
        col_subset_row = col_subset_row & (~col_present | col_in_row)
    same_support = row_subset_col & col_subset_row & (row_counts == col_counts)

    adjacent_support = (
        row_subset_col & (col_counts > row_counts)
    ) | (
        col_subset_row & (row_counts > col_counts)
    )
    boundary_support = torch.ones_like(adjacent_support)
    return same_support, adjacent_support, boundary_support


def dense_features_to_supports(
    atom_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense binary atom features to padded sparse support metadata.

    Args:
        atom_features: Dense point-atom feature matrix with shape
            ``(num_points, num_atoms)``. Positive values mark active supports.

    Returns:
        ``(support_indices, support_mask)`` where each row contains active atom
        indices padded with ``-1`` and a matching boolean validity mask.
    """

    if atom_features.ndim != 2:
        raise ValueError("atom_features must have shape (num_points, num_atoms)")
    active = atom_features > 0
    counts = active.sum(dim=1)
    max_support = int(counts.max().item()) if counts.numel() > 0 else 0
    support_indices = torch.full(
        (atom_features.shape[0], max_support),
        -1,
        dtype=torch.long,
        device=atom_features.device,
    )
    support_mask = torch.zeros(
        (atom_features.shape[0], max_support),
        dtype=torch.bool,
        device=atom_features.device,
    )
    if max_support == 0:
        return support_indices, support_mask

    point_rows, atom_cols = active.nonzero(as_tuple=True)
    if point_rows.numel() == 0:
        return support_indices, support_mask
    row_offsets = torch.cumsum(counts, dim=0) - counts
    slot_ids = torch.arange(
        point_rows.numel(),
        dtype=torch.long,
        device=atom_features.device,
    ) - row_offsets[point_rows]
    support_indices[point_rows, slot_ids] = atom_cols
    support_mask[point_rows, slot_ids] = True
    return support_indices, support_mask


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize vectors with a deterministic fallback for zero rows.

    Args:
        vectors: Tensor whose last dimension stores vector components.

    Returns:
        Unit vectors with zero-length rows replaced by the positive x-axis.
    """

    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    eps = torch.finfo(vectors.dtype).eps
    fallback = torch.zeros_like(vectors)
    fallback[..., 0] = 1
    return torch.where(norms > eps, vectors / norms.clamp_min(eps), fallback)


def _tangent_bases(normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct orthonormal tangent bases for normalized surface normals.

    Args:
        normals: Unit normal vectors with shape ``(..., 3)``.

    Returns:
        ``(basis_u, basis_v)`` tangent vectors orthogonal to ``normals``.
    """

    x_axis = torch.zeros_like(normals)
    x_axis[..., 0] = 1
    y_axis = torch.zeros_like(normals)
    y_axis[..., 1] = 1
    reference = torch.where(normals[..., :1].abs() < 0.9, x_axis, y_axis)
    basis_u = _normalize_vectors(torch.cross(normals, reference, dim=-1))
    basis_v = _normalize_vectors(torch.cross(normals, basis_u, dim=-1))
    return basis_u, basis_v


def _validate_positive_int(name: str, value: int) -> int:
    """Validate an integer option that must be strictly positive.

    Args:
        name: Option name used in error messages.
        value: Candidate integer value.

    Returns:
        ``value`` converted to a plain ``int``.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nearest_neighbor_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    max_distance: Optional[float],
    pairwise_element_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select nearest candidates without topology metadata.

    Args:
        points: Point coordinates with shape ``(num_points, 3)``.
        normals: Unit normals aligned with ``points``.
        neighbors: Number of candidates returned per point.
        weight_mode: Metric used to rank candidates.
        max_distance: Optional distance cutoff in the selected metric.
        pairwise_element_budget: Approximate maximum temporary distance-matrix
            size per block.

    Returns:
        ``(indices, metric_distances, euclidean_distances)`` candidate tensors.
    """

    num_points = int(points.shape[0])
    effective_budget = _effective_pairwise_element_budget(
        points.device,
        pairwise_element_budget,
    )
    rows_per_block = max(1, min(num_points, effective_budget // num_points))
    point_sq_norms = points.square().sum(dim=-1).view(1, -1)
    all_indices = []
    all_distances = []
    all_euclidean_distances = []

    for start in range(0, num_points, rows_per_block):
        stop = min(start + rows_per_block, num_points)
        block = points[start:stop]
        sq_distances = (
            block.square().sum(dim=-1, keepdim=True)
            + point_sq_norms
            - 2 * (block @ points.transpose(0, 1))
        ).clamp_min(0)
        block_rows = torch.arange(stop - start, device=points.device)
        self_cols = torch.arange(start, stop, device=points.device)
        sq_distances[block_rows, self_cols] = float("inf")
        sq_distances = torch.where(
            torch.isfinite(sq_distances),
            sq_distances,
            torch.full_like(sq_distances, float("inf")),
        )
        euclidean_distances = torch.sqrt(sq_distances)
        metric_distances = _pairwise_metric_distances(
            euclidean_distances,
            normals[start:stop],
            normals,
            weight_mode,
        )
        valid = torch.isfinite(metric_distances)
        if max_distance is not None:
            valid = valid & (metric_distances <= float(max_distance))
        block_indices, block_distances, block_euclidean_distances = _masked_topk_candidates(
            neighbors,
            metric_distances,
            euclidean_distances,
            valid,
        )
        all_indices.append(block_indices)
        all_distances.append(block_distances)
        all_euclidean_distances.append(block_euclidean_distances)

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_distances, dim=0),
        torch.cat(all_euclidean_distances, dim=0),
    )


def _nearest_topology_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
    max_distance: Optional[float],
    allow_disjoint_single_support_edges: bool,
    pairwise_element_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select nearest candidates after applying topology masks.

    Args:
        points: Point coordinates with shape ``(num_points, 3)``.
        normals: Unit normals aligned with ``points``.
        neighbors: Number of candidates returned per point.
        weight_mode: Metric used to rank candidates.
        support_indices: Optional padded atom-support indices per point.
        support_mask: Optional support validity mask.
        block_types: Optional analytic block-family tags per point.
        block_indices: Optional source block row ids per point.
        max_distance: Optional distance cutoff in the selected metric.
        allow_disjoint_single_support_edges: Whether one-support points from
            different supports may be considered topologically adjacent.
        pairwise_element_budget: Approximate maximum temporary distance-matrix
            size per block.

    Returns:
        ``(indices, metric_distances, euclidean_distances)`` candidate tensors.
    """

    if support_indices is None and block_types is None:
        return _nearest_neighbor_candidates(
            points,
            normals,
            neighbors,
            weight_mode=weight_mode,
            max_distance=max_distance,
            pairwise_element_budget=pairwise_element_budget,
        )

    num_points = int(points.shape[0])
    effective_budget = _effective_pairwise_element_budget(
        points.device,
        pairwise_element_budget,
    )
    topology_element_multiplier = 1
    if support_indices is not None and support_mask is not None:
        topology_element_multiplier += max(1, int(support_indices.shape[1]))
    if block_types is not None and block_indices is not None:
        topology_element_multiplier += 1
    rows_per_block = max(
        1,
        min(
            num_points,
            effective_budget // max(1, num_points * topology_element_multiplier),
        ),
    )
    point_sq_norms = points.square().sum(dim=-1).view(1, -1)
    all_indices = []
    all_distances = []
    all_euclidean_distances = []

    for start in range(0, num_points, rows_per_block):
        stop = min(start + rows_per_block, num_points)
        block = points[start:stop]
        sq_distances = (
            block.square().sum(dim=-1, keepdim=True)
            + point_sq_norms
            - 2 * (block @ points.transpose(0, 1))
        ).clamp_min(0)
        block_rows = torch.arange(start, stop, dtype=torch.long, device=points.device)
        self_rows = torch.arange(stop - start, device=points.device)
        sq_distances[self_rows, block_rows] = float("inf")
        topology_allowed = _topology_allowed_matrix(
            block_rows,
            num_points,
            support_indices=support_indices,
            support_mask=support_mask,
            block_types=block_types,
            block_indices=block_indices,
            allow_disjoint_single_support_edges=allow_disjoint_single_support_edges,
        )
        finite = torch.isfinite(sq_distances)
        sq_distances = torch.where(
            finite,
            sq_distances,
            torch.full_like(sq_distances, float("inf")),
        )
        euclidean_distances = torch.sqrt(sq_distances)
        metric_distances = _pairwise_metric_distances(
            euclidean_distances,
            normals[start:stop],
            normals,
            weight_mode,
        )
        valid_metric = torch.isfinite(metric_distances)
        if max_distance is not None:
            valid_metric = valid_metric & (metric_distances <= float(max_distance))

        block_indices_out, block_distances, block_euclidean = _masked_topk_candidates(
            neighbors,
            metric_distances,
            euclidean_distances,
            topology_allowed & valid_metric,
        )
        all_indices.append(block_indices_out)
        all_distances.append(block_distances)
        all_euclidean_distances.append(block_euclidean)

    return (
        torch.cat(all_indices, dim=0),
        torch.cat(all_distances, dim=0),
        torch.cat(all_euclidean_distances, dim=0),
    )


def _masked_topk_candidates(
    neighbors: int,
    metric_distances: torch.Tensor,
    euclidean_distances: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest valid candidates from a dense pairwise block.

    Args:
        neighbors: Number of candidates requested per row.
        metric_distances: Distances used for ranking.
        euclidean_distances: Euclidean distances gathered for final weights and
            tangent tests.
        valid: Boolean mask selecting candidate entries that may be ranked.

    Returns:
        ``(indices, metric_distances, euclidean_distances)`` for the selected
        top-k slots.
    """

    if neighbors <= 0:
        rows = int(metric_distances.shape[0])
        empty_indices = torch.empty(
            (rows, 0),
            dtype=torch.long,
            device=metric_distances.device,
        )
        empty_distances = torch.empty(
            (rows, 0),
            dtype=metric_distances.dtype,
            device=metric_distances.device,
        )
        return empty_indices, empty_distances, empty_distances

    masked_distances = torch.where(
        valid,
        metric_distances,
        torch.full_like(metric_distances, float("inf")),
    )
    block_distances, block_indices = torch.topk(
        masked_distances,
        min(int(neighbors), int(metric_distances.shape[1])),
        dim=1,
        largest=False,
        sorted=True,
    )
    block_euclidean_distances = torch.gather(euclidean_distances, 1, block_indices)
    return block_indices, block_distances, block_euclidean_distances


def _topology_allowed_matrix(
    rows: torch.Tensor,
    num_points: int,
    *,
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
    allow_disjoint_single_support_edges: bool,
) -> torch.Tensor:
    """Build a dense topology mask for a block of source rows.

    Args:
        rows: Source point indices represented by the current pairwise block.
        num_points: Total number of points in the graph.
        support_indices: Optional padded atom-support indices per point.
        support_mask: Optional support validity mask.
        block_types: Optional analytic block-family tags per point.
        block_indices: Optional source block row ids per point.
        allow_disjoint_single_support_edges: Whether disjoint one-support pairs
            should be allowed.

    Returns:
        Boolean matrix with shape ``(len(rows), num_points)``.
    """

    if block_types is not None and block_indices is not None:
        allowed = (
            block_types[rows].unsqueeze(1) == block_types.unsqueeze(0)
        ) & (
            block_indices[rows].unsqueeze(1) == block_indices.unsqueeze(0)
        )
    else:
        allowed = torch.ones(
            (int(rows.numel()), int(num_points)),
            dtype=torch.bool,
            device=rows.device,
        )

    if support_indices is None or support_mask is None:
        return allowed

    same_support, adjacent_support, boundary_support = _support_relationships_for_pairs(
        support_indices[rows].unsqueeze(1),
        support_mask[rows].unsqueeze(1),
        support_indices.unsqueeze(0),
        support_mask.unsqueeze(0),
    )
    disjoint_single_support = (
        _disjoint_single_support_relationships(
            support_indices[rows].unsqueeze(1),
            support_mask[rows].unsqueeze(1),
            support_indices.unsqueeze(0),
            support_mask.unsqueeze(0),
        )
        if allow_disjoint_single_support_edges
        else torch.zeros_like(same_support)
    )
    support_allowed = (
        same_support
        | (adjacent_support & boundary_support)
        | disjoint_single_support
    )
    if block_types is not None and block_indices is not None:
        return allowed | support_allowed
    return support_allowed


def _effective_pairwise_element_budget(
    device: torch.device,
    pairwise_element_budget: int,
) -> int:
    """Choose the effective pairwise block budget for a device.

    Args:
        device: Device on which pairwise tensors are allocated.
        pairwise_element_budget: Caller-provided element budget.

    Returns:
        The budget, raised to the GPU floor for CUDA devices.
    """

    if torch.device(device).type == "cuda":
        return max(int(pairwise_element_budget), _DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET)
    return int(pairwise_element_budget)


def _edge_weights(
    distances: torch.Tensor,
    normal_cosines: torch.Tensor,
    weight_mode: AdjacencyWeightMode,
) -> torch.Tensor:
    """Convert chord distances to edge weights.

    Args:
        distances: Euclidean chord distances.
        normal_cosines: Dot products between endpoint normals.
        weight_mode: ``"euclidean"`` for chord distances or ``"geodesic"`` for
            a local normal-angle approximation.

    Returns:
        Edge weights with the same shape as ``distances``.
    """

    if weight_mode == "euclidean":
        return distances

    angles = torch.acos(normal_cosines.clamp(-1, 1))
    half_angles = 0.5 * angles
    denominators = 2 * torch.sin(half_angles)
    small = angles <= math.sqrt(torch.finfo(distances.dtype).eps)
    scale = torch.where(
        small,
        torch.ones_like(angles),
        angles / denominators.clamp_min(torch.finfo(distances.dtype).eps),
    )
    return distances * scale


def _pairwise_metric_distances(
    euclidean_distances: torch.Tensor,
    row_normals: torch.Tensor,
    col_normals: torch.Tensor,
    weight_mode: AdjacencyWeightMode,
) -> torch.Tensor:
    """Compute ranking distances for a pairwise row/column block.

    Args:
        euclidean_distances: Euclidean pairwise distances.
        row_normals: Normals for block rows.
        col_normals: Normals for all candidate columns.
        weight_mode: Ranking metric.

    Returns:
        Distances in the requested metric.
    """

    if weight_mode == "euclidean":
        return euclidean_distances
    normal_cosines = (
        row_normals.unsqueeze(1) * col_normals.unsqueeze(0)
    ).sum(dim=-1).clamp(-1, 1)
    return _edge_weights(euclidean_distances, normal_cosines, weight_mode)


def _select_angular_neighbors(
    normals: torch.Tensor,
    directions: torch.Tensor,
    neighbor_distances: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_neighbors: int,
) -> torch.Tensor:
    """Keep nearest valid neighbors across tangent-plane angular sectors.

    Args:
        normals: Unit point normals with shape ``(num_points, 3)``.
        directions: Unit chord directions for candidate edges.
        neighbor_distances: Candidate metric distances.
        valid: Boolean mask for geometrically valid candidates.
        max_neighbors: Maximum selected candidates per point.

    Returns:
        Boolean keep mask aligned with ``neighbor_distances``.
    """

    num_points, num_neighbors = neighbor_distances.shape
    if num_points == 0 or num_neighbors == 0:
        return valid
    if max_neighbors < 3:
        return _select_nearest_valid_candidates(
            neighbor_distances,
            valid,
            max_neighbors=max_neighbors,
        )

    sector_count = min(int(max_neighbors), 6)
    basis_u, basis_v = _tangent_bases(normals)
    tangent_directions = directions - (
        directions * normals.unsqueeze(1)
    ).sum(dim=-1, keepdim=True) * normals.unsqueeze(1)
    tangent_directions = _normalize_vectors(tangent_directions)
    angles = torch.atan2(
        (tangent_directions * basis_v.unsqueeze(1)).sum(dim=-1),
        (tangent_directions * basis_u.unsqueeze(1)).sum(dim=-1),
    )
    normalized_angles = torch.remainder(angles + math.pi, 2 * math.pi)
    sectors = torch.div(
        normalized_angles * sector_count,
        2 * math.pi,
        rounding_mode="floor",
    ).to(dtype=torch.long).clamp_max(sector_count - 1)

    selected = torch.zeros_like(valid)
    for sector in range(sector_count):
        sector_valid = valid & (sectors == sector)
        selected = selected | _select_nearest_valid_candidates(
            neighbor_distances,
            sector_valid,
            max_neighbors=1,
        )

    return selected


def _select_nearest_valid_candidates(
    neighbor_distances: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_neighbors: int,
) -> torch.Tensor:
    """Keep the nearest valid candidates in each row.

    Args:
        neighbor_distances: Candidate ranking distances.
        valid: Boolean mask for candidates that may be kept.
        max_neighbors: Maximum selected candidates per row.

    Returns:
        Boolean keep mask aligned with ``neighbor_distances``.
    """

    if max_neighbors <= 0 or neighbor_distances.shape[1] == 0:
        return torch.zeros_like(valid)

    rank_distances = torch.where(
        valid,
        neighbor_distances,
        torch.full_like(neighbor_distances, float("inf")),
    )
    selected_distances, selected_positions = torch.topk(
        rank_distances,
        min(int(max_neighbors), int(neighbor_distances.shape[1])),
        dim=1,
        largest=False,
        sorted=True,
    )
    selected = torch.zeros_like(valid)
    selected.scatter_(1, selected_positions, torch.isfinite(selected_distances))
    return selected & valid


def _unique_undirected_edges(
    first: torch.Tensor,
    second: torch.Tensor,
    weights: torch.Tensor,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deduplicate undirected edge arrays by endpoint key.

    Args:
        first: First endpoint for each edge, expected to be ``<= second``.
        second: Second endpoint for each edge.
        weights: Edge weights aligned with endpoints.
        num_points: Number of graph vertices used to build unique keys.

    Returns:
        Deduplicated ``(first, second, weights)`` arrays.
    """

    keys = first * int(num_points) + second
    order = torch.argsort(keys)
    ordered_keys = keys[order]
    unique = torch.ones_like(ordered_keys, dtype=torch.bool)
    unique[1:] = ordered_keys[1:] != ordered_keys[:-1]
    selected = order[unique]
    return first[selected], second[selected], weights[selected]


def _empty_adjacency(num_points: int, like: torch.Tensor) -> torch.Tensor:
    """Create an empty sparse adjacency matrix.

    Args:
        num_points: Number of graph vertices.
        like: Tensor that supplies dtype and device for the matrix values.

    Returns:
        Coalesced sparse COO tensor with no stored edges.
    """

    indices = torch.empty((2, 0), dtype=torch.long, device=like.device)
    values = torch.empty((0,), dtype=like.dtype, device=like.device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_points, num_points),
        dtype=like.dtype,
        device=like.device,
    ).coalesce()


__all__ = [
    "build_surface_adjacency",
    "dense_features_to_supports",
]
