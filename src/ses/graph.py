"""Surface graph construction for sampled SES point clouds."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch


AdjacencyWeightMode = Literal["euclidean", "geodesic"]

_DEFAULT_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET = 64_000_000
_FAST_TOPOLOGY_MIN_POINTS = 4096
_FAST_TOPOLOGY_MIN_CANDIDATES = 64
_FAST_TOPOLOGY_CANDIDATE_MULTIPLIER = 8
_WINDOWED_TOPOLOGY_MIN_POINTS = 50_000
_WINDOWED_TOPOLOGY_MIN_RADIUS = 8
_WINDOWED_TOPOLOGY_GPU_MIN_POINTS = 4096
_WINDOWED_TOPOLOGY_MAX_SUPPORT = 3
_WINDOWED_NEIGHBOR_GPU_MIN_POINTS = 8192
_WINDOWED_NEIGHBOR_MIN_RADIUS = 64
_WINDOWED_NEIGHBOR_CANDIDATE_MULTIPLIER = 8


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
    with torch.no_grad():
        selection_points = points.detach()
        selection_normals = normals.detach()
        (
            neighbor_indices,
            neighbor_distances,
            neighbor_euclidean_distances,
        ) = _nearest_topology_candidates(
            selection_points,
            selection_normals,
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

        row_indices = torch.arange(
            num_points,
            dtype=torch.long,
            device=points.device,
        ).view(-1, 1)
        deltas = selection_points[neighbor_indices] - selection_points.unsqueeze(1)
        eps = torch.finfo(selection_points.dtype).eps
        safe_distances = neighbor_euclidean_distances.clamp_min(eps)
        directions = deltas / safe_distances.unsqueeze(-1)

        row_normals = selection_normals.unsqueeze(1)
        col_normals = selection_normals[neighbor_indices]
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
                selection_normals,
                directions,
                neighbor_distances,
                valid,
                max_neighbors=kept_neighbors,
            )
            edge_arrays = _candidate_edges_from_keep(
                row_indices,
                neighbor_indices,
                neighbor_euclidean_distances,
                normal_cosines,
                keep,
                weight_mode=weight_mode,
                like=points,
            )
        else:
            selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
                neighbor_distances,
                valid,
                max_neighbors=kept_neighbors,
            )
            edge_arrays = _candidate_edges_from_positions(
                row_indices,
                neighbor_indices,
                neighbor_euclidean_distances,
                normal_cosines,
                selected_positions,
                selected_valid,
                weight_mode=weight_mode,
            )
        unique_first, unique_second, _ = _unique_undirected_edges(
            *edge_arrays,
            num_points,
        )
    unique_weights = _edge_weights_for_edges(
        points,
        normals,
        unique_first,
        unique_second,
        weight_mode,
    )

    return _sparse_adjacency_from_edges(
        unique_first,
        unique_second,
        unique_weights,
        num_points,
        points,
    )


def _edge_weights_for_edges(
    points: torch.Tensor,
    normals: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    weight_mode: AdjacencyWeightMode,
) -> torch.Tensor:
    """Compute differentiable weights for already selected graph edges.

    Args:
        points: SES point coordinates, shape ``(num_points, 3)``.
        normals: Unit normals aligned with ``points``.
        first: First endpoint indices for undirected edges.
        second: Second endpoint indices for undirected edges.
        weight_mode: Edge-weight metric, either ``"euclidean"`` or
            ``"geodesic"``.

    Returns:
        Edge weights with gradients flowing only through ``points`` and
        ``normals`` at the selected endpoints.
    """

    if first.numel() == 0:
        return torch.empty((0,), dtype=points.dtype, device=points.device)
    deltas = points[second] - points[first]
    distances = torch.linalg.norm(deltas, dim=-1)
    normal_cosines = (normals[first] * normals[second]).sum(dim=-1).clamp(-1, 1)
    return _edge_weights(distances, normal_cosines, weight_mode)


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


def _candidate_edges_from_positions(
    row_indices: torch.Tensor,
    neighbor_indices: torch.Tensor,
    neighbor_distances: torch.Tensor,
    normal_cosines: torch.Tensor,
    selected_positions: torch.Tensor,
    selected_valid: torch.Tensor,
    *,
    weight_mode: AdjacencyWeightMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert selected per-row candidate positions into undirected edges."""

    if selected_positions.numel() == 0:
        empty_indices = torch.empty((0,), dtype=torch.long, device=neighbor_indices.device)
        empty_weights = torch.empty(
            (0,),
            dtype=neighbor_distances.dtype,
            device=neighbor_distances.device,
        )
        return empty_indices, empty_indices, empty_weights

    rows = row_indices.expand_as(neighbor_indices)
    selected_rows = torch.gather(rows, 1, selected_positions)
    selected_cols = torch.gather(neighbor_indices, 1, selected_positions)
    selected_distances = torch.gather(neighbor_distances, 1, selected_positions)
    selected_cosines = torch.gather(normal_cosines, 1, selected_positions)

    first = torch.minimum(selected_rows, selected_cols).reshape(-1)
    second = torch.maximum(selected_rows, selected_cols).reshape(-1)
    valid_edges = selected_valid.reshape(-1) & (first != second)
    distances = selected_distances.reshape(-1)
    cosines = selected_cosines.reshape(-1)
    weights = _edge_weights(distances[valid_edges], cosines[valid_edges], weight_mode)
    return first[valid_edges], second[valid_edges], weights


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
    flat_offsets = point_rows * int(max_support) + slot_ids
    support_indices = support_indices.reshape(-1).scatter(
        0,
        flat_offsets,
        atom_cols,
    ).view_as(support_indices)
    support_mask = support_mask.reshape(-1).scatter(
        0,
        flat_offsets,
        torch.ones_like(flat_offsets, dtype=torch.bool),
    ).view_as(support_mask)
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
    if _should_use_windowed_neighbor_candidates(points, num_points):
        return _nearest_windowed_neighbor_candidates(
            points,
            normals,
            neighbors,
            weight_mode=weight_mode,
            max_distance=max_distance,
        )

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
        self_cols = torch.arange(num_points, dtype=torch.long, device=points.device)
        block_rows = torch.arange(start, stop, dtype=torch.long, device=points.device)
        sq_distances = sq_distances.masked_fill(
            self_cols.view(1, -1) == block_rows.view(-1, 1),
            float("inf"),
        )
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


def _nearest_windowed_neighbor_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    max_distance: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select local spatial candidates without materializing all point pairs."""

    num_points = int(points.shape[0])
    if num_points <= 1 or neighbors <= 0:
        return _empty_candidate_tensors(num_points, 0, points)

    radius = max(
        _WINDOWED_NEIGHBOR_MIN_RADIUS,
        int(neighbors) * _WINDOWED_NEIGHBOR_CANDIDATE_MULTIPLIER,
    )
    offsets = torch.cat(
        (
            torch.arange(-radius, 0, dtype=torch.long, device=points.device),
            torch.arange(1, radius + 1, dtype=torch.long, device=points.device),
        ),
        dim=0,
    )
    order = _spatial_order(points)
    inverse_order = torch.argsort(order)
    sorted_points = points[order]
    sorted_normals = normals[order]

    sorted_rows = torch.arange(num_points, dtype=torch.long, device=points.device)
    raw_neighbor_positions = sorted_rows.view(-1, 1) + offsets.view(1, -1)
    in_bounds = (raw_neighbor_positions >= 0) & (raw_neighbor_positions < num_points)
    neighbor_positions = raw_neighbor_positions.clamp(0, num_points - 1)
    neighbor_indices = order[neighbor_positions]
    non_self = neighbor_indices != order.view(-1, 1)

    neighbor_points = points[neighbor_indices]
    deltas = neighbor_points - sorted_points.unsqueeze(1)
    euclidean_distances = torch.linalg.norm(deltas, dim=-1)
    if weight_mode == "geodesic":
        normal_cosines = (
            sorted_normals.unsqueeze(1) * normals[neighbor_indices]
        ).sum(dim=-1).clamp(-1, 1)
        metric_distances = _edge_weights(
            euclidean_distances,
            normal_cosines,
            weight_mode,
        )
    else:
        metric_distances = euclidean_distances

    valid = (
        in_bounds
        & non_self
        & torch.isfinite(metric_distances)
        & torch.isfinite(euclidean_distances)
    )
    if max_distance is not None:
        valid = valid & (metric_distances <= float(max_distance))

    selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
        metric_distances,
        valid,
        max_neighbors=min(int(neighbors), int(metric_distances.shape[1])),
    )
    selected_indices = torch.gather(neighbor_indices, 1, selected_positions)
    selected_distances = torch.gather(metric_distances, 1, selected_positions)
    selected_euclidean = torch.gather(euclidean_distances, 1, selected_positions)
    selected_distances = torch.where(
        selected_valid,
        selected_distances,
        torch.full_like(selected_distances, float("inf")),
    )
    selected_euclidean = torch.where(
        selected_valid,
        selected_euclidean,
        torch.full_like(selected_euclidean, float("inf")),
    )
    return (
        selected_indices[inverse_order],
        selected_distances[inverse_order],
        selected_euclidean[inverse_order],
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
    if _single_support_topology_is_unrestricted(
        support_indices,
        support_mask,
        block_types,
        block_indices,
        allow_disjoint_single_support_edges,
    ):
        return _nearest_neighbor_candidates(
            points,
            normals,
            neighbors,
            weight_mode=weight_mode,
            max_distance=max_distance,
            pairwise_element_budget=pairwise_element_budget,
        )

    if (
        _should_use_windowed_topology_candidates(points, num_points)
        and not allow_disjoint_single_support_edges
        and block_types is not None
        and block_indices is not None
        and support_indices is not None
        and support_mask is not None
        and support_indices.shape[1] <= _WINDOWED_TOPOLOGY_MAX_SUPPORT
    ):
        return _nearest_windowed_topology_candidates(
            points,
            normals,
            neighbors,
            weight_mode=weight_mode,
            support_indices=support_indices,
            support_mask=support_mask,
            block_types=block_types,
            block_indices=block_indices,
            max_distance=max_distance,
        )

    effective_budget = _effective_pairwise_element_budget(
        points.device,
        pairwise_element_budget,
    )
    if _should_prefilter_topology_candidates(num_points):
        return _nearest_prefiltered_topology_candidates(
            points,
            normals,
            neighbors,
            weight_mode=weight_mode,
            support_indices=support_indices,
            support_mask=support_mask,
            block_types=block_types,
            block_indices=block_indices,
            max_distance=max_distance,
            allow_disjoint_single_support_edges=allow_disjoint_single_support_edges,
            pairwise_element_budget=pairwise_element_budget,
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
        self_cols = torch.arange(num_points, dtype=torch.long, device=points.device)
        sq_distances = sq_distances.masked_fill(
            self_cols.view(1, -1) == block_rows.view(-1, 1),
            float("inf"),
        )
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


def _nearest_windowed_topology_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    block_types: torch.Tensor,
    block_indices: torch.Tensor,
    max_distance: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select topology-local candidates with fully tensorized GPU windows."""

    block_group_ids = _metadata_group_ids(
        torch.stack((block_types, block_indices), dim=-1),
    )
    block_candidates = _nearest_windowed_group_candidates(
        points,
        normals,
        block_group_ids,
        neighbors,
        weight_mode=weight_mode,
        max_distance=max_distance,
    )

    if support_indices.shape[1] == 0:
        return block_candidates

    support_candidates = _nearest_windowed_support_candidates(
        points,
        normals,
        support_indices,
        support_mask,
        block_types,
        block_indices,
        neighbors,
        weight_mode=weight_mode,
        max_distance=max_distance,
    )

    combined_indices = torch.cat((block_candidates[0], support_candidates[0]), dim=1)
    combined_distances = torch.cat((block_candidates[1], support_candidates[1]), dim=1)
    combined_euclidean = torch.cat((block_candidates[2], support_candidates[2]), dim=1)
    return _select_candidate_tensors(
        combined_indices,
        combined_distances,
        combined_euclidean,
        neighbors,
    )


def _metadata_group_ids(metadata: torch.Tensor) -> torch.Tensor:
    _, inverse = torch.unique(metadata.to(dtype=torch.long), dim=0, return_inverse=True)
    return inverse.to(dtype=torch.long)


def _spatial_group_order(points: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    spatial_order = _spatial_order(points)
    group_order = torch.argsort(group_ids[spatial_order], stable=True)
    return spatial_order[group_order]


def _spatial_order(points: torch.Tensor) -> torch.Tensor:
    mins = points.min(dim=0).values
    spans = (points.max(dim=0).values - mins).clamp_min(torch.finfo(points.dtype).eps)
    scaled = ((points - mins.view(1, 3)) / spans.view(1, 3)).clamp(0, 1)
    quantized = torch.floor(scaled * 1023).to(dtype=torch.long).clamp(0, 1023)
    spatial_keys = (quantized[:, 0] * 1024 + quantized[:, 1]) * 1024 + quantized[:, 2]
    return torch.argsort(spatial_keys, stable=True)


def _nearest_windowed_group_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    group_ids: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    max_distance: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = int(points.shape[0])
    if num_points <= 1 or neighbors <= 0:
        return _empty_candidate_tensors(num_points, 0, points)

    radius = max(_WINDOWED_TOPOLOGY_MIN_RADIUS, int(neighbors) * 2)
    offsets = torch.cat(
        (
            torch.arange(-radius, 0, dtype=torch.long, device=points.device),
            torch.arange(1, radius + 1, dtype=torch.long, device=points.device),
        ),
        dim=0,
    )
    order = _spatial_group_order(points, group_ids)
    inverse_order = torch.argsort(order)
    sorted_group_ids = group_ids[order]
    sorted_points = points[order]
    sorted_normals = normals[order]

    sorted_rows = torch.arange(num_points, dtype=torch.long, device=points.device)
    raw_neighbor_positions = sorted_rows.view(-1, 1) + offsets.view(1, -1)
    in_bounds = (raw_neighbor_positions >= 0) & (raw_neighbor_positions < num_points)
    neighbor_positions = raw_neighbor_positions.clamp(0, num_points - 1)
    neighbor_indices = order[neighbor_positions]
    same_group = (
        sorted_group_ids[neighbor_positions]
        == sorted_group_ids.view(-1, 1)
    )
    non_self = neighbor_indices != order.view(-1, 1)

    neighbor_points = points[neighbor_indices]
    deltas = neighbor_points - sorted_points.unsqueeze(1)
    euclidean_distances = torch.linalg.norm(deltas, dim=-1)
    if weight_mode == "geodesic":
        normal_cosines = (
            sorted_normals.unsqueeze(1) * normals[neighbor_indices]
        ).sum(dim=-1).clamp(-1, 1)
        metric_distances = _edge_weights(
            euclidean_distances,
            normal_cosines,
            weight_mode,
        )
    else:
        metric_distances = euclidean_distances

    valid = (
        in_bounds
        & same_group
        & non_self
        & torch.isfinite(metric_distances)
        & torch.isfinite(euclidean_distances)
    )
    if max_distance is not None:
        valid = valid & (metric_distances <= float(max_distance))

    selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
        metric_distances,
        valid,
        max_neighbors=min(int(neighbors), int(metric_distances.shape[1])),
    )
    selected_indices = torch.gather(neighbor_indices, 1, selected_positions)
    selected_distances = torch.gather(metric_distances, 1, selected_positions)
    selected_euclidean = torch.gather(euclidean_distances, 1, selected_positions)
    selected_distances = torch.where(
        selected_valid,
        selected_distances,
        torch.full_like(selected_distances, float("inf")),
    )
    selected_euclidean = torch.where(
        selected_valid,
        selected_euclidean,
        torch.full_like(selected_euclidean, float("inf")),
    )
    return (
        selected_indices[inverse_order],
        selected_distances[inverse_order],
        selected_euclidean[inverse_order],
    )


def _nearest_windowed_support_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    block_types: torch.Tensor,
    block_indices: torch.Tensor,
    neighbors: int,
    *,
    weight_mode: AdjacencyWeightMode,
    max_distance: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = int(points.shape[0])
    max_support = int(support_indices.shape[1])
    if num_points <= 1 or neighbors <= 0 or max_support == 0:
        return _empty_candidate_tensors(num_points, 0, points)

    radius = max(_WINDOWED_TOPOLOGY_MIN_RADIUS, int(neighbors))
    offsets = torch.cat(
        (
            torch.arange(-radius, 0, dtype=torch.long, device=points.device),
            torch.arange(1, radius + 1, dtype=torch.long, device=points.device),
        ),
        dim=0,
    )
    point_rows = torch.arange(num_points, dtype=torch.long, device=points.device)
    membership_rows = point_rows.view(-1, 1).expand(num_points, max_support).reshape(-1)
    membership_valid = support_mask.reshape(-1)
    membership_groups = torch.where(
        membership_valid,
        support_indices.reshape(-1),
        torch.full_like(membership_rows, torch.iinfo(torch.long).max),
    )

    membership_points = points[membership_rows]
    order = _spatial_group_order(membership_points, membership_groups)
    inverse_order = torch.argsort(order)
    sorted_rows = membership_rows[order]
    sorted_groups = membership_groups[order]
    sorted_valid = membership_valid[order]
    sorted_points = points[sorted_rows]
    sorted_normals = normals[sorted_rows]

    membership_count = int(membership_rows.shape[0])
    sorted_positions = torch.arange(
        membership_count,
        dtype=torch.long,
        device=points.device,
    )
    raw_neighbor_positions = sorted_positions.view(-1, 1) + offsets.view(1, -1)
    in_bounds = (
        (raw_neighbor_positions >= 0)
        & (raw_neighbor_positions < membership_count)
    )
    neighbor_positions = raw_neighbor_positions.clamp(0, membership_count - 1)
    neighbor_rows = sorted_rows[neighbor_positions]
    source_rows = sorted_rows.view(-1, 1).expand_as(neighbor_rows)
    same_support_atom = (
        sorted_valid.view(-1, 1)
        & sorted_valid[neighbor_positions]
        & (sorted_groups[neighbor_positions] == sorted_groups.view(-1, 1))
    )
    topology_allowed = _topology_allowed_candidates(
        source_rows,
        neighbor_rows,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        allow_disjoint_single_support_edges=False,
    )
    non_self = source_rows != neighbor_rows

    neighbor_points = points[neighbor_rows]
    deltas = neighbor_points - sorted_points.unsqueeze(1)
    euclidean_distances = torch.linalg.norm(deltas, dim=-1)
    if weight_mode == "geodesic":
        normal_cosines = (
            sorted_normals.unsqueeze(1) * normals[neighbor_rows]
        ).sum(dim=-1).clamp(-1, 1)
        metric_distances = _edge_weights(
            euclidean_distances,
            normal_cosines,
            weight_mode,
        )
    else:
        metric_distances = euclidean_distances

    valid = (
        in_bounds
        & same_support_atom
        & topology_allowed
        & non_self
        & torch.isfinite(metric_distances)
        & torch.isfinite(euclidean_distances)
    )
    if max_distance is not None:
        valid = valid & (metric_distances <= float(max_distance))

    selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
        metric_distances,
        valid,
        max_neighbors=min(int(neighbors), int(metric_distances.shape[1])),
    )
    selected_indices = torch.gather(neighbor_rows, 1, selected_positions)
    selected_distances = torch.gather(metric_distances, 1, selected_positions)
    selected_euclidean = torch.gather(euclidean_distances, 1, selected_positions)
    selected_distances = torch.where(
        selected_valid,
        selected_distances,
        torch.full_like(selected_distances, float("inf")),
    )
    selected_euclidean = torch.where(
        selected_valid,
        selected_euclidean,
        torch.full_like(selected_euclidean, float("inf")),
    )

    unsorted_indices = selected_indices[inverse_order]
    unsorted_distances = selected_distances[inverse_order]
    unsorted_euclidean = selected_euclidean[inverse_order]
    candidate_columns = int(unsorted_indices.shape[1])
    if candidate_columns == 0:
        return _empty_candidate_tensors(num_points, 0, points)
    return _select_candidate_tensors(
        unsorted_indices.reshape(num_points, max_support * candidate_columns),
        unsorted_distances.reshape(num_points, max_support * candidate_columns),
        unsorted_euclidean.reshape(num_points, max_support * candidate_columns),
        neighbors,
    )


def _empty_candidate_tensors(
    rows: int,
    columns: int,
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.empty((rows, columns), dtype=torch.long, device=like.device)
    distances = torch.empty((rows, columns), dtype=like.dtype, device=like.device)
    return indices, distances, distances


def _select_candidate_tensors(
    neighbor_indices: torch.Tensor,
    metric_distances: torch.Tensor,
    euclidean_distances: torch.Tensor,
    neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
        metric_distances,
        torch.isfinite(metric_distances) & torch.isfinite(euclidean_distances),
        max_neighbors=min(int(neighbors), int(metric_distances.shape[1])),
    )
    selected_indices = torch.gather(neighbor_indices, 1, selected_positions)
    selected_distances = torch.gather(metric_distances, 1, selected_positions)
    selected_euclidean = torch.gather(euclidean_distances, 1, selected_positions)
    selected_distances = torch.where(
        selected_valid,
        selected_distances,
        torch.full_like(selected_distances, float("inf")),
    )
    selected_euclidean = torch.where(
        selected_valid,
        selected_euclidean,
        torch.full_like(selected_euclidean, float("inf")),
    )
    return selected_indices, selected_distances, selected_euclidean


def _single_support_topology_is_unrestricted(
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
    allow_disjoint_single_support_edges: bool,
) -> bool:
    if (
        not allow_disjoint_single_support_edges
        or support_indices is None
        or support_mask is None
        or block_types is not None
        or block_indices is not None
        or support_indices.shape[1] != 1
    ):
        return False
    return bool(support_mask.all().item())


def _should_prefilter_topology_candidates(num_points: int) -> bool:
    return int(num_points) >= _FAST_TOPOLOGY_MIN_POINTS


def _should_use_windowed_topology_candidates(
    points: torch.Tensor,
    num_points: int,
) -> bool:
    if torch.device(points.device).type == "cuda":
        return int(num_points) >= _WINDOWED_TOPOLOGY_GPU_MIN_POINTS
    return int(num_points) >= _WINDOWED_TOPOLOGY_MIN_POINTS


def _should_use_windowed_neighbor_candidates(
    points: torch.Tensor,
    num_points: int,
) -> bool:
    return (
        torch.device(points.device).type == "cuda"
        and int(num_points) >= _WINDOWED_NEIGHBOR_GPU_MIN_POINTS
    )


def _nearest_prefiltered_topology_candidates(
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
    prefilter_neighbors = min(
        int(points.shape[0]) - 1,
        max(
            int(neighbors),
            _FAST_TOPOLOGY_MIN_CANDIDATES,
            int(neighbors) * _FAST_TOPOLOGY_CANDIDATE_MULTIPLIER,
        ),
    )
    neighbor_indices, metric_distances, euclidean_distances = _nearest_neighbor_candidates(
        points,
        normals,
        prefilter_neighbors,
        weight_mode=weight_mode,
        max_distance=max_distance,
        pairwise_element_budget=pairwise_element_budget,
    )
    rows = torch.arange(
        int(points.shape[0]),
        dtype=torch.long,
        device=points.device,
    ).view(-1, 1).expand_as(neighbor_indices)
    topology_allowed = _topology_allowed_candidates(
        rows,
        neighbor_indices,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
        allow_disjoint_single_support_edges=allow_disjoint_single_support_edges,
    )
    valid = topology_allowed & torch.isfinite(metric_distances)
    if max_distance is not None:
        valid = valid & (metric_distances <= float(max_distance))
    return _masked_topk_candidates(
        neighbors,
        metric_distances,
        euclidean_distances,
        valid,
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


def _topology_allowed_candidates(
    rows: torch.Tensor,
    cols: torch.Tensor,
    *,
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
    allow_disjoint_single_support_edges: bool,
) -> torch.Tensor:
    if block_types is not None and block_indices is not None:
        allowed = (
            block_types[rows] == block_types[cols]
        ) & (
            block_indices[rows] == block_indices[cols]
        )
    else:
        allowed = torch.ones_like(cols, dtype=torch.bool)

    if support_indices is None or support_mask is None:
        return allowed

    same_support, adjacent_support, boundary_support = _support_relationships_for_pairs(
        support_indices[rows],
        support_mask[rows],
        support_indices[cols],
        support_mask[cols],
    )
    disjoint_single_support = (
        _disjoint_single_support_relationships(
            support_indices[rows],
            support_mask[rows],
            support_indices[cols],
            support_mask[cols],
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

    sector_ids = torch.arange(sector_count, dtype=torch.long, device=valid.device)
    sector_valid = valid.unsqueeze(-1) & (
        sectors.unsqueeze(-1) == sector_ids.view(1, 1, -1)
    )
    rank_distances = torch.where(
        sector_valid,
        neighbor_distances.unsqueeze(-1),
        torch.full(
            (1, 1, 1),
            float("inf"),
            dtype=neighbor_distances.dtype,
            device=neighbor_distances.device,
        ),
    )
    selected_distances, selected_positions = rank_distances.min(dim=1)
    slots = torch.arange(num_neighbors, dtype=torch.long, device=valid.device)
    selected = (
        (slots.view(1, -1, 1) == selected_positions.unsqueeze(1))
        & torch.isfinite(selected_distances).unsqueeze(1)
    ).any(dim=-1)
    return selected & valid


def _select_nearest_valid_candidate_positions(
    neighbor_distances: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest valid candidate column positions in each row.

    Args:
        neighbor_distances: Candidate ranking distances.
        valid: Boolean mask for candidates that may be kept.
        max_neighbors: Maximum selected candidate positions per row.

    Returns:
        ``(positions, valid_positions)`` where both tensors have shape
        ``(rows, selected_columns)``.
    """

    if max_neighbors <= 0 or neighbor_distances.shape[1] == 0:
        rows = int(neighbor_distances.shape[0])
        return (
            torch.empty((rows, 0), dtype=torch.long, device=neighbor_distances.device),
            torch.empty((rows, 0), dtype=torch.bool, device=neighbor_distances.device),
        )

    finite_valid = valid & torch.isfinite(neighbor_distances)
    rank_distances = torch.where(
        finite_valid,
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
    return selected_positions, torch.isfinite(selected_distances)


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

    selected_positions, selected_valid = _select_nearest_valid_candidate_positions(
        neighbor_distances,
        valid,
        max_neighbors=max_neighbors,
    )
    if selected_positions.numel() == 0:
        return torch.zeros_like(valid)
    slots = torch.arange(
        int(neighbor_distances.shape[1]),
        dtype=torch.long,
        device=neighbor_distances.device,
    )
    selected = (
        (slots.view(1, -1, 1) == selected_positions.unsqueeze(1))
        & selected_valid.unsqueeze(1)
    ).any(dim=-1)
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
    if ordered_keys.numel() == 0:
        return first, second, weights
    unique = torch.cat(
        (
            torch.ones((1,), dtype=torch.bool, device=ordered_keys.device),
            ordered_keys[1:] != ordered_keys[:-1],
        ),
        dim=0,
    )
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
