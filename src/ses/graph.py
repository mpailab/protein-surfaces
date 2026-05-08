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
    neighbors: int = 8,
    candidate_neighbors: Optional[int] = None,
    min_normal_cosine: float = -0.25,
    max_tangent_component: float = 0.8,
    max_distance: Optional[float] = None,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Build a sparse undirected adjacency matrix over SES surface samples.

    Edges start from nearest-neighbor candidates in Euclidean space.  If SES
    block/support metadata is supplied, candidates are first constrained to
    points on the same analytic patch or on adjacent patch families: atom-pair
    and pair-probe.  They then pass two local surface tests.  Neighbor normals
    must be reasonably compatible, and the chord between points must lie close
    to both tangent planes.  This removes many shortcuts through narrow molecular
    gaps while keeping local mesh-like links along the sampled SES.

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
        candidate_neighbors: Number of Euclidean nearest-neighbor candidates
            tested per point.  Defaults to a larger value than ``neighbors`` so
            tangent/normal filtering can discard bad shortcuts.
        min_normal_cosine: Minimum dot product between endpoint normals.
        max_tangent_component: Maximum absolute projection of the chord direction
            onto either endpoint normal.
        max_distance: Optional hard Euclidean distance cutoff for candidate edges.
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
    if candidate_neighbors is None:
        candidate_neighbors = max(3 * neighbors, neighbors + 8)
    else:
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

    kept_neighbors = min(neighbors, num_points - 1)
    candidate_neighbors = min(candidate_neighbors, num_points - 1)
    neighbor_indices, neighbor_distances = _nearest_neighbor_candidates(
        points,
        candidate_neighbors,
        pairwise_element_budget=pairwise_element_budget,
    )

    row_indices = torch.arange(num_points, dtype=torch.long, device=points.device).view(
        -1,
        1,
    )
    deltas = points[neighbor_indices] - points.unsqueeze(1)
    eps = torch.finfo(points.dtype).eps
    safe_distances = neighbor_distances.clamp_min(eps)
    directions = deltas / safe_distances.unsqueeze(-1)

    row_normals = normals.unsqueeze(1)
    col_normals = normals[neighbor_indices]
    normal_cosines = (row_normals * col_normals).sum(dim=-1).clamp(-1, 1)
    row_tangent_components = (directions * row_normals).sum(dim=-1).abs()
    col_tangent_components = (directions * col_normals).sum(dim=-1).abs()
    topology_allowed = _topology_allowed_candidates(
        neighbor_indices,
        support_indices=support_indices,
        support_mask=support_mask,
        block_types=block_types,
        block_indices=block_indices,
    )

    valid = (
        torch.isfinite(neighbor_distances)
        & (neighbor_distances > eps)
        & topology_allowed
        & (normal_cosines >= float(min_normal_cosine))
        & (row_tangent_components <= float(max_tangent_component))
        & (col_tangent_components <= float(max_tangent_component))
    )
    if max_distance is not None:
        valid = valid & (neighbor_distances <= float(max_distance))

    valid_ranks = valid.to(torch.long).cumsum(dim=1)
    keep = valid & (valid_ranks <= kept_neighbors)
    if not bool(keep.any().item()):
        return _empty_adjacency(num_points, points)

    rows = row_indices.expand_as(neighbor_indices)[keep]
    cols = neighbor_indices[keep]
    distances = neighbor_distances[keep]
    normal_cosines = normal_cosines[keep]
    weights = _edge_weights(distances, normal_cosines, weight_mode)

    first = torch.minimum(rows, cols)
    second = torch.maximum(rows, cols)
    non_self = first != second
    if not bool(non_self.any().item()):
        return _empty_adjacency(num_points, points)
    first = first[non_self]
    second = second[non_self]
    weights = weights[non_self]

    unique_first, unique_second, unique_weights = _unique_undirected_edges(
        first,
        second,
        weights,
        num_points,
    )
    if unique_weights.numel() == 0:
        return _empty_adjacency(num_points, points)

    indices = torch.cat(
        (
            torch.stack((unique_first, unique_second), dim=0),
            torch.stack((unique_second, unique_first), dim=0),
        ),
        dim=1,
    )
    values = torch.cat((unique_weights, unique_weights), dim=0)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_points, num_points),
        dtype=points.dtype,
        device=points.device,
    ).coalesce()


def _prepare_graph_inputs(
    points: torch.Tensor,
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (num_points, 3)")
    if normals.shape != points.shape:
        raise ValueError("normals must have the same shape as points")

    point_dtype = points.dtype if points.is_floating_point() else torch.float32
    normal_dtype = normals.dtype if normals.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(point_dtype, normal_dtype)
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


def _topology_allowed_candidates(
    neighbor_indices: torch.Tensor,
    *,
    support_indices: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    block_types: Optional[torch.Tensor],
    block_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    allowed = torch.ones_like(neighbor_indices, dtype=torch.bool)
    if block_types is not None and block_indices is not None:
        same_block = (
            block_types.unsqueeze(1) == block_types[neighbor_indices]
        ) & (
            block_indices.unsqueeze(1) == block_indices[neighbor_indices]
        )
        allowed = same_block

    if support_indices is None or support_mask is None:
        return allowed

    same_support, adjacent_support = _support_relationships(
        neighbor_indices,
        support_indices,
        support_mask,
    )
    support_allowed = same_support | adjacent_support
    if block_types is not None and block_indices is not None:
        return allowed | adjacent_support
    return support_allowed


def _support_relationships(
    neighbor_indices: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_indices = support_indices.unsqueeze(1)
    row_mask = support_mask.unsqueeze(1)
    col_indices = support_indices[neighbor_indices]
    col_mask = support_mask[neighbor_indices]
    row_counts = row_mask.sum(dim=-1)
    col_counts = col_mask.sum(dim=-1)

    matches = row_indices.unsqueeze(-1) == col_indices.unsqueeze(-2)
    row_in_col = (matches & col_mask.unsqueeze(-2)).any(dim=-1) | ~row_mask
    col_in_row = (matches & row_mask.unsqueeze(-1)).any(dim=-2) | ~col_mask
    row_subset_col = row_in_col.all(dim=-1)
    col_subset_row = col_in_row.all(dim=-1)
    same_support = row_subset_col & col_subset_row & (row_counts == col_counts)
    adjacent_support = (
        row_subset_col & (col_counts == row_counts + 1)
    ) | (
        col_subset_row & (row_counts == col_counts + 1)
    )
    return same_support, adjacent_support


def dense_features_to_supports(
    atom_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense binary atom features to padded sparse support metadata."""

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
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    eps = torch.finfo(vectors.dtype).eps
    fallback = torch.zeros_like(vectors)
    fallback[..., 0] = 1
    return torch.where(norms > eps, vectors / norms.clamp_min(eps), fallback)


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nearest_neighbor_candidates(
    points: torch.Tensor,
    neighbors: int,
    *,
    pairwise_element_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_points = int(points.shape[0])
    effective_budget = _effective_pairwise_element_budget(
        points.device,
        pairwise_element_budget,
    )
    rows_per_block = max(1, min(num_points, effective_budget // num_points))
    point_sq_norms = points.square().sum(dim=-1).view(1, -1)
    all_indices = []
    all_distances = []

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
        block_distances, block_indices = torch.topk(
            sq_distances,
            neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        all_indices.append(block_indices)
        all_distances.append(torch.sqrt(block_distances))

    return torch.cat(all_indices, dim=0), torch.cat(all_distances, dim=0)


def _effective_pairwise_element_budget(
    device: torch.device,
    pairwise_element_budget: int,
) -> int:
    if torch.device(device).type == "cuda":
        return max(int(pairwise_element_budget), _DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET)
    return int(pairwise_element_budget)


def _edge_weights(
    distances: torch.Tensor,
    normal_cosines: torch.Tensor,
    weight_mode: AdjacencyWeightMode,
) -> torch.Tensor:
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


def _unique_undirected_edges(
    first: torch.Tensor,
    second: torch.Tensor,
    weights: torch.Tensor,
    num_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = first * int(num_points) + second
    order = torch.argsort(keys)
    ordered_keys = keys[order]
    unique = torch.ones_like(ordered_keys, dtype=torch.bool)
    unique[1:] = ordered_keys[1:] != ordered_keys[:-1]
    selected = order[unique]
    return first[selected], second[selected], weights[selected]


def _empty_adjacency(num_points: int, like: torch.Tensor) -> torch.Tensor:
    indices = torch.empty((2, 0), dtype=torch.long, device=like.device)
    values = torch.empty((0,), dtype=like.dtype, device=like.device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_points, num_points),
        dtype=like.dtype,
        device=like.device,
    ).coalesce()


__all__ = ["build_surface_adjacency", "dense_features_to_supports"]
