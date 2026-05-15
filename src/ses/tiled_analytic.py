"""Speed-first tiled analytic SES point sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._outputs import SamplerOutput, _format_sample_outputs
from .graph import AdjacencyWeightMode, build_surface_adjacency
from .analytic import (
    ATOM_BLOCK_TYPE,
    AnalyticBlocks,
    PAIR_BLOCK_TYPE,
    PROBE_BLOCK_TYPE,
    AnalyticSamples,
    _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    _ProbeBlocks,
    _build_exterior_context,
    _candidate_pair_indices,
    _compute_triple_probe_centers,
    _concat_samples,
    _deduplicate_centers,
    _dense_atom_features,
    _dense_atom_weights,
    _empty_samples,
    _extract_probe_blocks,
    _packed_fibonacci_directions,
    _pad_columns,
    _pair_points_from_params,
    _pair_sample_counts,
    _probe_center_supports,
    _sample_atom_blocks,
    _sample_pair_blocks,
    _sample_probe_blocks,
    build_analytic_blocks,
)
from .projection import (
    _atom_cell_query_keys,
    _build_atom_cell_table,
    _centers_feasible_against_all_atoms,
    _probe_centers_accessible_from_exterior,
    _segment_clearance_mask,
)


_DEFAULT_TILE_SIZE: Union[float, str] = "auto"
_DEFAULT_TILE_OVERLAP: Union[float, str] = "auto"
_DEFAULT_DEDUP_TOLERANCE = 0.05
_DEFAULT_ATOM_DENSITY_SCALE = 1.0
_DEFAULT_PAIR_DENSITY_SCALE = 1.0
_DEFAULT_PROBE_DENSITY_SCALE = 1.0
_DEFAULT_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET = 64_000_000
_DEFAULT_TILE_MEMORY_BUDGET_BYTES = 3 * 1024**3
_DEFAULT_MAX_GRID_POINTS = 500_000
_DEFAULT_MAX_PROBE_TRIPLES = 5_000_000
_AUTO_TILE_ATOM_THRESHOLD = 2_000
_AUTO_SMALL_TILE_SIZE = 64.0
_AUTO_SMALL_TILE_OVERLAP = 4.0
_AUTO_LARGE_TILE_SIZE = 40.0
_AUTO_LARGE_TILE_OVERLAP = 4.0
_RADIAL_LINE_GRID_MIN_SEGMENTS = 512
_RADIAL_LINE_GRID_MAX_STEPS = 192
_RADIAL_LINE_GRID_STEP_FRACTION = 0.5
_RADIAL_LINE_GRID_CELL_SPAN = 2
_AUTO_TILE_MEMORY_SIZES = (
    24.0,
    32.0,
    40.0,
    48.0,
    56.0,
    64.0,
    80.0,
    96.0,
    128.0,
    160.0,
    192.0,
    256.0,
    320.0,
    512.0,
)
_PAIR_AREA_BYTES_PER_MEMBERSHIP_FLOAT32 = 3584.0
_PAIR_SAMPLE_BYTES_FLOAT32 = 72.0
_TYPICAL_PAIR_PATCH_AREA = 8.5


@dataclass(frozen=True)
class _TileGrid:
    indices: torch.Tensor
    bbox_min: torch.Tensor
    dims: torch.Tensor
    process_min: torch.Tensor
    process_max: torch.Tensor


def _validate_scale(name: str, value: float, *, allow_zero: bool = False) -> float:
    scale = float(value)
    if allow_zero:
        if scale < 0:
            raise ValueError(f"{name} must be non-negative")
    elif scale <= 0:
        raise ValueError(f"{name} must be positive")
    return scale


def _effective_pairwise_element_budget(
    device: torch.device,
    pairwise_element_budget: int,
) -> int:
    if torch.device(device).type == "cuda":
        return max(int(pairwise_element_budget), _DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET)
    return int(pairwise_element_budget)


def _build_tile_grid(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    tile_size: float,
    tile_overlap: float,
) -> _TileGrid:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_overlap < 0:
        raise ValueError("tile_overlap must be non-negative")

    margin = atom_radii.max() + float(probe_radius)
    bbox_min = atom_coords.min(dim=0).values - margin
    bbox_max = atom_coords.max(dim=0).values + margin
    spans = (bbox_max - bbox_min).clamp_min(torch.finfo(atom_coords.dtype).eps)
    dims = torch.ceil(spans / float(tile_size)).to(torch.long).clamp_min(1)
    axes = [
        torch.arange(int(dims[axis].item()), dtype=torch.long, device=atom_coords.device)
        for axis in range(3)
    ]
    indices = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    core_min = bbox_min + indices.to(atom_coords.dtype) * float(tile_size)
    core_max = torch.minimum(core_min + float(tile_size), bbox_max.view(1, 3))
    return _TileGrid(
        indices=indices,
        bbox_min=bbox_min,
        dims=dims,
        process_min=core_min - float(tile_overlap),
        process_max=core_max + float(tile_overlap),
    )


def _resolve_tile_parameters(
    num_atoms: int,
    tile_size: Union[float, str],
    tile_overlap: Union[float, str],
) -> tuple[float, float]:
    if isinstance(tile_size, str):
        if tile_size != "auto":
            raise ValueError("tile_size must be positive or 'auto'")
        if int(num_atoms) <= _AUTO_TILE_ATOM_THRESHOLD:
            resolved_size = _AUTO_SMALL_TILE_SIZE
            resolved_overlap = _AUTO_SMALL_TILE_OVERLAP
        else:
            resolved_size = _AUTO_LARGE_TILE_SIZE
            resolved_overlap = _AUTO_LARGE_TILE_OVERLAP
        if isinstance(tile_overlap, str):
            if tile_overlap != "auto":
                raise ValueError("tile_overlap must be non-negative or 'auto'")
            return resolved_size, resolved_overlap
        return resolved_size, float(tile_overlap)

    if isinstance(tile_overlap, str):
        if tile_overlap == "auto":
            raise ValueError("tile_overlap='auto' requires tile_size='auto'")
        raise ValueError("tile_overlap must be non-negative or 'auto'")
    return float(tile_size), float(tile_overlap)


def _tile_atom_intersection_mask(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    grid: _TileGrid,
) -> torch.Tensor:
    influence = (atom_radii + float(probe_radius)).unsqueeze(-1)
    atom_min = atom_coords - influence
    atom_max = atom_coords + influence
    return (
        (atom_max.unsqueeze(0) >= grid.process_min.unsqueeze(1))
        & (atom_min.unsqueeze(0) <= grid.process_max.unsqueeze(1))
    ).all(dim=-1)


def _active_tile_ids(atom_tile_mask: torch.Tensor) -> torch.Tensor:
    if atom_tile_mask.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device)
    active = atom_tile_mask.any(dim=-1).nonzero(as_tuple=False).reshape(-1)
    if active.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device)
    return active


def _candidate_tile_ids(
    atom_tile_mask: torch.Tensor,
    *,
    min_atoms: int,
) -> torch.Tensor:
    if atom_tile_mask.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device)
    counts = atom_tile_mask.sum(dim=-1)
    return (counts >= int(min_atoms)).nonzero(as_tuple=False).reshape(-1)


def _point_owner_tile_ids(
    points: torch.Tensor,
    *,
    grid: _TileGrid,
    tile_size: float,
) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=points.device)
    owner = torch.floor((points.detach() - grid.bbox_min) / float(tile_size)).to(torch.long)
    owner = torch.minimum(
        torch.maximum(owner, torch.zeros_like(owner)),
        grid.dims.view(1, 3) - 1,
    )
    return (owner[:, 0] * grid.dims[1] + owner[:, 1]) * grid.dims[2] + owner[:, 2]


def _apply_tile_locality(
    samples: AnalyticSamples,
    *,
    grid: _TileGrid,
    atom_tile_mask: torch.Tensor,
    tile_size: float,
) -> AnalyticSamples:
    if samples.points.shape[0] == 0:
        return samples
    owner_ids = _point_owner_tile_ids(samples.points, grid=grid, tile_size=tile_size)
    safe_supports = samples.support_indices.clamp_min(0)
    owner_atoms = atom_tile_mask[
        owner_ids.unsqueeze(-1).expand_as(safe_supports),
        safe_supports,
    ]
    keep = (owner_atoms | ~samples.support_mask).all(dim=-1)
    return _index_samples(samples, keep)


def _pair_block_ids_from_supports(
    support_indices: torch.Tensor,
    *,
    num_atoms: int,
) -> torch.Tensor:
    if support_indices.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=support_indices.device)
    first = support_indices[:, 0].clamp_min(0)
    second = support_indices[:, 1].clamp_min(0)
    lo = torch.minimum(first, second)
    hi = torch.maximum(first, second)
    return lo * int(num_atoms) + hi


def _probe_block_ids_from_supports(
    support_indices: torch.Tensor,
    probe_centers: torch.Tensor,
    atom_coords: torch.Tensor,
    *,
    num_atoms: int,
) -> torch.Tensor:
    if support_indices.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=support_indices.device)

    safe_supports = support_indices.clamp_min(0)
    sorted_supports = torch.sort(safe_supports, dim=-1).values
    first = sorted_supports[:, 0]
    second = sorted_supports[:, 1]
    third = sorted_supports[:, 2]
    base_ids = ((first * int(num_atoms)) + second) * int(num_atoms) + third

    first_coords = atom_coords[first]
    second_coords = atom_coords[second]
    third_coords = atom_coords[third]
    normals = torch.cross(
        second_coords - first_coords,
        third_coords - first_coords,
        dim=-1,
    )
    side = ((probe_centers - first_coords) * normals).sum(dim=-1) >= 0
    return base_ids * 2 + side.to(dtype=torch.long)


def _sample_contact_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    local_indices: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float,
    atom_density_scale: float,
    include_normals: bool = False,
) -> AnalyticSamples:
    if local_indices.numel() == 0:
        context = _build_exterior_context(atom_coords, atom_radii, probe_radius)
        return _empty_samples(context, max_support=1)

    with torch.no_grad():
        areas = 4 * math.pi * atom_radii[local_indices].detach().square()
        counts = torch.ceil(
            areas * float(atom_density_scale) / float(point_area),
        ).to(torch.long).clamp_min(1)
        owner_rows = torch.repeat_interleave(
            torch.arange(local_indices.shape[0], device=atom_coords.device),
            counts,
        )
        owners = local_indices[owner_rows]
        directions = _packed_fibonacci_directions(
            counts,
            dtype=atom_coords.dtype,
            device=atom_coords.device,
        )

    probe_centers = (
        atom_coords[owners]
        + (atom_radii[owners] + float(probe_radius)).unsqueeze(-1) * directions
    )
    points = atom_coords[owners] + atom_radii[owners].unsqueeze(-1) * directions
    normals = directions if include_normals else None
    support_indices = owners.unsqueeze(-1)
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)
    support_weights = torch.ones(
        support_indices.shape,
        dtype=atom_coords.dtype,
        device=atom_coords.device,
    )
    return AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            ATOM_BLOCK_TYPE,
            dtype=torch.long,
            device=atom_coords.device,
        ),
        block_indices=owner_rows,
        support_indices=support_indices,
        support_mask=support_mask,
        support_weights=support_weights,
        normals=normals,
        probe_centers=probe_centers,
    )


def _sample_pair_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    local_indices: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float,
    pair_density_scale: float,
    include_normals: bool = False,
) -> AnalyticSamples:
    local_context = _build_exterior_context(
        atom_coords[local_indices],
        atom_radii[local_indices],
        probe_radius,
    )
    empty = _empty_samples(local_context, max_support=2)
    if local_indices.numel() < 2 or pair_density_scale <= 0:
        return empty

    pair_indices = _candidate_pair_indices(local_context)
    if pair_indices.numel() == 0:
        return empty

    with torch.no_grad():
        effective_point_area = float(point_area) / float(pair_density_scale)
        counts = _pair_sample_counts(
            local_context,
            pair_indices,
            point_area=effective_point_area,
            oversample_factor=1.0,
        )
        pair_rows, theta_values, arc_fracs = _packed_pair_parameters_uniform_arc(
            counts,
            dtype=atom_coords.dtype,
            device=atom_coords.device,
        )
    if pair_rows.numel() == 0:
        return empty

    selected_local_pairs = pair_indices[pair_rows]
    probe_centers, points, first_weights, second_weights = _pair_points_from_params(
        atom_coords[local_indices],
        atom_radii[local_indices] + float(probe_radius),
        probe_radius,
        selected_local_pairs,
        theta_values,
        arc_fracs,
    )
    normals = (probe_centers - points) / float(probe_radius) if include_normals else None
    selected_pairs = local_indices[selected_local_pairs]
    support_weights = torch.stack(
        (first_weights.detach(), second_weights.detach()),
        dim=-1,
    )
    support_weights = support_weights / support_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(atom_coords.dtype).eps)
    support_mask = torch.ones_like(selected_pairs, dtype=torch.bool)
    return AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            PAIR_BLOCK_TYPE,
            dtype=torch.long,
            device=atom_coords.device,
        ),
        block_indices=pair_rows,
        support_indices=selected_pairs,
        support_mask=support_mask,
        support_weights=support_weights,
        normals=normals,
        probe_centers=probe_centers,
    )


def _sample_probe_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    local_indices: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float,
    probe_density_scale: float,
    max_grid_points: int,
    max_probe_triples: Optional[int],
    include_normals: bool = False,
) -> AnalyticSamples:
    if local_indices.numel() < 3 or probe_density_scale <= 0:
        global_context = _build_exterior_context(atom_coords, atom_radii, probe_radius)
        empty = _empty_samples(global_context, max_support=3)
        return empty

    local_context = _build_exterior_context(
        atom_coords[local_indices],
        atom_radii[local_indices],
        probe_radius,
        max_grid_points=max_grid_points,
    )
    empty = _empty_samples(local_context, max_support=3)
    local_pair_indices = _candidate_pair_indices(local_context)
    probe_blocks = _extract_probe_blocks(
        local_context,
        pair_indices=local_pair_indices,
        max_triples=max_probe_triples,
    )
    if probe_blocks.probe_seed_indices.numel() == 0:
        return empty

    blocks = AnalyticBlocks(
        atom_indices=torch.empty((0,), dtype=torch.long, device=atom_coords.device),
        pair_indices=torch.empty((0, 2), dtype=torch.long, device=atom_coords.device),
        probe_seed_indices=probe_blocks.probe_seed_indices,
        probe_center_signs=probe_blocks.probe_center_signs,
        probe_support_indices=probe_blocks.probe_support_indices,
        probe_support_mask=probe_blocks.probe_support_mask,
        probe_center_hints=probe_blocks.probe_center_hints,
    )
    probe_samples = _sample_probe_blocks(
        local_context,
        blocks,
        point_area=point_area,
        oversample_factor=1.0,
        probe_density_scale=probe_density_scale,
        include_normals=include_normals,
    )
    if probe_samples.points.shape[0] == 0:
        return empty

    safe_local_supports = probe_samples.support_indices.clamp_min(0)
    global_supports = local_indices[safe_local_supports]
    global_supports = torch.where(
        probe_samples.support_mask,
        global_supports,
        torch.full_like(global_supports, -1),
    )
    probe_centers = probe_blocks.probe_center_hints[
        probe_samples.block_indices
    ].to(dtype=atom_coords.dtype, device=atom_coords.device)
    with torch.no_grad():
        global_seed_supports = local_indices[probe_blocks.probe_seed_indices.clamp_min(0)]
        block_indices = _probe_block_ids_from_supports(
            global_seed_supports[probe_samples.block_indices],
            probe_centers,
            atom_coords,
            num_atoms=atom_coords.shape[0],
        )
    return AnalyticSamples(
        points=probe_samples.points,
        block_types=probe_samples.block_types,
        block_indices=block_indices,
        support_indices=global_supports,
        support_mask=probe_samples.support_mask,
        support_weights=probe_samples.support_weights,
        normals=probe_samples.normals,
        probe_centers=probe_centers,
    )


def _packed_pair_parameters_uniform_arc(
    counts: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack speed-first torus samples without building per-pair arc CDF tables."""

    counts = counts.to(device=device, dtype=torch.long)
    if counts.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )

    total_count = int(counts.sum().item())
    if total_count <= 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )

    rows = torch.arange(counts.shape[0], dtype=torch.long, device=device)
    pair_rows = torch.repeat_interleave(rows, counts, output_size=total_count)
    row_starts = torch.cumsum(counts, dim=0) - counts
    sample_offsets = (
        torch.arange(total_count, dtype=dtype, device=device)
        - row_starts[pair_rows].to(dtype=dtype)
        + 0.5
    )
    row_counts = counts[pair_rows].to(dtype=dtype).clamp_min(1)
    theta_values = 2 * math.pi * sample_offsets / row_counts
    golden_ratio_conjugate = torch.as_tensor(
        0.6180339887498949,
        dtype=dtype,
        device=device,
    )
    arc_fracs = torch.frac(sample_offsets * golden_ratio_conjugate)
    return pair_rows, theta_values, arc_fracs


def _tile_pair_memberships(
    atom_tile_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    active_tile_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair_indices.shape[0] == 0 or atom_tile_mask.shape[0] == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
        )

    if active_tile_ids is None:
        active_tile_ids = _active_tile_ids(atom_tile_mask)
    if active_tile_ids.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
        )

    active_mask = atom_tile_mask[active_tile_ids]
    membership = (
        active_mask[:, pair_indices[:, 0]]
        & active_mask[:, pair_indices[:, 1]]
    )
    active_rows, pair_rows = membership.nonzero(as_tuple=True)
    if active_rows.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
            torch.empty((0,), dtype=torch.long, device=atom_tile_mask.device),
        )
    return active_tile_ids[active_rows], pair_rows


def _tile_pair_membership_count(
    atom_tile_mask: torch.Tensor,
    pair_indices: torch.Tensor,
    active_tile_ids: torch.Tensor,
) -> int:
    if pair_indices.shape[0] == 0 or active_tile_ids.numel() == 0:
        return 0
    active_mask = atom_tile_mask[active_tile_ids]
    membership = (
        active_mask[:, pair_indices[:, 0]]
        & active_mask[:, pair_indices[:, 1]]
    )
    return int(membership.sum().item())


def _estimate_tile_pair_work_bytes(
    membership_count: int,
    *,
    point_area: float,
    pair_density_scale: float,
    dtype: torch.dtype,
) -> float:
    dtype_scale = 2.0 if dtype == torch.float64 else 1.0
    area_bytes = (
        float(membership_count)
        * _PAIR_AREA_BYTES_PER_MEMBERSHIP_FLOAT32
        * dtype_scale
    )
    estimated_samples = float(membership_count) * max(
        1.0,
        _TYPICAL_PAIR_PATCH_AREA * float(pair_density_scale) / float(point_area),
    )
    sample_bytes = estimated_samples * _PAIR_SAMPLE_BYTES_FLOAT32 * dtype_scale
    return area_bytes + sample_bytes


def _resolve_memory_aware_tile_parameters(
    context,
    *,
    tile_size: Union[float, str],
    tile_overlap: Union[float, str],
    point_area: float,
    pair_density_scale: float,
    pair_indices: Optional[torch.Tensor] = None,
    tile_memory_budget_bytes: int = _DEFAULT_TILE_MEMORY_BUDGET_BYTES,
) -> tuple[float, float]:
    resolved_size, resolved_overlap = _resolve_tile_parameters(
        context.num_atoms,
        tile_size,
        tile_overlap,
    )
    if not isinstance(tile_size, str) or tile_size != "auto":
        return resolved_size, resolved_overlap
    if pair_density_scale <= 0:
        return float(_AUTO_TILE_MEMORY_SIZES[-1]), resolved_overlap

    if pair_indices is None:
        pair_indices = _candidate_pair_indices(context)
    else:
        pair_indices = pair_indices.to(device=context.device, dtype=torch.long)
    if pair_indices.numel() == 0:
        return float(_AUTO_TILE_MEMORY_SIZES[-1]), resolved_overlap

    largest_fit_size: Optional[float] = None
    best_over_budget_size = resolved_size
    best_over_budget_bytes = float("inf")
    for candidate_size in _AUTO_TILE_MEMORY_SIZES:
        if candidate_size < resolved_size:
            continue
        grid = _build_tile_grid(
            context.atom_coords.detach(),
            context.atom_radii.detach(),
            context.probe_radius,
            tile_size=float(candidate_size),
            tile_overlap=resolved_overlap,
        )
        atom_tile_mask = _tile_atom_intersection_mask(
            context.atom_coords.detach(),
            context.atom_radii.detach(),
            context.probe_radius,
            grid,
        )
        active_ids = _candidate_tile_ids(atom_tile_mask, min_atoms=2)
        membership_count = _tile_pair_membership_count(
            atom_tile_mask,
            pair_indices,
            active_ids,
        )
        estimate = _estimate_tile_pair_work_bytes(
            membership_count,
            point_area=point_area,
            pair_density_scale=pair_density_scale,
            dtype=context.dtype,
        )
        if estimate < best_over_budget_bytes:
            best_over_budget_bytes = estimate
            best_over_budget_size = float(candidate_size)
        if estimate <= int(tile_memory_budget_bytes):
            largest_fit_size = float(candidate_size)

    if largest_fit_size is not None:
        return largest_fit_size, resolved_overlap
    return best_over_budget_size, resolved_overlap


def _candidate_tile_triple_indices(
    num_atoms: int,
    pair_indices: torch.Tensor,
    tile_ids: torch.Tensor,
    *,
    max_triples: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pair_indices.device
    empty_tiles = torch.empty((0,), dtype=torch.long, device=device)
    empty_triples = torch.empty((0, 3), dtype=torch.long, device=device)
    if pair_indices.numel() == 0 or tile_ids.numel() == 0:
        return empty_tiles, empty_triples
    if max_triples is not None and max_triples <= 0:
        return empty_tiles, empty_triples

    pairs = torch.sort(pair_indices.to(dtype=torch.long), dim=-1).values
    tile_ids = tile_ids.to(dtype=torch.long, device=device)
    in_range = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 1] < int(num_atoms))
        & (pairs[:, 0] < pairs[:, 1])
        & (tile_ids >= 0)
    )
    pairs = pairs[in_range]
    tile_ids = tile_ids[in_range]
    if pairs.numel() == 0:
        return empty_tiles, empty_triples

    pair_keys = ((tile_ids * int(num_atoms)) + pairs[:, 0]) * int(num_atoms) + pairs[:, 1]
    order = torch.argsort(pair_keys)
    pair_keys = pair_keys[order]
    pairs = pairs[order]
    tile_ids = tile_ids[order]
    unique = torch.cat(
        (
            torch.ones((1,), dtype=torch.bool, device=device),
            pair_keys[1:] != pair_keys[:-1],
        ),
        dim=0,
    )
    pair_keys = pair_keys[unique]
    pairs = pairs[unique]
    tile_ids = tile_ids[unique]

    first_keys = tile_ids * int(num_atoms) + pairs[:, 0]
    group_keys, counts = torch.unique_consecutive(first_keys, return_counts=True)
    max_degree = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_degree < 2:
        return empty_tiles, empty_triples

    offsets = torch.cat(
        (
            torch.zeros((1,), dtype=torch.long, device=device),
            counts.cumsum(dim=0),
        ),
        dim=0,
    )
    max_combos = max_degree * (max_degree - 1) // 2
    rows_per_batch = max(
        1,
        min(
            group_keys.shape[0],
            _DEFAULT_GPU_PAIRWISE_ELEMENT_BUDGET // max(1, max_combos),
        ),
    )
    combo_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tile_chunks = []
    triple_chunks = []
    triple_count = 0
    for start in range(0, group_keys.shape[0], rows_per_batch):
        stop = min(start + rows_per_batch, group_keys.shape[0])
        batch_counts = counts[start:stop]
        batch_degree = int(batch_counts.max().item()) if batch_counts.numel() > 0 else 0
        if batch_degree < 2:
            continue

        first_pair = int(offsets[start].item())
        last_pair = int(offsets[stop].item())
        if last_pair <= first_pair:
            continue

        neighbors = torch.full(
            (stop - start, batch_degree),
            -1,
            dtype=torch.long,
            device=device,
        )
        local_group_rows = torch.repeat_interleave(
            torch.arange(stop - start, dtype=torch.long, device=device),
            batch_counts,
        )
        local_slots = (
            torch.arange(first_pair, last_pair, dtype=torch.long, device=device)
            - torch.repeat_interleave(offsets[start:stop], batch_counts)
        )
        slot_mask = local_slots < batch_degree
        neighbors[local_group_rows[slot_mask], local_slots[slot_mask]] = pairs[
            first_pair:last_pair
        ][slot_mask, 1]

        combos = combo_cache.get(batch_degree)
        if combos is None:
            combos = torch.triu_indices(
                batch_degree,
                batch_degree,
                offset=1,
                dtype=torch.long,
                device=device,
            )
            combo_cache[batch_degree] = (combos[0], combos[1])
        combo_first, combo_second = combo_cache[batch_degree]
        combo_count = int(combo_first.numel())
        group_rows = torch.arange(stop - start, dtype=torch.long, device=device)
        first_atoms = group_keys[start:stop].remainder(int(num_atoms))
        tile_columns = torch.div(group_keys[start:stop], int(num_atoms), rounding_mode="floor")
        first_column = first_atoms.repeat_interleave(combo_count)
        tile_column = tile_columns.repeat_interleave(combo_count)
        second_column = neighbors[:, combo_first].reshape(-1)
        third_column = neighbors[:, combo_second].reshape(-1)
        del group_rows

        valid = third_column >= 0
        second_valid = second_column[valid]
        third_valid = third_column[valid]
        tile_valid = tile_column[valid]
        if second_valid.numel() == 0:
            continue
        candidate_keys = (
            (tile_valid * int(num_atoms) + second_valid) * int(num_atoms)
            + third_valid
        )
        positions = torch.searchsorted(pair_keys, candidate_keys)
        in_bounds = positions < pair_keys.shape[0]
        safe_positions = positions.clamp_max(max(pair_keys.shape[0] - 1, 0))
        matches = in_bounds & (pair_keys[safe_positions] == candidate_keys)

        chunk_tiles = tile_valid[matches]
        chunk = torch.stack(
            (
                first_column[valid][matches],
                second_valid[matches],
                third_valid[matches],
            ),
            dim=-1,
        )
        if chunk.shape[0] == 0:
            continue
        if max_triples is not None:
            remaining = int(max_triples) - triple_count
            if remaining <= 0:
                break
            if chunk.shape[0] > remaining:
                chunk = chunk[:remaining]
                chunk_tiles = chunk_tiles[:remaining]
        tile_chunks.append(chunk_tiles)
        triple_chunks.append(chunk)
        triple_count += int(chunk.shape[0])
        if max_triples is not None and triple_count >= int(max_triples):
            break

    if not triple_chunks:
        return empty_tiles, empty_triples
    return torch.cat(tile_chunks, dim=0), torch.cat(triple_chunks, dim=0)


def _deduplicate_centers_by_tile(
    centers: torch.Tensor,
    tile_ids: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    if centers.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=centers.device)
    center_keys = torch.round(centers.detach() / float(tolerance)).to(torch.long)
    keys = torch.cat((tile_ids.to(torch.long).unsqueeze(-1), center_keys), dim=-1)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    indices = torch.arange(centers.shape[0], dtype=torch.long, device=centers.device)
    first_indices = torch.full(
        (int(inverse.max().item()) + 1,),
        centers.shape[0],
        dtype=torch.long,
        device=centers.device,
    )
    first_indices.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    first_indices = first_indices[first_indices < centers.shape[0]]
    return torch.sort(first_indices).values


def _extract_tile_probe_blocks(
    context,
    atom_tile_mask: torch.Tensor,
    *,
    pair_indices: Optional[torch.Tensor] = None,
    max_support_atoms: int = _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    support_tolerance: float = 1e-3,
    dedup_tolerance: float = 1e-4,
    max_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
) -> tuple[_ProbeBlocks, torch.Tensor]:
    empty = _empty_tile_probe_blocks(context.device, max_support_atoms)
    if context.num_atoms < 3:
        return empty

    if pair_indices is None:
        pair_indices = _candidate_pair_indices(context)
    else:
        pair_indices = pair_indices.to(device=context.device, dtype=torch.long)
    active_tile_ids = _candidate_tile_ids(atom_tile_mask, min_atoms=3)
    tile_pair_ids, base_pair_rows = _tile_pair_memberships(
        atom_tile_mask,
        pair_indices,
        active_tile_ids,
    )
    if base_pair_rows.numel() == 0:
        return empty

    triple_tile_ids, triple_indices = _candidate_tile_triple_indices(
        context.num_atoms,
        pair_indices[base_pair_rows],
        tile_pair_ids,
        max_triples=max_triples,
    )
    if triple_indices.numel() == 0:
        return empty

    centers, valid = _compute_triple_probe_centers(
        context.atom_coords,
        context.expanded_radii,
        triple_indices[:, 0],
        triple_indices[:, 1],
        triple_indices[:, 2],
    )
    flat_centers = centers.reshape(-1, 3)
    flat_valid = valid.reshape(-1) & torch.isfinite(flat_centers).all(dim=-1)
    flat_valid_indices = flat_valid.nonzero(as_tuple=False).reshape(-1)
    if flat_valid_indices.numel() == 0:
        return empty
    flat_centers = flat_centers[flat_valid]
    flat_seed_rows = torch.div(flat_valid_indices, 2, rounding_mode="floor")
    flat_seeds = triple_indices[flat_seed_rows]
    flat_tile_ids = triple_tile_ids[flat_seed_rows]
    flat_signs = flat_valid_indices.remainder(2)
    feasible = context.centers_feasible(flat_centers)
    accessible = context.centers_accessible(flat_centers, feasible)
    flat_centers = flat_centers[accessible]
    if flat_centers.shape[0] == 0:
        return empty
    flat_seeds = flat_seeds[accessible]
    flat_tile_ids = flat_tile_ids[accessible]
    flat_signs = flat_signs[accessible]
    keep = _deduplicate_centers_by_tile(flat_centers, flat_tile_ids, dedup_tolerance)
    flat_centers = flat_centers[keep]
    flat_seeds = flat_seeds[keep]
    flat_tile_ids = flat_tile_ids[keep]
    flat_signs = flat_signs[keep]
    support_indices, support_mask = _probe_center_supports(
        flat_centers,
        flat_seeds,
        context.atom_coords,
        context.expanded_radii,
        max_support_atoms=max_support_atoms,
        tolerance=support_tolerance,
    )
    support_counts = support_mask.sum(dim=-1)
    enough_support = support_counts >= 3
    kept_seeds = flat_seeds[enough_support]
    if kept_seeds.shape[0] == 0:
        return empty
    kept_signs = flat_signs[enough_support]
    kept_support_indices = support_indices[enough_support]
    kept_support_mask = support_mask[enough_support]
    kept_centers = flat_centers[enough_support]
    kept_tile_ids = flat_tile_ids[enough_support]

    return (
        _ProbeBlocks(
            probe_seed_indices=kept_seeds,
            probe_center_signs=kept_signs,
            probe_support_indices=kept_support_indices,
            probe_support_mask=kept_support_mask,
            probe_center_hints=kept_centers.detach(),
        ),
        kept_tile_ids,
    )


def _empty_tile_probe_blocks(
    device: torch.device,
    max_support_atoms: int,
) -> tuple[_ProbeBlocks, torch.Tensor]:
    blocks = _ProbeBlocks(
        probe_seed_indices=torch.empty((0, 3), dtype=torch.long, device=device),
        probe_center_signs=torch.empty((0,), dtype=torch.long, device=device),
        probe_support_indices=torch.empty((0, max_support_atoms), dtype=torch.long, device=device),
        probe_support_mask=torch.empty((0, max_support_atoms), dtype=torch.bool, device=device),
        probe_center_hints=torch.empty((0, 3), dtype=torch.float32, device=device),
    )
    tile_ids = torch.empty((0,), dtype=torch.long, device=device)
    return blocks, tile_ids


def _sample_tile_local_pair_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    atom_tile_mask: torch.Tensor,
    grid: _TileGrid,
    probe_radius: float,
    *,
    point_area: float,
    pair_density_scale: float,
    tile_size: float,
    include_normals: bool = False,
    pair_indices: Optional[torch.Tensor] = None,
) -> AnalyticSamples:
    context = _build_exterior_context(atom_coords, atom_radii, probe_radius)
    empty = _empty_samples(context, max_support=2)
    if atom_tile_mask.shape[0] == 0 or pair_density_scale <= 0:
        return empty

    if pair_indices is None:
        pair_indices = _candidate_pair_indices(context)
    else:
        pair_indices = pair_indices.to(device=context.device, dtype=torch.long)
    if pair_indices.numel() == 0:
        return empty

    with torch.no_grad():
        active_tile_ids = _candidate_tile_ids(atom_tile_mask, min_atoms=2)
        tile_ids, base_pair_rows = _tile_pair_memberships(
            atom_tile_mask,
            pair_indices,
            active_tile_ids,
        )
        if base_pair_rows.numel() == 0:
            return empty

        effective_point_area = float(point_area) / float(pair_density_scale)
        counts = _pair_sample_counts(
            context,
            pair_indices[base_pair_rows],
            point_area=effective_point_area,
            oversample_factor=1.0,
        ).to(dtype=torch.long)
        pair_rows, theta_values, arc_fracs = _packed_pair_parameters_uniform_arc(
            counts,
            dtype=atom_coords.dtype,
            device=atom_coords.device,
        )
    if pair_rows.numel() == 0:
        return empty

    selected_rows = base_pair_rows[pair_rows]
    selected_pairs = pair_indices[selected_rows]
    probe_centers, points, first_weights, second_weights = _pair_points_from_params(
        atom_coords,
        atom_radii + float(probe_radius),
        probe_radius,
        selected_pairs,
        theta_values,
        arc_fracs,
    )
    with torch.no_grad():
        keep = _point_owner_tile_ids(
            points,
            grid=grid,
            tile_size=tile_size,
        ) == tile_ids[pair_rows]
    normals = (probe_centers - points) / float(probe_radius) if include_normals else None
    support_weights = torch.stack(
        (first_weights.detach(), second_weights.detach()),
        dim=-1,
    )
    support_weights = support_weights / support_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(atom_coords.dtype).eps)
    samples = AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            PAIR_BLOCK_TYPE,
            dtype=torch.long,
            device=atom_coords.device,
        ),
        block_indices=_pair_block_ids_from_supports(
            selected_pairs,
            num_atoms=atom_coords.shape[0],
        ),
        support_indices=selected_pairs,
        support_mask=torch.ones_like(selected_pairs, dtype=torch.bool),
        support_weights=support_weights,
        normals=normals,
        probe_centers=probe_centers,
    )
    return _index_samples(samples, keep)


def _sample_tile_local_probe_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    atom_tile_mask: torch.Tensor,
    grid: _TileGrid,
    probe_radius: float,
    *,
    point_area: float,
    probe_density_scale: float,
    tile_size: float,
    max_grid_points: int,
    max_probe_triples: Optional[int],
    include_normals: bool = False,
    pair_indices: Optional[torch.Tensor] = None,
) -> AnalyticSamples:
    context = _build_exterior_context(
        atom_coords,
        atom_radii,
        probe_radius,
        max_grid_points=max_grid_points,
    )
    empty = _empty_samples(context, max_support=3)
    if atom_tile_mask.shape[0] == 0 or probe_density_scale <= 0:
        return empty

    probe_blocks, block_tile_ids = _extract_tile_probe_blocks(
        context,
        atom_tile_mask,
        pair_indices=pair_indices,
        max_triples=max_probe_triples,
    )
    if probe_blocks.probe_seed_indices.numel() == 0:
        return empty

    blocks = AnalyticBlocks(
        atom_indices=torch.empty((0,), dtype=torch.long, device=atom_coords.device),
        pair_indices=torch.empty((0, 2), dtype=torch.long, device=atom_coords.device),
        probe_seed_indices=probe_blocks.probe_seed_indices,
        probe_center_signs=probe_blocks.probe_center_signs,
        probe_support_indices=probe_blocks.probe_support_indices,
        probe_support_mask=probe_blocks.probe_support_mask,
        probe_center_hints=probe_blocks.probe_center_hints,
    )
    probe_samples = _sample_probe_blocks(
        context,
        blocks,
        point_area=point_area,
        oversample_factor=1.0,
        probe_density_scale=probe_density_scale,
        include_normals=include_normals,
    )
    if probe_samples.points.shape[0] == 0:
        return empty

    with torch.no_grad():
        sample_tile_ids = block_tile_ids[probe_samples.block_indices]
        keep = _point_owner_tile_ids(
            probe_samples.points,
            grid=grid,
            tile_size=tile_size,
        ) == sample_tile_ids
    probe_samples = _index_samples(probe_samples, keep)
    if probe_samples.points.shape[0] == 0:
        return empty

    probe_centers = probe_blocks.probe_center_hints[
        probe_samples.block_indices
    ].to(dtype=atom_coords.dtype, device=atom_coords.device)
    with torch.no_grad():
        block_indices = _probe_block_ids_from_supports(
            probe_blocks.probe_seed_indices[probe_samples.block_indices],
            probe_centers,
            atom_coords,
            num_atoms=atom_coords.shape[0],
        )
    return AnalyticSamples(
        points=probe_samples.points,
        block_types=probe_samples.block_types,
        block_indices=block_indices,
        support_indices=probe_samples.support_indices,
        support_mask=probe_samples.support_mask,
        support_weights=probe_samples.support_weights,
        normals=probe_samples.normals,
        probe_centers=probe_centers,
    )


def _points_outside_atoms(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    *,
    pairwise_element_budget: int,
) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    if atom_coords.shape[0] == 0:
        return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)

    grid_mask = _points_outside_atoms_grid(
        points,
        atom_coords,
        atom_radii,
        pairwise_element_budget=pairwise_element_budget,
    )
    if grid_mask is not None:
        return grid_mask

    effective_pairwise_budget = _effective_pairwise_element_budget(
        points.device,
        pairwise_element_budget,
    )
    rows = max(1, min(points.shape[0], effective_pairwise_budget // atom_coords.shape[0]))
    masks = []
    radii_sq = atom_radii.square().unsqueeze(0)
    for start in range(0, points.shape[0], rows):
        stop = min(start + rows, points.shape[0])
        sq_dists = torch.cdist(points[start:stop], atom_coords).square()
        tol = (
            256
            * torch.finfo(points.dtype).eps
            * torch.maximum(sq_dists, radii_sq).clamp_min(1)
        )
        masks.append((sq_dists >= radii_sq - tol).all(dim=-1))
    return torch.cat(masks, dim=0)


def _points_outside_atoms_grid(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    *,
    pairwise_element_budget: int,
) -> Optional[torch.Tensor]:
    if points.shape[0] < 1024 or atom_coords.shape[0] < 64:
        return None

    max_radius = atom_radii.max().clamp_min(torch.finfo(points.dtype).eps)
    cell_size = float(max_radius.item())
    table = _build_atom_cell_table(atom_coords, cell_size)
    if table is None:
        return None

    cell_span = int(math.ceil(float(max_radius.item()) / cell_size)) + 1
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-cell_span, cell_span + 1, device=points.device),
            torch.arange(-cell_span, cell_span + 1, device=points.device),
            torch.arange(-cell_span, cell_span + 1, device=points.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    query_slots = offsets.shape[0] * table.max_occupancy
    if query_slots >= atom_coords.shape[0]:
        return None

    rows_per_block = max(
        1,
        min(
            points.shape[0],
            _effective_pairwise_element_budget(points.device, pairwise_element_budget)
            // max(1, query_slots),
        ),
    )
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=points.device)
    radii_sq = atom_radii.square()
    block_masks = []

    for start in range(0, points.shape[0], rows_per_block):
        stop = min(start + rows_per_block, points.shape[0])
        block_points = points[start:stop]
        point_cells = torch.floor(block_points / table.cell_size).to(torch.long)
        query_cells = point_cells.unsqueeze(1) + offsets.view(1, -1, 3)
        flat_keys, flat_valid_cells = _atom_cell_query_keys(
            query_cells.reshape(-1, 3),
            table,
        )
        keys = flat_keys.reshape(block_points.shape[0], offsets.shape[0])
        valid_cells = flat_valid_cells.reshape(block_points.shape[0], offsets.shape[0])
        starts_in_table = torch.searchsorted(table.sorted_keys, keys, right=False)
        stops_in_table = torch.searchsorted(table.sorted_keys, keys, right=True)

        positions = starts_in_table.unsqueeze(-1) + slot_offsets.view(1, 1, -1)
        has_atom = valid_cells.unsqueeze(-1) & (positions < stops_in_table.unsqueeze(-1))

        safe_positions = positions.clamp_max(table.sorted_atom_indices.shape[0] - 1)
        atom_indices = table.sorted_atom_indices[safe_positions].reshape(
            block_points.shape[0],
            -1,
        )
        candidate_coords = atom_coords[atom_indices]
        sq_dists = (
            block_points.unsqueeze(1) - candidate_coords
        ).square().sum(dim=-1)
        local_radii_sq = radii_sq[atom_indices]
        tol = (
            256
            * torch.finfo(points.dtype).eps
            * torch.maximum(sq_dists, local_radii_sq).clamp_min(1)
        )
        blocked = (
            has_atom.reshape(block_points.shape[0], -1)
            & (sq_dists < local_radii_sq - tol)
        )
        block_masks.append(~blocked.any(dim=-1))

    return torch.cat(block_masks, dim=0)


def _segment_clearance_mask_line_grid(
    starts: torch.Tensor,
    ends: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii_sq: torch.Tensor,
    *,
    pairwise_element_budget: int,
) -> Optional[torch.Tensor]:
    if starts.shape[0] < _RADIAL_LINE_GRID_MIN_SEGMENTS or atom_coords.shape[0] < 64:
        return None

    max_radius = torch.sqrt(expanded_radii_sq.max().clamp_min(0))
    cell_size = float(max_radius.clamp_min(torch.finfo(starts.dtype).eps).item())
    table = _build_atom_cell_table(atom_coords, cell_size)
    if table is None:
        return None

    segment_dirs = ends - starts
    segment_lens = torch.linalg.norm(segment_dirs, dim=-1)
    max_segment_len = float(segment_lens.max().item())
    step_size = max(
        _RADIAL_LINE_GRID_STEP_FRACTION * cell_size,
        float(torch.finfo(starts.dtype).eps),
    )
    max_steps = int(math.ceil(max_segment_len / step_size)) + 1
    if max_steps <= 1 or max_steps > _RADIAL_LINE_GRID_MAX_STEPS:
        return None

    cell_span = _RADIAL_LINE_GRID_CELL_SPAN
    axes = [
        torch.arange(-cell_span, cell_span + 1, device=starts.device)
        for _ in range(3)
    ]
    offsets = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    query_slots = max_steps * offsets.shape[0] * table.max_occupancy
    if query_slots >= max(1, atom_coords.shape[0] * 4):
        return None

    clear_mask = torch.ones(starts.shape[0], dtype=torch.bool, device=starts.device)
    rows_per_block = max(
        1,
        min(
            starts.shape[0],
            _effective_pairwise_element_budget(starts.device, pairwise_element_budget)
            // max(1, query_slots * 2),
        ),
    )
    sample_offsets = torch.arange(max_steps, dtype=starts.dtype, device=starts.device)
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=starts.device)
    segment_lens_sq = segment_dirs.square().sum(dim=-1).clamp_min(
        torch.finfo(starts.dtype).eps,
    )

    for start in range(0, starts.shape[0], rows_per_block):
        stop = min(start + rows_per_block, starts.shape[0])
        block_starts = starts[start:stop]
        block_dirs = segment_dirs[start:stop]
        block_lens = segment_lens[start:stop].clamp_min(torch.finfo(starts.dtype).eps)
        sample_counts = torch.ceil(block_lens / step_size).to(torch.long) + 1
        valid_steps = sample_offsets.view(1, -1) < sample_counts.view(-1, 1)
        sample_fracs = (sample_offsets.view(1, -1) * step_size) / block_lens.view(-1, 1)
        sample_fracs = sample_fracs.clamp(0, 1)
        line_points = (
            block_starts.unsqueeze(1)
            + sample_fracs.unsqueeze(-1) * block_dirs.unsqueeze(1)
        )
        line_cells = torch.floor(line_points / table.cell_size).to(torch.long)
        query_cells = line_cells.unsqueeze(2) + offsets.view(1, 1, -1, 3)

        flat_keys, flat_valid_cells = _atom_cell_query_keys(
            query_cells.reshape(-1, 3),
            table,
        )
        keys = flat_keys.reshape(block_starts.shape[0], max_steps, offsets.shape[0])
        valid_cells = flat_valid_cells.reshape(
            block_starts.shape[0],
            max_steps,
            offsets.shape[0],
        )
        valid_cells = valid_cells & valid_steps.unsqueeze(-1)
        starts_in_table = torch.searchsorted(table.sorted_keys, keys, right=False)
        stops_in_table = torch.searchsorted(table.sorted_keys, keys, right=True)
        positions = starts_in_table.unsqueeze(-1) + slot_offsets.view(1, 1, 1, -1)
        has_atom = valid_cells.unsqueeze(-1) & (positions < stops_in_table.unsqueeze(-1))

        safe_positions = positions.clamp_max(table.sorted_atom_indices.shape[0] - 1)
        atom_indices = table.sorted_atom_indices[safe_positions].reshape(
            block_starts.shape[0],
            -1,
        )
        candidate_coords = atom_coords[atom_indices]
        start_to_atoms = candidate_coords - block_starts.unsqueeze(1)
        nearest_params = (
            start_to_atoms * block_dirs.unsqueeze(1)
        ).sum(dim=-1) / segment_lens_sq[start:stop].unsqueeze(-1)
        nearest_params = nearest_params.clamp(0, 1)
        closest_points = (
            block_starts.unsqueeze(1)
            + nearest_params.unsqueeze(-1) * block_dirs.unsqueeze(1)
        )
        closest_dists_sq = (closest_points - candidate_coords).square().sum(dim=-1)
        local_radii_sq = expanded_radii_sq[atom_indices]
        tol = (
            256
            * torch.finfo(starts.dtype).eps
            * torch.maximum(closest_dists_sq, local_radii_sq).clamp_min(1)
        )
        blocked = (
            has_atom.reshape(block_starts.shape[0], -1)
            & (closest_dists_sq < local_radii_sq - tol)
        )
        clear_mask[start:stop] = ~blocked.any(dim=-1)

    return clear_mask


def _fast_radial_accessible(
    probe_centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    expanded_radii_sq: Optional[torch.Tensor] = None,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    if probe_centers.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=probe_centers.device)
    if atom_coords.shape[0] < 4:
        return torch.ones((probe_centers.shape[0],), dtype=torch.bool, device=probe_centers.device)

    molecule_center = atom_coords.mean(dim=0, keepdim=True)
    exterior_dirs = probe_centers - molecule_center
    dir_norms = torch.linalg.norm(exterior_dirs, dim=-1, keepdim=True)
    fallback = torch.zeros_like(exterior_dirs)
    fallback[:, 0] = 1
    exterior_dirs = torch.where(
        dir_norms > torch.finfo(probe_centers.dtype).eps,
        exterior_dirs / dir_norms.clamp_min(torch.finfo(probe_centers.dtype).eps),
        fallback,
    )
    exterior_radius = (
        torch.linalg.norm(atom_coords - molecule_center, dim=-1).max()
        + expanded_radii.max()
        + 4.0
    )
    exterior_points = molecule_center + exterior_dirs * exterior_radius
    if expanded_radii_sq is None:
        expanded_radii_sq = expanded_radii.square()
    line_grid_clear = _segment_clearance_mask_line_grid(
        probe_centers,
        exterior_points,
        atom_coords,
        expanded_radii_sq,
        pairwise_element_budget=pairwise_element_budget,
    )
    if line_grid_clear is not None:
        return line_grid_clear
    return _segment_clearance_mask(
        probe_centers,
        exterior_points,
        atom_coords,
        expanded_radii_sq,
    )


def _filter_samples(
    samples: AnalyticSamples,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    exact_accessibility: bool,
    grid_spacing: Optional[float],
    max_grid_points: int,
    pairwise_element_budget: int,
    check_points_outside_atoms: bool = False,
    trust_probe_blocks: bool = False,
) -> AnalyticSamples:
    if samples.points.shape[0] == 0:
        return samples
    return _index_samples(
        samples,
        _sample_filter_mask(
            samples,
            atom_coords,
            atom_radii,
            probe_radius,
            exact_accessibility=exact_accessibility,
            grid_spacing=grid_spacing,
            max_grid_points=max_grid_points,
            pairwise_element_budget=pairwise_element_budget,
            check_points_outside_atoms=check_points_outside_atoms,
            trust_probe_blocks=trust_probe_blocks,
        ),
    )


def _sample_filter_mask(
    samples: AnalyticSamples,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    exact_accessibility: bool,
    grid_spacing: Optional[float],
    max_grid_points: int,
    pairwise_element_budget: int,
    check_points_outside_atoms: bool = False,
    trust_probe_blocks: bool = False,
) -> torch.Tensor:
    probe_centers = samples.probe_centers
    if probe_centers is None:
        raise ValueError("tiled analytic samples require probe centers")
    return _filter_geometry_mask(
        samples.points,
        probe_centers,
        samples.block_types,
        atom_coords,
        atom_radii,
        probe_radius,
        exact_accessibility=exact_accessibility,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=pairwise_element_budget,
        check_points_outside_atoms=check_points_outside_atoms,
        trust_probe_blocks=trust_probe_blocks,
    )


def _filter_geometry_mask(
    points: torch.Tensor,
    probe_centers: torch.Tensor,
    block_types: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    exact_accessibility: bool,
    grid_spacing: Optional[float],
    max_grid_points: int,
    pairwise_element_budget: int,
    check_points_outside_atoms: bool = False,
    trust_probe_blocks: bool = False,
) -> torch.Tensor:
    expanded_radii = atom_radii + float(probe_radius)
    expanded_radii_sq = expanded_radii.square()
    with torch.no_grad():
        finite = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(probe_centers).all(dim=-1)
        )
        valid = finite.clone()
        if exact_accessibility:
            needs_center_filter = valid
            if trust_probe_blocks:
                needs_center_filter = needs_center_filter & (
                    block_types != PROBE_BLOCK_TYPE
                )
            feasible = _centers_feasible_against_all_atoms(
                probe_centers,
                atom_coords,
                expanded_radii_sq,
            )
            valid = valid & (~needs_center_filter | feasible)

        if check_points_outside_atoms:
            outside = _points_outside_atoms(
                points,
                atom_coords,
                atom_radii,
                pairwise_element_budget=pairwise_element_budget,
            )
            valid = valid & outside

        needs_accessibility = valid
        if trust_probe_blocks:
            needs_accessibility = needs_accessibility & (block_types != PROBE_BLOCK_TYPE)
        candidate_indices = needs_accessibility.nonzero(as_tuple=False).reshape(-1)
        if candidate_indices.numel() > 0:
            if exact_accessibility:
                accessible = _probe_centers_accessible_from_exterior(
                    probe_centers,
                    atom_coords,
                    atom_radii,
                    float(probe_radius),
                    valid_center_mask=needs_accessibility,
                    grid_spacing=grid_spacing,
                    max_grid_points=int(max_grid_points),
                    assume_centers_feasible=True,
                )
                local_accessible = accessible[candidate_indices]
            else:
                local_accessible = _fast_radial_accessible(
                    probe_centers[candidate_indices],
                    atom_coords,
                    expanded_radii,
                    expanded_radii_sq,
                    pairwise_element_budget=int(pairwise_element_budget),
                )
            accessible = torch.zeros_like(valid).scatter(
                0,
                candidate_indices,
                local_accessible,
            )
            valid = valid & (~needs_accessibility | accessible)

    return valid


def _filter_sample_groups(
    samples: list[AnalyticSamples],
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    exact_accessibility: bool,
    grid_spacing: Optional[float],
    max_grid_points: int,
    pairwise_element_budget: int,
    check_points_outside_atoms: bool = False,
    trust_probe_blocks: bool = False,
) -> list[AnalyticSamples]:
    nonempty = [sample for sample in samples if sample.points.shape[0] > 0]
    if not nonempty:
        return samples

    for sample in nonempty:
        if sample.probe_centers is None:
            raise ValueError("tiled analytic samples require probe centers")
    points = torch.cat([sample.points for sample in nonempty], dim=0)
    probe_centers = torch.cat([sample.probe_centers for sample in nonempty], dim=0)
    block_types = torch.cat([sample.block_types for sample in nonempty], dim=0)
    keep = _filter_geometry_mask(
        points,
        probe_centers,
        block_types,
        atom_coords,
        atom_radii,
        probe_radius,
        exact_accessibility=exact_accessibility,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=pairwise_element_budget,
        check_points_outside_atoms=check_points_outside_atoms,
        trust_probe_blocks=trust_probe_blocks,
    )

    filtered: list[AnalyticSamples] = []
    offset = 0
    for sample in samples:
        count = int(sample.points.shape[0])
        if count == 0:
            filtered.append(sample)
            continue
        filtered.append(_index_samples(sample, keep[offset : offset + count]))
        offset += count
    return filtered


def _index_samples(samples: AnalyticSamples, mask_or_indices: torch.Tensor) -> AnalyticSamples:
    return AnalyticSamples(
        points=samples.points[mask_or_indices],
        block_types=samples.block_types[mask_or_indices],
        block_indices=samples.block_indices[mask_or_indices],
        support_indices=samples.support_indices[mask_or_indices],
        support_mask=samples.support_mask[mask_or_indices],
        support_weights=samples.support_weights[mask_or_indices],
        normals=None if samples.normals is None else samples.normals[mask_or_indices],
        atom_weights=None,
        blocks=None,
        probe_centers=(
            None
            if samples.probe_centers is None
            else samples.probe_centers[mask_or_indices]
        ),
    )


def _concat_tiled_samples(
    samples: list[AnalyticSamples],
    context,
) -> AnalyticSamples:
    nonempty = [sample for sample in samples if sample.points.shape[0] > 0]
    if not nonempty:
        return _empty_samples(context, max_support=2)

    max_support = max(sample.support_indices.shape[1] for sample in nonempty)
    points = torch.cat([sample.points for sample in nonempty], dim=0)
    block_types = torch.cat([sample.block_types for sample in nonempty], dim=0)
    block_indices = torch.cat([sample.block_indices for sample in nonempty], dim=0)
    support_indices = torch.cat(
        [_pad_columns(sample.support_indices, max_support, fill_value=-1) for sample in nonempty],
        dim=0,
    )
    support_mask = torch.cat(
        [_pad_columns(sample.support_mask, max_support, fill_value=False) for sample in nonempty],
        dim=0,
    )
    support_weights = torch.cat(
        [_pad_columns(sample.support_weights, max_support, fill_value=0.0) for sample in nonempty],
        dim=0,
    )
    normals = (
        torch.cat([sample.normals for sample in nonempty if sample.normals is not None], dim=0)
        if all(sample.normals is not None for sample in nonempty)
        else None
    )
    probe_centers = torch.cat(
        [
            sample.probe_centers
            if sample.probe_centers is not None
            else torch.empty((0, 3), dtype=context.dtype, device=context.device)
            for sample in nonempty
        ],
        dim=0,
    )
    return AnalyticSamples(
        points=points,
        block_types=block_types,
        block_indices=block_indices,
        support_indices=support_indices,
        support_mask=support_mask,
        support_weights=support_weights,
        normals=normals,
        probe_centers=probe_centers,
    )


def _geometry_only_samples(
    points: torch.Tensor,
    normals: Optional[torch.Tensor],
    context,
) -> AnalyticSamples:
    row_count = points.shape[0]
    support_indices = torch.empty(
        (row_count, 0),
        dtype=torch.long,
        device=context.device,
    )
    return AnalyticSamples(
        points=points,
        block_types=torch.empty((row_count,), dtype=torch.long, device=context.device),
        block_indices=torch.empty((row_count,), dtype=torch.long, device=context.device),
        support_indices=support_indices,
        support_mask=torch.empty((row_count, 0), dtype=torch.bool, device=context.device),
        support_weights=torch.empty((row_count, 0), dtype=context.dtype, device=context.device),
        normals=normals,
        probe_centers=None,
    )


def _concat_tiled_geometry_samples(
    samples: list[AnalyticSamples],
    context,
    *,
    dedup_tolerance: float,
    include_normals: bool,
) -> AnalyticSamples:
    nonempty = [sample for sample in samples if sample.points.shape[0] > 0]
    if not nonempty:
        empty_points = torch.empty((0, 3), dtype=context.dtype, device=context.device)
        empty_normals = (
            torch.empty((0, 3), dtype=context.dtype, device=context.device)
            if include_normals
            else None
        )
        return _geometry_only_samples(empty_points, empty_normals, context)

    points = torch.cat([sample.points for sample in nonempty], dim=0)
    normals = None
    if include_normals and all(sample.normals is not None for sample in nonempty):
        normals = torch.cat([sample.normals for sample in nonempty], dim=0)
    keep = _deduplicate_centers(points, float(dedup_tolerance))
    points = points[keep]
    if normals is not None:
        normals = normals[keep]
    return _geometry_only_samples(points, normals, context)


def _sample_single_tile_analytic_samples(
    context,
    *,
    point_area: float,
    atom_density_scale: float,
    pair_density_scale: float,
    probe_density_scale: float,
    dedup_tolerance: float,
    include_atom_weights: bool,
    include_normals: bool,
    keep_metadata: bool,
    grid_spacing: Optional[float],
    max_grid_points: int,
    max_probe_triples: Optional[int],
) -> AnalyticSamples:
    blocks, analytic_context = build_analytic_blocks(
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        max_probe_support_atoms=_DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
        dedup_tolerance=float(dedup_tolerance),
        max_probe_triples=max_probe_triples,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    atom_samples = (
        _sample_atom_blocks(
            analytic_context,
            blocks.atom_indices,
            point_area=float(point_area),
            oversample_factor=float(atom_density_scale),
            include_normals=include_normals,
        )
        if atom_density_scale > 0
        else _empty_samples(analytic_context, max_support=1)
    )
    pair_samples = (
        _sample_pair_blocks(
            analytic_context,
            blocks.pair_indices,
            point_area=float(point_area),
            oversample_factor=float(pair_density_scale),
            include_normals=include_normals,
        )
        if pair_density_scale > 0
        else _empty_samples(analytic_context, max_support=2)
    )
    probe_samples = (
        _sample_probe_blocks(
            analytic_context,
            blocks,
            point_area=float(point_area),
            oversample_factor=1.0,
            probe_density_scale=float(probe_density_scale),
            include_normals=include_normals,
        )
        if probe_density_scale > 0
        else _empty_samples(
            analytic_context,
            max_support=max(1, int(blocks.probe_support_indices.shape[1])),
        )
    )
    samples = _concat_samples(
        (atom_samples, pair_samples, probe_samples),
        context=analytic_context,
        include_atom_weights=include_atom_weights,
        blocks=blocks,
    )
    if keep_metadata:
        return samples
    return _geometry_only_samples(samples.points, samples.normals, analytic_context)


def _deduplicate_samples(samples: AnalyticSamples, tolerance: float) -> AnalyticSamples:
    if samples.points.shape[0] == 0:
        return samples
    keep = _deduplicate_centers(samples.points, tolerance)
    return _index_samples(samples, keep)


def _with_atom_weights(
    samples: AnalyticSamples,
    atom_weights: Optional[torch.Tensor],
) -> AnalyticSamples:
    return AnalyticSamples(
        points=samples.points,
        block_types=samples.block_types,
        block_indices=samples.block_indices,
        support_indices=samples.support_indices,
        support_mask=samples.support_mask,
        support_weights=samples.support_weights,
        normals=samples.normals,
        atom_weights=atom_weights,
        blocks=samples.blocks,
        probe_centers=samples.probe_centers,
    )


def _sample_tiled_analytic_samples(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float = 1.0,
    tile_size: Union[float, str] = _DEFAULT_TILE_SIZE,
    tile_overlap: Union[float, str] = _DEFAULT_TILE_OVERLAP,
    atom_density_scale: float = _DEFAULT_ATOM_DENSITY_SCALE,
    pair_density_scale: float = _DEFAULT_PAIR_DENSITY_SCALE,
    probe_density_scale: float = _DEFAULT_PROBE_DENSITY_SCALE,
    dedup_tolerance: float = _DEFAULT_DEDUP_TOLERANCE,
    exact_accessibility: bool = False,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
    max_probe_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
    include_atom_weights: bool = False,
    include_normals: bool = False,
    keep_metadata: bool = True,
) -> AnalyticSamples:
    """Return tiled analytic samples with sparse support metadata."""

    if point_area <= 0:
        raise ValueError("point_area must be positive")
    if dedup_tolerance <= 0:
        raise ValueError("dedup_tolerance must be positive")
    if max_grid_points <= 0:
        raise ValueError("max_grid_points must be positive")
    if max_probe_triples is not None and max_probe_triples <= 0:
        raise ValueError("max_probe_triples must be positive or None")
    if pairwise_element_budget <= 0:
        raise ValueError("pairwise_element_budget must be positive")

    atom_density_scale = _validate_scale("atom_density_scale", atom_density_scale)
    pair_density_scale = _validate_scale(
        "pair_density_scale",
        pair_density_scale,
        allow_zero=True,
    )
    probe_density_scale = _validate_scale(
        "probe_density_scale",
        probe_density_scale,
        allow_zero=True,
    )
    keep_metadata = bool(keep_metadata or include_atom_weights)

    context = _build_exterior_context(
        atom_coords,
        atom_radii,
        probe_radius,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    pairwise_element_budget = _effective_pairwise_element_budget(
        context.device,
        int(pairwise_element_budget),
    )
    if context.num_atoms == 0:
        empty = (
            _empty_samples(context, max_support=3)
            if keep_metadata
            else _geometry_only_samples(
                torch.empty((0, 3), dtype=context.dtype, device=context.device),
                torch.empty((0, 3), dtype=context.dtype, device=context.device)
                if include_normals
                else None,
                context,
            )
        )
        if not include_atom_weights:
            return empty
        return _with_atom_weights(
            empty,
            torch.empty((0, 0), dtype=context.dtype, device=context.device),
        )

    needs_pair_graph = pair_density_scale > 0 or probe_density_scale > 0
    pair_indices: Optional[torch.Tensor] = None
    with torch.no_grad():
        if needs_pair_graph and isinstance(tile_size, str) and tile_size == "auto":
            pair_indices = _candidate_pair_indices(context)
        resolved_tile_size, resolved_tile_overlap = _resolve_memory_aware_tile_parameters(
            context,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            point_area=float(point_area),
            pair_density_scale=pair_density_scale,
            pair_indices=pair_indices,
        )
        grid = _build_tile_grid(
            context.atom_coords.detach(),
            context.atom_radii.detach(),
            context.probe_radius,
            tile_size=resolved_tile_size,
            tile_overlap=resolved_tile_overlap,
        )
        atom_tile_mask = _tile_atom_intersection_mask(
            context.atom_coords.detach(),
            context.atom_radii.detach(),
            context.probe_radius,
            grid,
        )

    if int(grid.indices.shape[0]) == 1:
        return _sample_single_tile_analytic_samples(
            context,
            point_area=float(point_area),
            atom_density_scale=atom_density_scale,
            pair_density_scale=pair_density_scale,
            probe_density_scale=probe_density_scale,
            dedup_tolerance=float(dedup_tolerance),
            include_atom_weights=include_atom_weights,
            include_normals=include_normals,
            keep_metadata=keep_metadata,
            grid_spacing=grid_spacing,
            max_grid_points=max_grid_points,
            max_probe_triples=None if max_probe_triples is None else int(max_probe_triples),
        )

    atom_indices = torch.arange(context.num_atoms, dtype=torch.long, device=context.device)
    use_tile_local_candidates = int(grid.indices.shape[0]) > 1
    if needs_pair_graph and pair_indices is None:
        with torch.no_grad():
            pair_indices = _candidate_pair_indices(context)
    contact = _sample_contact_candidates(
        context.atom_coords,
        context.atom_radii,
        atom_indices,
        context.probe_radius,
        point_area=float(point_area),
        atom_density_scale=atom_density_scale,
        include_normals=include_normals,
    )
    contact = _apply_tile_locality(
        contact,
        grid=grid,
        atom_tile_mask=atom_tile_mask,
        tile_size=resolved_tile_size,
    )

    if pair_density_scale > 0 and use_tile_local_candidates:
        pair = _sample_tile_local_pair_candidates(
            context.atom_coords,
            context.atom_radii,
            atom_tile_mask,
            grid,
            context.probe_radius,
            point_area=float(point_area),
            pair_density_scale=pair_density_scale,
            tile_size=resolved_tile_size,
            include_normals=include_normals,
            pair_indices=pair_indices,
        )
    else:
        pair = (
            _sample_pair_candidates(
                context.atom_coords,
                context.atom_radii,
                atom_indices,
                context.probe_radius,
                point_area=float(point_area),
                pair_density_scale=pair_density_scale,
                include_normals=include_normals,
            )
            if pair_density_scale > 0
            else _empty_samples(context, max_support=2)
        )
        pair = _apply_tile_locality(
            pair,
            grid=grid,
            atom_tile_mask=atom_tile_mask,
            tile_size=resolved_tile_size,
        )

    if probe_density_scale > 0 and use_tile_local_candidates:
        probe = _sample_tile_local_probe_candidates(
            context.atom_coords,
            context.atom_radii,
            atom_tile_mask,
            grid,
            context.probe_radius,
            point_area=float(point_area),
            probe_density_scale=float(probe_density_scale),
            tile_size=resolved_tile_size,
            max_grid_points=max_grid_points,
            max_probe_triples=None if max_probe_triples is None else int(max_probe_triples),
            include_normals=include_normals,
            pair_indices=pair_indices,
        )
    else:
        probe = (
            _sample_probe_candidates(
                context.atom_coords,
                context.atom_radii,
                atom_indices,
                context.probe_radius,
                point_area=float(point_area),
                probe_density_scale=float(probe_density_scale),
                max_grid_points=max_grid_points,
                max_probe_triples=None if max_probe_triples is None else int(max_probe_triples),
                include_normals=include_normals,
            )
            if probe_density_scale > 0
            else _empty_samples(context, max_support=3)
        )
        probe = _apply_tile_locality(
            probe,
            grid=grid,
            atom_tile_mask=atom_tile_mask,
            tile_size=resolved_tile_size,
        )
    contact, pair, probe = _filter_sample_groups(
        [contact, pair, probe],
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        exact_accessibility=bool(exact_accessibility),
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=int(pairwise_element_budget),
        trust_probe_blocks=resolved_tile_overlap >= float(context.probe_radius),
    )

    if keep_metadata:
        samples = _concat_tiled_samples([contact, pair, probe], context)
        samples = _deduplicate_samples(samples, float(dedup_tolerance))
    else:
        samples = _concat_tiled_geometry_samples(
            [contact, pair, probe],
            context,
            dedup_tolerance=float(dedup_tolerance),
            include_normals=include_normals,
        )
    if include_atom_weights:
        samples = _with_atom_weights(
            samples,
            _dense_atom_weights(
                samples.support_indices,
                samples.support_mask,
                samples.support_weights,
                num_atoms=context.num_atoms,
            ),
        )
    return samples


def sample_tiled_analytic_points(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float = 1.0,
    tile_size: Union[float, str] = _DEFAULT_TILE_SIZE,
    tile_overlap: Union[float, str] = _DEFAULT_TILE_OVERLAP,
    atom_density_scale: float = _DEFAULT_ATOM_DENSITY_SCALE,
    pair_density_scale: float = _DEFAULT_PAIR_DENSITY_SCALE,
    probe_density_scale: float = _DEFAULT_PROBE_DENSITY_SCALE,
    dedup_tolerance: float = _DEFAULT_DEDUP_TOLERANCE,
    exact_accessibility: bool = False,
    include_atom_features: bool = False,
    include_normals: bool = False,
    include_adjacency: bool = False,
    adjacency_weight: AdjacencyWeightMode = "euclidean",
    adjacency_neighbors: int = 6,
    adjacency_candidate_neighbors: Optional[int] = None,
    adjacency_prune_redundant: bool = False,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
    max_probe_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> SamplerOutput:
    """Sample SES points with a tiled, speed-first analytic approximation.

    The sampler partitions the molecule into overlapping 3D tiles, samples
    local atom-contact and optional pair-torus candidates, keeps samples whose
    SES point belongs to the tile core, and then applies global molecule
    filters.  The default settings are intentionally approximate: they target
    fast exterior point clouds rather than exact analytic block coverage.
    When the resolved grid contains a single tile, the sampler uses the same
    analytic block pipeline as :func:`ses.analytic.sample_analytic_points`.
    Set ``include_normals=True`` to also return outward SES normals aligned with
    the sampled points.
    Set ``include_adjacency=True`` to receive a sparse symmetric SES surface
    adjacency matrix with shape ``(num_points, num_points)``.  By default the
    graph builds a compact topology-aware candidate pool before normal/tangent
    filtering.
    """

    need_sample_normals = include_normals or include_adjacency
    need_sample_metadata = include_atom_features or include_adjacency
    samples = _sample_tiled_analytic_samples(
        atom_coords,
        atom_radii,
        probe_radius,
        point_area=point_area,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        atom_density_scale=atom_density_scale,
        pair_density_scale=pair_density_scale,
        probe_density_scale=probe_density_scale,
        dedup_tolerance=dedup_tolerance,
        exact_accessibility=exact_accessibility,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        max_probe_triples=max_probe_triples,
        pairwise_element_budget=pairwise_element_budget,
        include_normals=need_sample_normals,
        keep_metadata=need_sample_metadata,
    )
    graph_normals = samples.normals
    normals = graph_normals if include_normals else None
    adjacency = (
        build_surface_adjacency(
            samples.points,
            graph_normals,
            support_indices=samples.support_indices,
            support_mask=samples.support_mask,
            block_types=samples.block_types,
            block_indices=samples.block_indices,
            weight_mode=adjacency_weight,
            neighbors=adjacency_neighbors,
            candidate_neighbors=adjacency_candidate_neighbors,
            prune_redundant_edges=adjacency_prune_redundant,
            pairwise_element_budget=pairwise_element_budget,
        )
        if include_adjacency
        else None
    )
    if not include_atom_features:
        return _format_sample_outputs(
            samples.points,
            normals=normals,
            adjacency=adjacency,
        )

    atom_features = _dense_atom_features(
        samples.support_indices,
        samples.support_mask,
        num_atoms=int(atom_coords.shape[0]),
        dtype=samples.points.dtype,
    )
    return _format_sample_outputs(
        samples.points,
        atom_features=atom_features,
        normals=normals,
        adjacency=adjacency,
    )


__all__ = ["sample_tiled_analytic_points"]
