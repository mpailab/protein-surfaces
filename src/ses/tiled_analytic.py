"""Speed-first tiled analytic SES point sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from ._outputs import SamplerOutput, _format_sample_outputs
from .analytic import (
    ATOM_BLOCK_TYPE,
    AnalyticBlocks,
    PAIR_BLOCK_TYPE,
    PROBE_BLOCK_TYPE,
    AnalyticSamples,
    _build_exterior_context,
    _candidate_pair_indices,
    _deduplicate_centers,
    _dense_atom_features,
    _dense_atom_weights,
    _empty_samples,
    _extract_probe_blocks,
    _packed_fibonacci_directions,
    _packed_pair_parameters,
    _pad_columns,
    _pair_points_from_params,
    _pair_sample_counts,
    _sample_probe_blocks,
)
from .projection import (
    _atom_cell_query_keys,
    _build_atom_cell_table,
    _centers_feasible_against_all_atoms,
    _probe_centers_accessible_from_exterior,
    _segment_clearance_mask,
)


_DEFAULT_TILE_SIZE = 64.0
_DEFAULT_TILE_OVERLAP = 4.0
_DEFAULT_DEDUP_TOLERANCE = 0.05
_DEFAULT_ATOM_DENSITY_SCALE = 1.55
_DEFAULT_PAIR_DENSITY_SCALE = 1.55
_DEFAULT_PROBE_DENSITY_SCALE = 1.55
_DEFAULT_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_DEFAULT_MAX_GRID_POINTS = 500_000
_DEFAULT_MAX_PROBE_TRIPLES = 5_000_000
_AUTO_TILE_ATOM_THRESHOLD = 2_000
_AUTO_SMALL_TILE_SIZE = 64.0
_AUTO_SMALL_TILE_OVERLAP = 6.0
_AUTO_LARGE_TILE_SIZE = 24.0
_AUTO_LARGE_TILE_OVERLAP = 3.0


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


def _sample_contact_candidates(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    local_indices: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float,
    atom_density_scale: float,
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
    normals = directions
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
        pair_rows, theta_values, arc_fracs = _packed_pair_parameters(
            counts,
            local_context,
            pair_indices,
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
    normals = (probe_centers - points) / float(probe_radius)
    selected_pairs = local_indices[selected_local_pairs]
    support_weights = torch.stack((first_weights, second_weights), dim=-1)
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
) -> AnalyticSamples:
    global_context = _build_exterior_context(atom_coords, atom_radii, probe_radius)
    empty = _empty_samples(global_context, max_support=3)
    if local_indices.numel() < 3 or probe_density_scale <= 0:
        return empty

    local_context = _build_exterior_context(
        atom_coords[local_indices],
        atom_radii[local_indices],
        probe_radius,
        max_grid_points=max_grid_points,
    )
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
    return AnalyticSamples(
        points=probe_samples.points,
        block_types=probe_samples.block_types,
        block_indices=probe_samples.block_indices,
        support_indices=global_supports,
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

    rows = max(1, min(points.shape[0], int(pairwise_element_budget) // atom_coords.shape[0]))
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

    outside = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    rows_per_block = max(
        1,
        min(points.shape[0], int(pairwise_element_budget) // max(1, query_slots)),
    )
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=points.device)
    radii_sq = atom_radii.square()

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
        if not bool(has_atom.any().item()):
            continue

        point_rows, cell_rows, slot_rows = has_atom.nonzero(as_tuple=True)
        atom_indices = table.sorted_atom_indices[
            positions[point_rows, cell_rows, slot_rows]
        ]
        sq_dists = (
            block_points[point_rows] - atom_coords[atom_indices]
        ).square().sum(dim=-1)
        local_radii_sq = radii_sq[atom_indices]
        tol = (
            256
            * torch.finfo(points.dtype).eps
            * torch.maximum(sq_dists, local_radii_sq).clamp_min(1)
        )
        blocked = sq_dists < local_radii_sq - tol
        if bool(blocked.any().item()):
            block_outside = outside[start:stop]
            block_outside[point_rows[blocked]] = False
            outside[start:stop] = block_outside

    return outside


def _fast_radial_accessible(
    probe_centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
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
    return _segment_clearance_mask(
        probe_centers,
        exterior_points,
        atom_coords,
        expanded_radii.square(),
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
    probe_centers = samples.probe_centers
    if probe_centers is None:
        raise ValueError("tiled analytic samples require probe centers")

    expanded_radii = atom_radii + float(probe_radius)
    with torch.no_grad():
        finite = torch.isfinite(samples.points).all(dim=-1) & torch.isfinite(probe_centers).all(dim=-1)
        valid = finite.clone()
        if bool(valid.any().item()):
            needs_center_filter = valid
            if trust_probe_blocks:
                needs_center_filter = needs_center_filter & (samples.block_types != PROBE_BLOCK_TYPE)
            candidate_indices = needs_center_filter.nonzero(as_tuple=False).reshape(-1)
        else:
            candidate_indices = torch.empty((0,), dtype=torch.long, device=samples.points.device)
        if candidate_indices.numel() > 0:
            feasible = _centers_feasible_against_all_atoms(
                probe_centers[candidate_indices],
                atom_coords,
                expanded_radii.square(),
            )
            valid[candidate_indices] = feasible

        if check_points_outside_atoms and bool(valid.any().item()):
            candidate_indices = valid.nonzero(as_tuple=False).reshape(-1)
            outside = _points_outside_atoms(
                samples.points[candidate_indices],
                atom_coords,
                atom_radii,
                pairwise_element_budget=pairwise_element_budget,
            )
            valid[candidate_indices] = outside

        if bool(valid.any().item()):
            needs_accessibility = valid
            if trust_probe_blocks:
                needs_accessibility = needs_accessibility & (samples.block_types != PROBE_BLOCK_TYPE)
            candidate_indices = needs_accessibility.nonzero(as_tuple=False).reshape(-1)
        else:
            candidate_indices = torch.empty((0,), dtype=torch.long, device=samples.points.device)
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
                )
            valid[candidate_indices] = local_accessible

    return _index_samples(samples, valid)


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
        probe_centers=None if samples.probe_centers is None else samples.probe_centers[mask_or_indices],
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

    context = _build_exterior_context(
        atom_coords,
        atom_radii,
        probe_radius,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    if context.num_atoms == 0:
        empty = _empty_samples(context, max_support=3)
        if not include_atom_weights:
            return empty
        return _with_atom_weights(
            empty,
            torch.empty((0, 0), dtype=context.dtype, device=context.device),
        )

    resolved_tile_size, resolved_tile_overlap = _resolve_tile_parameters(
        context.num_atoms,
        tile_size,
        tile_overlap,
    )

    grid = _build_tile_grid(
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        tile_size=resolved_tile_size,
        tile_overlap=resolved_tile_overlap,
    )
    atom_tile_mask = _tile_atom_intersection_mask(
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        grid,
    )

    atom_indices = torch.arange(context.num_atoms, dtype=torch.long, device=context.device)
    contact = _sample_contact_candidates(
        context.atom_coords,
        context.atom_radii,
        atom_indices,
        context.probe_radius,
        point_area=float(point_area),
        atom_density_scale=atom_density_scale,
    )
    contact = _apply_tile_locality(
        contact,
        grid=grid,
        atom_tile_mask=atom_tile_mask,
        tile_size=resolved_tile_size,
    )
    contact = _filter_samples(
        contact,
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        exact_accessibility=bool(exact_accessibility),
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=int(pairwise_element_budget),
    )

    pair = (
        _sample_pair_candidates(
            context.atom_coords,
            context.atom_radii,
            atom_indices,
            context.probe_radius,
            point_area=float(point_area),
            pair_density_scale=pair_density_scale,
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
    pair = _filter_samples(
        pair,
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        exact_accessibility=bool(exact_accessibility),
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=int(pairwise_element_budget),
    )

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
    probe = _filter_samples(
        probe,
        context.atom_coords,
        context.atom_radii,
        context.probe_radius,
        exact_accessibility=bool(exact_accessibility),
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
        pairwise_element_budget=int(pairwise_element_budget),
        trust_probe_blocks=resolved_tile_overlap >= float(context.probe_radius),
    )

    samples = _concat_tiled_samples([contact, pair, probe], context)
    samples = _deduplicate_samples(samples, float(dedup_tolerance))
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
    Set ``include_normals=True`` to also return outward SES normals aligned with
    the sampled points.
    """

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
    )
    normals = samples.normals if include_normals else None
    if not include_atom_features:
        return _format_sample_outputs(samples.points, normals=normals)

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
    )


__all__ = ["sample_tiled_analytic_points"]
