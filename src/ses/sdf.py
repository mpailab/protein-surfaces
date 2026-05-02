"""SDF level-set solvent-excluded surface sampling.

This module implements a lightweight PyTorch variant of the surface generation
idea used by dMaSIF: start from an over-sampled cloud around atoms, iteratively
move candidates onto a smooth distance-function level set, reject inaccessible
probe-center samples and shift surviving centers back by one probe radius to
obtain SES points.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from .projection import (
    _centers_feasible_against_all_atoms,
    _probe_centers_accessible_from_exterior,
    sample_atom_points,
)


_DEFAULT_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_DEFAULT_MAX_GRID_POINTS = 2_000_000


def _prepare_sdf_inputs(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    m: int,
    probe_radius: float,
    smoothness: float,
    iterations: int,
    level_tolerance: float,
    subsample_spacing: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and dtype-normalize public SDF sampler inputs."""

    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError("m must be an integer")
    if m < 0:
        raise ValueError("m must be non-negative")
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if probe_radius <= 0:
        raise ValueError("probe_radius must be positive")
    if smoothness <= 0:
        raise ValueError("smoothness must be positive")
    if level_tolerance < 0:
        raise ValueError("level_tolerance must be non-negative")
    if subsample_spacing is not None and subsample_spacing <= 0:
        raise ValueError("subsample_spacing must be positive")
    if atom_coords.ndim != 2 or atom_coords.shape[-1] != 3:
        raise ValueError("atom_coords must have shape (n, 3)")

    coord_dtype = atom_coords.dtype if atom_coords.is_floating_point() else torch.float32
    radii_dtype = atom_radii.dtype if atom_radii.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(coord_dtype, radii_dtype)
    common_device = atom_coords.device

    coords = atom_coords.to(dtype=common_dtype, device=common_device)
    radii = atom_radii.reshape(-1).to(dtype=common_dtype, device=common_device)
    if radii.numel() != coords.shape[0]:
        raise ValueError("atom_radii must contain one radius per atom")
    if bool((radii < 0).any().item()):
        raise ValueError("atom_radii must be non-negative")
    return coords, radii


def _max_sdf_rows(num_atoms: int, pairwise_element_budget: int) -> int:
    """Return a chunk size that keeps point-atom distance matrices bounded."""

    if num_atoms <= 0:
        return 1
    return max(1, int(pairwise_element_budget) // max(1, int(num_atoms)))


def _normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    """Normalize row vectors with a deterministic fallback for zero rows."""

    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    eps = torch.finfo(vectors.dtype).eps
    fallback = torch.zeros_like(vectors)
    fallback[..., 0] = 1
    return torch.where(norms > eps, vectors / norms.clamp_min(eps), fallback)


def _soft_expanded_atom_sdf(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    smoothness: float,
) -> torch.Tensor:
    """Smooth minimum of signed distances to expanded atom spheres."""

    center_distances = torch.linalg.norm(
        points.unsqueeze(1) - atom_coords.unsqueeze(0),
        dim=-1,
    )
    signed_distances = center_distances - expanded_atom_radii.unsqueeze(0)
    return -float(smoothness) * torch.logsumexp(
        -signed_distances / float(smoothness),
        dim=-1,
    )


def _sdf_values(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    smoothness: float,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Evaluate smooth SDF values in chunks."""

    if points.shape[0] == 0:
        return torch.empty((0,), dtype=points.dtype, device=points.device)

    rows = _max_sdf_rows(atom_coords.shape[0], pairwise_element_budget)
    values = []
    for start in range(0, points.shape[0], rows):
        stop = min(start + rows, points.shape[0])
        values.append(
            _soft_expanded_atom_sdf(
                points[start:stop],
                atom_coords,
                expanded_atom_radii,
                smoothness,
            )
        )
    return torch.cat(values, dim=0)


def _sdf_values_and_normals(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    smoothness: float,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate smooth SDF values and outward normals in chunks."""

    if points.shape[0] == 0:
        empty_values = torch.empty((0,), dtype=points.dtype, device=points.device)
        empty_normals = torch.empty((0, 3), dtype=points.dtype, device=points.device)
        return empty_values, empty_normals

    rows = _max_sdf_rows(atom_coords.shape[0], pairwise_element_budget)
    values = []
    normals = []
    with torch.enable_grad():
        for start in range(0, points.shape[0], rows):
            stop = min(start + rows, points.shape[0])
            block = points[start:stop].detach().clone().requires_grad_(True)
            block_values = _soft_expanded_atom_sdf(
                block,
                atom_coords,
                expanded_atom_radii,
                smoothness,
            )
            block_grads = torch.autograd.grad(block_values.sum(), block)[0]
            values.append(block_values.detach())
            normals.append(_normalize_vectors(block_grads.detach()))
    return torch.cat(values, dim=0), torch.cat(normals, dim=0)


def _project_centers_to_sdf_level(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    smoothness: float,
    iterations: int,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Move probe-center candidates onto the zero level set of the smooth SDF."""

    projected = centers.detach()
    for _ in range(iterations):
        sdf_values, normals = _sdf_values_and_normals(
            projected,
            atom_coords,
            expanded_atom_radii,
            smoothness,
            pairwise_element_budget=pairwise_element_budget,
        )
        projected = projected - sdf_values.unsqueeze(-1) * normals
    return projected.detach()


def _grid_subsample(points: torch.Tensor, spacing: Optional[float]) -> torch.Tensor:
    """Average points inside cubic grid cells, preserving dtype and device."""

    if spacing is None or points.shape[0] == 0:
        return points

    origin = points.min(dim=0).values
    labels = torch.floor((points - origin) / float(spacing)).to(torch.long)
    _, inverse = torch.unique(labels, dim=0, return_inverse=True)
    num_cells = int(inverse.max().item()) + 1

    sums = torch.zeros((num_cells, 3), dtype=points.dtype, device=points.device)
    counts = torch.zeros((num_cells, 1), dtype=points.dtype, device=points.device)
    sums.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), points)
    counts.scatter_add_(
        0,
        inverse.unsqueeze(-1),
        torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device),
    )
    return sums / counts.clamp_min(1)


def _points_outside_atoms(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Return whether SES points are outside every atom van der Waals sphere."""

    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    if atom_coords.shape[0] == 0:
        return torch.ones((points.shape[0],), dtype=torch.bool, device=points.device)

    rows = _max_sdf_rows(atom_coords.shape[0], pairwise_element_budget)
    masks = []
    radii_sq = atom_radii.square().unsqueeze(0)
    for start in range(0, points.shape[0], rows):
        stop = min(start + rows, points.shape[0])
        sq_dists = (
            points[start:stop].unsqueeze(1) - atom_coords.unsqueeze(0)
        ).square().sum(dim=-1)
        tol = (
            256
            * torch.finfo(points.dtype).eps
            * torch.maximum(sq_dists, radii_sq).clamp_min(1)
        )
        masks.append((sq_dists >= radii_sq - tol).all(dim=-1))
    return torch.cat(masks, dim=0)


def _sdf_atom_features(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    smoothness: float,
    feature_threshold: float,
    *,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> torch.Tensor:
    """Build binary atom-support features from smooth SDF ownership weights."""

    num_points = centers.shape[0]
    num_atoms = atom_coords.shape[0]
    features = torch.zeros(
        (num_points, num_atoms),
        dtype=centers.dtype,
        device=centers.device,
    )
    if num_points == 0 or num_atoms == 0:
        return features

    rows = _max_sdf_rows(num_atoms, pairwise_element_budget)
    for start in range(0, num_points, rows):
        stop = min(start + rows, num_points)
        center_distances = torch.linalg.norm(
            centers[start:stop].unsqueeze(1) - atom_coords.unsqueeze(0),
            dim=-1,
        )
        signed_distances = center_distances - expanded_atom_radii.unsqueeze(0)
        weights = torch.softmax(-signed_distances / float(smoothness), dim=-1)
        strongest = weights == weights.max(dim=-1, keepdim=True).values
        supported = (weights >= float(feature_threshold)) | strongest
        features[start:stop] = supported.to(dtype=features.dtype)
    return features


def sample_sdf_points(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    m: int,
    probe_radius: float,
    *,
    smoothness: float = 0.3,
    iterations: int = 6,
    level_tolerance: float = 0.05,
    subsample_spacing: Optional[float] = None,
    feature_threshold: float = 0.1,
    include_atom_features: bool = False,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
    pairwise_element_budget: int = _DEFAULT_PAIRWISE_ELEMENT_BUDGET,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Sample visible SES points through a smooth SDF level-set interface.

    The sampler works in probe-center space.  It places ``m`` deterministic
    Fibonacci-lattice candidates on every atom sphere expanded by
    ``probe_radius``, projects them to the zero level set of a smooth minimum of
    signed distances to those expanded spheres, keeps only exterior-accessible
    probe centers and shifts them inward by ``probe_radius`` along the SDF
    normal.

    Args:
        atom_coords: Atom center coordinates with shape ``(n, 3)``.
        atom_radii: Atom radii with one value per atom.
        m: Number of level-set candidates seeded per atom.
        probe_radius: Solvent probe radius.
        smoothness: Positive soft-min temperature for the SDF, in coordinate
            units.  Smaller values more closely follow the hard expanded-sphere
            union.
        iterations: Number of Newton-style SDF projection iterations.
        level_tolerance: Maximum accepted absolute SDF value after projection.
        subsample_spacing: Optional cubic-grid spacing used to average nearby
            probe centers before the final normal shift.
        feature_threshold: Smooth ownership threshold for binary atom-support
            features.  The strongest atom is always kept.
        include_atom_features: If true, return ``(points, atom_features)``.
        grid_spacing: Optional exterior flood-fill grid spacing.
        max_grid_points: Safety cap for exterior flood-fill grids.
        pairwise_element_budget: Approximate maximum point-atom distance matrix
            size used in chunked SDF calculations.

    Returns:
        ``points`` when ``include_atom_features`` is false, otherwise
        ``(points, atom_features)``.
    """

    if feature_threshold <= 0 or feature_threshold > 1:
        raise ValueError("feature_threshold must be in the interval (0, 1]")
    if max_grid_points <= 0:
        raise ValueError("max_grid_points must be positive")
    if pairwise_element_budget <= 0:
        raise ValueError("pairwise_element_budget must be positive")

    coords, radii = _prepare_sdf_inputs(
        atom_coords,
        atom_radii,
        m,
        probe_radius,
        smoothness,
        iterations,
        level_tolerance,
        subsample_spacing,
    )
    num_atoms = coords.shape[0]
    if m == 0 or num_atoms == 0:
        empty_points = torch.empty((0, 3), dtype=coords.dtype, device=coords.device)
        if not include_atom_features:
            return empty_points
        empty_features = torch.empty(
            (0, num_atoms),
            dtype=coords.dtype,
            device=coords.device,
        )
        return empty_points, empty_features

    expanded_radii = radii + float(probe_radius)
    initial_centers = sample_atom_points(coords, expanded_radii, m).reshape(-1, 3)
    centers = _project_centers_to_sdf_level(
        initial_centers,
        coords,
        expanded_radii,
        float(smoothness),
        iterations,
        pairwise_element_budget=pairwise_element_budget,
    )

    sdf_values = _sdf_values(
        centers,
        coords,
        expanded_radii,
        float(smoothness),
        pairwise_element_budget=pairwise_element_budget,
    )
    finite_mask = torch.isfinite(centers).all(dim=-1) & torch.isfinite(sdf_values)
    level_mask = sdf_values.abs() <= float(level_tolerance)
    feasible_mask = _centers_feasible_against_all_atoms(
        centers,
        coords,
        expanded_radii.square(),
    )
    valid_center_mask = finite_mask & level_mask & feasible_mask
    accessible_mask = _probe_centers_accessible_from_exterior(
        centers,
        coords,
        radii,
        float(probe_radius),
        valid_center_mask=valid_center_mask,
        grid_spacing=grid_spacing,
        max_grid_points=int(max_grid_points),
        assume_centers_feasible=True,
    )
    centers = centers[accessible_mask]
    centers = _grid_subsample(centers, subsample_spacing)

    _, normals = _sdf_values_and_normals(
        centers,
        coords,
        expanded_radii,
        float(smoothness),
        pairwise_element_budget=pairwise_element_budget,
    )
    points = centers - float(probe_radius) * normals
    outside_mask = _points_outside_atoms(
        points,
        coords,
        radii,
        pairwise_element_budget=pairwise_element_budget,
    )
    points = points[outside_mask]
    centers = centers[outside_mask]

    if not include_atom_features:
        return points

    atom_features = _sdf_atom_features(
        centers,
        coords,
        expanded_radii,
        float(smoothness),
        float(feature_threshold),
        pairwise_element_budget=pairwise_element_budget,
    )
    return points, atom_features


__all__ = [
    "sample_sdf_points",
]
