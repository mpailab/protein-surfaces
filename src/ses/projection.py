"""Solvent-excluded surface sampling and projection interfaces.

External interface:
    sample_projected_points: return SES point coordinates and, optionally,
        dense atom-assignment features.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import NamedTuple, Optional, Union

import torch


_PAIRWISE_ELEMENT_BUDGET = 8_000_000
_EXTERNAL_GRID_CACHE_SIZE = 4
_LOCAL_ATOM_INDEX_CACHE_SIZE = 4
_EXTERNAL_GRID_CACHE: "OrderedDict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]" = OrderedDict()
_LOCAL_ATOM_INDEX_CACHE: "OrderedDict[tuple[object, ...], torch.Tensor]" = OrderedDict()


def _tensor_cache_token(tensor: torch.Tensor) -> tuple[object, ...]:
    """
    Return a conservative identity token for tensor-backed geometry caches.
    """
    return (
        str(tensor.device),
        str(tensor.dtype),
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor._version),
    )


class _ProjectionInputs(NamedTuple):
    """
    Validated and dtype-normalized inputs for SES projection.

    Attributes:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,).
    """

    points: torch.Tensor
    atom_coords: torch.Tensor
    atom_radii: torch.Tensor


class _PointAtomGeometry(NamedTuple):
    """
    Geometry shared by point/atom distance and plane tests.

    Attributes:
        valid_point_mask: Whether each sampled point is outside every atom sphere,
            shape (m, n).
        coord_sq_norms: Squared atom-center norms, shape (n,).
        point_coord_dots: Dot products between sampled points and atom centers,
            shape (m, n, n).
    """

    valid_point_mask: torch.Tensor
    coord_sq_norms: torch.Tensor
    point_coord_dots: torch.Tensor


class _PairGeometry(NamedTuple):
    """
    Pairwise geometry for atom pairs.

    Attributes:
        atom_ext_radii: Atom radii expanded by the probe radius, shape (n,).
        atom_ext_radii_sq: Squared expanded atom radii, shape (n,).
        pair_coord_diffs: Pairwise center differences c_i - c_j, shape (n, n, 3).
        pair_coord_diffs_sq: Squared pairwise center distances, shape (n, n).
        valid_atom_pair_mask: Whether expanded atom spheres intersect in a
            non-degenerate circle, shape (n, n).
    """

    atom_ext_radii: torch.Tensor
    atom_ext_radii_sq: torch.Tensor
    pair_coord_diffs: torch.Tensor
    pair_coord_diffs_sq: torch.Tensor
    valid_atom_pair_mask: torch.Tensor


class _TangencyGeometry(NamedTuple):
    """
    Tangency-plane quantities for sampled points and atom pairs.

    Attributes:
        point_coord_diffs_dots: Dot products of points with c_i - c_j, shape
            (m, n, n).
        tangency_plane_bias: Biases of the tangency planes on the owner atom
            sphere, shape (n, n).
        tangency_plane_mask: Whether each point lies on the accepted side of
            each tangency plane, shape (m, n, n).
    """

    point_coord_diffs_dots: torch.Tensor
    tangency_plane_bias: torch.Tensor
    tangency_plane_mask: torch.Tensor


def sample_atom_points(
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n, 1), (n,), or n elements
    m: int,                    # number of points sampled on each atom sphere
) -> torch.Tensor:             # sampled points, shape (m, n, 3)
    """
    Sample evenly distributed points on every atom sphere using a Fibonacci lattice.

    Args:
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n, 1), (n,), or any shape with n elements.
        m: Number of points sampled on each atom sphere.

    Returns:
        sampled_points: Sphere-surface point coordinates, shape (m, n, 3).
    """
    if isinstance(m, bool) or not isinstance(m, int):
        raise TypeError("m must be an integer")
    if m < 0:
        raise ValueError("m must be non-negative")

    if atom_coords.ndim != 2 or atom_coords.shape[-1] != 3:
        raise ValueError("atom_coords must have shape (n, 3)")

    coord_dtype = atom_coords.dtype if atom_coords.is_floating_point() else torch.float32
    radii_dtype = atom_radii.dtype if atom_radii.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(coord_dtype, radii_dtype)
    common_device = atom_coords.device

    atom_coords = atom_coords.to(dtype=common_dtype, device=common_device)
    atom_radii = atom_radii.reshape(-1).to(dtype=common_dtype, device=common_device)
    num_atoms = atom_coords.shape[0]

    if atom_radii.numel() != num_atoms:
        raise ValueError("atom_radii must contain one radius per atom")
    if bool((atom_radii < 0).any().item()):
        raise ValueError("atom_radii must be non-negative")
    if m == 0 or num_atoms == 0:
        return torch.empty((m, num_atoms, 3), dtype=common_dtype, device=common_device)

    sample_indices = torch.arange(m, dtype=common_dtype, device=common_device)
    z_coords = 1 - 2 * (sample_indices + 0.5) / m
    radial_coords = torch.sqrt((1 - z_coords.square()).clamp_min(0))
    golden_angle = torch.as_tensor(
        math.pi * (3.0 - math.sqrt(5.0)),
        dtype=common_dtype,
        device=common_device,
    )
    azimuths = sample_indices * golden_angle
    unit_directions = torch.stack(
        (
            torch.cos(azimuths) * radial_coords,
            torch.sin(azimuths) * radial_coords,
            z_coords,
        ),
        dim=-1,
    )

    return (
        atom_coords.unsqueeze(0)
        + atom_radii.view(1, num_atoms, 1) * unit_directions.view(m, 1, 3)
    )


def _sample_projected_grid(
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n, 1), (n,), or n elements
    m: int,                    # number of points sampled on each atom sphere
    probe_radius: float,       # radius of the probe sphere
) -> tuple[torch.Tensor,       # projected SES points, shape (m, n, 3)
           torch.Tensor]:      # validity mask, shape (m, n)
    """
    Sample atom-sphere points and project them onto a structured SES grid.

    Args:
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n, 1), (n,), or any shape with n elements.
        m: Number of points sampled on each atom sphere.
        probe_radius: Probe sphere radius, scalar.

    Returns:
        projected_points_on_ses: Projected SES points, shape (m, n, 3).
        valid_point_mask: Whether each projected point has an externally
            accessible probe center, shape (m, n).
    """
    points = sample_atom_points(atom_coords, atom_radii, m)
    return project_points(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=probe_radius,
    )


def sample_projected_points(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    m: int,
    probe_radius: float,
    *,
    include_atom_features: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Sample visible SES points with the projection-based interface.

    This is the package-level projection scenario: callers receive a flat point
    cloud with shape ``(num_points, 3)``.  If atom binding is needed, set
    ``include_atom_features=True`` to also receive a dense one-hot feature
    tensor with shape ``(num_points, num_atoms)``.  A value of one means the SES
    point came from the corresponding atom's sampled sphere before projection.

    Args:
        atom_coords: Atom center coordinates, shape ``(n, 3)``.
        atom_radii: Atom radii, shape ``(n,)`` or any shape with ``n`` values.
        m: Number of starting points sampled on each atom sphere.
        probe_radius: Solvent probe radius.
        include_atom_features: If true, also return dense atom-assignment
            features aligned with the returned point rows.

    Returns:
        ``points`` when ``include_atom_features`` is false, otherwise
        ``(points, atom_features)``.
    """

    projected_points, valid_mask = _sample_projected_grid(
        atom_coords,
        atom_radii,
        m,
        probe_radius,
    )
    finite_mask = torch.isfinite(projected_points).all(dim=-1)
    visible_mask = valid_mask & finite_mask
    points = projected_points[visible_mask]

    if not include_atom_features:
        return points

    num_atoms = projected_points.shape[1]
    owner_indices = (
        torch.arange(num_atoms, device=projected_points.device)
        .view(1, num_atoms)
        .expand(projected_points.shape[0], num_atoms)[visible_mask]
    )
    atom_features = torch.zeros(
        (points.shape[0], num_atoms),
        dtype=points.dtype,
        device=points.device,
    )
    if owner_indices.numel() > 0:
        atom_features.scatter_(1, owner_indices.unsqueeze(1), 1)
    return points, atom_features


def _select_suitable_atom_indices(
    suitable_atom_mask: torch.Tensor,        # shape (m, n, n), dtype bool
    geodesic_distances: torch.Tensor,        # shape (m, n, n), dtype float
    pair_normals: torch.Tensor,              # shape (n, n, 3), dtype float
) -> torch.Tensor:                           # shape (m, n, 2), dtype torch.int
    """
    For each `(sample, atom)` row in a boolean mask of shape `(m, n, n)`,
    return the two valid atom indices whose corresponding tangency circles are
    farthest from the sampled point by shortest geodesic distance on the owner
    atom sphere.
    `suitable_atom_mask` is assumed to already have a zero diagonal, so self-pairs
    are excluded before this helper is called. Missing atom indices fall back to
    the owner atom index.

    Args:
        suitable_atom_mask: Candidate neighbor mask, shape (m, n, n).
        geodesic_distances: Distances used to rank candidates, shape (m, n, n).
        pair_normals: Unit normals of pair planes, shape (n, n, 3).

    Returns:
        Selected atom indices, shape (m, n, 2).
    """
    num_samples, num_atoms, _ = suitable_atom_mask.shape
    device = suitable_atom_mask.device
    index_dtype = torch.int

    # Check that the number of atoms can be safely represented in the output dtype.
    max_supported_atoms = torch.iinfo(index_dtype).max + 1
    if num_atoms > max_supported_atoms:
        raise ValueError("suitable_atom_mask.shape[1] must fit in torch.int")

    # The owner indices are used when there are no valid atoms or only one valid atom.
    partner_indices = torch.arange(num_atoms, device=device, dtype=index_dtype)
    owner_indices = partner_indices.view(1, num_atoms, 1).expand(
        num_samples,
        num_atoms,
        2,
    )

    # If there are fewer than 2 atoms, skip the sorting and return the owner indices.
    if num_atoms < 2:
        return owner_indices

    if geodesic_distances.shape != suitable_atom_mask.shape:
        raise ValueError(
            "geodesic_distances must have the same shape as suitable_atom_mask"
        )
    if pair_normals.shape != (num_atoms, num_atoms, 3):
        raise ValueError("pair_normals must have shape (n, n, 3)")

    # Invalid entries get -inf, so max-distance selection ignores them.
    # `argmax` preserves the lowest atom index for exact ties, matching the
    # previous deterministic tie-break.
    distances = torch.where(
        suitable_atom_mask,
        geodesic_distances,
        torch.full((), float("-inf"), dtype=geodesic_distances.dtype, device=device),
    )
    first_indices = distances.argmax(dim=-1)
    first_valid = torch.gather(
        suitable_atom_mask,
        -1,
        first_indices.unsqueeze(-1),
    ).squeeze(-1)
    owner_rows = partner_indices.to(torch.long).view(1, num_atoms).expand(
        num_samples,
        num_atoms,
    )
    candidate_indices = partner_indices.to(torch.long).view(1, 1, num_atoms).expand(
        num_samples,
        num_atoms,
        num_atoms,
    )
    normal_cosines = (
        pair_normals[owner_rows, first_indices].unsqueeze(-2)
        * pair_normals[owner_rows.unsqueeze(-1), candidate_indices]
    ).sum(dim=-1)
    first_distances = torch.gather(
        distances,
        -1,
        first_indices.unsqueeze(-1),
    ).squeeze(-1)
    second_blocked = (
        first_valid.unsqueeze(-1)
        & suitable_atom_mask
        & (normal_cosines > 0)
        & (distances <= first_distances.unsqueeze(-1) * normal_cosines)
    )
    second_blocked.scatter_(-1, first_indices.unsqueeze(-1), False)
    second_mask = suitable_atom_mask & ~second_blocked
    second_mask.scatter_(-1, first_indices.unsqueeze(-1), False)
    second_distances = torch.where(
        second_mask,
        distances,
        torch.full((), float("-inf"), dtype=geodesic_distances.dtype, device=device),
    )
    second_indices = second_distances.argmax(dim=-1)
    second_valid = torch.gather(
        second_mask,
        -1,
        second_indices.unsqueeze(-1),
    ).squeeze(-1)

    selected_first = torch.where(
        first_valid,
        first_indices.to(index_dtype),
        owner_indices[..., 0],
    )
    selected_second = torch.where(
        second_valid,
        second_indices.to(index_dtype),
        owner_indices[..., 1],
    )
    return torch.stack((selected_first, selected_second), dim=-1)


def _prepare_projection_inputs(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
) -> _ProjectionInputs:
    """
    Validate projection inputs and move tensors to a common dtype/device.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii before reshaping, shape (n,) or any shape with
            n elements.
        probe_radius: Probe sphere radius, scalar.

    Returns:
        _ProjectionInputs with points shape (m, n, 3), atom_coords shape (n, 3),
        and atom_radii shape (n,).
    """
    if probe_radius <= 0:
        raise ValueError("probe_radius must be positive")

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (m, n, 3)")
    if atom_coords.ndim != 2 or atom_coords.shape[-1] != 3:
        raise ValueError("atom_coords must have shape (n, 3)")

    if atom_coords.shape[0] != points.shape[1]:
        raise ValueError("points.shape[1] must match atom_coords.shape[0]")

    point_dtype = points.dtype if points.is_floating_point() else torch.float32
    coord_dtype = atom_coords.dtype if atom_coords.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(point_dtype, coord_dtype)
    common_device = atom_coords.device

    points = points.to(dtype=common_dtype, device=common_device)
    atom_coords = atom_coords.to(dtype=common_dtype, device=common_device)
    atom_radii = atom_radii.reshape(-1).to(dtype=common_dtype, device=common_device)

    if atom_radii.numel() != points.shape[1]:
        raise ValueError("radii must contain one radius per atom")
    if bool((atom_radii < 0).any().item()):
        raise ValueError("atom_radii must be non-negative")

    return _ProjectionInputs(points, atom_coords, atom_radii)


def _compute_point_atom_geometry(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
) -> _PointAtomGeometry:
    """
    Compute point/atom products and the point validity mask.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,).

    Returns:
        _PointAtomGeometry with valid_point_mask shape (m, n), coord_sq_norms
        shape (n,), and point_coord_dots shape (m, n, n).
    """
    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    coord_sq_norms = atom_coords.square().sum(dim=-1)
    point_coord_dots = points @ atom_coords.transpose(0, 1)
    point_atom_sq_dists = (
        point_sq_norms + coord_sq_norms.view(1, 1, -1) - 2 * point_coord_dots
    )
    radii_sq = atom_radii.square().view(1, 1, -1)
    sq_dist_tol = (
        100
        * torch.finfo(point_atom_sq_dists.dtype).eps
        * torch.maximum(point_atom_sq_dists.abs(), radii_sq).clamp_min(1)
    )
    valid_point_mask = (
        point_atom_sq_dists >= radii_sq - sq_dist_tol
    ).all(dim=-1) # shape (m, n)

    return _PointAtomGeometry(valid_point_mask, coord_sq_norms, point_coord_dots)


def _compute_pair_geometry(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
) -> _PairGeometry:
    """
    Compute pairwise atom geometry for expanded atom spheres.

    Args:
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,).
        probe_radius: Probe sphere radius, scalar.

    Returns:
        _PairGeometry with atom_ext_radii shape (n,), atom_ext_radii_sq shape
        (n,), pair_coord_diffs shape (n, n, 3), pair_coord_diffs_sq shape
        (n, n), and valid_atom_pair_mask shape (n, n).
    """
    atom_ext_radii = atom_radii + probe_radius
    atom_ext_radii_sq = atom_ext_radii.square()
    pair_radius_sums_sq = (atom_ext_radii[:, None] + atom_ext_radii[None, :]).square()
    pair_radius_diffs_sq = (atom_ext_radii[:, None] - atom_ext_radii[None, :]).square()
    pair_coord_diffs = atom_coords[:, None, :] - atom_coords[None, :, :]
    pair_coord_diffs_sq = pair_coord_diffs.square().sum(dim=-1)
    valid_atom_pair_mask = (
        (pair_coord_diffs_sq > pair_radius_diffs_sq)
        & (pair_coord_diffs_sq < pair_radius_sums_sq)
    ) # shape (n, n)

    return _PairGeometry(
        atom_ext_radii,
        atom_ext_radii_sq,
        pair_coord_diffs,
        pair_coord_diffs_sq,
        valid_atom_pair_mask,
    )


def _compute_tangency_geometry(
    point_geometry: _PointAtomGeometry,
    pair_geometry: _PairGeometry,
    atom_radii: torch.Tensor,
) -> _TangencyGeometry:
    """
    Compute tangency-plane biases and masks for every point and atom pair.

    Args:
        point_geometry: Point/atom geometry with valid_point_mask shape (m, n),
            coord_sq_norms shape (n,), and point_coord_dots shape (m, n, n).
        pair_geometry: Pairwise atom geometry with pair_coord_diffs shape
            (n, n, 3), pair_coord_diffs_sq shape (n, n), atom_ext_radii shape
            (n,), and atom_ext_radii_sq shape (n,).
        atom_radii: Atom radii, shape (n,).

    Returns:
        _TangencyGeometry with point_coord_diffs_dots shape (m, n, n),
        tangency_plane_bias shape (n, n), and tangency_plane_mask shape
        (m, n, n).
    """
    point_coord_diffs_dots = (
        point_geometry.point_coord_dots.diagonal(dim1=1, dim2=2).unsqueeze(-1)
        - point_geometry.point_coord_dots
    )
    tangency_plane_bias = 0.5 * (
        pair_geometry.pair_coord_diffs_sq
        + point_geometry.coord_sq_norms[:, None]
        - point_geometry.coord_sq_norms[None, :]
        - (atom_radii / pair_geometry.atom_ext_radii)[:, None]
        * (
            pair_geometry.pair_coord_diffs_sq
            + pair_geometry.atom_ext_radii_sq[:, None]
            - pair_geometry.atom_ext_radii_sq[None, :]
        )
    )
    tangency_plane_mask = point_coord_diffs_dots <= tangency_plane_bias

    return _TangencyGeometry(
        point_coord_diffs_dots,
        tangency_plane_bias,
        tangency_plane_mask,
    )


def _compute_geodesic_distances(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    pair_geometry: _PairGeometry,
    tangency_geometry: _TangencyGeometry,
) -> torch.Tensor:
    """
    Compute geodesic distances from owner-sphere points to tangency circles.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,).
        pair_geometry: Pairwise atom geometry with pair_coord_diffs shape
            (n, n, 3) and pair_coord_diffs_sq shape (n, n).
        tangency_geometry: Tangency geometry with point_coord_diffs_dots shape
            (m, n, n) and tangency_plane_bias shape (n, n).

    Returns:
        Geodesic distances on each owner atom sphere, shape (m, n, n).
    """
    num_atoms = atom_coords.shape[0]
    pair_coord_dists = torch.sqrt(pair_geometry.pair_coord_diffs_sq)
    safe_pair_coord_dists = pair_coord_dists.masked_fill(pair_coord_dists == 0, 1)
    point_owner_diffs = points - atom_coords.unsqueeze(0)
    point_owner_dists = torch.linalg.norm(point_owner_diffs, dim=-1)
    safe_point_owner_dists = point_owner_dists.masked_fill(point_owner_dists == 0, 1)
    plane_center_dots = (
        pair_geometry.pair_coord_diffs * atom_coords[:, None, :]
    ).sum(dim=-1)
    safe_atom_radii = atom_radii.masked_fill(atom_radii == 0, 1)

    point_plane_cosines = (
        (tangency_geometry.point_coord_diffs_dots - plane_center_dots.unsqueeze(0))
        / (safe_point_owner_dists.unsqueeze(-1) * safe_pair_coord_dists.unsqueeze(0))
    ).clamp(-1, 1)
    tangency_plane_cosines = (
        (tangency_geometry.tangency_plane_bias - plane_center_dots)
        / (safe_atom_radii[:, None] * safe_pair_coord_dists)
    ).clamp(-1, 1)
    geodesic_distances = atom_radii.view(1, num_atoms, 1) * torch.abs(
        torch.acos(point_plane_cosines)
        - torch.acos(tangency_plane_cosines).unsqueeze(0)
    )

    valid_geodesic_mask = (
        (atom_radii > 0).view(1, num_atoms, 1)
        & (point_owner_dists > 0).unsqueeze(-1)
        & (pair_geometry.pair_coord_diffs_sq > 0).unsqueeze(0)
    )
    return torch.where(
        valid_geodesic_mask,
        geodesic_distances,
        torch.zeros_like(geodesic_distances),
    )


def _compute_suitable_atom_indices(
    point_geometry: _PointAtomGeometry,
    pair_geometry: _PairGeometry,
    tangency_geometry: _TangencyGeometry,
    geodesic_distances: torch.Tensor,
) -> torch.Tensor:
    """
    Combine geometric masks and select two neighbor atoms for each point.

    Args:
        point_geometry: Point/atom geometry with valid_point_mask shape (m, n).
        pair_geometry: Pairwise atom geometry with valid_atom_pair_mask shape
            (n, n).
        tangency_geometry: Tangency geometry with tangency_plane_mask shape
            (m, n, n).
        geodesic_distances: Distances used to rank candidate neighbors, shape
            (m, n, n).

    Returns:
        Selected atom indices, shape (m, n, 2).
    """
    suitable_atom_mask = (
        point_geometry.valid_point_mask.unsqueeze(-1)
        & pair_geometry.valid_atom_pair_mask.unsqueeze(0)
        & tangency_geometry.tangency_plane_mask
    ) # shape (m, n, n)
    suitable_atom_mask.diagonal(dim1=-2, dim2=-1).zero_()
    pair_dists = torch.sqrt(pair_geometry.pair_coord_diffs_sq).clamp_min(
        torch.finfo(pair_geometry.pair_coord_diffs.dtype).eps,
    )

    return _select_suitable_atom_indices(
        suitable_atom_mask,
        geodesic_distances,
        pair_geometry.pair_coord_diffs / pair_dists.unsqueeze(-1),
    )


def _build_affine_projector(
    num_samples: int,
    point_geometry: _PointAtomGeometry,
    pair_geometry: _PairGeometry,
    suitable_atom_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build affine projectors induced by the selected pair planes.

    Args:
        num_samples: Number of point samples m.
        point_geometry: Point/atom geometry with coord_sq_norms shape (n,).
        pair_geometry: Pairwise atom geometry with pair_coord_diffs shape
            (n, n, 3) and atom_ext_radii_sq shape (n,).
        suitable_atom_indices: Selected atom indices, shape (m, n, 2).

    Returns:
        affine_projection: Linear projection part, shape (m, n, 3, 3).
        affine_shift: Affine shift part, shape (m, n, 3).
    """
    num_atoms = pair_geometry.pair_coord_diffs.shape[0]
    device = pair_geometry.pair_coord_diffs.device
    owner_indices = torch.arange(num_atoms, device=device, dtype=torch.long)
    owner_indices = owner_indices.view(1, num_atoms, 1).expand(num_samples, -1, 2)

    plane_normals = pair_geometry.pair_coord_diffs[
        owner_indices,
        suitable_atom_indices,
    ]
    plane_bias = 0.5 * (
        point_geometry.coord_sq_norms[:, None]
        - point_geometry.coord_sq_norms[None, :]
        - pair_geometry.atom_ext_radii_sq[:, None]
        + pair_geometry.atom_ext_radii_sq[None, :]
    )
    plane_bias = plane_bias[owner_indices, suitable_atom_indices].unsqueeze(-1)
    grams = plane_normals @ plane_normals.transpose(-1, -2)
    grams_inv = torch.linalg.pinv(grams, hermitian=True)
    affine_projection = plane_normals.transpose(-1, -2) @ grams_inv @ plane_normals
    affine_shift = (plane_normals.transpose(-1, -2) @ grams_inv @ plane_bias).squeeze(-1)

    return affine_projection, affine_shift


def _apply_affine_projector(
    x: torch.Tensor,
    affine_projection: torch.Tensor,
    affine_shift: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the precomputed affine projector to point-like coordinates.

    Args:
        x: Coordinates to project, shape (m, n, 3).
        affine_projection: Linear projection part, shape (m, n, 3, 3).
        affine_shift: Affine shift part, shape (m, n, 3).

    Returns:
        Projected coordinates, shape (m, n, 3).
    """
    return x - (affine_projection @ x.unsqueeze(-1)).squeeze(-1) + affine_shift


def _recover_probe_centers(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_ext_radii_sq: torch.Tensor,
    affine_projection: torch.Tensor,
    affine_shift: torch.Tensor,
) -> torch.Tensor:
    """
    Recover probe centers after projecting points and owner atom centers.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_ext_radii_sq: Squared expanded atom radii, shape (n,).
        affine_projection: Linear projection part, shape (m, n, 3, 3).
        affine_shift: Affine shift part, shape (m, n, 3).

    Returns:
        Probe center coordinates, shape (m, n, 3).
    """
    num_samples = points.shape[0]
    projected_points = _apply_affine_projector(points, affine_projection, affine_shift)
    broadcasted_atom_coords = atom_coords.unsqueeze(0).expand(num_samples, -1, -1)
    projected_atoms = _apply_affine_projector(
        broadcasted_atom_coords,
        affine_projection,
        affine_shift,
    )
    probe_directions = projected_points - projected_atoms
    projected_point_atom_dists = torch.linalg.norm(probe_directions, dim=-1)
    projected_atom_sq_dists = (
        projected_atoms - broadcasted_atom_coords
    ).square().sum(dim=-1)
    probe_offset_sq = atom_ext_radii_sq.unsqueeze(0) - projected_atom_sq_dists
    probe_offset_dists = torch.sqrt(probe_offset_sq.clamp_min(0))
    probe_shift_denoms = projected_point_atom_dists.clamp_min(
        torch.finfo(projected_point_atom_dists.dtype).eps,
    )
    probe_shift_coefs = probe_offset_dists / probe_shift_denoms

    return projected_atoms + probe_directions * probe_shift_coefs.unsqueeze(-1)


def _compute_pair_probe_centers(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_ext_radii: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Build probe centers on expanded-sphere intersection circles for atom pairs.
    """
    eps = torch.finfo(points.dtype).eps

    first_centers = atom_coords[first_indices]
    second_centers = atom_coords[second_indices]
    first_radii = atom_ext_radii[first_indices]
    second_radii = atom_ext_radii[second_indices]
    axes = second_centers - first_centers
    axis_dists = torch.linalg.norm(axes, dim=-1).clamp_min(eps)
    axis_dirs = axes / axis_dists.unsqueeze(-1)

    circle_offsets = (
        first_radii.square()
        - second_radii.square()
        + axis_dists.square()
    ) / (2 * axis_dists)
    circle_centers = first_centers + circle_offsets.unsqueeze(-1) * axis_dirs
    circle_radii = torch.sqrt(
        (first_radii.square() - circle_offsets.square()).clamp_min(0),
    )

    pair_points = points
    while pair_points.ndim < circle_centers.ndim:
        pair_points = pair_points.unsqueeze(-2)

    radial_dirs = pair_points - circle_centers
    radial_dirs = radial_dirs - (radial_dirs * axis_dirs).sum(
        dim=-1,
        keepdim=True,
    ) * axis_dirs
    radial_dir_norms = torch.linalg.norm(radial_dirs, dim=-1, keepdim=True)
    z_ref = torch.zeros_like(axis_dirs)
    z_ref[..., 2] = 1
    y_ref = torch.zeros_like(axis_dirs)
    y_ref[..., 1] = 1
    refs = torch.where(axis_dirs[..., 2:].abs() > 0.9, y_ref, z_ref)
    fallback_dirs = torch.cross(axis_dirs, refs, dim=-1)
    fallback_dirs = fallback_dirs / torch.linalg.norm(
        fallback_dirs,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    radial_dirs = torch.where(
        radial_dir_norms > eps,
        radial_dirs / radial_dir_norms.clamp_min(eps),
        fallback_dirs,
    )
    return circle_centers + circle_radii.unsqueeze(-1) * radial_dirs


def _build_local_atom_indices(
    atom_coords: torch.Tensor,
    max_neighbors: int = 256,
) -> torch.Tensor:
    """
    Build a fixed-width nearest-neighbor table for local SES candidate checks.

    The projection code only needs atoms close enough to interact with the
    owner's expanded sphere. A capped nearest-neighbor table keeps memory
    linear in the number of atoms instead of materializing `(n, n, 3)` geometry
    for large benchmark molecules.
    """
    num_atoms = atom_coords.shape[0]
    if num_atoms == 0:
        return torch.empty((0, 0), dtype=torch.long, device=atom_coords.device)

    cache_key = (_tensor_cache_token(atom_coords), int(max_neighbors))
    cached = _LOCAL_ATOM_INDEX_CACHE.get(cache_key)
    if cached is not None:
        _LOCAL_ATOM_INDEX_CACHE.move_to_end(cache_key)
        return cached

    neighbor_count = min(num_atoms, max_neighbors)
    if neighbor_count == num_atoms:
        neighbor_indices = torch.arange(
            num_atoms,
            device=atom_coords.device,
            dtype=torch.long,
        ).view(1, num_atoms).expand(num_atoms, -1)
        _LOCAL_ATOM_INDEX_CACHE[cache_key] = neighbor_indices
        _LOCAL_ATOM_INDEX_CACHE.move_to_end(cache_key)
        while len(_LOCAL_ATOM_INDEX_CACHE) > _LOCAL_ATOM_INDEX_CACHE_SIZE:
            _LOCAL_ATOM_INDEX_CACHE.popitem(last=False)
        return neighbor_indices

    block_size = max(1, min(num_atoms, 1024))
    neighbor_indices = torch.empty(
        (num_atoms, neighbor_count),
        dtype=torch.long,
        device=atom_coords.device,
    )
    all_indices = torch.arange(num_atoms, device=atom_coords.device)
    for start in range(0, num_atoms, block_size):
        stop = min(start + block_size, num_atoms)
        distances = torch.cdist(atom_coords[start:stop], atom_coords)
        _, local_indices = torch.topk(
            distances,
            k=neighbor_count,
            dim=-1,
            largest=False,
        )
        # `topk` includes the zero self-distance, but exact ties can make the
        # order nondeterministic. Put the owner in the first slot explicitly.
        owners = all_indices[start:stop]
        self_slots = local_indices == owners.unsqueeze(-1)
        has_self = self_slots.any(dim=-1)
        missing_self = ~has_self
        local_indices[missing_self, -1] = owners[missing_self]
        self_slots = local_indices == owners.unsqueeze(-1)
        self_positions = self_slots.to(torch.long).argmax(dim=-1)
        first_values = local_indices[:, 0].clone()
        local_indices[:, 0] = owners
        local_indices[
            torch.arange(stop - start, device=atom_coords.device),
            self_positions,
        ] = first_values
        neighbor_indices[start:stop] = local_indices

    _LOCAL_ATOM_INDEX_CACHE[cache_key] = neighbor_indices
    _LOCAL_ATOM_INDEX_CACHE.move_to_end(cache_key)
    while len(_LOCAL_ATOM_INDEX_CACHE) > _LOCAL_ATOM_INDEX_CACHE_SIZE:
        _LOCAL_ATOM_INDEX_CACHE.popitem(last=False)
    return neighbor_indices


def _squared_feasibility_mask(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_ext_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Check that probe centers do not overlap local expanded atom spheres.
    """
    center_sq_dists = (centers - atom_coords).square().sum(dim=-1)
    ext_radii_sq = atom_ext_radii.square()
    sq_dist_tol = (
        100
        * torch.finfo(center_sq_dists.dtype).eps
        * torch.maximum(center_sq_dists.abs(), ext_radii_sq).clamp_min(1)
    )
    return center_sq_dists >= ext_radii_sq - sq_dist_tol


def _valid_points_against_local_atoms(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Check whether owner-sphere samples lie outside all local atom interiors.
    """
    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    coord_sq_norms = atom_coords.square().sum(dim=-1)
    point_coord_dots = (points.unsqueeze(-2) * atom_coords).sum(dim=-1)
    point_sq_dists = point_sq_norms + coord_sq_norms - 2 * point_coord_dots
    radii_sq = atom_radii.square()
    sq_dist_tol = (
        100
        * torch.finfo(point_sq_dists.dtype).eps
        * torch.maximum(point_sq_dists.abs(), radii_sq).clamp_min(1)
    )
    return (point_sq_dists >= radii_sq - sq_dist_tol).all(dim=-1)


def _valid_points_against_all_atoms(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Full validity check matching the public projection mask formula exactly.
    """
    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    coord_sq_norms = atom_coords.square().sum(dim=-1).view(1, -1)
    point_coord_dots = points @ atom_coords.transpose(0, 1)
    point_sq_dists = point_sq_norms + coord_sq_norms - 2 * point_coord_dots
    radii_sq = atom_radii.square().view(1, -1)
    sq_dist_tol = (
        100
        * torch.finfo(point_sq_dists.dtype).eps
        * torch.maximum(point_sq_dists.abs(), radii_sq).clamp_min(1)
    )
    return (point_sq_dists >= radii_sq - sq_dist_tol).all(dim=-1)


def _centers_feasible_against_local_atoms(
    centers: torch.Tensor,
    local_atom_coords: torch.Tensor,
    local_atom_ext_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized local feasibility for candidate probe centers.
    """
    row_count = centers.shape[0]
    candidate_shape = centers.shape[1:-1]
    flat_centers = centers.reshape(row_count, -1, 3)
    center_sq_norms = flat_centers.square().sum(dim=-1, keepdim=True)
    atom_sq_norms = local_atom_coords.square().sum(dim=-1).unsqueeze(-2)
    center_atom_dots = flat_centers @ local_atom_coords.transpose(-1, -2)
    center_sq_dists = center_sq_norms + atom_sq_norms - 2 * center_atom_dots
    ext_radii_sq = local_atom_ext_radii.square().unsqueeze(-2)
    sq_dist_tol = (
        100
        * torch.finfo(center_sq_dists.dtype).eps
        * torch.maximum(center_sq_dists.abs(), ext_radii_sq).clamp_min(1)
    )
    feasible = (center_sq_dists >= ext_radii_sq - sq_dist_tol).all(dim=-1)
    return feasible.reshape(row_count, *candidate_shape)


def _compute_triple_probe_centers(
    atom_coords: torch.Tensor,
    atom_ext_radii: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    third_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the two possible probe centers touching each atom triple.
    """
    eps = torch.finfo(atom_coords.dtype).eps
    first_centers = atom_coords[first_indices]
    second_centers = atom_coords[second_indices]
    third_centers = atom_coords[third_indices]
    first_radii = atom_ext_radii[first_indices]
    second_radii = atom_ext_radii[second_indices]
    third_radii = atom_ext_radii[third_indices]

    first_to_second = second_centers - first_centers
    first_to_third = third_centers - first_centers
    second_dists = torch.linalg.norm(first_to_second, dim=-1)
    safe_second_dists = second_dists.clamp_min(eps)
    ex = first_to_second / safe_second_dists.unsqueeze(-1)
    i = (ex * first_to_third).sum(dim=-1)
    third_plane_vec = first_to_third - i.unsqueeze(-1) * ex
    j = torch.linalg.norm(third_plane_vec, dim=-1)
    safe_j = j.clamp_min(eps)
    ey = third_plane_vec / safe_j.unsqueeze(-1)
    ez = torch.cross(ex, ey, dim=-1)

    x = (
        first_radii.square()
        - second_radii.square()
        + second_dists.square()
    ) / (2 * safe_second_dists)
    y = (
        first_radii.square()
        - third_radii.square()
        + i.square()
        + j.square()
        - 2 * i * x
    ) / (2 * safe_j)
    z_sq = first_radii.square() - x.square() - y.square()
    z = torch.sqrt(z_sq.clamp_min(0))
    base_centers = first_centers + x.unsqueeze(-1) * ex + y.unsqueeze(-1) * ey
    centers = torch.stack(
        (
            base_centers + z.unsqueeze(-1) * ez,
            base_centers - z.unsqueeze(-1) * ez,
        ),
        dim=-2,
    )

    scale = torch.maximum(
        first_radii,
        torch.maximum(second_radii, third_radii),
    ).clamp_min(1)
    tol = 1000 * eps * scale.square()
    valid = (second_dists > eps) & (j > eps) & (z_sq >= -tol)
    return centers, valid.unsqueeze(-1).expand_as(z.unsqueeze(-1).expand(*z.shape, 2))


def _pair_patch_membership(
    points: torch.Tensor,
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_ext_radii: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Check that projected directions lie on the pair's toroidal SES arc.
    """
    eps = torch.finfo(points.dtype).eps
    first_normals = (
        centers - atom_coords[first_indices]
    ) / atom_ext_radii[first_indices].unsqueeze(-1).clamp_min(eps)
    second_normals = (
        centers - atom_coords[second_indices]
    ) / atom_ext_radii[second_indices].unsqueeze(-1).clamp_min(eps)
    point_dirs = points.unsqueeze(-2) - centers
    point_dirs = point_dirs / torch.linalg.norm(
        point_dirs,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    target_normals = -point_dirs

    normal_cosines = (first_normals * second_normals).sum(dim=-1).clamp(-1, 1)
    det = (1 - normal_cosines.square()).clamp_min(eps)
    rhs_first = (target_normals * first_normals).sum(dim=-1)
    rhs_second = (target_normals * second_normals).sum(dim=-1)
    first_coefs = (rhs_first - normal_cosines * rhs_second) / det
    second_coefs = (rhs_second - normal_cosines * rhs_first) / det
    reconstructed = (
        first_coefs.unsqueeze(-1) * first_normals
        + second_coefs.unsqueeze(-1) * second_normals
    )
    residuals = torch.linalg.norm(reconstructed - target_normals, dim=-1)
    coef_tol = 1000 * eps
    residual_tol = torch.sqrt(torch.as_tensor(eps, dtype=points.dtype, device=points.device))
    return (
        (first_coefs >= -coef_tol)
        & (second_coefs >= -coef_tol)
        & (residuals <= residual_tol)
    )


def _triple_patch_membership(
    points: torch.Tensor,
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_ext_radii: torch.Tensor,
    first_indices: torch.Tensor,
    second_indices: torch.Tensor,
    third_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Check that projected directions lie in a triple-contact normal cone.
    """
    eps = torch.finfo(points.dtype).eps
    first_normals = (
        centers - atom_coords[first_indices].unsqueeze(-2)
    ) / atom_ext_radii[first_indices].unsqueeze(-1).unsqueeze(-1).clamp_min(eps)
    second_normals = (
        centers - atom_coords[second_indices].unsqueeze(-2)
    ) / atom_ext_radii[second_indices].unsqueeze(-1).unsqueeze(-1).clamp_min(eps)
    third_normals = (
        centers - atom_coords[third_indices].unsqueeze(-2)
    ) / atom_ext_radii[third_indices].unsqueeze(-1).unsqueeze(-1).clamp_min(eps)
    point_dirs = points.unsqueeze(-2).unsqueeze(-2) - centers
    point_dirs = point_dirs / torch.linalg.norm(
        point_dirs,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    target_normals = -point_dirs
    second_cross_third = torch.cross(second_normals, third_normals, dim=-1)
    determinants = (first_normals * second_cross_third).sum(dim=-1)
    nonsingular = determinants.abs() > 1000 * eps
    safe_determinants = determinants.masked_fill(~nonsingular, 1)
    first_coefs = (
        target_normals * second_cross_third
    ).sum(dim=-1) / safe_determinants
    second_coefs = (
        first_normals * torch.cross(target_normals, third_normals, dim=-1)
    ).sum(dim=-1) / safe_determinants
    third_coefs = (
        first_normals * torch.cross(second_normals, target_normals, dim=-1)
    ).sum(dim=-1) / safe_determinants
    coefs = torch.stack((first_coefs, second_coefs, third_coefs), dim=-1)
    reconstructed = (
        first_coefs.unsqueeze(-1) * first_normals
        + second_coefs.unsqueeze(-1) * second_normals
        + third_coefs.unsqueeze(-1) * third_normals
    )
    residuals = torch.linalg.norm(reconstructed - target_normals, dim=-1)
    coef_tol = 1000 * eps
    residual_tol = torch.sqrt(torch.as_tensor(eps, dtype=points.dtype, device=points.device))
    return (
        nonsingular
        & (coefs >= -coef_tol).all(dim=-1)
        & (residuals <= residual_tol)
    )


def _replace_better_probe_centers(
    best_centers: torch.Tensor,
    best_scores: torch.Tensor,
    candidate_centers: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select lower-score candidate centers row-wise.
    """
    masked_scores = torch.where(
        candidate_mask,
        candidate_scores,
        torch.full((), float("inf"), dtype=best_scores.dtype, device=best_scores.device),
    )
    candidate_best_scores, candidate_best_indices = masked_scores.min(dim=-1)
    candidate_best_centers = torch.gather(
        candidate_centers,
        1,
        candidate_best_indices.view(-1, 1, 1).expand(-1, 1, 3),
    ).squeeze(1)
    use_candidate = candidate_best_scores < best_scores
    return (
        torch.where(use_candidate.unsqueeze(-1), candidate_best_centers, best_centers),
        torch.where(use_candidate, candidate_best_scores, best_scores),
    )


def _compute_active_set_probe_centers(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Choose nearest feasible 1-, 2-, or 3-contact probe centers for point samples.
    """
    num_samples, num_atoms, _ = points.shape
    device = points.device
    dtype = points.dtype
    atom_ext_radii = atom_radii + probe_radius
    local_atom_indices = _build_local_atom_indices(atom_coords)
    max_active_neighbors = min(8, max(num_atoms - 1, 0))
    probe_centers = torch.empty_like(points)
    valid_point_mask = torch.empty(points.shape[:2], dtype=torch.bool, device=device)
    if num_atoms == 0:
        return probe_centers, valid_point_mask

    flat_points = points.reshape(-1, 3)
    flat_centers = probe_centers.reshape(-1, 3)
    flat_valid_mask = valid_point_mask.reshape(-1)
    total_rows = flat_points.shape[0]
    block_rows = 8192
    atom_range = torch.arange(num_atoms, device=device, dtype=torch.long)
    pair_combinations = torch.combinations(
        torch.arange(max_active_neighbors + 1, device=device, dtype=torch.long),
        r=2,
    )
    triple_combinations = torch.combinations(
        torch.arange(max_active_neighbors + 1, device=device, dtype=torch.long),
        r=3,
    )

    for start in range(0, total_rows, block_rows):
        stop = min(start + block_rows, total_rows)
        block_points = flat_points[start:stop]
        owner_indices = atom_range[(torch.arange(start, stop, device=device) % num_atoms)]
        owner_coords = atom_coords[owner_indices]
        owner_radii = atom_radii[owner_indices]
        owner_ext_radii = atom_ext_radii[owner_indices]
        local_indices = local_atom_indices[owner_indices]
        local_coords = atom_coords[local_indices]
        local_radii = atom_radii[local_indices]
        local_ext_radii = atom_ext_radii[local_indices]

        owner_directions = block_points - owner_coords
        owner_direction_norms = torch.linalg.norm(
            owner_directions,
            dim=-1,
            keepdim=True,
        )
        fallback_directions = torch.zeros_like(owner_directions)
        fallback_directions[:, 0] = 1
        owner_directions = torch.where(
            owner_direction_norms > torch.finfo(dtype).eps,
            owner_directions / owner_direction_norms.clamp_min(torch.finfo(dtype).eps),
            fallback_directions,
        )
        direct_centers = owner_coords + owner_ext_radii.unsqueeze(-1) * owner_directions

        if local_atom_indices.shape[1] == num_atoms:
            point_is_valid = _valid_points_against_all_atoms(
                block_points,
                atom_coords,
                atom_radii,
            )
        else:
            point_is_valid = _valid_points_against_local_atoms(
                block_points,
                local_coords,
                local_radii,
            )
        direct_is_feasible = _squared_feasibility_mask(
            direct_centers.unsqueeze(-2),
            local_coords,
            local_ext_radii,
        ).all(dim=-1)

        best_centers = direct_centers
        best_scores = torch.where(
            point_is_valid & direct_is_feasible,
            torch.zeros((stop - start,), dtype=dtype, device=device),
            torch.full((stop - start,), float("inf"), dtype=dtype, device=device),
        )

        unresolved_mask = point_is_valid & ~direct_is_feasible
        unresolved_indices = unresolved_mask.nonzero(as_tuple=False).reshape(-1)
        if unresolved_indices.numel() > 0 and max_active_neighbors > 0:
            unresolved_points = block_points[unresolved_indices]
            unresolved_direct_centers = direct_centers[unresolved_indices]
            unresolved_local_indices = local_indices[unresolved_indices]
            unresolved_local_coords = local_coords[unresolved_indices]
            unresolved_local_ext_radii = local_ext_radii[unresolved_indices]
            unresolved_owner_indices = owner_indices[unresolved_indices]

            center_local_dists = torch.linalg.norm(
                unresolved_direct_centers.unsqueeze(-2) - unresolved_local_coords,
                dim=-1,
            )
            penetrations = unresolved_local_ext_radii - center_local_dists
            penetrations = torch.where(
                unresolved_local_indices != unresolved_owner_indices.unsqueeze(-1),
                penetrations,
                torch.full((), float("-inf"), dtype=dtype, device=device),
            )
            active_count = max_active_neighbors + 1
            _, blocker_slots = torch.topk(
                penetrations,
                k=max_active_neighbors,
                dim=-1,
                largest=True,
            )
            blocker_indices = torch.gather(
                unresolved_local_indices,
                1,
                blocker_slots,
            )
            active_indices = torch.cat(
                (unresolved_owner_indices.unsqueeze(-1), blocker_indices),
                dim=-1,
            )
            unresolved_best_centers = best_centers[unresolved_indices]
            unresolved_best_scores = best_scores[unresolved_indices]

            if pair_combinations.numel() > 0:
                pair_slots = pair_combinations[
                    (pair_combinations < active_count).all(dim=-1)
                ]
                first_indices = active_indices[:, pair_slots[:, 0]]
                second_indices = active_indices[:, pair_slots[:, 1]]
                first_coords = atom_coords[first_indices]
                second_coords = atom_coords[second_indices]
                first_radii = atom_ext_radii[first_indices]
                second_radii = atom_ext_radii[second_indices]
                pair_dists_sq = (first_coords - second_coords).square().sum(dim=-1)
                pair_valid = (
                    (first_indices != second_indices)
                    & (pair_dists_sq > (first_radii - second_radii).square())
                    & (pair_dists_sq < (first_radii + second_radii).square())
                )
                pair_centers = _compute_pair_probe_centers(
                    unresolved_points,
                    atom_coords,
                    atom_ext_radii,
                    first_indices,
                    second_indices,
                )
                pair_feasible = _centers_feasible_against_local_atoms(
                    pair_centers,
                    unresolved_local_coords,
                    unresolved_local_ext_radii,
                )
                pair_scores = torch.abs(
                    torch.linalg.norm(
                        unresolved_points.unsqueeze(-2) - pair_centers,
                        dim=-1,
                    )
                    - probe_radius
                )
                pair_membership = _pair_patch_membership(
                    unresolved_points,
                    pair_centers,
                    atom_coords,
                    atom_ext_radii,
                    first_indices,
                    second_indices,
                )
                unresolved_best_centers, unresolved_best_scores = (
                    _replace_better_probe_centers(
                        unresolved_best_centers,
                        unresolved_best_scores,
                        pair_centers,
                        pair_scores,
                        pair_valid & pair_feasible & pair_membership,
                    )
                )

            still_unresolved = ~unresolved_best_scores.isfinite()
            triple_rows = (
                still_unresolved
                | (unresolved_best_scores > 10 * torch.finfo(dtype).eps)
            ).nonzero(as_tuple=False).reshape(-1)
            if triple_rows.numel() > 0 and triple_combinations.numel() > 0:
                triple_active_indices = active_indices[triple_rows]
                triple_points = unresolved_points[triple_rows]
                triple_local_coords = unresolved_local_coords[triple_rows]
                triple_local_ext_radii = unresolved_local_ext_radii[triple_rows]
                triple_slots = triple_combinations[
                    (triple_combinations < active_count).all(dim=-1)
                ]
                first_indices = triple_active_indices[:, triple_slots[:, 0]]
                second_indices = triple_active_indices[:, triple_slots[:, 1]]
                third_indices = triple_active_indices[:, triple_slots[:, 2]]
                triple_centers, triple_valid = _compute_triple_probe_centers(
                    atom_coords,
                    atom_ext_radii,
                    first_indices,
                    second_indices,
                    third_indices,
                )
                distinct_triples = (
                    (first_indices != second_indices)
                    & (first_indices != third_indices)
                    & (second_indices != third_indices)
                ).unsqueeze(-1)
                triple_feasible = _centers_feasible_against_local_atoms(
                    triple_centers,
                    triple_local_coords,
                    triple_local_ext_radii,
                )
                triple_scores = torch.abs(
                    torch.linalg.norm(
                        triple_points.unsqueeze(-2).unsqueeze(-2) - triple_centers,
                        dim=-1,
                    )
                    - probe_radius
                )
                triple_membership = _triple_patch_membership(
                    triple_points,
                    triple_centers,
                    atom_coords,
                    atom_ext_radii,
                    first_indices,
                    second_indices,
                    third_indices,
                )
                flat_triple_centers = triple_centers.reshape(
                    triple_centers.shape[0],
                    -1,
                    3,
                )
                flat_triple_scores = triple_scores.reshape(
                    triple_scores.shape[0],
                    -1,
                )
                flat_strict_mask = (
                    distinct_triples
                    & triple_valid
                    & triple_feasible
                    & triple_membership
                ).reshape(triple_centers.shape[0], -1)
                triple_best_centers, triple_best_scores = _replace_better_probe_centers(
                    unresolved_best_centers[triple_rows],
                    unresolved_best_scores[triple_rows],
                    flat_triple_centers,
                    flat_triple_scores,
                    flat_strict_mask,
                )
                unresolved_best_centers[triple_rows] = triple_best_centers
                unresolved_best_scores[triple_rows] = triple_best_scores

            use_direct_fallback = ~unresolved_best_scores.isfinite()
            unresolved_best_centers = torch.where(
                use_direct_fallback.unsqueeze(-1),
                unresolved_direct_centers,
                unresolved_best_centers,
            )
            best_centers[unresolved_indices] = unresolved_best_centers
            best_scores[unresolved_indices] = unresolved_best_scores

        flat_centers[start:stop] = best_centers
        flat_valid_mask[start:stop] = point_is_valid

    return probe_centers, valid_point_mask


def _prefer_pair_only_probe_centers(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    pair_geometry: _PairGeometry,
    suitable_atom_indices: torch.Tensor,
    probe_radius: float,
    owner_probe_centers: torch.Tensor,
) -> torch.Tensor:
    """
    Prefer probe centers supported only by two selected neighbors when valid.
    """
    num_samples, num_atoms, _ = points.shape
    device = points.device
    dtype = points.dtype
    eps = torch.finfo(dtype).eps

    owner_indices = torch.arange(num_atoms, device=device, dtype=torch.long)
    owner_indices = owner_indices.view(1, num_atoms).expand(num_samples, -1)
    first_indices = suitable_atom_indices[..., 0].to(torch.long)
    second_indices = suitable_atom_indices[..., 1].to(torch.long)
    pair_only_mask = (
        (first_indices != owner_indices)
        & (second_indices != owner_indices)
        & (first_indices != second_indices)
        & pair_geometry.valid_atom_pair_mask[first_indices, second_indices]
    )

    pair_probe_centers = _compute_pair_probe_centers(
        points,
        atom_coords,
        pair_geometry.atom_ext_radii,
        first_indices,
        second_indices,
    )

    center_atom_dists = torch.linalg.norm(
        pair_probe_centers.unsqueeze(-2) - atom_coords,
        dim=-1,
    )
    feasibility_tol = 100 * eps * pair_geometry.atom_ext_radii.max().clamp_min(1)
    pair_feasible = (
        center_atom_dists
        >= pair_geometry.atom_ext_radii.view(1, 1, -1) - feasibility_tol
    ).all(dim=-1)
    owner_center_atom_dists = torch.linalg.norm(
        owner_probe_centers.unsqueeze(-2) - atom_coords,
        dim=-1,
    )
    owner_feasible = (
        owner_center_atom_dists
        >= pair_geometry.atom_ext_radii.view(1, 1, -1) - feasibility_tol
    ).all(dim=-1)
    owner_errors = torch.abs(
        torch.linalg.norm(points - owner_probe_centers, dim=-1) - probe_radius,
    )
    pair_errors = torch.abs(
        torch.linalg.norm(points - pair_probe_centers, dim=-1) - probe_radius,
    )
    use_pair = pair_only_mask & pair_feasible & (
        ~owner_feasible | (pair_errors < owner_errors)
    )
    return torch.where(
        use_pair.unsqueeze(-1),
        pair_probe_centers,
        owner_probe_centers,
    )


def _prefer_any_pair_only_probe_centers(
    points: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    pair_geometry: _PairGeometry,
    tangency_geometry: _TangencyGeometry,
    probe_radius: float,
    current_probe_centers: torch.Tensor,
) -> torch.Tensor:
    """
    Repair infeasible centers by trying probe supports from any two non-owner atoms.
    """
    num_samples, num_atoms, _ = points.shape
    if num_atoms < 3:
        return current_probe_centers

    device = points.device
    dtype = points.dtype
    eps = torch.finfo(dtype).eps
    pair_indices = torch.combinations(
        torch.arange(num_atoms, device=device, dtype=torch.long),
        r=2,
    )
    first_indices = pair_indices[:, 0].view(1, 1, -1).expand(
        num_samples,
        num_atoms,
        -1,
    )
    second_indices = pair_indices[:, 1].view(1, 1, -1).expand(
        num_samples,
        num_atoms,
        -1,
    )
    owner_indices = torch.arange(num_atoms, device=device, dtype=torch.long).view(
        1,
        num_atoms,
        1,
    )
    pair_only_mask = (
        (first_indices != owner_indices)
        & (second_indices != owner_indices)
        & pair_geometry.valid_atom_pair_mask[first_indices, second_indices]
    )

    pair_probe_centers = _compute_pair_probe_centers(
        points,
        atom_coords,
        pair_geometry.atom_ext_radii,
        first_indices,
        second_indices,
    )

    feasibility_tol = 100 * eps * pair_geometry.atom_ext_radii.max().clamp_min(1)
    first_normals = pair_geometry.pair_coord_diffs[first_indices, second_indices]
    second_normals = -first_normals
    normal_dists = torch.linalg.norm(first_normals, dim=-1).clamp_min(eps)
    owner_coords = atom_coords[owner_indices.squeeze(-1)].unsqueeze(-2)
    owner_radii = atom_radii.view(1, num_atoms, 1).clamp_min(eps)
    pair_points = points.unsqueeze(-2)
    first_biases = tangency_geometry.tangency_plane_bias[
        first_indices,
        second_indices,
    ]
    second_biases = tangency_geometry.tangency_plane_bias[
        second_indices,
        first_indices,
    ]
    first_circle_cosines = (
        first_biases - (first_normals * owner_coords).sum(dim=-1)
    ) / (owner_radii * normal_dists)
    second_circle_cosines = (
        second_biases - (second_normals * owner_coords).sum(dim=-1)
    ) / (owner_radii * normal_dists)
    contact_plane_mask = (
        (first_circle_cosines.abs() <= 1)
        & (second_circle_cosines.abs() <= 1)
        & ((pair_points * first_normals).sum(dim=-1) <= first_biases)
        & ((pair_points * second_normals).sum(dim=-1) <= second_biases)
    )
    current_feasible = (
        torch.linalg.norm(current_probe_centers.unsqueeze(-2) - atom_coords, dim=-1)
        >= pair_geometry.atom_ext_radii.view(1, 1, -1) - feasibility_tol
    ).all(dim=-1)
    pair_feasible = (
        torch.linalg.norm(pair_probe_centers.unsqueeze(-2) - atom_coords, dim=-1)
        >= pair_geometry.atom_ext_radii.view(1, 1, 1, -1) - feasibility_tol
    ).all(dim=-1)
    pair_errors = torch.abs(
        torch.linalg.norm(points.unsqueeze(-2) - pair_probe_centers, dim=-1)
        - probe_radius,
    )
    pair_scores = torch.where(
        pair_only_mask
        & contact_plane_mask
        & pair_feasible
        & ~current_feasible.unsqueeze(-1),
        pair_errors,
        torch.full((), float("inf"), dtype=dtype, device=device),
    )
    best_scores, best_indices = pair_scores.min(dim=-1)
    best_probe_centers = torch.gather(
        pair_probe_centers,
        2,
        best_indices.view(num_samples, num_atoms, 1, 1).expand(-1, -1, 1, 3),
    ).squeeze(2)

    return torch.where(
        best_scores.isfinite().unsqueeze(-1),
        best_probe_centers,
        current_probe_centers,
    )


def _compute_probe_centers(
    points: torch.Tensor,      # coordinates of sampled points, shape (m, n, 3)
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n,)
    probe_radius: float,       # radius of the probe sphere
) -> tuple[torch.Tensor,       # probe centers, shape (m, n, 3)
           torch.Tensor]:      # validity mask, shape (m, n)
    """
    Find probe centers used to project sampled points onto the solvent-excluded surface.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,).
        probe_radius: Probe sphere radius, scalar.

    Returns:
        probe_centers: Probe center coordinates, shape (m, n, 3).
        valid_point_mask: Whether each sampled point is outside every atom sphere,
            shape (m, n).
    """
    return _compute_active_set_probe_centers(
        points,
        atom_coords,
        atom_radii,
        probe_radius,
    )


def _segment_clearance_mask(
    starts: torch.Tensor,
    ends: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
) -> torch.Tensor:
    """
    Check whether each segment avoids the expanded atom spheres.
    """
    if starts.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=starts.device)
    if atom_coords.shape[0] == 0:
        return torch.ones((starts.shape[0],), dtype=torch.bool, device=starts.device)

    grid_clear = _segment_clearance_mask_with_atom_grid(
        starts,
        ends,
        atom_coords,
        expanded_atom_radii_sq,
    )
    if grid_clear is not None:
        return grid_clear

    max_rows = _max_pairwise_rows(
        starts.shape[0],
        atom_coords.shape[0],
    )
    if starts.shape[0] > max_rows:
        clear_mask = torch.empty(
            starts.shape[0],
            dtype=torch.bool,
            device=starts.device,
        )
        for start in range(0, starts.shape[0], max_rows):
            stop = min(start + max_rows, starts.shape[0])
            clear_mask[start:stop] = _segment_clearance_mask(
                starts[start:stop],
                ends[start:stop],
                atom_coords,
                expanded_atom_radii_sq,
            )
        return clear_mask

    segment_dirs = ends - starts
    segment_lens_sq = segment_dirs.square().sum(dim=-1).clamp_min(
        torch.finfo(starts.dtype).eps,
    )
    start_to_atoms = atom_coords.unsqueeze(0) - starts.unsqueeze(1)
    nearest_params = (
        start_to_atoms * segment_dirs.unsqueeze(1)
    ).sum(dim=-1) / segment_lens_sq.unsqueeze(-1)
    nearest_params = nearest_params.clamp(0, 1)
    closest_points = (
        starts.unsqueeze(1) + nearest_params.unsqueeze(-1) * segment_dirs.unsqueeze(1)
    )
    closest_dists_sq = (
        closest_points - atom_coords.unsqueeze(0)
    ).square().sum(dim=-1)
    tol = (
        256
        * torch.finfo(starts.dtype).eps
        * torch.maximum(closest_dists_sq, expanded_atom_radii_sq.unsqueeze(0)).clamp_min(1)
    )
    return (closest_dists_sq >= expanded_atom_radii_sq.unsqueeze(0) - tol).all(dim=-1)


def _segment_clearance_mask_with_atom_grid(
    starts: torch.Tensor,
    ends: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Segment/sphere clearance using a torch atom-cell broad phase.

    The grid path is device-generic and helps short local segments.  Long
    segments can touch too many cells, so callers fall back to the blocked
    all-pairs implementation when the broad phase would be wider than useful.
    """

    if starts.shape[0] < 64 or atom_coords.shape[0] < 64:
        return None

    max_radius = torch.sqrt(expanded_atom_radii_sq.max().clamp_min(0))
    cell_size = float(max_radius.clamp_min(torch.finfo(starts.dtype).eps).item())
    segment_dirs = ends - starts
    segment_lens = torch.linalg.norm(segment_dirs, dim=-1)
    max_query_radius = float((0.5 * segment_lens.max() + max_radius).item())
    cell_span = int(math.ceil(max_query_radius / cell_size))
    if cell_span > 3:
        return None

    table = _build_atom_cell_table(atom_coords, cell_size)
    if table is None:
        return None

    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-cell_span, cell_span + 1, device=starts.device),
            torch.arange(-cell_span, cell_span + 1, device=starts.device),
            torch.arange(-cell_span, cell_span + 1, device=starts.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    if offsets.shape[0] * table.max_occupancy >= atom_coords.shape[0]:
        return None

    clear_mask = torch.ones(starts.shape[0], dtype=torch.bool, device=starts.device)
    midpoints = 0.5 * (starts + ends)
    midpoint_cells = torch.floor(midpoints / cell_size).to(torch.long)
    segment_lens_sq = segment_dirs.square().sum(dim=-1).clamp_min(
        torch.finfo(starts.dtype).eps,
    )
    rows_per_block = max(
        1,
        min(
            starts.shape[0],
            _PAIRWISE_ELEMENT_BUDGET // max(1, offsets.shape[0] * table.max_occupancy),
        ),
    )
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=starts.device)
    for start in range(0, starts.shape[0], rows_per_block):
        stop = min(start + rows_per_block, starts.shape[0])
        query_cells = midpoint_cells[start:stop].unsqueeze(1) + offsets.view(1, -1, 3)
        flat_keys, flat_valid_cells = _atom_cell_query_keys(
            query_cells.reshape(-1, 3),
            table,
        )
        keys = flat_keys.reshape(stop - start, offsets.shape[0])
        valid_cells = flat_valid_cells.reshape(stop - start, offsets.shape[0])
        starts_in_table = torch.searchsorted(table.sorted_keys, keys, right=False)
        stops_in_table = torch.searchsorted(table.sorted_keys, keys, right=True)
        positions = starts_in_table.unsqueeze(-1) + slot_offsets.view(1, 1, -1)
        has_atom = valid_cells.unsqueeze(-1) & (positions < stops_in_table.unsqueeze(-1))
        atom_indices = table.sorted_atom_indices[
            positions.clamp_max(table.sorted_atom_indices.shape[0] - 1)
        ]
        candidate_coords = atom_coords[atom_indices]
        block_starts = starts[start:stop]
        block_dirs = segment_dirs[start:stop]
        start_to_atoms = candidate_coords - block_starts.view(-1, 1, 1, 3)
        nearest_params = (
            start_to_atoms * block_dirs.view(-1, 1, 1, 3)
        ).sum(dim=-1) / segment_lens_sq[start:stop].view(-1, 1, 1)
        nearest_params = nearest_params.clamp(0, 1)
        closest_points = (
            block_starts.view(-1, 1, 1, 3)
            + nearest_params.unsqueeze(-1) * block_dirs.view(-1, 1, 1, 3)
        )
        closest_dists_sq = (closest_points - candidate_coords).square().sum(dim=-1)
        local_radii_sq = expanded_atom_radii_sq[atom_indices]
        tol = (
            256
            * torch.finfo(starts.dtype).eps
            * torch.maximum(closest_dists_sq, local_radii_sq).clamp_min(1)
        )
        blocked = has_atom & (closest_dists_sq < local_radii_sq - tol)
        clear_mask[start:stop] = ~blocked.any(dim=(1, 2))
    return clear_mask


def _max_pairwise_rows(row_count: int, column_count: int) -> int:
    """
    Bound row chunks for temporary `(rows, atoms)` distance matrices.
    """
    if row_count <= 0:
        return 1
    if column_count <= 0:
        return row_count
    return max(1, min(row_count, _PAIRWISE_ELEMENT_BUDGET // column_count))


def _centers_feasible_against_all_atoms(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    tol_scale: float = 256,
) -> torch.Tensor:
    """
    Check probe-center feasibility without materializing an unbounded full matrix.
    """
    if centers.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=centers.device)
    if atom_coords.shape[0] == 0:
        return torch.ones((centers.shape[0],), dtype=torch.bool, device=centers.device)

    grid_feasible = _centers_feasible_against_all_atoms_grid(
        centers,
        atom_coords,
        expanded_atom_radii_sq,
        tol_scale=tol_scale,
    )
    if grid_feasible is not None:
        return grid_feasible

    feasible = torch.empty(centers.shape[0], dtype=torch.bool, device=centers.device)
    max_rows = _max_pairwise_rows(centers.shape[0], atom_coords.shape[0])
    for start in range(0, centers.shape[0], max_rows):
        stop = min(start + max_rows, centers.shape[0])
        center_sq_dists = (
            centers[start:stop].unsqueeze(1) - atom_coords.unsqueeze(0)
        ).square().sum(dim=-1)
        tol = (
            tol_scale
            * torch.finfo(center_sq_dists.dtype).eps
            * torch.maximum(
                center_sq_dists,
                expanded_atom_radii_sq.unsqueeze(0),
            ).clamp_min(1)
        )
        feasible[start:stop] = (
            center_sq_dists >= expanded_atom_radii_sq.unsqueeze(0) - tol
        ).all(dim=-1)
    return feasible


class _AtomCellTable(NamedTuple):
    origin: torch.Tensor
    dims: torch.Tensor
    cell_size: float
    sorted_keys: torch.Tensor
    sorted_atom_indices: torch.Tensor
    max_occupancy: int


def _build_atom_cell_table(
    atom_coords: torch.Tensor,
    cell_size: float,
) -> Optional[_AtomCellTable]:
    """Build a sparse atom-cell table using torch tensors on the input device."""

    if atom_coords.shape[0] == 0 or cell_size <= 0:
        return None

    atom_cells = torch.floor(atom_coords / cell_size).to(torch.long)
    origin = atom_cells.min(dim=0).values - 1
    max_cell = atom_cells.max(dim=0).values + 1
    dims = (max_cell - origin + 1).clamp_min(1)
    if bool((dims <= 0).any().item()):
        return None

    shifted = atom_cells - origin.unsqueeze(0)
    keys = _linear_cell_keys(shifted, dims)
    order = torch.argsort(keys)
    sorted_keys = keys[order].contiguous()
    _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    max_occupancy = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_occupancy <= 0:
        return None
    return _AtomCellTable(
        origin=origin,
        dims=dims,
        cell_size=float(cell_size),
        sorted_keys=sorted_keys,
        sorted_atom_indices=order.to(torch.long).contiguous(),
        max_occupancy=max_occupancy,
    )


def _linear_cell_keys(shifted_cells: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Linearize already-shifted integer grid cells."""

    return (shifted_cells[..., 0] * dims[1] + shifted_cells[..., 1]) * dims[2] + shifted_cells[..., 2]


def _atom_cell_query_keys(
    query_cells: torch.Tensor,
    table: _AtomCellTable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return linear query keys and whether each query cell is inside the table."""

    shifted = query_cells - table.origin.view(1, 3)
    valid = ((shifted >= 0) & (shifted < table.dims.view(1, 3))).all(dim=-1)
    safe_shifted = torch.where(valid.unsqueeze(-1), shifted, torch.zeros_like(shifted))
    return _linear_cell_keys(safe_shifted, table.dims), valid


def _centers_feasible_against_all_atoms_grid(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    tol_scale: float = 256,
) -> Optional[torch.Tensor]:
    """
    Device-generic point/sphere feasibility using a sparse atom-cell table.
    """

    if centers.shape[0] < 1024 or atom_coords.shape[0] < 64:
        return None

    max_radius = torch.sqrt(expanded_atom_radii_sq.max().clamp_min(0))
    cell_size = float(max_radius.clamp_min(torch.finfo(centers.dtype).eps).item())
    table = _build_atom_cell_table(atom_coords, cell_size)
    if table is None or table.max_occupancy * 27 >= atom_coords.shape[0]:
        return None

    return _centers_feasible_with_atom_table(
        centers,
        atom_coords,
        expanded_atom_radii_sq,
        table,
        tol_scale=tol_scale,
    )


def _centers_feasible_with_atom_table(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    table: _AtomCellTable,
    *,
    tol_scale: float = 256,
) -> torch.Tensor:
    """Point/sphere feasibility using a prebuilt sparse atom-cell table."""

    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-1, 2, device=centers.device),
            torch.arange(-1, 2, device=centers.device),
            torch.arange(-1, 2, device=centers.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    feasible = torch.ones(centers.shape[0], dtype=torch.bool, device=centers.device)
    rows_per_block = max(
        1,
        min(centers.shape[0], _PAIRWISE_ELEMENT_BUDGET // max(1, 27 * table.max_occupancy)),
    )
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=centers.device)

    for start in range(0, centers.shape[0], rows_per_block):
        stop = min(start + rows_per_block, centers.shape[0])
        block_centers = centers[start:stop]
        center_cells = torch.floor(block_centers / table.cell_size).to(torch.long)
        query_cells = center_cells.unsqueeze(1) + offsets.view(1, -1, 3)
        flat_keys, flat_valid_cells = _atom_cell_query_keys(
            query_cells.reshape(-1, 3),
            table,
        )
        keys = flat_keys.reshape(block_centers.shape[0], offsets.shape[0])
        valid_cells = flat_valid_cells.reshape(block_centers.shape[0], offsets.shape[0])
        starts_in_table = torch.searchsorted(table.sorted_keys, keys, right=False)
        stops_in_table = torch.searchsorted(table.sorted_keys, keys, right=True)

        positions = starts_in_table.unsqueeze(-1) + slot_offsets.view(1, 1, -1)
        has_atom = valid_cells.unsqueeze(-1) & (positions < stops_in_table.unsqueeze(-1))
        safe_positions = positions.clamp_max(table.sorted_atom_indices.shape[0] - 1)
        atom_indices = table.sorted_atom_indices[safe_positions]
        candidate_coords = atom_coords[atom_indices]
        sq_dists = (
            block_centers.view(-1, 1, 1, 3) - candidate_coords
        ).square().sum(dim=-1)
        local_radii_sq = expanded_atom_radii_sq[atom_indices]
        tol = (
            float(tol_scale)
            * torch.finfo(centers.dtype).eps
            * torch.maximum(sq_dists, local_radii_sq).clamp_min(1)
        )
        blocked = has_atom & (sq_dists < local_radii_sq - tol)
        feasible[start:stop] = ~blocked.any(dim=(1, 2))

    return feasible


def _grid_free_mask(
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    bbox_min: torch.Tensor,
    grid_spacing: float,
    dims: torch.Tensor,
    grid_points: int,
) -> torch.Tensor:
    """
    Mark grid points outside all expanded atom spheres.
    """
    return _grid_free_mask_with_torch(
        atom_coords,
        expanded_atom_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )


def _grid_free_mask_with_kdtree(
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    bbox_min: torch.Tensor,
    grid_spacing: float,
    dims: torch.Tensor,
    grid_points: int,
) -> torch.Tensor:
    """
    Compatibility wrapper for the device-generic occupancy implementation.
    """

    return _grid_free_mask_with_torch(
        atom_coords,
        expanded_atom_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )


def _grid_free_mask_with_torch(
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    bbox_min: torch.Tensor,
    grid_spacing: float,
    dims: torch.Tensor,
    grid_points: int,
) -> torch.Tensor:
    """
    Device-generic occupancy fallback with bounded temporary matrices.
    """
    dtype = atom_coords.dtype
    device = atom_coords.device
    raster_free = _grid_free_mask_by_atom_raster(
        atom_coords,
        expanded_atom_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )
    if raster_free is not None:
        return raster_free

    flat_free = torch.empty(grid_points, dtype=torch.bool, device=device)
    nx, ny, nz = (int(dim.item()) for dim in dims)
    yz_size = ny * nz
    block_size = _max_pairwise_rows(grid_points, atom_coords.shape[0])
    max_radius = torch.sqrt(expanded_atom_radii_sq.max().clamp_min(0))
    cell_size = float(max_radius.clamp_min(torch.finfo(dtype).eps).item())
    atom_table = _build_atom_cell_table(atom_coords, cell_size)
    if atom_table is not None and atom_table.max_occupancy * 27 < atom_coords.shape[0]:
        block_size = max(
            1,
            min(grid_points, _PAIRWISE_ELEMENT_BUDGET // max(1, 27 * atom_table.max_occupancy)),
        )
    for start in range(0, grid_points, block_size):
        stop = min(start + block_size, grid_points)
        flat_indices = torch.arange(start, stop, device=device, dtype=torch.long)
        ix = flat_indices // yz_size
        rem = flat_indices - ix * yz_size
        iy = rem // nz
        iz = rem - iy * nz
        grid_centers = torch.stack(
            (
                bbox_min[0] + ix.to(dtype) * grid_spacing,
                bbox_min[1] + iy.to(dtype) * grid_spacing,
                bbox_min[2] + iz.to(dtype) * grid_spacing,
            ),
            dim=-1,
        )
        if atom_table is not None and atom_table.max_occupancy * 27 < atom_coords.shape[0]:
            flat_free[start:stop] = _centers_feasible_with_atom_table(
                grid_centers,
                atom_coords,
                expanded_atom_radii_sq,
                atom_table,
            )
        else:
            sq_dists = (
                grid_centers.unsqueeze(1) - atom_coords.unsqueeze(0)
            ).square().sum(dim=-1)
            flat_free[start:stop] = (
                sq_dists >= expanded_atom_radii_sq.unsqueeze(0)
            ).all(dim=-1)
    return flat_free


def _grid_free_mask_by_atom_raster(
    atom_coords: torch.Tensor,
    expanded_atom_radii_sq: torch.Tensor,
    bbox_min: torch.Tensor,
    grid_spacing: float,
    dims: torch.Tensor,
    grid_points: int,
) -> Optional[torch.Tensor]:
    """Mark occupied grid cells by rasterizing each expanded atom sphere."""

    if atom_coords.shape[0] == 0:
        return torch.ones((grid_points,), dtype=torch.bool, device=atom_coords.device)

    dtype = atom_coords.dtype
    device = atom_coords.device
    expanded_atom_radii = torch.sqrt(expanded_atom_radii_sq.clamp_min(0))
    max_radius = float(expanded_atom_radii.max().item())
    max_step = int(math.ceil(max_radius / float(grid_spacing)))
    if max_step <= 0:
        return torch.ones((grid_points,), dtype=torch.bool, device=device)
    offset_count = (2 * max_step + 1) ** 3
    if offset_count <= 0:
        return None

    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-max_step, max_step + 1, device=device),
            torch.arange(-max_step, max_step + 1, device=device),
            torch.arange(-max_step, max_step + 1, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    # Keep temporary `(atoms, offsets)` tensors bounded for both CPU and GPU.
    atom_chunk = max(1, min(atom_coords.shape[0], _PAIRWISE_ELEMENT_BUDGET // offset_count))
    occupied = torch.zeros((grid_points,), dtype=torch.bool, device=device)
    dims_long = dims.to(device=device, dtype=torch.long)
    ny = int(dims_long[1].item())
    nz = int(dims_long[2].item())

    for start in range(0, atom_coords.shape[0], atom_chunk):
        stop = min(start + atom_chunk, atom_coords.shape[0])
        chunk_coords = atom_coords[start:stop]
        center_cells = torch.round((chunk_coords - bbox_min.view(1, 3)) / grid_spacing).to(
            torch.long,
        )
        cells = center_cells.unsqueeze(1) + offsets.view(1, -1, 3)
        valid = ((cells >= 0) & (cells < dims_long.view(1, 1, 3))).all(dim=-1)
        safe_cells = torch.where(valid.unsqueeze(-1), cells, torch.zeros_like(cells))
        grid_coords = bbox_min.view(1, 1, 3) + safe_cells.to(dtype) * float(grid_spacing)
        sq_dists = (grid_coords - chunk_coords.unsqueeze(1)).square().sum(dim=-1)
        inside = valid & (sq_dists < expanded_atom_radii_sq[start:stop].view(-1, 1))
        flat_indices = (
            (safe_cells[..., 0] * ny + safe_cells[..., 1]) * nz + safe_cells[..., 2]
        )
        occupied[flat_indices[inside]] = True

    return ~occupied


def _external_reachable_grid(
    atom_coords: torch.Tensor,
    expanded_atom_radii: torch.Tensor,
    query_centers: torch.Tensor,
    grid_spacing: float,
    max_grid_points: int,
    cache_key: Optional[tuple[object, ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Build a flood-filled grid of center-space connected to the exterior.
    """
    if cache_key is not None:
        cached = _EXTERNAL_GRID_CACHE.get(cache_key)
        if cached is not None:
            _EXTERNAL_GRID_CACHE.move_to_end(cache_key)
            return cached

    dtype = atom_coords.dtype
    device = atom_coords.device
    margin = max(2.0 * grid_spacing, float(expanded_atom_radii.max().item()) + grid_spacing)
    sphere_min = atom_coords - expanded_atom_radii.unsqueeze(-1)
    sphere_max = atom_coords + expanded_atom_radii.unsqueeze(-1)
    bbox_min = sphere_min.min(dim=0).values - margin
    bbox_max = sphere_max.max(dim=0).values + margin

    extents = (bbox_max - bbox_min).clamp_min(grid_spacing)
    dims = torch.ceil(extents / grid_spacing).to(torch.long) + 1
    grid_points = int(dims.prod().item())
    if grid_points > max_grid_points:
        scale = (grid_points / max_grid_points) ** (1.0 / 3.0)
        grid_spacing *= scale
        dims = torch.ceil(extents / grid_spacing).to(torch.long) + 1
        grid_points = int(dims.prod().item())

    nx, ny, nz = (int(dim.item()) for dim in dims)
    expanded_atom_radii_sq = expanded_atom_radii.square()
    flat_free = _grid_free_mask(
        atom_coords,
        expanded_atom_radii,
        expanded_atom_radii_sq,
        bbox_min,
        grid_spacing,
        dims,
        grid_points,
    )

    free_grid = flat_free.reshape(nx, ny, nz)
    reached_grid = _flood_fill_reachable_grid(free_grid)

    result = (
        reached_grid,
        bbox_min,
        grid_spacing,
        dims.to(device=device),
    )
    if cache_key is not None:
        _EXTERNAL_GRID_CACHE[cache_key] = result
        _EXTERNAL_GRID_CACHE.move_to_end(cache_key)
        while len(_EXTERNAL_GRID_CACHE) > _EXTERNAL_GRID_CACHE_SIZE:
            _EXTERNAL_GRID_CACHE.popitem(last=False)
    return result


def _flood_fill_reachable_grid(free_grid: torch.Tensor) -> torch.Tensor:
    """Flood-fill exterior-reachable cells using torch operations on-device."""

    seed_grid = torch.zeros_like(free_grid)
    seed_grid[0, :, :] = free_grid[0, :, :]
    seed_grid[-1, :, :] = free_grid[-1, :, :]
    seed_grid[:, 0, :] = free_grid[:, 0, :]
    seed_grid[:, -1, :] = free_grid[:, -1, :]
    seed_grid[:, :, 0] = free_grid[:, :, 0]
    seed_grid[:, :, -1] = free_grid[:, :, -1]

    reached_grid = seed_grid
    max_iterations = sum(int(size) for size in free_grid.shape)
    for _ in range(max_iterations):
        expanded = reached_grid.clone()
        expanded[1:, :, :] = expanded[1:, :, :] | reached_grid[:-1, :, :]
        expanded[:-1, :, :] = expanded[:-1, :, :] | reached_grid[1:, :, :]
        expanded[:, 1:, :] = expanded[:, 1:, :] | reached_grid[:, :-1, :]
        expanded[:, :-1, :] = expanded[:, :-1, :] | reached_grid[:, 1:, :]
        expanded[:, :, 1:] = expanded[:, :, 1:] | reached_grid[:, :, :-1]
        expanded[:, :, :-1] = expanded[:, :, :-1] | reached_grid[:, :, 1:]
        expanded = expanded & free_grid
        if bool(torch.equal(expanded, reached_grid)):
            return reached_grid
        reached_grid = expanded
    return reached_grid


def _probe_centers_accessible_from_exterior(
    probe_centers: torch.Tensor,
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    valid_center_mask: Optional[torch.Tensor] = None,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = 2_000_000,
    assume_centers_feasible: bool = False,
) -> torch.Tensor:
    """
    Return whether probe centers lie in the exterior component of center-space.
    """
    if valid_center_mask is None:
        candidate_mask = torch.isfinite(probe_centers).all(dim=-1)
    else:
        candidate_mask = valid_center_mask & torch.isfinite(probe_centers).all(dim=-1)

    accessible_mask = torch.zeros_like(candidate_mask)
    flat_candidate_indices = candidate_mask.reshape(-1).nonzero(
        as_tuple=False,
    ).reshape(-1)
    if flat_candidate_indices.numel() == 0:
        return accessible_mask
    if atom_coords.shape[0] < 4:
        return candidate_mask

    calc_device = atom_coords.device
    calc_dtype = atom_coords.dtype if atom_coords.dtype in (torch.float32, torch.float64) else torch.float32
    calc_centers = probe_centers.reshape(-1, 3)[flat_candidate_indices].to(
        device=calc_device,
        dtype=calc_dtype,
    )
    calc_atom_coords = atom_coords.to(dtype=calc_dtype)
    calc_atom_radii = atom_radii.to(dtype=calc_dtype)
    expanded_atom_radii = calc_atom_radii + probe_radius
    expanded_atom_radii_sq = expanded_atom_radii.square()

    if assume_centers_feasible:
        feasible_centers = torch.ones(
            calc_centers.shape[0],
            dtype=torch.bool,
            device=calc_device,
        )
    else:
        feasible_centers = _centers_feasible_against_all_atoms(
            calc_centers,
            calc_atom_coords,
            expanded_atom_radii_sq,
        )
    feasible_indices = feasible_centers.nonzero(as_tuple=False).reshape(-1)
    if feasible_indices.numel() == 0:
        return accessible_mask

    effective_spacing = grid_spacing
    if effective_spacing is None:
        effective_spacing = min(0.25, max(0.12, float(probe_radius) / 5.0))
    effective_spacing = float(effective_spacing)
    if effective_spacing <= 0:
        raise ValueError("grid_spacing must be positive")
    if max_grid_points <= 0:
        raise ValueError("max_grid_points must be positive")

    grid_cache_key: Optional[tuple[object, ...]] = (
        _tensor_cache_token(calc_atom_coords),
        _tensor_cache_token(calc_atom_radii),
        float(probe_radius),
        effective_spacing,
        int(max_grid_points),
    )

    feasible_query_centers = calc_centers[feasible_indices]
    molecule_center = calc_atom_coords.mean(dim=0, keepdim=True)
    exterior_dirs = feasible_query_centers - molecule_center
    exterior_dir_norms = torch.linalg.norm(exterior_dirs, dim=-1, keepdim=True)
    fallback_dirs = torch.zeros_like(exterior_dirs)
    fallback_dirs[:, 0] = 1
    exterior_dirs = torch.where(
        exterior_dir_norms > torch.finfo(calc_dtype).eps,
        exterior_dirs / exterior_dir_norms.clamp_min(torch.finfo(calc_dtype).eps),
        fallback_dirs,
    )
    exterior_radius = (
        torch.linalg.norm(calc_atom_coords - molecule_center, dim=-1).max()
        + expanded_atom_radii.max()
        + 4.0 * effective_spacing
    )
    exterior_points = molecule_center + exterior_dirs * exterior_radius
    query_accessible = _segment_clearance_mask(
        feasible_query_centers,
        exterior_points,
        calc_atom_coords,
        expanded_atom_radii_sq,
    )
    unresolved_query_mask = ~query_accessible
    unresolved_query_indices = unresolved_query_mask.nonzero(
        as_tuple=False,
    ).reshape(-1)
    if unresolved_query_indices.numel() == 0:
        flat_accessible = accessible_mask.reshape(-1)
        flat_accessible[flat_candidate_indices[feasible_indices[query_accessible]]] = True
        return accessible_mask
    grid_query_centers = feasible_query_centers[unresolved_query_indices]

    reachable_grid, bbox_min, effective_spacing, dims = _external_reachable_grid(
        calc_atom_coords,
        expanded_atom_radii,
        grid_query_centers,
        effective_spacing,
        max_grid_points,
        cache_key=grid_cache_key,
    )

    frac_indices = (grid_query_centers - bbox_min) / effective_spacing
    base_indices = torch.floor(frac_indices).to(torch.long)
    dims_long = dims.to(dtype=torch.long)
    grid_accessible = torch.zeros(
        grid_query_centers.shape[0],
        dtype=torch.bool,
        device=calc_device,
    )

    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offsets = torch.tensor([dx, dy, dz], dtype=torch.long, device=calc_device)
                corner_indices = torch.minimum(
                    torch.maximum(
                        base_indices + offsets,
                        torch.zeros((1, 3), dtype=torch.long, device=calc_device),
                    ),
                    (dims_long - 1).unsqueeze(0),
                )
                reached_corner_mask = reachable_grid[
                    corner_indices[:, 0],
                    corner_indices[:, 1],
                    corner_indices[:, 2],
                ]
                unresolved = reached_corner_mask & ~grid_accessible
                unresolved_indices = unresolved.nonzero(as_tuple=False).reshape(-1)
                if unresolved_indices.numel() == 0:
                    continue
                corner_points = bbox_min + corner_indices[unresolved].to(calc_dtype) * effective_spacing
                segment_clear = _segment_clearance_mask(
                    grid_query_centers[unresolved],
                    corner_points,
                    calc_atom_coords,
                    expanded_atom_radii_sq,
                )
                grid_accessible[unresolved_indices[segment_clear]] = True

    query_accessible[unresolved_query_indices[grid_accessible]] = True

    flat_accessible = accessible_mask.reshape(-1)
    flat_accessible[flat_candidate_indices[feasible_indices[query_accessible]]] = True
    return accessible_mask


def project_points(
    points: torch.Tensor,      # coordinates of sampled points, shape (m, n, 3)
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n,)
    probe_radius: float,       # radius of the probe sphere
) -> tuple[torch.Tensor,       # projected points, shape (m, n, 3)
           torch.Tensor]:      # validity mask, shape (m, n)
    """
    Project sampled points onto the solvent-excluded surface.

    Args:
        points: Sampled point coordinates, shape (m, n, 3).
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n,) or any shape with n elements.
        probe_radius: Probe sphere radius, scalar.

    Returns:
        projected_points_on_ses: Projected SES points, shape (m, n, 3).
        valid_point_mask: Whether each projected point has an externally
            accessible probe center, shape (m, n).
    """
    prepared = _prepare_projection_inputs(
        points,
        atom_coords,
        atom_radii,
        probe_radius,
    )
    points = prepared.points
    atom_coords = prepared.atom_coords
    atom_radii = prepared.atom_radii
    num_samples, num_atoms, _ = points.shape
    common_device = atom_coords.device

    if num_samples == 0 or num_atoms == 0:
        empty_mask = torch.empty(points.shape[:2], dtype=torch.bool, device=common_device)
        return points, empty_mask

    # Compute the probe centers and validity mask for the sampled points.
    probe_centers, valid_point_mask = _compute_probe_centers(
        points,
        atom_coords,
        atom_radii,
        probe_radius,
    )
    valid_point_mask = valid_point_mask & _probe_centers_accessible_from_exterior(
        probe_centers,
        atom_coords,
        atom_radii,
        probe_radius,
        valid_point_mask,
    )

    # Finally, project the original points onto the probe spheres.
    point_probe_center_diffs = points - probe_centers
    point_probe_center_dists = torch.linalg.norm(point_probe_center_diffs, dim=-1)
    projected_points_on_ses = (
        probe_centers
        + point_probe_center_diffs
        * (probe_radius / point_probe_center_dists).unsqueeze(-1)
    )

    return projected_points_on_ses, valid_point_mask


__all__ = [
    "project_points",
    "sample_atom_points",
    "sample_projected_points",
]
