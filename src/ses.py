"""Solvent-excluded surface sampling and projection interfaces.

External interfaces:
    sample_atom_sphere_points: sample evenly distributed points on atom spheres.
    sample_ses_points: sample atom-sphere points and project them onto SES.
    project_points_to_ses: project sampled atom-sphere points onto SES.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch


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


def sample_atom_sphere_points(
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


def sample_ses_points(
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n, 1), (n,), or n elements
    m: int,                    # number of points sampled on each atom sphere
    probe_radius: float,       # radius of the probe sphere
) -> tuple[torch.Tensor,       # projected SES points, shape (m, n, 3)
           torch.Tensor]:      # validity mask, shape (m, n)
    """
    Sample atom-sphere points and project them onto the solvent-excluded surface.

    Args:
        atom_coords: Atom center coordinates, shape (n, 3).
        atom_radii: Atom radii, shape (n, 1), (n,), or any shape with n elements.
        m: Number of points sampled on each atom sphere.
        probe_radius: Probe sphere radius, scalar.

    Returns:
        projected_points_on_ses: Projected SES points, shape (m, n, 3).
        valid_point_mask: Whether each sampled point is outside every atom sphere,
            shape (m, n).
    """
    points = sample_atom_sphere_points(atom_coords, atom_radii, m)
    return project_points_to_ses(
        points=points,
        atom_coords=atom_coords,
        atom_radii=atom_radii,
        probe_radius=probe_radius,
    )


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
    num_samples = points.shape[0]

    # Step 1: Compute the validity mask for each point with respect to each atom.
    # A point is valid if it lies outside all atom spheres.
    point_geometry = _compute_point_atom_geometry(points, atom_coords, atom_radii)

    # Step 2: Compute the validity mask for each pair of atoms.
    # A pair of atoms (i, j) is valid if the expanded spheres (atom + probe)
    # intersect in a non-degenerate circle.
    pair_geometry = _compute_pair_geometry(atom_coords, atom_radii, probe_radius)

    # Step 3: Compute the tangency plane masks for each point and atom pair.
    # The tangency plane is a plane that contains the circle of tangency between
    # the atom sphere containing the point and the probe sphere.
    # Reuse the point/atom dot products from step 1 instead of running a
    # second contraction against `pair_coord_diffs`.
    tangency_geometry = _compute_tangency_geometry(
        point_geometry,
        pair_geometry,
        atom_radii,
    )

    # Step 4: Compute shortest geodesic distances from each sampled point on
    # the owner atom sphere to each pair's tangency circle. The rolling-probe
    # center plane is parallel to this tangency plane; the tangency circle is
    # the actual trace of that pair constraint on the owner atom sphere.
    geodesic_distances = _compute_geodesic_distances(
        points,
        atom_coords,
        atom_radii,
        pair_geometry,
        tangency_geometry,
    )

    # Step 5: Combine the masks to find suitable atom indices for each point.
    suitable_atom_indices = _compute_suitable_atom_indices(
        point_geometry,
        pair_geometry,
        tangency_geometry,
        geodesic_distances,
    )

    # Step 6: Build the affine projector onto the intersection of the pair planes
    # corresponding to the suitable atom indices.
    affine_projection, affine_shift = _build_affine_projector(
        num_samples,
        point_geometry,
        pair_geometry,
        suitable_atom_indices,
    )

    # Step 7: Find the probe centers by projecting the points onto the affine plane
    # and shifting them outwards from the projected atom centers.
    probe_centers = _recover_probe_centers(
        points,
        atom_coords,
        pair_geometry.atom_ext_radii_sq,
        affine_projection,
        affine_shift,
    )
    probe_centers = _prefer_pair_only_probe_centers(
        points,
        atom_coords,
        pair_geometry,
        suitable_atom_indices,
        probe_radius,
        probe_centers,
    )
    probe_centers = _prefer_any_pair_only_probe_centers(
        points,
        atom_coords,
        atom_radii,
        pair_geometry,
        tangency_geometry,
        probe_radius,
        probe_centers,
    )

    return probe_centers, point_geometry.valid_point_mask


def project_points_to_ses(
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
        valid_point_mask: Whether each sampled point is outside every atom sphere,
            shape (m, n).
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

    # Finally, project the original points onto the probe spheres.
    point_probe_center_diffs = points - probe_centers
    point_probe_center_dists = torch.linalg.norm(point_probe_center_diffs, dim=-1)
    projected_points_on_ses = (
        probe_centers
        + point_probe_center_diffs
        * (probe_radius / point_probe_center_dists).unsqueeze(-1)
    )

    return projected_points_on_ses, valid_point_mask
