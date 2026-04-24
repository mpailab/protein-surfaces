from __future__ import annotations

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


def _select_suitable_atom_indices(
    suitable_atom_mask: torch.Tensor,        # shape (m, n, n), dtype bool
    geodesic_distances: torch.Tensor,        # shape (m, n, n), dtype float
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

    # Invalid entries get -inf, so sorting by descending distance pushes them
    # behind every valid candidate. Stable sorting preserves atom-index order for
    # exact ties, matching the previous deterministic tie-break.
    distances = torch.where(
        suitable_atom_mask,
        geodesic_distances,
        torch.full((), float("-inf"), dtype=geodesic_distances.dtype, device=device),
    )
    ordered_indices = torch.argsort(distances, dim=-1, descending=True, stable=True)

    first_two_indices = ordered_indices[..., :2].to(index_dtype)
    first_two_mask = torch.gather(suitable_atom_mask, -1, ordered_indices[..., :2])

    return torch.where(first_two_mask, first_two_indices, owner_indices)


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
    valid_point_mask = (
        point_atom_sq_dists >= atom_radii.square().view(1, 1, -1)
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

    return _select_suitable_atom_indices(
        suitable_atom_mask,
        geodesic_distances,
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
    probe_offset_dists = torch.sqrt(
        atom_ext_radii_sq.unsqueeze(0) - projected_atom_sq_dists
    )
    probe_shift_coefs = probe_offset_dists / projected_point_atom_dists

    return projected_atoms + probe_directions * probe_shift_coefs.unsqueeze(-1)


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
