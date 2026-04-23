from __future__ import annotations

import torch


def _select_suitable_atom_indices(
    suitable_atom_mask: torch.Tensor, # shape (m, n, n), dtype bool
) -> torch.Tensor:                    # shape (m, n, 2), dtype torch.int
    """
    For each `(sample, atom)` row in a boolean mask of shape `(m, n, n)`,
    return the two smallest valid atom indices along the last axis.
    `suitable_atom_mask` is assumed to already have a zero diagonal, so self-pairs
    are excluded before this helper is called. Missing atom indices fall back to
    the owner atom index.
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

    # If there are fewer than 2 atoms, we can skip the topk and return the owner indices.
    if num_atoms < 2:
        return owner_indices

    # Form candidate indices by replacing False entries in the mask with a sentinel value 
    # that is larger than any valid index. This way, when we take the top 2 smallest 
    # values, the invalid entries will be pushed to the end and we can easily detect 
    # if there were fewer than 2 valid indices.
    sentinel = num_atoms
    candidate_indices = torch.where(
        suitable_atom_mask,
        partner_indices.view(1, 1, num_atoms),
        torch.full((), sentinel, dtype=index_dtype, device=device),
    )

    # Find the two smallest valid atom indices along the last axis. If there are fewer 
    # than 2 valid indices, the sentinel value will be among the top 2, and we can use 
    # that to detect the fallback case.
    first_two_indices = torch.topk(candidate_indices, k=2, dim=-1, 
                                   largest=False, sorted=True).values

    # If the smallest index is the sentinel, it means there are no valid indices, 
    # so we return the owner indices.
    return torch.where(first_two_indices == sentinel, owner_indices, first_two_indices)


def project_points_to_ses(
    points: torch.Tensor,      # coordinates of sampled points, shape (m, n, 3)
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n,)
    probe_radius: float,       # radius of the probe sphere
) -> tuple[torch.Tensor,       # projected points, shape (m, n, 3)
           torch.Tensor]:      # validity mask, shape (m, n)
    """
    Project sampled points onto the solvent-excluded surface.
    """
    if probe_radius <= 0:
        raise ValueError("probe_radius must be positive")

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (m, n, 3)")
    if atom_coords.ndim != 2 or atom_coords.shape[-1] != 3:
        raise ValueError("atom_coords must have shape (n, 3)")

    num_samples, num_atoms, _ = points.shape
    if atom_coords.shape[0] != num_atoms:
        raise ValueError("points.shape[1] must match atom_coords.shape[0]")

    point_dtype = points.dtype if points.is_floating_point() else torch.float32
    coord_dtype = atom_coords.dtype if atom_coords.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(point_dtype, coord_dtype)
    common_device = atom_coords.device

    points = points.to(dtype=common_dtype, device=common_device)
    atom_coords = atom_coords.to(dtype=common_dtype, device=common_device)
    atom_radii = atom_radii.reshape(-1).to(dtype=common_dtype, device=common_device)

    if atom_radii.numel() != num_atoms:
        raise ValueError("radii must contain one radius per atom")
    if bool((atom_radii < 0).any().item()):
        raise ValueError("atom_radii must be non-negative")

    if num_samples == 0 or num_atoms == 0:
        empty_mask = torch.empty(points.shape[:2], dtype=torch.bool, device=common_device)
        return points, empty_mask
    
    # Step 1: Compute the validity mask for each point with respect to each atom.
    # A point is valid if it lies outside all atom spheres.
    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    coord_sq_norms = atom_coords.square().sum(dim=-1)
    point_coord_dots = torch.einsum("mik,jk->mij", points, atom_coords)
    point_atom_sq_dists = (
        point_sq_norms + coord_sq_norms.view(1, 1, -1) - 2 * point_coord_dots
    )
    valid_point_mask = (
        point_atom_sq_dists >= atom_radii.square().view(1, 1, -1)
    ).all(dim=-1) # shape (m, n)

    # Step 2: Compute the validity mask for each pair of atoms.
    # A pair of atoms (i, j) is valid if the expanded spheres (atom + probe) 
    # intersect in a non-degenerate circle.
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

    # Step 3: Compute the tangency plane masks for each point and atom pair. 
    # The tangency plane is a plane that contains the circle of tangency between 
    # the atom sphere containing the point and the probe sphere.
    # Reuse the point/atom dot products from step 1 instead of running a
    # second contraction against `pair_coord_diffs`.
    point_coord_diffs_dots = (
        point_coord_dots.diagonal(dim1=1, dim2=2).unsqueeze(-1) - point_coord_dots
    )
    tangency_plane_bias = 0.5 * (
        pair_coord_diffs_sq
        + coord_sq_norms[:, None]
        - coord_sq_norms[None, :]
        - (atom_radii / atom_ext_radii)[:, None]
        * (
            pair_coord_diffs_sq
            + atom_ext_radii_sq[:, None]
            - atom_ext_radii_sq[None, :]
        )
    )
    tangency_plane_mask = point_coord_diffs_dots <= tangency_plane_bias # shape (m, n, n)

    # Step 4: Combine the masks to find suitable atom indices for each point.
    suitable_atom_mask = (
        valid_point_mask.unsqueeze(-1) &
        valid_atom_pair_mask.unsqueeze(0) &
        tangency_plane_mask
    ) # shape (m, n, n)
    suitable_atom_mask.diagonal(dim1=-2, dim2=-1).zero_()
    suitable_atom_indices = _select_suitable_atom_indices(suitable_atom_mask)

    # Step 5: Build the affine projector onto the intersection of the pair planes 
    # corresponding to the suitable atom indices.
    owner_indices = torch.arange(num_atoms, device=common_device, dtype=torch.long)
    owner_indices = owner_indices.view(1, num_atoms, 1).expand(num_samples, -1, 2)
    plane_normals = pair_coord_diffs[owner_indices, suitable_atom_indices]
    plane_bias = 0.5 * (coord_sq_norms[:, None] - coord_sq_norms[None, :]
                        - atom_ext_radii_sq[:, None] + atom_ext_radii_sq[None, :])
    plane_bias = plane_bias[owner_indices, suitable_atom_indices].unsqueeze(-1)
    grams = plane_normals @ plane_normals.transpose(-1, -2)
    grams_inv = torch.linalg.pinv(grams, hermitian=True)
    affine_projection = plane_normals.transpose(-1, -2) @ grams_inv @ plane_normals
    affine_shift = (plane_normals.transpose(-1, -2) @ grams_inv @ plane_bias).squeeze(-1)
    def affine_projector(x: torch.Tensor) -> torch.Tensor:
        return x - (affine_projection @ x.unsqueeze(-1)).squeeze(-1) + affine_shift

    # Step 6: Find the probe centers by projecting the points onto the affine plane
    # and shifting them outwards from the projected atom centers.
    projected_points = affine_projector(points)
    broadcasted_atom_coords = atom_coords.unsqueeze(0).expand(num_samples, -1, -1)
    projected_atoms = affine_projector(broadcasted_atom_coords)
    probe_directions = projected_points - projected_atoms
    projected_point_atom_dists = torch.linalg.norm(probe_directions, dim=-1)
    projected_atom_sq_dists = (projected_atoms - broadcasted_atom_coords).square().sum(dim=-1)
    probe_offset_dists = torch.sqrt(atom_ext_radii_sq.unsqueeze(0) - projected_atom_sq_dists)
    probe_shift_coefs = probe_offset_dists / projected_point_atom_dists
    probe_centers = (projected_atoms + probe_directions * probe_shift_coefs.unsqueeze(-1))

    # Step 7: Finally, project the original points onto the probe spheres.
    point_probe_center_diffs = points - probe_centers
    point_probe_center_dists = torch.linalg.norm(point_probe_center_diffs, dim=-1)
    projected_points_on_ses = (
        probe_centers
        + point_probe_center_diffs
        * (probe_radius / point_probe_center_dists).unsqueeze(-1)
    )

    return projected_points_on_ses, valid_point_mask
