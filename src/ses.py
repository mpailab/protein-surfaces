from __future__ import annotations

import torch


def _compute_valid_point_mask(
    points: torch.Tensor,
    coords: torch.Tensor,
    atom_radii: torch.Tensor,
) -> torch.Tensor:
    """
    Return the boolean validity mask together with the squared point-atom
    distances used to build it.

    `points[k, i]` is treated as valid iff for every atom center `coords[j]`
    we have

        ||points[k, i] - coords[j]||^2 >= radii[j]^2.

    A point lying exactly on an atom sphere is therefore still considered
    valid.
    """
    point_sq_norms = points.square().sum(dim=-1, keepdim=True)
    coord_sq_norms = coords.square().sum(dim=-1).view(1, 1, -1)
    point_coord_dots = torch.einsum("mik,jk->mij", points, coords)
    point_atom_sq_dists = point_sq_norms + coord_sq_norms - 2 * point_coord_dots
    valid_point_mask = (
        point_atom_sq_dists >= atom_radii.square().view(1, 1, -1)
    ).all(dim=-1)
    return valid_point_mask


def _compute_intersection_plane_points(
    coords: torch.Tensor,
    pair_diffs: torch.Tensor,
    pair_sq_dists: torch.Tensor,
    safe_pair_sq_dists: torch.Tensor,
    sphere_radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each atom pair (i, j), compute the plane that contains the circle of
    intersection of the two spheres with radii `sphere_radii[i]` and
    `sphere_radii[j]`.

    Returns:
        valid_circle_mask:
            Boolean tensor of shape (n, n). True where the two spheres
            intersect in a non-degenerate circle.
        plane_points:
            Tensor of shape (n, n, 3). `plane_points[i, j]` is the formula-based
            point on the corresponding circle plane. For invalid pairs we still
            return this analytic continuation and let downstream masks suppress
            its effect.
    """
    sphere_radii_sq = sphere_radii.square()
    pair_radius_sums_sq = (sphere_radii[:, None] + sphere_radii[None, :]).square()
    pair_radius_diffs_sq = (
        sphere_radii[:, None] - sphere_radii[None, :]
    ).square()

    valid_circle_mask = (
        (pair_sq_dists > pair_radius_diffs_sq)
        & (pair_sq_dists < pair_radius_sums_sq)
    )
    circle_center_weights = (
        safe_pair_sq_dists
        + sphere_radii_sq[:, None]
        - sphere_radii_sq[None, :]
    ) / (2 * safe_pair_sq_dists)
    plane_points = (
        coords[:, None, :]
        - circle_center_weights.unsqueeze(-1) * pair_diffs
    )

    return valid_circle_mask, plane_points


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


def find_suitable_atom_indices(
    points: torch.Tensor,      # coordinates of sampled points, shape (m, n, 3)
    atom_coords: torch.Tensor, # coordinates of atom centers, shape (n, 3)
    atom_radii: torch.Tensor,  # radii of atoms, shape (n,)
    probe_radius: float,       # radius of the probe sphere
) -> torch.Tensor:             # shape (m, n, 2), dtype torch.int
    
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

    return _select_suitable_atom_indices(suitable_atom_mask)


def project_points_to_ses(
    points: torch.Tensor,
    coords: torch.Tensor,
    radii: torch.Tensor,
    probe_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project sampled points onto the solvent-excluded surface.

    The implementation is organized as a straight pipeline:
    1. validate inputs and normalize dtype/device,
    2. build point/atom validity masks,
    3. assemble pair-wise circle geometry and the affine projector,
    4. recover probe centers,
    5. project the original samples onto the probe sphere.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (m, n, 3)")
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (n, 3)")

    num_samples, num_atoms, _ = points.shape
    if coords.shape[0] != num_atoms:
        raise ValueError("points.shape[1] must match coords.shape[0]")

    point_dtype = points.dtype if points.is_floating_point() else torch.float32
    coord_dtype = coords.dtype if coords.is_floating_point() else torch.float32
    common_dtype = torch.promote_types(point_dtype, coord_dtype)
    common_device = coords.device

    points = points.to(dtype=common_dtype, device=common_device)
    coords = coords.to(dtype=common_dtype, device=common_device)
    atom_radii = radii.reshape(-1).to(dtype=common_dtype, device=common_device)

    if atom_radii.numel() != num_atoms:
        raise ValueError("radii must contain one radius per atom")

    if num_samples == 0 or num_atoms == 0:
        empty_mask = torch.empty(
            points.shape[:2],
            dtype=torch.bool,
            device=common_device,
        )
        return points, empty_mask

    atom_ext_radii = atom_radii + probe_radius
    atom_ext_radii_sq = atom_ext_radii.square()

    # ------------------------------------------------------------------
    # 1. Mark sampled points that are outside every atom sphere, then derive
    #    the per-atom mask used for the second index in pair-wise tests.
    # ------------------------------------------------------------------
    valid_point_mask = _compute_valid_point_mask(points, coords, atom_radii)
    valid_atom_mask = valid_point_mask.any(dim=0)
    valid_atom_pair_mask = valid_atom_mask[:, None] & valid_atom_mask[None, :]

    # ------------------------------------------------------------------
    # 2. Build the pair-wise geometry shared by the projector formulas.
    #    We work with both ordinary atom intersections and expanded
    #    atom+probe intersections.
    # ------------------------------------------------------------------
    pair_diffs = coords[:, None, :] - coords[None, :, :]
    pair_sq_dists = pair_diffs.square().sum(dim=-1)

    # Zero distances only appear on the diagonal or for duplicate centers, so
    # masking them to one is enough to stabilize the analytic formulas below.
    safe_pair_sq_dists = pair_sq_dists.masked_fill(pair_sq_dists == 0, 1)

    valid_probe_circle_mask, probe_circle_centers = (
        _compute_intersection_plane_points(
            coords=coords,
            pair_diffs=pair_diffs,
            pair_sq_dists=pair_sq_dists,
            safe_pair_sq_dists=safe_pair_sq_dists,
            sphere_radii=atom_ext_radii,
        )
    )
    valid_atom_circle_mask, atom_circle_centers = _compute_intersection_plane_points(
        coords=coords,
        pair_diffs=pair_diffs,
        pair_sq_dists=pair_sq_dists,
        safe_pair_sq_dists=safe_pair_sq_dists,
        sphere_radii=atom_radii,
    )

    # Invalid expanded-sphere pairs should not contribute to the linear
    # projector, so we zero both the plane points and their direction vectors.
    probe_circle_centers = probe_circle_centers.masked_fill(
        ~valid_probe_circle_mask.unsqueeze(-1),
        0,
    )
    pair_diffs = pair_diffs.masked_fill(~valid_probe_circle_mask.unsqueeze(-1), 0)
    circle_center_dots = (pair_diffs * probe_circle_centers).sum(dim=-1)

    atom_diff_grams = pair_diffs @ pair_diffs.transpose(-1, -2)
    atom_diff_grams_inv = torch.linalg.pinv(atom_diff_grams, hermitian=True)

    # ------------------------------------------------------------------
    # 3. For each sampled point points[k, i], decide which pair directions
    #    (i, j) are active. The point must lie between the tangency plane and
    #    the boundary plane along the corresponding pair axis.
    # ------------------------------------------------------------------

    # A point on the plane through the atom/probe tangency
    # circle for atom i and the probe motion induced by pair (i, j).
    tangency_plane_points = (
        probe_radius * coords[:, None, :]
        + atom_radii[:, None, None] * probe_circle_centers
    ) / atom_ext_radii[:, None, None]

    # The outer boundary plane comes from the ordinary atom-atom intersection
    # when it exists; otherwise we fall back to the expanded-sphere plane.
    boundary_plane_points = torch.where(
        valid_atom_circle_mask.unsqueeze(-1),
        atom_circle_centers,
        probe_circle_centers,
    )

    tangency_plane_biases = (pair_diffs * tangency_plane_points).sum(dim=-1)
    boundary_plane_biases = (pair_diffs * boundary_plane_points).sum(dim=-1)
    point_axis_dots = torch.einsum("mik,ijk->mij", points, pair_diffs)

    point_probe_circle_mask = (
        valid_probe_circle_mask.unsqueeze(0)
        & valid_atom_pair_mask.unsqueeze(0)
        & valid_point_mask.unsqueeze(-1)
        & (point_axis_dots >= boundary_plane_biases.unsqueeze(0))
        & (point_axis_dots <= tangency_plane_biases.unsqueeze(0))
    )
    point_probe_circle_selector = torch.diag_embed(
        point_probe_circle_mask.to(dtype=common_dtype)
    )

    # ------------------------------------------------------------------
    # 4. Turn the selected pair directions into the affine projector
    #
    #       X -> (I - P) X + c
    #
    #    and apply it both to the sampled points and to the atom centers.
    # ------------------------------------------------------------------
    point_probe_circle_product = (
        pair_diffs.transpose(-1, -2).unsqueeze(0)
        @ point_probe_circle_selector
        @ atom_diff_grams_inv.unsqueeze(0)
        @ point_probe_circle_selector
    )
    point_probe_circle_projection = (
        point_probe_circle_product @ pair_diffs.unsqueeze(0)
    )
    point_probe_circle_center_projection = (
        point_probe_circle_product @ circle_center_dots.unsqueeze(0).unsqueeze(-1)
    ).squeeze(-1)

    projected_points = (
        points
        - (point_probe_circle_projection @ points.unsqueeze(-1)).squeeze(-1)
        + point_probe_circle_center_projection
    )

    broadcasted_coords = coords.unsqueeze(0).expand(num_samples, -1, -1)
    projected_atom_centers = (
        broadcasted_coords
        - (point_probe_circle_projection @ broadcasted_coords.unsqueeze(-1)).squeeze(-1)
        + point_probe_circle_center_projection
    )

    # ------------------------------------------------------------------
    # 5. Recover the probe center from the right triangle formed by the true
    #    atom center, the projected atom center and the projected SES point.
    #    Then project the original sample onto that probe sphere.
    # ------------------------------------------------------------------
    projected_point_atom_center_dists = torch.linalg.norm(
        projected_points - projected_atom_centers,
        dim=-1,
    )
    projected_atom_center_coord_sq_dists = (
        projected_atom_centers - broadcasted_coords
    ).square().sum(dim=-1)
    projected_probe_directions = projected_points - projected_atom_centers

    projected_probe_offset_dists = torch.sqrt(
        atom_ext_radii_sq.unsqueeze(0) - projected_atom_center_coord_sq_dists
    )
    projected_probe_centers = (
        projected_atom_centers
        + projected_probe_directions
        * (
            projected_probe_offset_dists / projected_point_atom_center_dists
        ).unsqueeze(-1)
    )

    point_probe_center_diffs = points - projected_probe_centers
    point_probe_center_dists = torch.linalg.norm(point_probe_center_diffs, dim=-1)
    projected_points_on_ses = (
        projected_probe_centers
        + point_probe_center_diffs
        * (probe_radius / point_probe_center_dists).unsqueeze(-1)
    )

    return projected_points_on_ses, valid_point_mask
