"""Analytic solvent-excluded surface block sampling.

The solvent-excluded surface (SES) is the boundary traced by a spherical
solvent probe as it rolls around atoms.  This module represents that boundary
as explicit analytic blocks instead of projecting arbitrary atom-sphere points:

* atom contact patches, where the probe touches a single atom and the SES lies
  on the atom's van der Waals sphere;
* pair toroidal patches, where the probe touches two atoms and its center moves
  around the intersection circle of two expanded atom spheres;
* fixed-probe reentrant spherical patches, where the probe touches at least
  three atoms and the SES lies on the probe sphere itself.

The code deliberately separates topology discovery from differentiable point
generation.  Discovery asks questions such as "which pairs intersect?" and
"is this probe-center component exterior?" under ``torch.no_grad()`` because
those decisions are discrete.  Once blocks have been selected, sample
coordinates are rebuilt from the original ``atom_coords`` and ``atom_radii``
tensors so gradients still flow through the geometric paths that remain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from .projection import (
    _atom_cell_query_keys,
    _build_atom_cell_table,
    _centers_feasible_against_all_atoms,
    _compute_triple_probe_centers,
    _probe_centers_accessible_from_exterior,
)


ATOM_BLOCK_TYPE = 1
PAIR_BLOCK_TYPE = 2
PROBE_BLOCK_TYPE = 3

# Memory and combinatorial guards.  They bound all-pairs/all-triples style
# operations without changing the intended lightweight CPU verification path.
_DEFAULT_PAIR_BUDGET = 8_000_000
_DEFAULT_MAX_PROBE_TRIPLES = 5_000_000
_DEFAULT_MAX_PROBE_SUPPORT_ATOMS = 16
_DEFAULT_MAX_GRID_POINTS = 50_000_000
_DEFAULT_TRIPLE_COMBO_BUDGET = 8_000_000

# Probe-patch rejection sampling keeps a small pool of extra candidates for
# coarse patches and then greedily picks well-spaced directions from that pool.
_WELL_SPACED_SELECTION_LIMIT = 256
_WELL_SPACED_CANDIDATE_FACTOR = 12


@dataclass(frozen=True)
class ExteriorContext:
    """Reusable molecule geometry for exterior SES membership checks.

    Attributes:
        atom_coords: Floating atom coordinates with shape ``(n, 3)``.
        atom_radii: Atom radii with shape ``(n,)`` on the same device/dtype as
            ``atom_coords``.
        probe_radius: Radius of the rolling solvent probe.
        expanded_radii: Per-atom radii in probe-center space,
            ``atom_radii + probe_radius``.
        grid_spacing: Optional spacing override for the flood-fill based
            exterior test in
            :func:`ses.projection._probe_centers_accessible_from_exterior`.
        max_grid_points: Safety cap for the exterior flood-fill grid.
    """

    atom_coords: torch.Tensor
    atom_radii: torch.Tensor
    probe_radius: float
    expanded_radii: torch.Tensor
    grid_spacing: Optional[float] = None
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS

    @property
    def device(self) -> torch.device:
        """Device shared by all tensors stored in the context."""

        return self.atom_coords.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating dtype used for geometric calculations."""

        return self.atom_coords.dtype

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the prepared molecule."""

        return int(self.atom_coords.shape[0])

    def centers_feasible(self, centers: torch.Tensor) -> torch.Tensor:
        """Return whether probe centers avoid every expanded atom sphere.

        Args:
            centers: Probe-center coordinates with trailing shape ``(..., 3)``.

        Returns:
            Boolean mask with shape ``centers.shape[:-1]``.  A true value means
            the probe center is not strictly inside any expanded atom sphere.
        """

        return _centers_feasible_against_atoms(
            centers=centers,
            atom_coords=self.atom_coords,
            expanded_radii=self.expanded_radii,
        )

    def centers_accessible(
        self,
        centers: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return whether centers belong to the exterior center-space component.

        Args:
            centers: Probe-center coordinates with trailing shape ``(..., 3)``.
            valid_mask: Optional precomputed mask of centers that are finite and
                feasible.  Supplying it avoids asking the grid search to route
                through centers that are already known to be invalid.

        Returns:
            Boolean mask with shape ``centers.shape[:-1]``.  True entries are in
            the same connected component as the far exterior of probe-center
            space.
        """

        return _probe_centers_accessible_from_exterior(
            centers,
            self.atom_coords,
            self.atom_radii,
            self.probe_radius,
            valid_center_mask=valid_mask,
            grid_spacing=self.grid_spacing,
            max_grid_points=self.max_grid_points,
            assume_centers_feasible=valid_mask is not None,
        )

    def centers_exterior(self, centers: torch.Tensor) -> torch.Tensor:
        """Return the combined finite, feasible and exterior-accessible mask.

        This is the common predicate used to decide whether a contact, torus or
        reentrant probe sample belongs to the visible SES instead of an enclosed
        cavity or an atom-overlapping probe location.
        """

        finite_mask = torch.isfinite(centers).all(dim=-1)
        feasible_mask = self.centers_feasible(centers) & finite_mask
        return self.centers_accessible(centers, feasible_mask)


@dataclass(frozen=True)
class AnalyticBlocks:
    """Detached analytic SES block definitions.

    The tensors in this dataclass describe which analytic pieces exist.  They
    are treated as topology, so they may be cached and reused, but they are not
    expected to carry gradients.  Sampling functions recompute coordinates from
    the original differentiable atom tensors.

    Attributes:
        atom_indices: Atom patch owner indices, shape ``(a,)``.
        pair_indices: Toroidal patch support atom pairs, shape ``(b, 2)``.
        probe_seed_indices: Three atoms used to reconstruct each probe center,
            shape ``(c, 3)``.
        probe_center_signs: Which of the two trilateration roots to use for
            each probe block, shape ``(c,)``.
        probe_support_indices: Padded support atoms touching each fixed probe
            center, shape ``(c, k_max)``.
        probe_support_mask: Valid entries in ``probe_support_indices``, shape
            ``(c, k_max)``.
        probe_center_hints: Detached center coordinates used only for stable
            root selection and diagnostics, shape ``(c, 3)``.
    """

    atom_indices: torch.Tensor
    pair_indices: torch.Tensor
    probe_seed_indices: torch.Tensor
    probe_center_signs: torch.Tensor
    probe_support_indices: torch.Tensor
    probe_support_mask: torch.Tensor
    probe_center_hints: torch.Tensor

    @property
    def num_blocks(self) -> int:
        """Total number of atom, pair and fixed-probe blocks."""

        return (
            int(self.atom_indices.shape[0])
            + int(self.pair_indices.shape[0])
            + int(self.probe_seed_indices.shape[0])
        )


@dataclass(frozen=True)
class AnalyticSamples:
    """Generated SES points and sparse atom-support metadata.

    Attributes:
        points: Sampled SES coordinates, shape ``(m, 3)``.
        block_types: Integer block-family tags for each point.  Values are one
            of ``ATOM_BLOCK_TYPE``, ``PAIR_BLOCK_TYPE`` or ``PROBE_BLOCK_TYPE``.
        block_indices: Row index of the source block within the corresponding
            block tensor in :class:`AnalyticBlocks`.
        support_indices: Padded atom indices that define each sample, shape
            ``(m, k_max)``.  Padding entries are ``-1``.
        support_mask: Boolean validity mask for ``support_indices``.
        support_weights: Per-sample weights over support atoms.  Valid weights
            are normalized to sum to one.
        atom_weights: Optional dense ``(m, n_atoms)`` version of the sparse
            support weights.  This is convenient for tests and small molecules,
            but may be expensive for large systems.
        blocks: Optional block topology that produced these samples.
    """

    points: torch.Tensor
    block_types: torch.Tensor
    block_indices: torch.Tensor
    support_indices: torch.Tensor
    support_mask: torch.Tensor
    support_weights: torch.Tensor
    atom_weights: Optional[torch.Tensor] = None
    blocks: Optional[AnalyticBlocks] = None


def _build_exterior_context(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
) -> ExteriorContext:
    """Validate atom tensors and build reusable exterior-membership context.

    Args:
        atom_coords: Atom coordinates with shape ``(n, 3)``.  Integer tensors
            are promoted to floating point.
        atom_radii: Atom radii with one value per atom.  The tensor is flattened
            before validation.
        probe_radius: Positive solvent probe radius.
        grid_spacing: Optional spacing override for exterior flood-fill checks.
        max_grid_points: Maximum number of grid cells allowed during exterior
            flood fill.

    Returns:
        A :class:`ExteriorContext` with normalized dtype/device and cached
        expanded radii.

    Raises:
        ValueError: If shapes are invalid, radii are negative or
            ``probe_radius`` is not positive.
    """

    coords, radii = _prepare_atom_inputs(atom_coords, atom_radii, probe_radius)
    return ExteriorContext(
        atom_coords=coords,
        atom_radii=radii,
        probe_radius=float(probe_radius),
        expanded_radii=radii + float(probe_radius),
        grid_spacing=grid_spacing,
        max_grid_points=int(max_grid_points),
    )


def build_analytic_blocks(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    atom_filter_samples: int = 256,
    pair_filter_samples: int = 24,
    max_probe_support_atoms: int = _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    support_tolerance: float = 1e-3,
    dedup_tolerance: float = 1e-4,
    max_probe_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
) -> tuple[AnalyticBlocks, ExteriorContext]:
    """Extract and exterior-filter all analytic SES blocks for a molecule.

    The returned blocks are detached topology.  Pass them to
    :func:`sample_analytic_points` or to the per-family sampling functions
    to rebuild differentiable sample coordinates from the original atom tensors.

    Args:
        atom_coords: Atom coordinates with shape ``(n, 3)``.
        atom_radii: Atom radii with one value per atom.
        probe_radius: Positive solvent probe radius.
        atom_filter_samples: Number of probe-center directions tested around
            each atom when deciding whether the atom has any exterior contact
            patch.
        pair_filter_samples: Number of angular samples tested around each
            expanded-sphere intersection circle when deciding whether a pair
            has an exterior torus block.
        max_probe_support_atoms: Number of padded support-atom slots stored per
            fixed-probe block.  At least three slots are required.
        support_tolerance: Distance tolerance used to decide whether an atom
            touches a candidate fixed probe center.
        dedup_tolerance: Grid tolerance used to remove duplicate fixed probe
            centers found through different seed triples.
        max_probe_triples: Optional cap on candidate atom triples considered
            for fixed-probe blocks.  ``None`` means no explicit cap.
        grid_spacing: Optional exterior flood-fill spacing override.
        max_grid_points: Maximum flood-fill grid size for exterior checks.

    Returns:
        ``(blocks, context)`` where ``blocks`` contains detached topology and
        ``context`` contains validated geometry used by the samplers.
    """

    context = _build_exterior_context(
        atom_coords,
        atom_radii,
        probe_radius,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    with torch.no_grad():
        # Block discovery is discrete.  Keep it detached so autograd only tracks
        # the continuous coordinates regenerated by the sampling functions.
        candidate_pair_indices = _candidate_pair_indices(context)
        atom_indices = _extract_atom_blocks(
            context,
            filter_samples=atom_filter_samples,
        )
        pair_indices = _extract_pair_blocks(
            context,
            filter_samples=pair_filter_samples,
            pair_indices=candidate_pair_indices,
        )
        probe_blocks = _extract_probe_blocks(
            context,
            pair_indices=candidate_pair_indices,
            max_support_atoms=max_probe_support_atoms,
            support_tolerance=support_tolerance,
            dedup_tolerance=dedup_tolerance,
            max_triples=max_probe_triples,
        )

    blocks = AnalyticBlocks(
        atom_indices=atom_indices,
        pair_indices=pair_indices,
        probe_seed_indices=probe_blocks.probe_seed_indices,
        probe_center_signs=probe_blocks.probe_center_signs,
        probe_support_indices=probe_blocks.probe_support_indices,
        probe_support_mask=probe_blocks.probe_support_mask,
        probe_center_hints=probe_blocks.probe_center_hints,
    )
    return blocks, context


def _extract_atom_blocks(
    context: ExteriorContext,
    *,
    filter_samples: int = 64,
    atom_chunk_size: int = 2048,
) -> torch.Tensor:
    """Return atom indices with at least one exterior contact direction.

    Each candidate direction places the probe center on the atom's expanded
    sphere.  If any such center is feasible and exterior-accessible, the atom's
    van der Waals sphere contributes an SES contact patch.

    Args:
        context: Prepared molecule geometry.
        filter_samples: Number of Fibonacci directions tested per atom.

    Returns:
        Long tensor of exposed atom indices with shape ``(a,)``.

    Raises:
        ValueError: If ``filter_samples`` is not positive.
    """

    if context.num_atoms == 0:
        return torch.empty((0,), dtype=torch.long, device=context.device)
    if filter_samples <= 0:
        raise ValueError("filter_samples must be positive")
    if atom_chunk_size <= 0:
        raise ValueError("atom_chunk_size must be positive")

    directions = _fibonacci_unit_vectors(
        filter_samples,
        dtype=context.dtype,
        device=context.device,
    )
    exposed_chunks = []
    atom_range = torch.arange(context.num_atoms, dtype=torch.long, device=context.device)
    for start in range(0, context.num_atoms, atom_chunk_size):
        stop = min(start + atom_chunk_size, context.num_atoms)
        centers = (
            context.atom_coords[start:stop, None, :]
            + context.expanded_radii[start:stop, None, None] * directions[None, :, :]
        )
        exterior = context.centers_exterior(centers.reshape(-1, 3)).reshape(
            stop - start,
            filter_samples,
        )
        exposed = exterior.any(dim=-1)
        exposed_chunks.append(atom_range[start:stop][exposed])
    if not exposed_chunks:
        return torch.empty((0,), dtype=torch.long, device=context.device)
    return torch.cat(exposed_chunks, dim=0)


def _extract_pair_blocks(
    context: ExteriorContext,
    *,
    filter_samples: int = 12,
    pair_indices: Optional[torch.Tensor] = None,
    pair_chunk_size: int = 50_000,
) -> torch.Tensor:
    """Return atom pairs whose expanded-sphere intersection circle is exterior.

    A pair torus exists when two expanded atom spheres intersect in a circle and
    at least one sampled probe center on that circle is exterior-accessible.

    Args:
        context: Prepared molecule geometry.
        filter_samples: Number of angular probe-center samples per pair circle.

    Returns:
        Long tensor of atom pairs with shape ``(b, 2)``.

    Raises:
        ValueError: If ``filter_samples`` is not positive.
    """

    if filter_samples <= 0:
        raise ValueError("filter_samples must be positive")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    if pair_indices is None:
        pair_indices = _candidate_pair_indices(context)
    else:
        pair_indices = pair_indices.to(device=context.device, dtype=torch.long)
    if pair_indices.numel() == 0:
        return pair_indices

    centers, circle_radii, basis_u, basis_v, valid = _pair_circle_parameters(
        context.atom_coords,
        context.expanded_radii,
        pair_indices,
    )
    valid = valid & (circle_radii > _sqrt_eps(context.dtype))
    if not bool(valid.any().item()):
        return torch.empty((0, 2), dtype=torch.long, device=context.device)

    pair_indices = pair_indices[valid]
    centers = centers[valid]
    circle_radii = circle_radii[valid]
    basis_u = basis_u[valid]
    basis_v = basis_v[valid]
    angles = (
        2
        * math.pi
        * (torch.arange(filter_samples, dtype=context.dtype, device=context.device) + 0.5)
        / filter_samples
    )
    cos_angles = torch.cos(angles).view(1, -1, 1)
    sin_angles = torch.sin(angles).view(1, -1, 1)
    exterior_pair_chunks = []
    for start in range(0, pair_indices.shape[0], pair_chunk_size):
        stop = min(start + pair_chunk_size, pair_indices.shape[0])
        radial_dirs = (
            cos_angles * basis_u[start:stop, None, :]
            + sin_angles * basis_v[start:stop, None, :]
        )
        probe_centers = (
            centers[start:stop, None, :]
            + circle_radii[start:stop, None, None] * radial_dirs
        )
        exterior = context.centers_exterior(probe_centers.reshape(-1, 3)).reshape(
            stop - start,
            filter_samples,
        )
        keep = exterior.any(dim=-1)
        exterior_pair_chunks.append(pair_indices[start:stop][keep])
    if not exterior_pair_chunks:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)
    return torch.cat(exterior_pair_chunks, dim=0)


@dataclass(frozen=True)
class _ProbeBlocks:
    """Internal fixed-probe block container before public block assembly.

    This mirrors the fixed-probe fields in :class:`AnalyticBlocks` without
    carrying atom and pair blocks.  It lets extraction return a typed empty
    result without constructing unrelated public fields.
    """

    probe_seed_indices: torch.Tensor
    probe_center_signs: torch.Tensor
    probe_support_indices: torch.Tensor
    probe_support_mask: torch.Tensor
    probe_center_hints: torch.Tensor


def _extract_probe_blocks(
    context: ExteriorContext,
    *,
    pair_indices: Optional[torch.Tensor] = None,
    max_support_atoms: int = _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    support_tolerance: float = 1e-3,
    dedup_tolerance: float = 1e-4,
    max_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
) -> _ProbeBlocks:
    """Return fixed-probe blocks from exterior triple intersections.

    Fixed-probe blocks are seeded by atom triples whose expanded spheres have a
    common intersection.  Each triple can produce two probe centers, one on each
    side of the triple plane.  The function keeps only centers that are finite,
    feasible against all expanded atoms, accessible from the exterior, unique at
    ``dedup_tolerance`` and supported by at least three atoms.

    Args:
        context: Prepared molecule geometry.
        pair_indices: Optional candidate pair graph.  Passing the unfiltered
            pair graph keeps narrow reentrant regions that might not pass the
            torus exterior filter by themselves.
        max_support_atoms: Number of padded support-atom slots stored per
            fixed-probe center.
        support_tolerance: Distance tolerance for deciding which atoms touch a
            fixed probe center.
        dedup_tolerance: Rounding tolerance for duplicate center removal.
        max_triples: Optional cap on candidate triples enumerated from the pair
            graph.  ``None`` disables this guard.

    Returns:
        Internal fixed-probe block data.  Empty tensors are returned when no
        valid exterior fixed-probe centers are found.

    Raises:
        ValueError: If support capacity or tolerances are invalid.
    """

    if max_support_atoms < 3:
        raise ValueError("max_support_atoms must be at least 3")
    if support_tolerance <= 0:
        raise ValueError("support_tolerance must be positive")
    if dedup_tolerance <= 0:
        raise ValueError("dedup_tolerance must be positive")

    empty = _empty_probe_blocks(context.device, max_support_atoms)
    if context.num_atoms < 3:
        return empty

    if pair_indices is None:
        pair_indices = _candidate_pair_indices(context)
    # Triples are enumerated from the pair graph, then analytically solved as
    # intersections of three expanded atom spheres.
    triple_indices = _candidate_triple_indices(
        context.num_atoms,
        pair_indices,
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
    if not bool(flat_valid.any().item()):
        return empty

    flat_valid_indices = flat_valid.nonzero(as_tuple=False).reshape(-1)
    flat_centers = flat_centers[flat_valid]
    flat_seeds = triple_indices[torch.div(flat_valid_indices, 2, rounding_mode="floor")]
    flat_signs = flat_valid_indices.remainder(2)
    feasible = context.centers_feasible(flat_centers)
    accessible = context.centers_accessible(flat_centers, feasible)
    if not bool(accessible.any().item()):
        return empty

    flat_centers = flat_centers[accessible]
    flat_seeds = flat_seeds[accessible]
    flat_signs = flat_signs[accessible]
    # Different atom triples can describe the same physical probe center.  Keep
    # the first occurrence so downstream sampling sees one patch per center.
    keep = _deduplicate_centers(flat_centers, dedup_tolerance)
    flat_centers = flat_centers[keep]
    flat_seeds = flat_seeds[keep]
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
    if not bool(enough_support.any().item()):
        return empty

    return _ProbeBlocks(
        probe_seed_indices=flat_seeds[enough_support],
        probe_center_signs=flat_signs[enough_support],
        probe_support_indices=support_indices[enough_support],
        probe_support_mask=support_mask[enough_support],
        probe_center_hints=flat_centers[enough_support].detach(),
    )


def _sample_atom_blocks(
    context: ExteriorContext,
    atom_indices: torch.Tensor,
    *,
    point_area: float,
    oversample_factor: float = 1.25,
) -> AnalyticSamples:
    """Generate points on exterior atom-contact SES patches.

    Args:
        context: Prepared molecule geometry.
        atom_indices: Atom indices returned by :func:`_extract_atom_blocks`.
        point_area: Target surface area represented by one returned point.
        oversample_factor: Multiplier applied before exterior clipping.  Values
            above one preserve density after hidden directions are removed.

    Returns:
        :class:`AnalyticSamples` whose support width is one.  The sampled
        points remain differentiable with respect to the owning atom coordinate
        and radius.
    """

    _validate_point_area(point_area, oversample_factor)
    atom_indices = atom_indices.to(device=context.device, dtype=torch.long)
    empty = _empty_samples(context, max_support=1)
    if atom_indices.numel() == 0:
        return empty

    with torch.no_grad():
        areas = 4 * math.pi * context.atom_radii[atom_indices].detach().square()
        counts = _counts_from_areas(areas, point_area, oversample_factor)
        owner_rows = torch.repeat_interleave(
            torch.arange(atom_indices.shape[0], device=context.device),
            counts,
        )
        owners = atom_indices[owner_rows]
        directions = _packed_fibonacci_directions(counts, context.dtype, context.device)

    # A contact SES point lies on the atom sphere, while the corresponding
    # probe center lies farther out on the expanded sphere.  Exterior filtering
    # is applied to the probe centers, then the surviving directions are reused
    # for differentiable atom-sphere coordinates.
    probe_centers = (
        context.atom_coords[owners]
        + context.expanded_radii[owners].unsqueeze(-1) * directions
    )
    with torch.no_grad():
        exterior = context.centers_exterior(probe_centers)

    points = (
        context.atom_coords[owners]
        + context.atom_radii[owners].unsqueeze(-1) * directions
    )
    points = points[exterior]
    owners = owners[exterior]
    owner_rows = owner_rows[exterior]
    support_indices = owners.unsqueeze(-1)
    support_mask = torch.ones_like(support_indices, dtype=torch.bool)
    support_weights = torch.ones(
        support_indices.shape,
        dtype=context.dtype,
        device=context.device,
    )
    return AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            ATOM_BLOCK_TYPE,
            dtype=torch.long,
            device=context.device,
        ),
        block_indices=owner_rows,
        support_indices=support_indices,
        support_mask=support_mask,
        support_weights=support_weights,
    )


def _sample_pair_blocks(
    context: ExteriorContext,
    pair_indices: torch.Tensor,
    *,
    point_area: float,
    oversample_factor: float = 1.25,
) -> AnalyticSamples:
    """Generate points on exterior pair toroidal SES patches.

    Args:
        context: Prepared molecule geometry.
        pair_indices: Atom pairs returned by :func:`_extract_pair_blocks`.
        point_area: Target surface area represented by one returned point.
        oversample_factor: Multiplier applied before exterior clipping.

    Returns:
        :class:`AnalyticSamples` whose support width is two.  Support weights
        encode the spherical interpolation between the two atom-contact normals
        that generated each point on the rolling probe sphere.
    """

    _validate_point_area(point_area, oversample_factor)
    pair_indices = pair_indices.to(device=context.device, dtype=torch.long)
    empty = _empty_samples(context, max_support=2)
    if pair_indices.numel() == 0:
        return empty

    with torch.no_grad():
        counts = _pair_sample_counts(
            context,
            pair_indices,
            point_area=point_area,
            oversample_factor=oversample_factor,
        )
        pair_rows, theta_values, arc_fracs = _packed_pair_parameters(
            counts,
            context,
            pair_indices,
            dtype=context.dtype,
            device=context.device,
        )
    if pair_rows.numel() == 0:
        return empty

    selected_pairs = pair_indices[pair_rows]
    # ``theta_values`` move the probe center around the pair circle.  The
    # ``arc_fracs`` parameter then selects a point along the concave probe arc
    # between the two atom-contact normals.
    probe_centers, points, first_weights, second_weights = _pair_points_from_params(
        context.atom_coords,
        context.expanded_radii,
        context.probe_radius,
        selected_pairs,
        theta_values,
        arc_fracs,
    )
    with torch.no_grad():
        exterior = context.centers_exterior(probe_centers)

    points = points[exterior]
    selected_pairs = selected_pairs[exterior]
    pair_rows = pair_rows[exterior]
    first_weights = first_weights[exterior]
    second_weights = second_weights[exterior]
    support_weights = torch.stack((first_weights, second_weights), dim=-1)
    support_weights = support_weights / support_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(context.dtype).eps)
    support_mask = torch.ones_like(selected_pairs, dtype=torch.bool)
    return AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            PAIR_BLOCK_TYPE,
            dtype=torch.long,
            device=context.device,
        ),
        block_indices=pair_rows,
        support_indices=selected_pairs,
        support_mask=support_mask,
        support_weights=support_weights,
    )


def _sample_probe_blocks(
    context: ExteriorContext,
    blocks: AnalyticBlocks,
    *,
    point_area: float,
    oversample_factor: float = 1.25,
    probe_density_scale: float = 1.0,
) -> AnalyticSamples:
    """Generate points on fixed-probe reentrant spherical SES patches.

    Args:
        context: Prepared molecule geometry.
        blocks: Analytic block topology containing fixed-probe seed triples,
            root signs and support atoms.
        point_area: Target surface area represented by one returned point.
        oversample_factor: Multiplier used when converting estimated patch area
            to a sample count.
        probe_density_scale: Extra multiplier for fixed-probe patches only.
            This is useful for visualizing small multi-support patches without
            increasing atom and torus densities.

    Returns:
        :class:`AnalyticSamples` whose support width matches
        ``blocks.probe_support_indices.shape[1]``.  Points are rebuilt from the
        selected probe-center root and support normals so gradients flow through
        atom coordinates and radii.

    Raises:
        ValueError: If ``probe_density_scale`` is not positive.
    """

    _validate_point_area(point_area, oversample_factor)
    if probe_density_scale <= 0:
        raise ValueError("probe_density_scale must be positive")
    empty = _empty_samples(
        context,
        max_support=max(1, int(blocks.probe_support_indices.shape[1])),
    )
    if blocks.probe_seed_indices.numel() == 0:
        return empty

    seed_indices = blocks.probe_seed_indices.to(device=context.device, dtype=torch.long)
    signs = blocks.probe_center_signs.to(device=context.device, dtype=torch.long)
    support_indices = blocks.probe_support_indices.to(
        device=context.device,
        dtype=torch.long,
    )
    support_mask = blocks.probe_support_mask.to(device=context.device)

    centers_two, valid_two = _compute_triple_probe_centers(
        context.atom_coords,
        context.expanded_radii,
        seed_indices[:, 0],
        seed_indices[:, 1],
        seed_indices[:, 2],
    )
    block_rows = torch.arange(seed_indices.shape[0], device=context.device)
    centers = centers_two[block_rows, signs.clamp(0, 1)]
    center_valid = valid_two[block_rows, signs.clamp(0, 1)]

    with torch.no_grad():
        # Area and simplex-weight generation only decide where to sample within
        # already-selected patches.  Coordinates are recomputed with gradients
        # below using the same support weights.
        areas = _probe_patch_area_estimates(
            centers.detach(),
            support_indices,
            support_mask,
            context.atom_coords.detach(),
            context.expanded_radii.detach(),
            probe_radius=context.probe_radius,
        )
        counts = _counts_from_areas(
            areas * float(probe_density_scale),
            point_area,
            oversample_factor,
        )
        sample_block_rows, simplex_weights = _packed_probe_patch_weights(
            centers.detach(),
            support_indices,
            support_mask,
            context.atom_coords.detach(),
            context.expanded_radii.detach(),
            counts,
            dtype=context.dtype,
            device=context.device,
        )
    if sample_block_rows.numel() == 0:
        return empty

    safe_support_indices = support_indices.clamp_min(0)
    selected_support_indices = safe_support_indices[sample_block_rows]
    selected_support_mask = support_mask[sample_block_rows]
    selected_weights = simplex_weights
    selected_centers = centers[sample_block_rows]
    support_coords = context.atom_coords[selected_support_indices]
    support_radii = context.expanded_radii[selected_support_indices].clamp_min(
        torch.finfo(context.dtype).eps,
    )
    # Support normals point from each touching atom center to the fixed probe
    # center in expanded-sphere space.  Convex combinations of these normals
    # define directions on the reentrant probe patch.
    normals = (selected_centers.unsqueeze(1) - support_coords) / support_radii.unsqueeze(-1)
    normals = normals * selected_support_mask.unsqueeze(-1)
    probe_dirs = (selected_weights.unsqueeze(-1) * normals).sum(dim=1)
    probe_dirs = probe_dirs / torch.linalg.norm(
        probe_dirs,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(context.dtype).eps)
    points = selected_centers - float(context.probe_radius) * probe_dirs

    with torch.no_grad():
        finite = torch.isfinite(points).all(dim=-1)
        valid = center_valid[sample_block_rows] & finite
    points = points[valid]
    sample_block_rows = sample_block_rows[valid]
    selected_support_indices = selected_support_indices[valid]
    selected_support_mask = selected_support_mask[valid]
    selected_weights = selected_weights[valid]
    selected_weights = selected_weights * selected_support_mask
    selected_weights = selected_weights / selected_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(context.dtype).eps)

    return AnalyticSamples(
        points=points,
        block_types=torch.full(
            (points.shape[0],),
            PROBE_BLOCK_TYPE,
            dtype=torch.long,
            device=context.device,
        ),
        block_indices=sample_block_rows,
        support_indices=selected_support_indices,
        support_mask=selected_support_mask,
        support_weights=selected_weights,
    )


def _sample_analytic_samples(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float = 1.0,
    oversample_factor: float = 1.25,
    probe_density_scale: float = 1.0,
    include_atom_weights: bool = False,
    blocks: Optional[AnalyticBlocks] = None,
    atom_filter_samples: int = 256,
    pair_filter_samples: int = 24,
    max_probe_support_atoms: int = _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    support_tolerance: float = 1e-3,
    dedup_tolerance: float = 1e-4,
    max_probe_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
) -> AnalyticSamples:
    """Generate detailed SES samples from atom, pair and fixed-probe blocks.

    This internal entry point returns the full metadata container used by tests
    and visualization helpers.  When ``blocks`` is not supplied, topology is
    discovered first.  The three block families are then sampled independently
    and concatenated with compatible sparse support metadata.

    Args:
        atom_coords: Atom coordinates, shape ``(n, 3)``.
        atom_radii: Atom radii, shape ``(n,)`` or any shape with ``n`` values.
        probe_radius: Solvent probe radius.
        point_area: Target area per point in square Angstroms.
        oversample_factor: Candidate oversampling multiplier before exterior
            filtering. Values above one improve coverage near clipped patches.
        probe_density_scale: Extra multiplier for fixed-probe patch point
            counts. Useful when visualizing small multi-support patches whose
            boundaries should be emphasized.
        include_atom_weights: If true, also return a dense ``(points, atoms)``
            weighted atom-assignment matrix. The default sparse
            ``support_*`` fields are much cheaper for large molecules.
        blocks: Optional precomputed blocks from :func:`build_analytic_blocks`.
        atom_filter_samples: Contact-patch discovery samples per atom when
            ``blocks`` is not supplied.
        pair_filter_samples: Pair-circle discovery samples per pair when
            ``blocks`` is not supplied.
        max_probe_support_atoms: Number of padded support-atom slots stored per
            fixed-probe block during discovery.
        support_tolerance: Distance tolerance for fixed-probe support atoms.
        dedup_tolerance: Rounding tolerance for duplicate fixed probe centers.
        max_probe_triples: Optional cap on candidate triples considered during
            fixed-probe discovery.  ``None`` disables the cap.
        grid_spacing: Optional exterior flood-fill spacing override.
        max_grid_points: Maximum flood-fill grid size for exterior checks.

    Returns:
        :class:`AnalyticSamples` with point coordinates and source-atom
        assignments.
    """

    context = _build_exterior_context(
        atom_coords,
        atom_radii,
        probe_radius,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    if blocks is None:
        blocks, context = build_analytic_blocks(
            context.atom_coords,
            context.atom_radii,
            context.probe_radius,
            atom_filter_samples=atom_filter_samples,
            pair_filter_samples=pair_filter_samples,
            max_probe_support_atoms=max_probe_support_atoms,
            support_tolerance=support_tolerance,
            dedup_tolerance=dedup_tolerance,
            max_probe_triples=max_probe_triples,
            grid_spacing=grid_spacing,
            max_grid_points=max_grid_points,
        )

    atom_samples = _sample_atom_blocks(
        context,
        blocks.atom_indices,
        point_area=point_area,
        oversample_factor=oversample_factor,
    )
    pair_samples = _sample_pair_blocks(
        context,
        blocks.pair_indices,
        point_area=point_area,
        oversample_factor=oversample_factor,
    )
    probe_samples = _sample_probe_blocks(
        context,
        blocks,
        point_area=point_area,
        oversample_factor=oversample_factor,
        probe_density_scale=probe_density_scale,
    )
    samples = _concat_samples(
        (atom_samples, pair_samples, probe_samples),
        context=context,
        include_atom_weights=include_atom_weights,
        blocks=blocks,
    )
    return samples


def sample_analytic_points(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
    *,
    point_area: float = 1.0,
    oversample_factor: float = 1.25,
    probe_density_scale: float = 1.0,
    include_atom_features: bool = False,
    blocks: Optional[AnalyticBlocks] = None,
    atom_filter_samples: int = 256,
    pair_filter_samples: int = 24,
    max_probe_support_atoms: int = _DEFAULT_MAX_PROBE_SUPPORT_ATOMS,
    support_tolerance: float = 1e-3,
    dedup_tolerance: float = 1e-4,
    max_probe_triples: Optional[int] = _DEFAULT_MAX_PROBE_TRIPLES,
    grid_spacing: Optional[float] = None,
    max_grid_points: int = _DEFAULT_MAX_GRID_POINTS,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Sample visible SES points with the analytic block interface.

    This is the package-level analytic scenario: callers receive a flat point
    cloud with shape ``(num_points, 3)``.  If atom binding is needed, set
    ``include_atom_features=True`` to also receive a dense multi-hot feature
    tensor with shape ``(num_points, num_atoms)``.  A value of one means the
    corresponding atom supports that analytic SES sample.

    Args:
        atom_coords: Atom coordinates, shape ``(n, 3)``.
        atom_radii: Atom radii, shape ``(n,)`` or any shape with ``n`` values.
        probe_radius: Solvent probe radius.
        point_area: Target area per point in square Angstroms.
        oversample_factor: Candidate oversampling multiplier before exterior
            filtering.
        probe_density_scale: Extra multiplier for fixed-probe patch point
            counts.
        include_atom_features: If true, also return dense atom-assignment
            features aligned with the returned point rows.
        blocks: Optional precomputed blocks from :func:`build_analytic_blocks`.
        atom_filter_samples: Contact-patch discovery samples per atom when
            ``blocks`` is not supplied.
        pair_filter_samples: Pair-circle discovery samples per pair when
            ``blocks`` is not supplied.
        max_probe_support_atoms: Number of padded support-atom slots stored per
            fixed-probe block during discovery.
        support_tolerance: Distance tolerance for fixed-probe support atoms.
        dedup_tolerance: Rounding tolerance for duplicate fixed probe centers.
        max_probe_triples: Optional cap on candidate triples considered during
            fixed-probe discovery.  ``None`` disables the cap.
        grid_spacing: Optional exterior flood-fill spacing override.
        max_grid_points: Maximum flood-fill grid size for exterior checks.

    Returns:
        ``points`` when ``include_atom_features`` is false, otherwise
        ``(points, atom_features)``.
    """

    samples = _sample_analytic_samples(
        atom_coords,
        atom_radii,
        probe_radius,
        point_area=point_area,
        oversample_factor=oversample_factor,
        probe_density_scale=probe_density_scale,
        include_atom_weights=False,
        blocks=blocks,
        atom_filter_samples=atom_filter_samples,
        pair_filter_samples=pair_filter_samples,
        max_probe_support_atoms=max_probe_support_atoms,
        support_tolerance=support_tolerance,
        dedup_tolerance=dedup_tolerance,
        max_probe_triples=max_probe_triples,
        grid_spacing=grid_spacing,
        max_grid_points=max_grid_points,
    )
    if not include_atom_features:
        return samples.points

    atom_features = _dense_atom_features(
        samples.support_indices,
        samples.support_mask,
        num_atoms=int(atom_coords.shape[0]),
        dtype=samples.points.dtype,
    )
    return samples.points, atom_features


def _prepare_atom_inputs(
    atom_coords: torch.Tensor,
    atom_radii: torch.Tensor,
    probe_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize atom inputs to a common floating dtype and validate shapes."""

    if probe_radius <= 0:
        raise ValueError("probe_radius must be positive")
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


def _validate_point_area(point_area: float, oversample_factor: float) -> None:
    """Validate sampling-density parameters shared by all block samplers."""

    if point_area <= 0:
        raise ValueError("point_area must be positive")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be positive")


def _sqrt_eps(dtype: torch.dtype) -> float:
    """Return ``sqrt(machine epsilon)`` for tolerance comparisons."""

    return math.sqrt(torch.finfo(dtype).eps)


def _fibonacci_unit_vectors(
    count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return approximately uniform unit vectors on the sphere.

    The Fibonacci lattice gives deterministic, rotation-free coverage without
    random state.  It is used for coarse exterior checks and atom-patch samples.
    """

    if count <= 0:
        return torch.empty((0, 3), dtype=dtype, device=device)
    sample_indices = torch.arange(count, dtype=dtype, device=device)
    z_coords = 1 - 2 * (sample_indices + 0.5) / count
    radial_coords = torch.sqrt((1 - z_coords.square()).clamp_min(0))
    golden_angle = torch.as_tensor(
        math.pi * (3.0 - math.sqrt(5.0)),
        dtype=dtype,
        device=device,
    )
    azimuths = sample_indices * golden_angle
    return torch.stack(
        (
            torch.cos(azimuths) * radial_coords,
            torch.sin(azimuths) * radial_coords,
            z_coords,
        ),
        dim=-1,
    )


def _packed_fibonacci_directions(
    counts: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate one Fibonacci direction set for each requested block count."""

    directions = []
    cached_by_count: dict[int, torch.Tensor] = {}
    for count_tensor in counts:
        count = int(count_tensor.item())
        if count <= 0:
            continue
        cached = cached_by_count.get(count)
        if cached is None:
            cached = _fibonacci_unit_vectors(count, dtype=dtype, device=device)
            cached_by_count[count] = cached
        directions.append(cached)
    if not directions:
        return torch.empty((0, 3), dtype=dtype, device=device)
    return torch.cat(directions, dim=0)


def _counts_from_areas(
    areas: torch.Tensor,
    point_area: float,
    oversample_factor: float,
) -> torch.Tensor:
    """Convert per-block area estimates to at least one sample per block."""

    if areas.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=areas.device)
    counts = torch.ceil(areas * float(oversample_factor) / float(point_area)).to(
        torch.long,
    )
    return counts.clamp_min(1)


def _centers_feasible_against_atoms(
    centers: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    *,
    pair_budget: int = _DEFAULT_PAIR_BUDGET,
) -> torch.Tensor:
    """Check that probe centers are outside all expanded atom spheres.

    The implementation flattens arbitrary leading center dimensions and
    processes them in blocks so the temporary ``centers x atoms`` distance
    matrix stays within ``pair_budget``.
    """

    flat_centers = centers.reshape(-1, 3)
    if flat_centers.numel() == 0:
        return torch.empty(centers.shape[:-1], dtype=torch.bool, device=centers.device)
    if atom_coords.shape[0] == 0:
        return torch.ones(centers.shape[:-1], dtype=torch.bool, device=centers.device)
    if atom_coords.shape[0] >= 300 and flat_centers.shape[0] >= 1024:
        feasible = _centers_feasible_against_all_atoms(
            flat_centers,
            atom_coords,
            expanded_radii.square(),
        )
        return feasible.reshape(centers.shape[:-1])

    feasible = torch.empty(flat_centers.shape[0], dtype=torch.bool, device=centers.device)
    block_rows = max(1, min(flat_centers.shape[0], pair_budget // atom_coords.shape[0]))
    expanded_sq = expanded_radii.square()
    for start in range(0, flat_centers.shape[0], block_rows):
        stop = min(start + block_rows, flat_centers.shape[0])
        sq_dists = (
            flat_centers[start:stop].unsqueeze(1) - atom_coords.unsqueeze(0)
        ).square().sum(dim=-1)
        tol = (
            256
            * torch.finfo(sq_dists.dtype).eps
            * torch.maximum(sq_dists, expanded_sq.unsqueeze(0)).clamp_min(1)
        )
        feasible[start:stop] = (sq_dists >= expanded_sq.unsqueeze(0) - tol).all(dim=-1)
    return feasible.reshape(centers.shape[:-1])


def _candidate_pair_indices(context: ExteriorContext) -> torch.Tensor:
    """Return candidate atom pairs whose expanded spheres intersect."""

    if context.num_atoms < 2:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)
    pair_indices = _candidate_pair_indices_grid(context)
    if pair_indices is not None:
        return pair_indices
    return _candidate_pair_indices_torch(context)


def _candidate_pair_indices_kdtree(context: ExteriorContext) -> torch.Tensor:
    """Compatibility wrapper for the device-generic pair enumerator."""

    return _candidate_pair_indices(context)


def _candidate_pair_indices_grid(context: ExteriorContext) -> Optional[torch.Tensor]:
    """Find expanded-sphere intersections with a device-generic atom grid."""

    if context.num_atoms < 2:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)

    max_radius = context.expanded_radii.max()
    cell_size = float((2.0 * max_radius).clamp_min(torch.finfo(context.dtype).eps).item())
    table = _build_atom_cell_table(context.atom_coords, cell_size)
    if table is None or table.max_occupancy * 27 >= context.num_atoms:
        return None

    atom_cells = torch.floor(context.atom_coords / table.cell_size).to(torch.long)
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-1, 2, device=context.device),
            torch.arange(-1, 2, device=context.device),
            torch.arange(-1, 2, device=context.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    rows_per_block = max(
        1,
        min(context.num_atoms, _DEFAULT_PAIR_BUDGET // max(1, 27 * table.max_occupancy)),
    )
    pair_chunks = []
    atom_range = torch.arange(context.num_atoms, dtype=torch.long, device=context.device)
    slot_offsets = torch.arange(table.max_occupancy, dtype=torch.long, device=context.device)
    for start in range(0, context.num_atoms, rows_per_block):
        stop = min(start + rows_per_block, context.num_atoms)
        first_indices = atom_range[start:stop]
        first_cells = atom_cells[start:stop]
        query_cells = first_cells.unsqueeze(1) + offsets.view(1, -1, 3)
        flat_keys, flat_valid_cells = _atom_cell_query_keys(
            query_cells.reshape(-1, 3),
            table,
        )
        keys = flat_keys.reshape(first_indices.shape[0], offsets.shape[0])
        valid_cells = flat_valid_cells.reshape(first_indices.shape[0], offsets.shape[0])
        starts_in_table = torch.searchsorted(table.sorted_keys, keys, right=False)
        stops_in_table = torch.searchsorted(table.sorted_keys, keys, right=True)
        positions = starts_in_table.unsqueeze(-1) + slot_offsets.view(1, 1, -1)
        has_atom = valid_cells.unsqueeze(-1) & (positions < stops_in_table.unsqueeze(-1))
        second_indices = table.sorted_atom_indices[
            positions.clamp_max(table.sorted_atom_indices.shape[0] - 1)
        ]
        valid = has_atom & (second_indices > first_indices.view(-1, 1, 1))
        if not bool(valid.any().item()):
            continue

        first_valid = first_indices.view(-1, 1, 1).expand_as(second_indices)[valid]
        second_valid = second_indices[valid]
        diffs = context.atom_coords[first_valid] - context.atom_coords[second_valid]
        dist_sq = diffs.square().sum(dim=-1)
        first_radii = context.expanded_radii[first_valid]
        second_radii = context.expanded_radii[second_valid]
        pair_valid = (
            (dist_sq < (first_radii + second_radii).square())
            & (dist_sq > (first_radii - second_radii).square())
        )
        pair_chunks.append(
            torch.stack(
                (first_valid[pair_valid], second_valid[pair_valid]),
                dim=-1,
            )
        )
    if not pair_chunks:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)
    pairs = torch.cat(pair_chunks, dim=0)
    if pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)
    keys = pairs[:, 0] * context.num_atoms + pairs[:, 1]
    order = torch.argsort(keys)
    pairs = pairs[order]
    unique = torch.ones((pairs.shape[0],), dtype=torch.bool, device=context.device)
    unique[1:] = keys[order][1:] != keys[order][:-1]
    return pairs[unique]


def _candidate_pair_indices_torch(context: ExteriorContext) -> torch.Tensor:
    """Torch fallback for candidate expanded-sphere pair enumeration."""

    pair_chunks = []
    num_atoms = context.num_atoms
    rows_per_block = max(1, min(num_atoms, _DEFAULT_PAIR_BUDGET // num_atoms))
    atom_range = torch.arange(num_atoms, device=context.device)
    for start in range(0, num_atoms, rows_per_block):
        stop = min(start + rows_per_block, num_atoms)
        row_indices = atom_range[start:stop]
        diffs = context.atom_coords[start:stop, None, :] - context.atom_coords[None, :, :]
        dist_sq = diffs.square().sum(dim=-1)
        row_radii = context.expanded_radii[start:stop, None]
        radius_sums_sq = (row_radii + context.expanded_radii[None, :]).square()
        radius_diffs_sq = (row_radii - context.expanded_radii[None, :]).square()
        upper = atom_range.unsqueeze(0) > row_indices.unsqueeze(1)
        valid = upper & (dist_sq < radius_sums_sq) & (dist_sq > radius_diffs_sq)
        local_rows, cols = valid.nonzero(as_tuple=True)
        if local_rows.numel() > 0:
            pair_chunks.append(torch.stack((row_indices[local_rows], cols), dim=-1))
    if not pair_chunks:
        return torch.empty((0, 2), dtype=torch.long, device=context.device)
    return torch.cat(pair_chunks, dim=0)


def _pair_circle_parameters(
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    pair_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute intersection-circle geometry for expanded atom-sphere pairs.

    Returns:
        ``(centers, radii, basis_u, basis_v, valid)`` where ``basis_u`` and
        ``basis_v`` span the plane perpendicular to the atom-pair axis.  Invalid
        rows correspond to degenerate or non-intersecting sphere pairs.
    """

    dtype = atom_coords.dtype
    eps = torch.finfo(dtype).eps
    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    first_centers = atom_coords[first]
    second_centers = atom_coords[second]
    first_radii = expanded_radii[first]
    second_radii = expanded_radii[second]
    axes = second_centers - first_centers
    dists = torch.linalg.norm(axes, dim=-1)
    safe_dists = dists.clamp_min(eps)
    axis_dirs = axes / safe_dists.unsqueeze(-1)
    offsets = (
        first_radii.square()
        - second_radii.square()
        + dists.square()
    ) / (2 * safe_dists)
    circle_centers = first_centers + offsets.unsqueeze(-1) * axis_dirs
    radius_sq = first_radii.square() - offsets.square()
    circle_radii = torch.sqrt(radius_sq.clamp_min(0))

    z_ref = torch.zeros_like(axis_dirs)
    z_ref[:, 2] = 1
    y_ref = torch.zeros_like(axis_dirs)
    y_ref[:, 1] = 1
    refs = torch.where(axis_dirs[:, 2:].abs() > 0.9, y_ref, z_ref)
    basis_u = torch.cross(axis_dirs, refs, dim=-1)
    basis_u = basis_u / torch.linalg.norm(
        basis_u,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    basis_v = torch.cross(axis_dirs, basis_u, dim=-1)
    valid = (dists > eps) & (radius_sq >= -1000 * eps)
    return circle_centers, circle_radii, basis_u, basis_v, valid


def _pair_sample_counts(
    context: ExteriorContext,
    pair_indices: torch.Tensor,
    *,
    point_area: float,
    oversample_factor: float,
) -> torch.Tensor:
    """Estimate torus-patch areas and convert them to sample counts."""

    areas = _pair_patch_area_estimates(context, pair_indices)
    return _counts_from_areas(areas, point_area, oversample_factor)


def _pair_patch_area_estimates(
    context: ExteriorContext,
    pair_indices: torch.Tensor,
    *,
    arc_quadrature: int = 96,
) -> torch.Tensor:
    """Estimate visible pair torus patch area before exterior clipping.

    The area integral separates into a full turn around the pair circle and an
    arc on the probe sphere between the two atom-contact normals.  Quadrature is
    used along the probe-sphere arc because its projected radius varies.
    """

    centers, circle_radii, basis_u, _, valid = _pair_circle_parameters(
        context.atom_coords,
        context.expanded_radii,
        pair_indices,
    )
    if pair_indices.numel() == 0:
        return torch.empty((0,), dtype=context.dtype, device=context.device)

    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    probe_centers = centers + circle_radii.unsqueeze(-1) * basis_u
    first_normals = (
        probe_centers - context.atom_coords[first]
    ) / context.expanded_radii[first].unsqueeze(-1).clamp_min(torch.finfo(context.dtype).eps)
    second_normals = (
        probe_centers - context.atom_coords[second]
    ) / context.expanded_radii[second].unsqueeze(-1).clamp_min(torch.finfo(context.dtype).eps)
    arc_angles = torch.acos((first_normals * second_normals).sum(dim=-1).clamp(-1, 1))

    arc_fracs = (
        torch.arange(arc_quadrature, dtype=context.dtype, device=context.device) + 0.5
    ) / arc_quadrature
    directions = _slerp_unit_vectors(
        first_normals,
        second_normals,
        arc_angles,
        arc_fracs,
    )
    radial_components = (directions * basis_u.unsqueeze(1)).sum(dim=-1)
    surface_radii = (
        circle_radii.unsqueeze(-1) - float(context.probe_radius) * radial_components
    ).clamp_min(0)
    mean_surface_radii = surface_radii.mean(dim=-1)
    areas = 2 * math.pi * float(context.probe_radius) * arc_angles * mean_surface_radii
    areas = torch.where(valid, areas, torch.zeros_like(areas))
    return areas


def _packed_pair_parameters(
    counts: torch.Tensor,
    context: ExteriorContext,
    pair_indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-sample pair rows, circle angles and arc fractions.

    The angle around the pair circle is evenly spaced per block.  The arc
    fraction uses a low-discrepancy sequence and an area CDF so samples are
    closer to uniform on the curved toroidal patch.
    """

    row_chunks = []
    theta_chunks = []
    frac_chunks = []
    arc_cdfs = _pair_arc_cdfs(context, pair_indices)
    golden_ratio_conjugate = torch.as_tensor(
        0.6180339887498949,
        dtype=dtype,
        device=device,
    )
    grouped_rows: dict[int, list[int]] = {}
    for row, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        if count > 0:
            grouped_rows.setdefault(count, []).append(row)

    for count, rows_for_count in grouped_rows.items():
        rows = torch.as_tensor(rows_for_count, dtype=torch.long, device=device)
        sample_ids = torch.arange(count, dtype=dtype, device=device) + 0.5
        theta = 2 * math.pi * sample_ids / count
        area_fracs = torch.frac(sample_ids * golden_ratio_conjugate)
        values = area_fracs.view(1, -1).expand(rows.shape[0], -1).contiguous()
        cdf_rows = arc_cdfs[rows].contiguous()
        indices = torch.searchsorted(cdf_rows, values, right=False)
        indices = indices.clamp(1, cdf_rows.shape[1] - 1)
        lower = torch.gather(cdf_rows, 1, indices - 1)
        upper = torch.gather(cdf_rows, 1, indices)
        denom = (upper - lower).clamp_min(torch.finfo(dtype).eps)
        local = (values - lower) / denom
        frac = (indices.to(dtype) - 1 + local) / (cdf_rows.shape[1] - 1)
        row_chunks.append(rows.repeat_interleave(count))
        theta_chunks.append(theta.view(1, -1).expand(rows.shape[0], -1).reshape(-1))
        frac_chunks.append(frac.reshape(-1))
    if not row_chunks:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )
    return (
        torch.cat(row_chunks, dim=0),
        torch.cat(theta_chunks, dim=0),
        torch.cat(frac_chunks, dim=0),
    )


def _pair_arc_cdfs(
    context: ExteriorContext,
    pair_indices: torch.Tensor,
    *,
    arc_quadrature: int = 128,
) -> torch.Tensor:
    """Return normalized area CDFs along each pair's probe-sphere arc."""

    centers, circle_radii, basis_u, _, _ = _pair_circle_parameters(
        context.atom_coords,
        context.expanded_radii,
        pair_indices,
    )
    if pair_indices.numel() == 0:
        return torch.empty((0, arc_quadrature + 1), dtype=context.dtype, device=context.device)

    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    probe_centers = centers + circle_radii.unsqueeze(-1) * basis_u
    eps = torch.finfo(context.dtype).eps
    first_normals = (
        probe_centers - context.atom_coords[first]
    ) / context.expanded_radii[first].unsqueeze(-1).clamp_min(eps)
    second_normals = (
        probe_centers - context.atom_coords[second]
    ) / context.expanded_radii[second].unsqueeze(-1).clamp_min(eps)
    arc_angles = torch.acos((first_normals * second_normals).sum(dim=-1).clamp(-1, 1))
    arc_fracs = torch.linspace(0, 1, arc_quadrature + 1, dtype=context.dtype, device=context.device)
    directions = _slerp_unit_vectors(
        first_normals,
        second_normals,
        arc_angles,
        arc_fracs,
    )
    radial_components = (directions * basis_u.unsqueeze(1)).sum(dim=-1)
    densities = (
        circle_radii.unsqueeze(-1) - float(context.probe_radius) * radial_components
    ).clamp_min(0)
    segment_areas = 0.5 * (densities[:, 1:] + densities[:, :-1])
    cdf = torch.cat(
        (
            torch.zeros((pair_indices.shape[0], 1), dtype=context.dtype, device=context.device),
            segment_areas.cumsum(dim=-1),
        ),
        dim=-1,
    )
    totals = cdf[:, -1:].clamp_min(eps)
    return cdf / totals


def _invert_pair_arc_cdf(area_fracs: torch.Tensor, cdf: torch.Tensor) -> torch.Tensor:
    """Map uniform area fractions to probe-arc fractions through a tabulated CDF."""

    if area_fracs.numel() == 0:
        return area_fracs
    indices = torch.searchsorted(cdf.contiguous(), area_fracs.contiguous(), right=False)
    indices = indices.clamp(1, cdf.numel() - 1)
    lower = cdf[indices - 1]
    upper = cdf[indices]
    denom = (upper - lower).clamp_min(torch.finfo(area_fracs.dtype).eps)
    local = (area_fracs - lower) / denom
    return (indices.to(area_fracs.dtype) - 1 + local) / (cdf.numel() - 1)


def _slerp_unit_vectors(
    first_normals: torch.Tensor,
    second_normals: torch.Tensor,
    arc_angles: torch.Tensor,
    arc_fracs: torch.Tensor,
) -> torch.Tensor:
    """Spherically interpolate between two batches of unit vectors."""

    eps = torch.finfo(first_normals.dtype).eps
    angles = arc_angles.unsqueeze(-1)
    fracs = arc_fracs.view(1, -1)
    sin_angles = torch.sin(angles)
    first_weights = torch.sin((1 - fracs) * angles)
    second_weights = torch.sin(fracs * angles)
    lerp_first = 1 - fracs
    lerp_second = fracs
    use_lerp = sin_angles.abs() <= _sqrt_eps(first_normals.dtype)
    first_coefs = torch.where(
        use_lerp,
        lerp_first,
        first_weights / sin_angles.clamp_min(eps),
    )
    second_coefs = torch.where(
        use_lerp,
        lerp_second,
        second_weights / sin_angles.clamp_min(eps),
    )
    directions = (
        first_coefs.unsqueeze(-1) * first_normals.unsqueeze(1)
        + second_coefs.unsqueeze(-1) * second_normals.unsqueeze(1)
    )
    return directions / torch.linalg.norm(
        directions,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)


def _pair_points_from_params(
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    probe_radius: float,
    pair_indices: torch.Tensor,
    theta_values: torch.Tensor,
    arc_fracs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert pair-circle and arc parameters into probe centers and SES points."""

    centers, circle_radii, basis_u, basis_v, valid = _pair_circle_parameters(
        atom_coords,
        expanded_radii,
        pair_indices,
    )
    radial_dirs = (
        torch.cos(theta_values).unsqueeze(-1) * basis_u
        + torch.sin(theta_values).unsqueeze(-1) * basis_v
    )
    probe_centers = centers + circle_radii.unsqueeze(-1) * radial_dirs
    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    eps = torch.finfo(atom_coords.dtype).eps
    first_normals = (
        probe_centers - atom_coords[first]
    ) / expanded_radii[first].unsqueeze(-1).clamp_min(eps)
    second_normals = (
        probe_centers - atom_coords[second]
    ) / expanded_radii[second].unsqueeze(-1).clamp_min(eps)
    arc_angles = torch.acos((first_normals * second_normals).sum(dim=-1).clamp(-1, 1))
    sin_angles = torch.sin(arc_angles)
    first_weights = torch.sin((1 - arc_fracs) * arc_angles)
    second_weights = torch.sin(arc_fracs * arc_angles)
    lerp_first = 1 - arc_fracs
    lerp_second = arc_fracs
    use_lerp = sin_angles.abs() <= _sqrt_eps(atom_coords.dtype)
    first_coefs = torch.where(use_lerp, lerp_first, first_weights / sin_angles.clamp_min(eps))
    second_coefs = torch.where(use_lerp, lerp_second, second_weights / sin_angles.clamp_min(eps))
    probe_dirs = (
        first_coefs.unsqueeze(-1) * first_normals
        + second_coefs.unsqueeze(-1) * second_normals
    )
    probe_dirs = probe_dirs / torch.linalg.norm(
        probe_dirs,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    points = probe_centers - float(probe_radius) * probe_dirs
    finite = valid & torch.isfinite(points).all(dim=-1)
    points = torch.where(finite.unsqueeze(-1), points, torch.full_like(points, float("nan")))
    return probe_centers, points, first_weights, second_weights


def _candidate_triple_indices(
    num_atoms: int,
    pair_indices: torch.Tensor,
    *,
    max_triples: Optional[int],
) -> torch.Tensor:
    """Enumerate atom triples that form triangles in the candidate pair graph."""

    if pair_indices.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=pair_indices.device)
    if max_triples is not None and max_triples <= 0:
        return torch.empty((0, 3), dtype=torch.long, device=pair_indices.device)

    pairs = pair_indices.to(dtype=torch.long)
    pairs = torch.sort(pairs, dim=-1).values
    in_range = (
        (pairs[:, 0] >= 0)
        & (pairs[:, 1] < int(num_atoms))
        & (pairs[:, 0] < pairs[:, 1])
    )
    pairs = pairs[in_range]
    if pairs.numel() == 0:
        return torch.empty((0, 3), dtype=torch.long, device=pair_indices.device)

    pair_keys = pairs[:, 0] * int(num_atoms) + pairs[:, 1]
    order = torch.argsort(pair_keys)
    pair_keys = pair_keys[order]
    pairs = pairs[order]
    unique = torch.ones((pairs.shape[0],), dtype=torch.bool, device=pairs.device)
    unique[1:] = pair_keys[1:] != pair_keys[:-1]
    pairs = pairs[unique]
    pair_keys = pair_keys[unique]

    first_atoms = pairs[:, 0]
    counts = torch.bincount(first_atoms, minlength=int(num_atoms))
    max_degree = int(counts.max().item()) if counts.numel() > 0 else 0
    if max_degree < 2:
        return torch.empty((0, 3), dtype=torch.long, device=pair_indices.device)
    offsets = torch.cat(
        (
            torch.zeros((1,), dtype=torch.long, device=pair_indices.device),
            counts.cumsum(dim=0),
        ),
        dim=0,
    )
    # Real protein pair graphs are locally bounded, so enumerate triangles as
    # batched combinations inside each atom's forward neighborhood.  This keeps
    # the work proportional to local degree and maps cleanly to GPU kernels.
    max_combos = max_degree * (max_degree - 1) // 2
    rows_per_batch = max(
        1,
        min(int(num_atoms), _DEFAULT_TRIPLE_COMBO_BUDGET // max(1, max_combos)),
    )
    combo_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    triple_chunks = []
    triple_count = 0
    for start in range(0, int(num_atoms), rows_per_batch):
        stop = min(start + rows_per_batch, int(num_atoms))
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
            device=pair_indices.device,
        )
        local_pairs = pairs[first_pair:last_pair]
        local_first = local_pairs[:, 0] - start
        local_slots = (
            torch.arange(first_pair, last_pair, dtype=torch.long, device=pair_indices.device)
            - offsets[local_pairs[:, 0]]
        )
        slot_mask = local_slots < batch_degree
        neighbors[local_first[slot_mask], local_slots[slot_mask]] = local_pairs[slot_mask, 1]

        combos = combo_cache.get(batch_degree)
        if combos is None:
            combos = torch.triu_indices(
                batch_degree,
                batch_degree,
                offset=1,
                dtype=torch.long,
                device=pair_indices.device,
            )
            combo_cache[batch_degree] = (combos[0], combos[1])
        combo_first, combo_second = combo_cache[batch_degree]
        combo_count = int(combo_first.numel())
        first_column = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=pair_indices.device,
        ).repeat_interleave(combo_count)
        second_column = neighbors[:, combo_first].reshape(-1)
        third_column = neighbors[:, combo_second].reshape(-1)
        valid = third_column >= 0
        if not bool(valid.any().item()):
            continue

        second_valid = second_column[valid]
        third_valid = third_column[valid]
        candidate_keys = second_valid * int(num_atoms) + third_valid
        positions = torch.searchsorted(pair_keys, candidate_keys)
        in_bounds = positions < pair_keys.shape[0]
        safe_positions = positions.clamp_max(max(pair_keys.shape[0] - 1, 0))
        matches = in_bounds & (pair_keys[safe_positions] == candidate_keys)
        if not bool(matches.any().item()):
            continue

        chunk = torch.stack(
            (
                first_column[valid][matches],
                second_valid[matches],
                third_valid[matches],
            ),
            dim=-1,
        )
        if max_triples is not None:
            remaining = int(max_triples) - triple_count
            if remaining <= 0:
                break
            if chunk.shape[0] > remaining:
                chunk = chunk[:remaining]
        triple_chunks.append(chunk)
        triple_count += int(chunk.shape[0])
        if max_triples is not None and triple_count >= int(max_triples):
            break
    if not triple_chunks:
        return torch.empty((0, 3), dtype=torch.long, device=pair_indices.device)
    return torch.cat(triple_chunks, dim=0)


def _deduplicate_centers(centers: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Return first-seen indices after rounding centers onto a tolerance grid."""

    if centers.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=centers.device)
    keys = torch.round(centers.detach() / tolerance).to(torch.long)
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


def _probe_center_supports(
    centers: torch.Tensor,
    seeds: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    *,
    max_support_atoms: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find padded support atoms touching each fixed probe center.

    Seed atoms are always retained.  Additional atoms within ``tolerance`` of
    the expanded radius are included up to ``max_support_atoms``; when there are
    too many, the closest non-seed supports are kept.
    """

    support_rows = []
    mask_rows = []
    center_chunks = max(1, min(centers.shape[0], _DEFAULT_PAIR_BUDGET // atom_coords.shape[0]))
    for start in range(0, centers.shape[0], center_chunks):
        stop = min(start + center_chunks, centers.shape[0])
        dists = torch.linalg.norm(
            centers[start:stop].unsqueeze(1) - atom_coords.unsqueeze(0),
            dim=-1,
        )
        residuals = torch.abs(dists - expanded_radii.unsqueeze(0))
        near = residuals <= tolerance
        for local_row in range(stop - start):
            row = start + local_row
            touched = near[local_row].nonzero(as_tuple=False).reshape(-1)
            seed = seeds[row]
            merged = torch.unique(torch.cat((seed, touched))).to(torch.long)
            if merged.numel() > max_support_atoms:
                seed_member = (merged.unsqueeze(-1) == seed.unsqueeze(0)).any(dim=-1)
                nonseed_indices = merged[~seed_member]
                keep_count = max(0, max_support_atoms - int(seed.numel()))
                if nonseed_indices.numel() > 0 and keep_count > 0:
                    nonseed_residuals = residuals[local_row, nonseed_indices]
                    order = torch.argsort(nonseed_residuals)
                    keep_nonseed = nonseed_indices[order[:keep_count]]
                    merged = torch.unique(torch.cat((seed, keep_nonseed))).to(torch.long)
                else:
                    merged = torch.unique(seed).to(torch.long)
            padded = torch.full(
                (max_support_atoms,),
                -1,
                dtype=torch.long,
                device=centers.device,
            )
            mask = torch.zeros((max_support_atoms,), dtype=torch.bool, device=centers.device)
            count = min(max_support_atoms, int(merged.numel()))
            padded[:count] = merged[:count]
            mask[:count] = True
            support_rows.append(padded)
            mask_rows.append(mask)
    if not support_rows:
        return (
            torch.empty((0, max_support_atoms), dtype=torch.long, device=centers.device),
            torch.empty((0, max_support_atoms), dtype=torch.bool, device=centers.device),
        )
    return torch.stack(support_rows, dim=0), torch.stack(mask_rows, dim=0)


def _empty_probe_blocks(device: torch.device, max_support_atoms: int) -> _ProbeBlocks:
    """Build an empty fixed-probe block container with stable tensor shapes."""

    return _ProbeBlocks(
        probe_seed_indices=torch.empty((0, 3), dtype=torch.long, device=device),
        probe_center_signs=torch.empty((0,), dtype=torch.long, device=device),
        probe_support_indices=torch.empty(
            (0, max_support_atoms),
            dtype=torch.long,
            device=device,
        ),
        probe_support_mask=torch.empty(
            (0, max_support_atoms),
            dtype=torch.bool,
            device=device,
        ),
        probe_center_hints=torch.empty((0, 3), dtype=torch.float32, device=device),
    )


def _probe_patch_area_estimates(
    centers: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    *,
    probe_radius: float,
) -> torch.Tensor:
    """Estimate fixed-probe spherical patch areas.

    Support atom normals define a spherical polygon on the unit probe sphere.
    Three supports form one spherical triangle; larger support sets are ordered
    around their convex hull and fan-triangulated from the mean direction.
    """

    if support_indices.numel() == 0:
        return torch.empty((0,), dtype=centers.dtype, device=centers.device)
    safe_indices = support_indices.clamp_min(0)
    support_coords = atom_coords[safe_indices]
    support_radii = expanded_radii[safe_indices].clamp_min(torch.finfo(centers.dtype).eps)
    normals = (centers.unsqueeze(1) - support_coords) / support_radii.unsqueeze(-1)
    areas = []
    for row in range(normals.shape[0]):
        valid_normals = normals[row, support_mask[row]]
        if valid_normals.shape[0] < 3:
            areas.append(torch.zeros((), dtype=centers.dtype, device=centers.device))
        elif valid_normals.shape[0] == 3:
            areas.append(
                _spherical_triangle_area(
                    valid_normals[0],
                    valid_normals[1],
                    valid_normals[2],
                )
            )
        else:
            anchor = valid_normals.mean(dim=0)
            anchor = anchor / torch.linalg.norm(anchor).clamp_min(torch.finfo(centers.dtype).eps)
            order = _convex_hull_normal_order(valid_normals, anchor)
            ordered = valid_normals[order]
            area = torch.zeros((), dtype=centers.dtype, device=centers.device)
            for idx in range(ordered.shape[0]):
                area = area + _spherical_triangle_area(
                    anchor,
                    ordered[idx],
                    ordered[(idx + 1) % ordered.shape[0]],
                )
            areas.append(area)
    return torch.stack(areas).clamp_min(0) * float(probe_radius) ** 2


def _spherical_triangle_area(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Return the solid angle of a spherical triangle on the unit sphere."""

    eps = torch.finfo(a.dtype).eps
    determinant = torch.abs(torch.dot(a, torch.cross(b, c, dim=0)))
    denominator = 1 + torch.dot(a, b) + torch.dot(b, c) + torch.dot(c, a)
    return 2 * torch.atan2(determinant, denominator.clamp_min(eps))


def _sort_normals_around_mean(normals: torch.Tensor) -> torch.Tensor:
    """Sort normals by azimuth around their normalized mean direction."""

    axis = normals.mean(dim=0)
    axis = axis / torch.linalg.norm(axis).clamp_min(torch.finfo(normals.dtype).eps)
    ref = torch.zeros_like(axis)
    ref[2] = 1
    if bool(torch.abs(torch.dot(axis, ref)) > 0.9):
        ref = torch.zeros_like(axis)
        ref[1] = 1
    tangent_u = torch.cross(axis, ref, dim=0)
    tangent_u = tangent_u / torch.linalg.norm(tangent_u).clamp_min(torch.finfo(normals.dtype).eps)
    tangent_v = torch.cross(axis, tangent_u, dim=0)
    x = normals @ tangent_u
    y = normals @ tangent_v
    return torch.argsort(torch.atan2(y, x))


def _convex_hull_normal_order(normals: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Order support normals around the spherical polygon boundary.

    Normals are projected to the tangent plane at ``axis`` and ordered with a
    monotonic-chain convex hull.  If the projection is unstable, the simpler
    angular sort around the mean direction is used as a fallback.
    """

    if normals.shape[0] <= 3:
        return torch.arange(normals.shape[0], dtype=torch.long, device=normals.device)

    tangent_u, tangent_v = _orthonormal_tangent_basis(axis)
    denominators = normals @ axis
    if bool((denominators <= _sqrt_eps(normals.dtype)).any().item()):
        return _sort_normals_around_mean(normals)

    projected = torch.stack(
        (
            (normals @ tangent_u) / denominators,
            (normals @ tangent_v) / denominators,
        ),
        dim=-1,
    )
    y_order = torch.argsort(projected[:, 1], stable=True)
    order_tensor = y_order[torch.argsort(projected[y_order, 0], stable=True)]
    order = [int(value.item()) for value in order_tensor]
    if len(order) <= 3:
        return order_tensor

    def cross(origin: int, first: int, second: int) -> float:
        first_vec = projected[first] - projected[origin]
        second_vec = projected[second] - projected[origin]
        return float((first_vec[0] * second_vec[1] - first_vec[1] * second_vec[0]).item())

    tolerance = 1e-12
    lower: list[int] = []
    for idx in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], idx) <= tolerance:
            lower.pop()
        lower.append(idx)

    upper: list[int] = []
    for idx in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], idx) <= tolerance:
            upper.pop()
        upper.append(idx)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return _sort_normals_around_mean(normals)
    return torch.as_tensor(hull, dtype=torch.long, device=normals.device)


def _packed_simplex_weights(
    counts: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack generic simplex weights for rows with variable support counts."""

    max_support = int(support_mask.shape[1])
    row_chunks = []
    weight_chunks = []
    for row, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        if count <= 0:
            continue
        support_count = int(support_mask[row].sum().item())
        weights = _simplex_weight_rows(count, support_count, max_support, dtype, device)
        row_chunks.append(
            torch.full((weights.shape[0],), row, dtype=torch.long, device=device),
        )
        weight_chunks.append(weights)
    if not row_chunks:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0, max_support), dtype=dtype, device=device),
        )
    return torch.cat(row_chunks, dim=0), torch.cat(weight_chunks, dim=0)


def _packed_probe_patch_weights(
    centers: torch.Tensor,
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    atom_coords: torch.Tensor,
    expanded_radii: torch.Tensor,
    counts: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate packed support-weight rows for all fixed-probe patches.

    The returned weights describe convex combinations of support normals.  The
    caller later converts those weighted normals into directions on the probe
    sphere and therefore differentiable SES points.
    """

    max_support = int(support_mask.shape[1])
    if support_indices.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0, max_support), dtype=dtype, device=device),
        )

    safe_indices = support_indices.clamp_min(0)
    support_coords = atom_coords[safe_indices]
    support_radii = expanded_radii[safe_indices].clamp_min(torch.finfo(dtype).eps)
    normals = (centers.unsqueeze(1) - support_coords) / support_radii.unsqueeze(-1)

    row_chunks = []
    weight_chunks = []
    for row, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        if count <= 0:
            continue
        support_slots = support_mask[row].nonzero(as_tuple=False).reshape(-1)
        support_count = int(support_slots.numel())
        if support_count < 3:
            continue
        if support_count == 3:
            local_weights = _spherical_triangle_weight_rows(
                count,
                normals[row, support_slots],
                dtype,
                device,
            )
            weights = torch.zeros(
                (local_weights.shape[0], max_support),
                dtype=dtype,
                device=device,
            )
            weights[:, support_slots] = local_weights
        else:
            weights = _polygon_probe_patch_weights(
                count,
                normals[row, support_slots],
                support_slots,
                max_support,
                dtype,
                device,
            )
        if weights.shape[0] == 0:
            continue
        row_chunks.append(torch.full((weights.shape[0],), row, dtype=torch.long, device=device))
        weight_chunks.append(weights)

    if not row_chunks:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0, max_support), dtype=dtype, device=device),
        )
    return torch.cat(row_chunks, dim=0), torch.cat(weight_chunks, dim=0)


def _polygon_probe_patch_weights(
    count: int,
    normals: torch.Tensor,
    support_slots: torch.Tensor,
    max_support: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Generate support weights for one multi-support spherical polygon patch."""

    support_count = int(normals.shape[0])
    if support_count < 3 or count <= 0:
        return torch.empty((0, max_support), dtype=dtype, device=device)

    anchor_sum = normals.mean(dim=0)
    anchor_norm = torch.linalg.norm(anchor_sum).clamp_min(torch.finfo(dtype).eps)
    anchor = anchor_sum / anchor_norm
    order = _convex_hull_normal_order(normals, anchor)
    ordered_normals = normals[order]
    ordered_slots = support_slots[order]
    anchor_weights = torch.zeros((max_support,), dtype=dtype, device=device)
    anchor_weights[support_slots] = 1.0 / (support_count * anchor_norm)
    boundary_count = int(ordered_normals.shape[0])
    sample_count = count
    # Wide polygons need extra candidate directions because rejection sampling
    # can concentrate near edges when only a few points are requested.
    if support_count >= 5:
        sample_count = max(sample_count, 3 * boundary_count, 2 * support_count)
    elif support_count == 4:
        sample_count = max(sample_count, 2 * boundary_count)

    polygon_weights = _rejection_spherical_polygon_weight_rows(
        sample_count,
        ordered_normals,
        ordered_slots,
        anchor,
        anchor_weights,
        max_support,
        dtype,
        device,
    )
    if polygon_weights.shape[0] >= sample_count:
        return polygon_weights[:sample_count]

    # Fallback: triangulate the polygon fan from the anchor and sample each
    # spherical triangle in proportion to its solid angle.
    sub_areas = []
    for index in range(boundary_count):
        sub_areas.append(
            _spherical_triangle_area(
                anchor,
                ordered_normals[index],
                ordered_normals[(index + 1) % boundary_count],
            )
        )
    sub_areas_tensor = torch.stack(sub_areas).clamp_min(0)
    sub_counts = _allocate_counts_by_weights(sample_count, sub_areas_tensor)

    polygon_weights = []
    for index, sub_count_tensor in enumerate(sub_counts):
        sub_count = int(sub_count_tensor.item())
        if sub_count <= 0:
            continue
        triangle_normals = torch.stack(
            (
                anchor,
                ordered_normals[index],
                ordered_normals[(index + 1) % boundary_count],
            ),
            dim=0,
        )
        triangle_weights = _spherical_triangle_weight_rows(
            sub_count,
            triangle_normals,
            dtype,
            device,
        )
        weights = triangle_weights[:, 0:1] * anchor_weights.unsqueeze(0)
        weights[:, ordered_slots[index]] = (
            weights[:, ordered_slots[index]] + triangle_weights[:, 1]
        )
        weights[:, ordered_slots[(index + 1) % boundary_count]] = (
            weights[:, ordered_slots[(index + 1) % boundary_count]] + triangle_weights[:, 2]
        )
        polygon_weights.append(weights)

    if not polygon_weights:
        return torch.empty((0, max_support), dtype=dtype, device=device)
    return _deduplicate_weight_rows(torch.cat(polygon_weights, dim=0))


def _deduplicate_weight_rows(
    weights: torch.Tensor,
    *,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    """Remove duplicate support-weight rows after tolerance-grid rounding."""

    if weights.shape[0] <= 1:
        return weights

    keys = torch.round(weights.detach() / tolerance).to(torch.long)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    indices = torch.arange(weights.shape[0], dtype=torch.long, device=weights.device)
    first_indices = torch.full(
        (int(inverse.max().item()) + 1,),
        weights.shape[0],
        dtype=torch.long,
        device=weights.device,
    )
    first_indices.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    first_indices = first_indices[first_indices < weights.shape[0]]
    return weights[torch.sort(first_indices).values]


def _allocate_counts_by_weights(total_count: int, weights: torch.Tensor) -> torch.Tensor:
    """Allocate an integer total across positive weights while preserving sum."""

    if weights.numel() == 0 or total_count <= 0:
        return torch.zeros_like(weights, dtype=torch.long)
    positive = weights > 0
    if not bool(positive.any().item()):
        result = torch.zeros_like(weights, dtype=torch.long)
        result[0] = total_count
        return result

    normalized = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    raw_counts = normalized * total_count
    counts = torch.floor(raw_counts).to(torch.long)
    counts = torch.where(positive & (counts == 0), torch.ones_like(counts), counts)
    diff = int(total_count - int(counts.sum().item()))
    if diff > 0:
        remainders = raw_counts - torch.floor(raw_counts)
        order = torch.argsort(remainders, descending=True)
        for index in order[:diff]:
            counts[index] += 1
    elif diff < 0:
        removable = counts - torch.where(
            positive,
            torch.ones_like(counts),
            torch.zeros_like(counts),
        )
        order = torch.argsort(raw_counts - torch.floor(raw_counts))
        remaining = -diff
        for index in order:
            if remaining <= 0:
                break
            take = min(int(removable[index].item()), remaining)
            if take > 0:
                counts[index] -= take
                remaining -= take
    return counts


def _simplex_weight_rows(
    count: int,
    support_count: int,
    max_support: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return deterministic low-discrepancy weights on a support simplex."""

    weights = torch.zeros((count, max_support), dtype=dtype, device=device)
    if support_count <= 0:
        return weights
    if support_count == 1:
        weights[:, 0] = 1
        return weights
    if support_count == 3:
        return _triangle_weight_rows(count, max_support, dtype, device)
    sample_ids = torch.arange(count, dtype=dtype, device=device).unsqueeze(-1)
    bases = torch.arange(1, support_count + 1, dtype=dtype, device=device).unsqueeze(0)
    golden = torch.as_tensor(0.6180339887498949, dtype=dtype, device=device)
    raw = torch.frac((sample_ids + 1) * (bases * golden + 0.17320508075688773))
    raw = raw + 0.05
    raw = raw / raw.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    weights[:, :support_count] = raw
    return weights


def _spherical_triangle_weight_rows(
    count: int,
    vertices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return support weights for samples inside a spherical triangle."""

    if count <= 0:
        return torch.empty((0, 3), dtype=dtype, device=device)
    if count < 12:
        return _triangle_weight_rows(count, 3, dtype, device)[:, :3]

    vertices = vertices.to(dtype=dtype, device=device)
    vertices = vertices / torch.linalg.norm(
        vertices,
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(dtype).eps)
    area = _spherical_triangle_area(vertices[0], vertices[1], vertices[2])
    if not bool(torch.isfinite(area).item()) or float(area.item()) <= _sqrt_eps(dtype):
        return _triangle_weight_rows(count, 3, dtype, device)[:, :3]

    weights = _rejection_spherical_triangle_weight_rows(
        count,
        vertices,
        area,
        dtype,
        device,
    )
    if weights.shape[0] >= count:
        return weights[:count]

    fallback = _triangle_weight_rows(count, 3, dtype, device)[:, :3]
    weights = _deduplicate_weight_rows(torch.cat((weights, fallback), dim=0))
    return weights[:count]


def _rejection_spherical_triangle_weight_rows(
    count: int,
    vertices: torch.Tensor,
    area: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sample a spherical triangle by rejection from a containing cap.

    Accepted directions are converted to cone coefficients relative to the
    triangle vertices.  Those coefficients are later used as support weights for
    differentiable reconstruction.
    """

    axis = vertices.mean(dim=0)
    axis_norm = torch.linalg.norm(axis)
    if not bool((axis_norm > torch.finfo(dtype).eps).item()):
        axis = vertices[0]
    else:
        axis = axis / axis_norm

    cap_cos = float((vertices @ axis).min().clamp(-1, 1).item()) - 1e-6
    cap_cos = max(-1.0, min(1.0, cap_cos))
    if cap_cos <= 0.0:
        cap_cos = -1.0

    cap_area = max(2 * math.pi * (1.0 - cap_cos), float(area.item()))
    acceptance = max(min(float(area.item()) / cap_area, 1.0), 1e-3)
    batch_size = max(64, int(math.ceil(1.5 * count / acceptance)))
    target_candidates = _well_spaced_candidate_count(count)
    max_candidates = max(batch_size, 64) * 32
    offset = 0
    accepted_chunks = []
    accepted_direction_chunks = []
    accepted_count = 0
    while accepted_count < target_candidates and offset < max_candidates:
        # The deterministic cap sequence avoids random seeds while still giving
        # enough coverage for greedy well-spaced thinning.
        directions = _low_discrepancy_cap_directions(
            batch_size,
            axis,
            cap_cos,
            offset,
            dtype,
            device,
        )
        coefficients, inside = _triangle_cone_coefficients(directions, vertices)
        if bool(inside.any().item()):
            accepted = coefficients[inside]
            accepted_chunks.append(accepted)
            accepted_direction_chunks.append(directions[inside])
            accepted_count += int(accepted.shape[0])
        offset += batch_size
        if accepted_count == 0:
            batch_size *= 2

    if not accepted_chunks:
        return torch.empty((0, 3), dtype=dtype, device=device)
    weights = torch.cat(accepted_chunks, dim=0)
    if count <= _WELL_SPACED_SELECTION_LIMIT and weights.shape[0] > count:
        directions = torch.cat(accepted_direction_chunks, dim=0)
        selected = _well_spaced_direction_indices(directions, count, axis)
        weights = weights[selected]
    return weights[:count]


def _rejection_spherical_polygon_weight_rows(
    count: int,
    ordered_normals: torch.Tensor,
    ordered_slots: torch.Tensor,
    anchor: torch.Tensor,
    anchor_weights: torch.Tensor,
    max_support: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Sample a spherical polygon by cap rejection and fan assignment."""

    support_count = int(ordered_normals.shape[0])
    if count <= 0 or support_count < 3:
        return torch.empty((0, max_support), dtype=dtype, device=device)

    area = torch.zeros((), dtype=dtype, device=device)
    for index in range(support_count):
        area = area + _spherical_triangle_area(
            anchor,
            ordered_normals[index],
            ordered_normals[(index + 1) % support_count],
        )
    if not bool(torch.isfinite(area).item()) or float(area.item()) <= _sqrt_eps(dtype):
        return torch.empty((0, max_support), dtype=dtype, device=device)

    cap_cos = float((ordered_normals @ anchor).min().clamp(-1, 1).item()) - 1e-6
    cap_cos = max(-1.0, min(1.0, cap_cos))
    if cap_cos <= 0.0:
        cap_cos = -1.0
    cap_area = max(2 * math.pi * (1.0 - cap_cos), float(area.item()))
    acceptance = max(min(float(area.item()) / cap_area, 1.0), 1e-3)
    batch_size = max(64, int(math.ceil(1.5 * count / acceptance)))
    target_candidates = _well_spaced_candidate_count(count)
    max_candidates = max(batch_size, 64) * 32
    edge_normals = torch.cross(
        ordered_normals,
        torch.roll(ordered_normals, shifts=-1, dims=0),
        dim=-1,
    )
    # Orient half-space tests so points inside the polygon satisfy every edge.
    edge_signs = torch.sign(edge_normals @ anchor).clamp_min(0) * 2 - 1
    edge_normals = edge_normals * edge_signs.unsqueeze(-1)

    offset = 0
    accepted_chunks = []
    accepted_direction_chunks = []
    accepted_count = 0
    while accepted_count < target_candidates and offset < max_candidates:
        directions = _low_discrepancy_cap_directions(
            batch_size,
            anchor,
            cap_cos,
            offset,
            dtype,
            device,
        )
        inside_tol = 1000 * torch.finfo(dtype).eps
        inside_polygon = (
            directions @ edge_normals.transpose(0, 1) >= -inside_tol
        ).all(dim=-1)
        if bool(inside_polygon.any().item()):
            weights, assigned_directions = _polygon_direction_weights(
                directions[inside_polygon],
                ordered_normals,
                ordered_slots,
                anchor,
                anchor_weights,
                max_support,
                dtype,
                device,
            )
            if weights.shape[0] > 0:
                accepted_chunks.append(weights)
                accepted_direction_chunks.append(assigned_directions)
                accepted_count += int(weights.shape[0])
        offset += batch_size
        if accepted_count == 0:
            batch_size *= 2

    if not accepted_chunks:
        return torch.empty((0, max_support), dtype=dtype, device=device)
    weights = torch.cat(accepted_chunks, dim=0)
    if count <= _WELL_SPACED_SELECTION_LIMIT and weights.shape[0] > count:
        directions = torch.cat(accepted_direction_chunks, dim=0)
        selected = _well_spaced_direction_indices(directions, count, anchor)
        weights = weights[selected]
    return weights[:count]


def _polygon_direction_weights(
    directions: torch.Tensor,
    ordered_normals: torch.Tensor,
    ordered_slots: torch.Tensor,
    anchor: torch.Tensor,
    anchor_weights: torch.Tensor,
    max_support: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign accepted polygon directions to fan triangles and support weights."""

    support_count = int(ordered_normals.shape[0])
    if directions.numel() == 0:
        return (
            torch.empty((0, max_support), dtype=dtype, device=device),
            torch.empty((0, 3), dtype=dtype, device=device),
        )

    result = torch.zeros((directions.shape[0], max_support), dtype=dtype, device=device)
    assigned = torch.zeros((directions.shape[0],), dtype=torch.bool, device=device)
    for index in range(support_count):
        triangle_normals = torch.stack(
            (
                anchor,
                ordered_normals[index],
                ordered_normals[(index + 1) % support_count],
            ),
            dim=0,
        )
        triangle_weights, inside = _triangle_cone_coefficients(
            directions,
            triangle_normals,
        )
        inside = inside & ~assigned
        if not bool(inside.any().item()):
            continue

        weights = triangle_weights[inside, 0:1] * anchor_weights.unsqueeze(0)
        weights[:, ordered_slots[index]] = (
            weights[:, ordered_slots[index]] + triangle_weights[inside, 1]
        )
        weights[:, ordered_slots[(index + 1) % support_count]] = (
            weights[:, ordered_slots[(index + 1) % support_count]]
            + triangle_weights[inside, 2]
        )
        result[inside] = weights
        assigned[inside] = True

    return result[assigned], directions[assigned]


def _well_spaced_candidate_count(count: int) -> int:
    """Return how many rejection candidates to collect before thinning."""

    if count <= 0:
        return 0
    if count > _WELL_SPACED_SELECTION_LIMIT:
        return count
    return max(count + 32, count * _WELL_SPACED_CANDIDATE_FACTOR, 64)


def _well_spaced_direction_indices(
    directions: torch.Tensor,
    count: int,
    preferred_axis: torch.Tensor,
) -> torch.Tensor:
    """Select a greedily well-spaced subset of unit directions."""

    if count <= 0 or directions.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=directions.device)
    if directions.shape[0] <= count:
        return torch.arange(directions.shape[0], dtype=torch.long, device=directions.device)

    axis = preferred_axis / torch.linalg.norm(preferred_axis).clamp_min(
        torch.finfo(directions.dtype).eps,
    )
    selected = []
    first = int(torch.argmax(directions @ axis).item())
    selected.append(first)
    nearest_dot = directions @ directions[first]

    for _ in range(1, count):
        next_index = int(torch.argmin(nearest_dot).item())
        selected.append(next_index)
        candidate_dots = directions @ directions[next_index]
        nearest_dot = torch.maximum(nearest_dot, candidate_dots)

    return torch.as_tensor(selected, dtype=torch.long, device=directions.device)


def _low_discrepancy_cap_directions(
    count: int,
    axis: torch.Tensor,
    cap_cos: float,
    offset: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Generate deterministic directions inside a spherical cap."""

    if count <= 0:
        return torch.empty((0, 3), dtype=dtype, device=device)

    sample_ids = torch.arange(
        offset,
        offset + count,
        dtype=dtype,
        device=device,
    ) + 0.5
    z_seed = torch.frac(sample_ids * 0.7548776662466927)
    phi_seed = torch.frac(sample_ids * 0.5698402909980532 + 0.137)
    cos_theta = 1.0 - z_seed * (1.0 - float(cap_cos))
    sin_theta = torch.sqrt((1.0 - cos_theta.square()).clamp_min(0))
    phi = 2 * math.pi * phi_seed

    tangent_u, tangent_v = _orthonormal_tangent_basis(axis)
    return (
        cos_theta.unsqueeze(-1) * axis.unsqueeze(0)
        + sin_theta.unsqueeze(-1)
        * (
            torch.cos(phi).unsqueeze(-1) * tangent_u.unsqueeze(0)
            + torch.sin(phi).unsqueeze(-1) * tangent_v.unsqueeze(0)
        )
    )


def _orthonormal_tangent_basis(axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two perpendicular unit vectors spanning the plane normal to axis."""

    ref = torch.zeros_like(axis)
    ref[2] = 1
    if bool(torch.abs(torch.dot(axis, ref)) > 0.9):
        ref = torch.zeros_like(axis)
        ref[1] = 1
    tangent_u = torch.cross(axis, ref, dim=0)
    tangent_u = tangent_u / torch.linalg.norm(tangent_u).clamp_min(
        torch.finfo(axis.dtype).eps,
    )
    tangent_v = torch.cross(axis, tangent_u, dim=0)
    return tangent_u, tangent_v


def _triangle_cone_coefficients(
    directions: torch.Tensor,
    vertices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express directions as non-negative coefficients of three cone vertices."""

    if directions.numel() == 0:
        return directions.new_empty((0, 3)), directions.new_empty((0,), dtype=torch.bool)

    matrix = vertices.transpose(0, 1)
    determinant = torch.linalg.det(matrix)
    if bool(torch.abs(determinant) > 1000 * torch.finfo(vertices.dtype).eps):
        coefficients = torch.linalg.solve(matrix, directions.transpose(0, 1)).transpose(0, 1)
    else:
        coefficients = (torch.linalg.pinv(matrix) @ directions.transpose(0, 1)).transpose(0, 1)

    reconstructed = coefficients @ vertices
    residuals = torch.linalg.norm(reconstructed - directions, dim=-1)
    coef_tol = 1000 * torch.finfo(vertices.dtype).eps
    residual_tol = 10 * math.sqrt(torch.finfo(vertices.dtype).eps)
    inside = (
        torch.isfinite(coefficients).all(dim=-1)
        & (coefficients >= -coef_tol).all(dim=-1)
        & (residuals <= residual_tol)
    )
    coefficients = coefficients.clamp_min(0)
    coefficients = coefficients / coefficients.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(vertices.dtype).eps)
    return coefficients, inside


def _triangle_weight_rows(
    count: int,
    max_support: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic barycentric grid for planar/simplex fallbacks."""

    if count <= 0:
        return torch.zeros((0, max_support), dtype=dtype, device=device)

    level = 0
    while (level + 1) * (level + 2) // 2 < count:
        level += 1

    if level == 0:
        weights = torch.zeros((1, max_support), dtype=dtype, device=device)
        weights[:, :3] = 1.0 / 3.0
        return weights

    barycentric_rows = []
    for first in range(level + 1):
        for second in range(level + 1 - first):
            third = level - first - second
            barycentric_rows.append((first, second, third))
    barycentric = torch.as_tensor(barycentric_rows, dtype=dtype, device=device)
    barycentric = barycentric / level
    center = torch.full((1, 3), 1.0 / 3.0, dtype=dtype, device=device)
    center_exists = torch.isclose(
        barycentric,
        center,
        rtol=1e-6,
        atol=1e-12,
    ).all(dim=-1).any()
    if not bool(center_exists.item()):
        barycentric = torch.cat((center, barycentric), dim=0)

    weights = torch.zeros((barycentric.shape[0], max_support), dtype=dtype, device=device)
    weights[:, :3] = barycentric
    return weights


def _empty_samples(context: ExteriorContext, max_support: int) -> AnalyticSamples:
    """Create an empty sample container with the requested support width."""

    return AnalyticSamples(
        points=torch.empty((0, 3), dtype=context.dtype, device=context.device),
        block_types=torch.empty((0,), dtype=torch.long, device=context.device),
        block_indices=torch.empty((0,), dtype=torch.long, device=context.device),
        support_indices=torch.empty((0, max_support), dtype=torch.long, device=context.device),
        support_mask=torch.empty((0, max_support), dtype=torch.bool, device=context.device),
        support_weights=torch.empty((0, max_support), dtype=context.dtype, device=context.device),
    )


def _concat_samples(
    samples: tuple[AnalyticSamples, ...],
    *,
    context: ExteriorContext,
    include_atom_weights: bool,
    blocks: AnalyticBlocks,
) -> AnalyticSamples:
    """Concatenate block-family samples and harmonize support-column counts."""

    max_support = max((sample.support_indices.shape[1] for sample in samples), default=1)
    nonempty = [sample for sample in samples if sample.points.shape[0] > 0]
    if not nonempty:
        empty = _empty_samples(context, max_support=max_support)
        atom_weights = (
            torch.empty((0, context.num_atoms), dtype=context.dtype, device=context.device)
            if include_atom_weights
            else None
        )
        return AnalyticSamples(
            points=empty.points,
            block_types=empty.block_types,
            block_indices=empty.block_indices,
            support_indices=empty.support_indices,
            support_mask=empty.support_mask,
            support_weights=empty.support_weights,
            atom_weights=atom_weights,
            blocks=blocks,
        )

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
    atom_weights = (
        _dense_atom_weights(
            support_indices,
            support_mask,
            support_weights,
            num_atoms=context.num_atoms,
        )
        if include_atom_weights
        else None
    )
    return AnalyticSamples(
        points=points,
        block_types=block_types,
        block_indices=block_indices,
        support_indices=support_indices,
        support_mask=support_mask,
        support_weights=support_weights,
        atom_weights=atom_weights,
        blocks=blocks,
    )


def _pad_columns(
    values: torch.Tensor,
    columns: int,
    *,
    fill_value: Union[float, int, bool],
) -> torch.Tensor:
    """Right-pad a two-dimensional tensor to ``columns`` columns."""

    if values.shape[1] == columns:
        return values
    result = torch.full(
        (values.shape[0], columns),
        fill_value,
        dtype=values.dtype,
        device=values.device,
    )
    result[:, : values.shape[1]] = values
    return result


def _dense_atom_weights(
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    support_weights: torch.Tensor,
    *,
    num_atoms: int,
) -> torch.Tensor:
    """Scatter sparse support weights into a dense per-atom weight matrix."""

    dense = torch.zeros(
        (support_indices.shape[0], num_atoms),
        dtype=support_weights.dtype,
        device=support_weights.device,
    )
    if support_indices.numel() == 0 or num_atoms == 0:
        return dense
    safe_indices = support_indices.clamp_min(0)
    weighted = torch.where(support_mask, support_weights, torch.zeros_like(support_weights))
    dense.scatter_add_(1, safe_indices, weighted)
    return dense


def _dense_atom_features(
    support_indices: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    num_atoms: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Scatter sparse supports into dense binary atom-assignment features."""

    dense = torch.zeros(
        (support_indices.shape[0], num_atoms),
        dtype=dtype,
        device=support_indices.device,
    )
    if support_indices.numel() == 0 or num_atoms == 0:
        return dense
    safe_indices = support_indices.clamp_min(0)
    support_values = support_mask.to(dtype=dtype)
    dense.scatter_add_(1, safe_indices, support_values)
    return dense.clamp_max(1)


__all__ = [
    "ATOM_BLOCK_TYPE",
    "PAIR_BLOCK_TYPE",
    "PROBE_BLOCK_TYPE",
    "AnalyticBlocks",
    "AnalyticSamples",
    "ExteriorContext",
    "build_analytic_blocks",
    "sample_analytic_points",
]
