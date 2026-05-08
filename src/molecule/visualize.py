"""py3Dmol helpers for SES molecule demos."""

from __future__ import annotations

import numpy as np
import py3Dmol
import torch

from molecule.examples import ATOM_COLORS, COVALENT_RADII, get_molecule
from ses.analytic import (
    ATOM_BLOCK_TYPE,
    PAIR_BLOCK_TYPE,
    PROBE_BLOCK_TYPE,
    _dense_atom_features,
    _sample_analytic_samples,
)
from ses.graph import build_surface_adjacency
from ses.projection import sample_atom_points, _sample_projected_grid
from ses.sdf import sample_sdf_points
from ses.tiled_analytic import _sample_tiled_analytic_samples


ANALYTIC_BLOCK_COLORS = {
    ATOM_BLOCK_TYPE: "#2a9d8f",
    PAIR_BLOCK_TYPE: "#f4a261",
    PROBE_BLOCK_TYPE: "#9b5de5",
}

ANALYTIC_BLOCK_NAMES = {
    ATOM_BLOCK_TYPE: "atom",
    PAIR_BLOCK_TYPE: "pair",
    PROBE_BLOCK_TYPE: "probe",
}

SDF_SUPPORT_COLORS = {
    1: "#2a9d8f",
    2: "#f4a261",
    3: "#9b5de5",
}

DEFAULT_SURFACE_EDGE_RADIUS = 0.012
DEFAULT_SURFACE_EDGE_COLOR = "#202124"
DEFAULT_SURFACE_EDGE_OPACITY = 0.58
DEFAULT_SURFACE_NORMAL_LENGTH = 0.25
DEFAULT_SURFACE_NORMAL_RADIUS = 0.012
DEFAULT_SURFACE_NORMAL_COLOR = "#111111"
DEFAULT_SURFACE_NORMAL_OPACITY = 0.68
DEFAULT_SURFACE_NORMAL_START_OFFSET = 0.035
DEFAULT_SURFACE_NORMAL_TIP_START = 0.72
DEFAULT_SURFACE_NORMAL_TIP_RADIUS_RATIO = 2.0


def _adjacency_to_edge_arrays(adjacency):
    adjacency = adjacency.coalesce()
    if adjacency._nnz() == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    indices = adjacency.indices()
    values = adjacency.values()
    keep = indices[0] < indices[1]
    return (
        indices[:, keep].transpose(0, 1).detach().cpu().numpy(),
        values[keep].detach().cpu().numpy(),
    )


def _limited_rows(total_count, max_count):
    if max_count is None or total_count <= max_count:
        return np.arange(total_count, dtype=np.int64)
    max_count = max(0, int(max_count))
    if max_count == 0:
        return np.empty((0,), dtype=np.int64)
    return np.linspace(0, total_count - 1, max_count, dtype=np.int64)


def _with_optional_normals_and_edges(
    result,
    points,
    normals,
    *,
    include_normals,
    include_edges,
    adjacency=None,
):
    if include_normals:
        result["normals"] = normals.detach().cpu().numpy()
    if include_edges:
        edge_indices, edge_weights = _adjacency_to_edge_arrays(adjacency)
        result["edge_indices"] = edge_indices
        result["edge_weights"] = edge_weights
    return result


def sample_display_points(
    molecule,
    m,
    probe_radius,
    *,
    include_normals=False,
    include_edges=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
):
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    need_normals = include_normals or include_edges
    atom_points = sample_atom_points(coords, radii, m)
    projected_outputs = _sample_projected_grid(
        coords,
        radii,
        m,
        probe_radius,
        include_normals=need_normals,
    )
    if need_normals:
        ses_points, valid_mask, normals = projected_outputs
    else:
        ses_points, valid_mask = projected_outputs
        normals = None
    finite_mask = torch.isfinite(atom_points).all(dim=-1) & torch.isfinite(ses_points).all(dim=-1)
    if normals is not None:
        finite_mask = finite_mask & torch.isfinite(normals).all(dim=-1)
    display_mask = valid_mask & finite_mask

    raw_owner_indices = (
        torch.arange(atom_points.shape[1], device=atom_points.device)
        .view(1, -1)
        .expand(atom_points.shape[0], -1)
        .reshape(-1)
        .detach()
        .cpu()
        .numpy()
    )
    owner_indices_tensor = display_mask.nonzero(as_tuple=False)[:, 1]
    points = ses_points[display_mask]
    graph_normals = normals[display_mask] if normals is not None else None
    adjacency = None
    if include_edges:
        support_indices = owner_indices_tensor.unsqueeze(-1)
        support_mask = torch.ones_like(support_indices, dtype=torch.bool)
        adjacency = build_surface_adjacency(
            points,
            graph_normals,
            support_indices=support_indices,
            support_mask=support_mask,
            weight_mode=adjacency_weight,
            neighbors=adjacency_neighbors,
            candidate_neighbors=adjacency_candidate_neighbors,
            prune_redundant_edges=adjacency_prune_redundant,
            allow_disjoint_single_support_edges=True,
        )
    result = {
        "raw_atom_points": atom_points.reshape(-1, 3).detach().cpu().numpy(),
        "raw_owner_indices": raw_owner_indices,
        "atom_points": atom_points[display_mask].detach().cpu().numpy(),
        "ses_points": points.detach().cpu().numpy(),
        "owner_indices": owner_indices_tensor.detach().cpu().numpy(),
        "valid_fraction": display_mask.to(dtype=torch.float64).mean().item(),
    }
    return _with_optional_normals_and_edges(
        result,
        points,
        graph_normals,
        include_normals=include_normals,
        include_edges=include_edges,
        adjacency=adjacency,
    )


def sample_analytic_display_points(
    molecule,
    point_area=0.03,
    probe_radius=1.4,
    device=None,
    dtype=torch.float64,
    include_atom_weights=True,
    probe_density_scale=1.0,
    atom_filter_samples=256,
    pair_filter_samples=24,
    max_probe_triples=5_000_000,
    max_grid_points=50_000_000,
    include_normals=False,
    include_edges=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
):
    """Sample analytic SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
    need_normals = include_normals or include_edges
    samples = _sample_analytic_samples(
        coords,
        radii,
        probe_radius,
        point_area=point_area,
        probe_density_scale=probe_density_scale,
        include_atom_weights=include_atom_weights,
        include_normals=need_normals,
        atom_filter_samples=atom_filter_samples,
        pair_filter_samples=pair_filter_samples,
        max_probe_triples=max_probe_triples,
        max_grid_points=max_grid_points,
    )

    unique_types, type_counts = samples.block_types.detach().cpu().unique(
        return_counts=True,
    )
    block_type_counts = {
        ANALYTIC_BLOCK_NAMES[int(block_type)]: int(count)
        for block_type, count in zip(unique_types, type_counts)
    }
    atom_weights = None
    if samples.atom_weights is not None:
        atom_weights = samples.atom_weights.detach().cpu().numpy()

    adjacency = None
    if include_edges:
        adjacency = build_surface_adjacency(
            samples.points,
            samples.normals,
            support_indices=samples.support_indices,
            support_mask=samples.support_mask,
            block_types=samples.block_types,
            block_indices=samples.block_indices,
            weight_mode=adjacency_weight,
            neighbors=adjacency_neighbors,
            candidate_neighbors=adjacency_candidate_neighbors,
            prune_redundant_edges=adjacency_prune_redundant,
        )

    result = {
        "ses_points": samples.points.detach().cpu().numpy(),
        "block_types": samples.block_types.detach().cpu().numpy(),
        "block_indices": samples.block_indices.detach().cpu().numpy(),
        "support_indices": samples.support_indices.detach().cpu().numpy(),
        "support_mask": samples.support_mask.detach().cpu().numpy(),
        "support_weights": samples.support_weights.detach().cpu().numpy(),
        "atom_weights": atom_weights,
        "block_type_counts": block_type_counts,
        "num_blocks": samples.blocks.num_blocks if samples.blocks is not None else 0,
        "device": str(device),
    }
    return _with_optional_normals_and_edges(
        result,
        samples.points,
        samples.normals,
        include_normals=include_normals,
        include_edges=include_edges,
        adjacency=adjacency,
    )


def sample_sdf_display_points(
    molecule,
    m=128,
    probe_radius=1.4,
    smoothness=0.3,
    iterations=6,
    level_tolerance=0.05,
    subsample_spacing=None,
    feature_threshold=0.1,
    device=None,
    dtype=torch.float64,
    max_grid_points=2_000_000,
    include_normals=False,
    include_edges=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
):
    """Sample SDF level-set SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
    outputs = sample_sdf_points(
        coords,
        radii,
        m=m,
        probe_radius=probe_radius,
        smoothness=smoothness,
        iterations=iterations,
        level_tolerance=level_tolerance,
        subsample_spacing=subsample_spacing,
        feature_threshold=feature_threshold,
        include_atom_features=True,
        include_normals=include_normals,
        include_adjacency=include_edges,
        adjacency_weight=adjacency_weight,
        adjacency_neighbors=adjacency_neighbors,
        adjacency_candidate_neighbors=adjacency_candidate_neighbors,
        adjacency_prune_redundant=adjacency_prune_redundant,
        max_grid_points=max_grid_points,
    )
    if include_normals and include_edges:
        points, atom_features, normals, adjacency = outputs
    elif include_normals:
        points, atom_features, normals = outputs
        adjacency = None
    elif include_edges:
        points, atom_features, adjacency = outputs
        normals = None
    else:
        points, atom_features = outputs
        normals = None
        adjacency = None

    support_counts = atom_features.sum(dim=1).to(dtype=torch.long)
    unique_counts, count_counts = support_counts.detach().cpu().unique(return_counts=True)
    support_count_counts = {
        int(support_count): int(count)
        for support_count, count in zip(unique_counts, count_counts)
    }

    result = {
        "ses_points": points.detach().cpu().numpy(),
        "atom_features": atom_features.detach().cpu().numpy(),
        "support_counts": support_counts.detach().cpu().numpy(),
        "support_count_counts": support_count_counts,
        "device": str(device),
    }
    return _with_optional_normals_and_edges(
        result,
        points,
        normals,
        include_normals=include_normals,
        include_edges=include_edges,
        adjacency=adjacency,
    )


def sample_tiled_analytic_display_points(
    molecule,
    point_area=0.5,
    probe_radius=1.4,
    tile_size="auto",
    tile_overlap="auto",
    atom_density_scale=1.55,
    pair_density_scale=1.55,
    probe_density_scale=1.55,
    dedup_tolerance=0.05,
    exact_accessibility=False,
    device=None,
    dtype=torch.float64,
    max_grid_points=500_000,
    max_probe_triples=5_000_000,
    include_normals=False,
    include_edges=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
):
    """Sample tiled analytic SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
    need_normals = include_normals or include_edges
    samples = _sample_tiled_analytic_samples(
        coords,
        radii,
        probe_radius,
        point_area=point_area,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        atom_density_scale=atom_density_scale,
        pair_density_scale=pair_density_scale,
        probe_density_scale=probe_density_scale,
        dedup_tolerance=dedup_tolerance,
        exact_accessibility=exact_accessibility,
        max_grid_points=max_grid_points,
        max_probe_triples=max_probe_triples,
        include_atom_weights=True,
        include_normals=need_normals,
    )
    atom_features = _dense_atom_features(
        samples.support_indices,
        samples.support_mask,
        num_atoms=coords.shape[0],
        dtype=samples.points.dtype,
    )
    support_counts = atom_features.sum(dim=1).to(dtype=torch.long)
    unique_counts, count_counts = support_counts.detach().cpu().unique(return_counts=True)
    support_count_counts = {
        int(support_count): int(count)
        for support_count, count in zip(unique_counts, count_counts)
    }

    adjacency = None
    if include_edges:
        adjacency = build_surface_adjacency(
            samples.points,
            samples.normals,
            support_indices=samples.support_indices,
            support_mask=samples.support_mask,
            block_types=samples.block_types,
            block_indices=samples.block_indices,
            weight_mode=adjacency_weight,
            neighbors=adjacency_neighbors,
            candidate_neighbors=adjacency_candidate_neighbors,
            prune_redundant_edges=adjacency_prune_redundant,
        )

    result = {
        "ses_points": samples.points.detach().cpu().numpy(),
        "block_types": samples.block_types.detach().cpu().numpy(),
        "block_indices": samples.block_indices.detach().cpu().numpy(),
        "atom_features": atom_features.detach().cpu().numpy(),
        "atom_weights": samples.atom_weights.detach().cpu().numpy(),
        "support_indices": samples.support_indices.detach().cpu().numpy(),
        "support_mask": samples.support_mask.detach().cpu().numpy(),
        "support_weights": samples.support_weights.detach().cpu().numpy(),
        "support_counts": support_counts.detach().cpu().numpy(),
        "support_count_counts": support_count_counts,
        "device": str(device),
    }
    return _with_optional_normals_and_edges(
        result,
        samples.points,
        samples.normals,
        include_normals=include_normals,
        include_edges=include_edges,
        adjacency=adjacency,
    )


def add_sphere(view, center, radius, color, opacity=1.0):
    view.addSphere(
        {
            "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
            "radius": float(radius),
            "color": color,
            "opacity": float(opacity),
        }
    )


def add_cylinder(view, start, end, radius, color, opacity=1.0):
    view.addCylinder(
        {
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
            "radius": float(radius),
            "color": color,
            "opacity": float(opacity),
        }
    )


def add_arrow(
    view,
    start,
    end,
    radius,
    color,
    opacity=1.0,
    tip_start=DEFAULT_SURFACE_NORMAL_TIP_START,
    tip_radius_ratio=DEFAULT_SURFACE_NORMAL_TIP_RADIUS_RATIO,
):
    view.addArrow(
        {
            "start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
            "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
            "radius": float(radius),
            "color": color,
            "opacity": float(opacity),
            "mid": float(tip_start),
            "radiusRatio": float(tip_radius_ratio),
        }
    )


def add_bonds(view, molecule):
    coords = molecule["coords"]
    elements = molecule["elements"]
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            max_dist = 1.25 * (COVALENT_RADII[elements[i]] + COVALENT_RADII[elements[j]])
            if np.linalg.norm(coords[i] - coords[j]) <= max_dist:
                add_cylinder(view, coords[i], coords[j], radius=0.045, color="#777777", opacity=0.85)


def add_surface_edges(
    view,
    points,
    edge_indices,
    *,
    radius=DEFAULT_SURFACE_EDGE_RADIUS,
    color=DEFAULT_SURFACE_EDGE_COLOR,
    opacity=DEFAULT_SURFACE_EDGE_OPACITY,
    max_edges=None,
):
    if radius <= 0:
        return
    points = np.asarray(points)
    edge_indices = np.asarray(edge_indices, dtype=np.int64)
    if points.size == 0 or edge_indices.size == 0:
        return
    rows = _limited_rows(edge_indices.shape[0], max_edges)
    for first, second in edge_indices[rows]:
        if first < 0 or second < 0 or first >= points.shape[0] or second >= points.shape[0]:
            continue
        start = points[first]
        end = points[second]
        if np.isfinite(start).all() and np.isfinite(end).all():
            add_cylinder(view, start, end, radius, color, opacity=opacity)


def add_surface_normals(
    view,
    points,
    normals,
    *,
    length=DEFAULT_SURFACE_NORMAL_LENGTH,
    radius=DEFAULT_SURFACE_NORMAL_RADIUS,
    color=DEFAULT_SURFACE_NORMAL_COLOR,
    opacity=DEFAULT_SURFACE_NORMAL_OPACITY,
    max_normals=None,
    start_offset=DEFAULT_SURFACE_NORMAL_START_OFFSET,
):
    if length <= 0 or radius <= 0:
        return
    points = np.asarray(points)
    normals = np.asarray(normals)
    if points.size == 0 or normals.size == 0:
        return
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (normal_lengths > 0)
    )
    valid_rows = np.nonzero(valid)[0]
    rows = valid_rows[_limited_rows(valid_rows.shape[0], max_normals)]
    for row in rows:
        normal = normals[row] / normal_lengths[row]
        start = points[row] + min(float(start_offset), 0.8 * float(length)) * normal
        end = points[row] + float(length) * normal
        add_arrow(
            view,
            start,
            end,
            radius,
            color,
            opacity=opacity,
        )


def render_py3dmol_points(
    molecule_name="Methionine",
    m=120,
    probe_radius=1.4,
    atom_point_radius=0.035,
    ses_point_radius=0.045,
    projection_line_radius=0.012,
    show_unfiltered_atom_points=False,
    show_edges=False,
    show_normals=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
    edge_radius=DEFAULT_SURFACE_EDGE_RADIUS,
    edge_color=DEFAULT_SURFACE_EDGE_COLOR,
    edge_opacity=DEFAULT_SURFACE_EDGE_OPACITY,
    max_edges=None,
    normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
    normal_radius=DEFAULT_SURFACE_NORMAL_RADIUS,
    normal_color=DEFAULT_SURFACE_NORMAL_COLOR,
    normal_opacity=DEFAULT_SURFACE_NORMAL_OPACITY,
    max_normals=None,
    width=760,
    height=580,
):
    molecule = get_molecule(molecule_name)
    sampled = sample_display_points(
        molecule,
        m=m,
        probe_radius=probe_radius,
        include_normals=show_normals,
        include_edges=show_edges,
        adjacency_weight=adjacency_weight,
        adjacency_neighbors=adjacency_neighbors,
        adjacency_candidate_neighbors=adjacency_candidate_neighbors,
        adjacency_prune_redundant=adjacency_prune_redundant,
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.46)

    atom_points_key = "raw_atom_points" if show_unfiltered_atom_points else "atom_points"
    owner_indices_key = "raw_owner_indices" if show_unfiltered_atom_points else "owner_indices"
    atom_point_opacity = 0.36 if show_unfiltered_atom_points else 0.58
    for point, atom_index in zip(sampled[atom_points_key], sampled[owner_indices_key]):
        element = molecule["elements"][atom_index]
        point_color = ATOM_COLORS[element]
        add_sphere(view, point, atom_point_radius, point_color, opacity=atom_point_opacity)

    for atom_point, ses_point, atom_index in zip(
        sampled["atom_points"],
        sampled["ses_points"],
        sampled["owner_indices"],
    ):
        element = molecule["elements"][atom_index]
        point_color = ATOM_COLORS[element]
        add_cylinder(view, atom_point, ses_point, projection_line_radius, point_color, opacity=0.42)

    if show_edges:
        add_surface_edges(
            view,
            sampled["ses_points"],
            sampled["edge_indices"],
            radius=edge_radius,
            color=edge_color,
            opacity=edge_opacity,
            max_edges=max_edges,
        )

    for point, atom_index in zip(sampled["ses_points"], sampled["owner_indices"]):
        element = molecule["elements"][atom_index]
        add_sphere(view, point, ses_point_radius, ATOM_COLORS[element], opacity=0.74)

    if show_normals:
        add_surface_normals(
            view,
            sampled["ses_points"],
            sampled["normals"],
            length=normal_length,
            radius=normal_radius,
            color=normal_color,
            opacity=normal_opacity,
            max_normals=max_normals,
        )

    view.addLabel(
        f"{molecule_name}: {len(sampled['ses_points'])} valid samples ({sampled['valid_fraction']:.0%})",
        {
            "position": {
                "x": float(molecule["coords"][:, 0].min() - 1.4),
                "y": float(molecule["coords"][:, 1].min() - 1.2),
                "z": float(molecule["coords"][:, 2].max() + 1.2),
            },
            "fontSize": 11,
            "fontColor": "#333333",
            "backgroundOpacity": 0,
        },
    )

    view.zoomTo()
    view.show()
    return view


def render_py3dmol_analytic_ses_points(
    molecule_name="Methionine",
    point_area=0.03,
    probe_radius=1.4,
    probe_density_scale=1.0,
    ses_point_radius=0.045,
    color_by="block_type",
    show_edges=False,
    show_normals=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
    edge_radius=DEFAULT_SURFACE_EDGE_RADIUS,
    edge_color=DEFAULT_SURFACE_EDGE_COLOR,
    edge_opacity=DEFAULT_SURFACE_EDGE_OPACITY,
    max_edges=None,
    normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
    normal_radius=DEFAULT_SURFACE_NORMAL_RADIUS,
    normal_color=DEFAULT_SURFACE_NORMAL_COLOR,
    normal_opacity=DEFAULT_SURFACE_NORMAL_OPACITY,
    max_normals=None,
    width=760,
    height=580,
    device=None,
    atom_filter_samples=256,
    pair_filter_samples=24,
    max_probe_triples=5_000_000,
    max_grid_points=50_000_000,
):
    """Render points generated by the analytic SES block sampler."""

    molecule = get_molecule(molecule_name)
    sampled = sample_analytic_display_points(
        molecule,
        point_area=point_area,
        probe_radius=probe_radius,
        probe_density_scale=probe_density_scale,
        device=device,
        atom_filter_samples=atom_filter_samples,
        pair_filter_samples=pair_filter_samples,
        max_probe_triples=max_probe_triples,
        max_grid_points=max_grid_points,
        include_normals=show_normals,
        include_edges=show_edges,
        adjacency_weight=adjacency_weight,
        adjacency_neighbors=adjacency_neighbors,
        adjacency_candidate_neighbors=adjacency_candidate_neighbors,
        adjacency_prune_redundant=adjacency_prune_redundant,
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

    if show_edges:
        add_surface_edges(
            view,
            sampled["ses_points"],
            sampled["edge_indices"],
            radius=edge_radius,
            color=edge_color,
            opacity=edge_opacity,
            max_edges=max_edges,
        )

    for row, point in enumerate(sampled["ses_points"]):
        block_type = int(sampled["block_types"][row])
        color = ANALYTIC_BLOCK_COLORS.get(block_type, "#555555")
        if color_by == "dominant_atom":
            support_mask = sampled["support_mask"][row]
            if support_mask.any():
                support_indices = sampled["support_indices"][row][support_mask]
                support_weights = sampled["support_weights"][row][support_mask]
                atom_index = int(support_indices[int(np.argmax(support_weights))])
                color = ATOM_COLORS[molecule["elements"][atom_index]]
        add_sphere(view, point, ses_point_radius, color, opacity=0.76)

    if show_normals:
        add_surface_normals(
            view,
            sampled["ses_points"],
            sampled["normals"],
            length=normal_length,
            radius=normal_radius,
            color=normal_color,
            opacity=normal_opacity,
            max_normals=max_normals,
        )

    counts = ", ".join(
        f"{name}: {count}" for name, count in sampled["block_type_counts"].items()
    )
    view.addLabel(
        (
            f"{molecule_name}: {len(sampled['ses_points'])} analytic SES samples"
            f" ({counts})"
        ),
        {
            "position": {
                "x": float(molecule["coords"][:, 0].min() - 1.4),
                "y": float(molecule["coords"][:, 1].min() - 1.2),
                "z": float(molecule["coords"][:, 2].max() + 1.2),
            },
            "fontSize": 11,
            "fontColor": "#333333",
            "backgroundOpacity": 0,
        },
    )

    view.zoomTo()
    view.show()
    return view


def render_py3dmol_sdf_ses_points(
    molecule_name="Methionine",
    m=128,
    probe_radius=1.4,
    smoothness=0.3,
    iterations=6,
    level_tolerance=0.05,
    subsample_spacing=None,
    feature_threshold=0.1,
    ses_point_radius=0.045,
    color_by="support_count",
    show_edges=False,
    show_normals=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
    edge_radius=DEFAULT_SURFACE_EDGE_RADIUS,
    edge_color=DEFAULT_SURFACE_EDGE_COLOR,
    edge_opacity=DEFAULT_SURFACE_EDGE_OPACITY,
    max_edges=None,
    normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
    normal_radius=DEFAULT_SURFACE_NORMAL_RADIUS,
    normal_color=DEFAULT_SURFACE_NORMAL_COLOR,
    normal_opacity=DEFAULT_SURFACE_NORMAL_OPACITY,
    max_normals=None,
    width=760,
    height=580,
    device=None,
    max_grid_points=2_000_000,
):
    """Render points generated by the SDF level-set SES sampler."""

    molecule = get_molecule(molecule_name)
    sampled = sample_sdf_display_points(
        molecule,
        m=m,
        probe_radius=probe_radius,
        smoothness=smoothness,
        iterations=iterations,
        level_tolerance=level_tolerance,
        subsample_spacing=subsample_spacing,
        feature_threshold=feature_threshold,
        device=device,
        max_grid_points=max_grid_points,
        include_normals=show_normals,
        include_edges=show_edges,
        adjacency_weight=adjacency_weight,
        adjacency_neighbors=adjacency_neighbors,
        adjacency_candidate_neighbors=adjacency_candidate_neighbors,
        adjacency_prune_redundant=adjacency_prune_redundant,
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

    if show_edges:
        add_surface_edges(
            view,
            sampled["ses_points"],
            sampled["edge_indices"],
            radius=edge_radius,
            color=edge_color,
            opacity=edge_opacity,
            max_edges=max_edges,
        )

    for row, point in enumerate(sampled["ses_points"]):
        support_count = int(sampled["support_counts"][row])
        color = SDF_SUPPORT_COLORS.get(min(support_count, 3), "#555555")
        if color_by == "dominant_atom":
            active_atoms = np.nonzero(sampled["atom_features"][row])[0]
            if active_atoms.size > 0:
                atom_index = int(active_atoms[0])
                color = ATOM_COLORS[molecule["elements"][atom_index]]
        add_sphere(view, point, ses_point_radius, color, opacity=0.76)

    if show_normals:
        add_surface_normals(
            view,
            sampled["ses_points"],
            sampled["normals"],
            length=normal_length,
            radius=normal_radius,
            color=normal_color,
            opacity=normal_opacity,
            max_normals=max_normals,
        )

    counts = ", ".join(
        f"support {support_count}: {count}"
        for support_count, count in sampled["support_count_counts"].items()
    )
    view.addLabel(
        (
            f"{molecule_name}: {len(sampled['ses_points'])} SDF SES samples"
            f" ({counts})"
        ),
        {
            "position": {
                "x": float(molecule["coords"][:, 0].min() - 1.4),
                "y": float(molecule["coords"][:, 1].min() - 1.2),
                "z": float(molecule["coords"][:, 2].max() + 1.2),
            },
            "fontSize": 11,
            "fontColor": "#333333",
            "backgroundOpacity": 0,
        },
    )

    view.zoomTo()
    view.show()
    return view


def render_py3dmol_tiled_analytic_ses_points(
    molecule_name="Methionine",
    point_area=0.5,
    probe_radius=1.4,
    tile_size="auto",
    tile_overlap="auto",
    atom_density_scale=1.55,
    pair_density_scale=1.55,
    probe_density_scale=1.55,
    dedup_tolerance=0.05,
    exact_accessibility=False,
    ses_point_radius=0.045,
    color_by="dominant_atom",
    show_edges=False,
    show_normals=False,
    adjacency_weight="euclidean",
    adjacency_neighbors=6,
    adjacency_candidate_neighbors=None,
    adjacency_prune_redundant=False,
    edge_radius=DEFAULT_SURFACE_EDGE_RADIUS,
    edge_color=DEFAULT_SURFACE_EDGE_COLOR,
    edge_opacity=DEFAULT_SURFACE_EDGE_OPACITY,
    max_edges=None,
    normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
    normal_radius=DEFAULT_SURFACE_NORMAL_RADIUS,
    normal_color=DEFAULT_SURFACE_NORMAL_COLOR,
    normal_opacity=DEFAULT_SURFACE_NORMAL_OPACITY,
    max_normals=None,
    width=760,
    height=580,
    device=None,
    max_grid_points=500_000,
    max_probe_triples=5_000_000,
):
    """Render points generated by the tiled analytic SES sampler."""

    molecule = get_molecule(molecule_name)
    sampled = sample_tiled_analytic_display_points(
        molecule,
        point_area=point_area,
        probe_radius=probe_radius,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        atom_density_scale=atom_density_scale,
        pair_density_scale=pair_density_scale,
        probe_density_scale=probe_density_scale,
        dedup_tolerance=dedup_tolerance,
        exact_accessibility=exact_accessibility,
        device=device,
        max_grid_points=max_grid_points,
        max_probe_triples=max_probe_triples,
        include_normals=show_normals,
        include_edges=show_edges,
        adjacency_weight=adjacency_weight,
        adjacency_neighbors=adjacency_neighbors,
        adjacency_candidate_neighbors=adjacency_candidate_neighbors,
        adjacency_prune_redundant=adjacency_prune_redundant,
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

    if show_edges:
        add_surface_edges(
            view,
            sampled["ses_points"],
            sampled["edge_indices"],
            radius=edge_radius,
            color=edge_color,
            opacity=edge_opacity,
            max_edges=max_edges,
        )

    for row, point in enumerate(sampled["ses_points"]):
        support_count = int(sampled["support_counts"][row])
        color = SDF_SUPPORT_COLORS.get(min(support_count, 3), "#555555")
        if color_by == "block_type":
            block_type = int(sampled["block_types"][row])
            color = ANALYTIC_BLOCK_COLORS.get(block_type, "#555555")
        if color_by == "dominant_atom":
            support_mask = sampled["support_mask"][row]
            if support_mask.any():
                support_indices = sampled["support_indices"][row][support_mask]
                support_weights = sampled["support_weights"][row][support_mask]
                atom_index = int(support_indices[int(np.argmax(support_weights))])
                color = ATOM_COLORS[molecule["elements"][atom_index]]
        add_sphere(view, point, ses_point_radius, color, opacity=0.76)

    if show_normals:
        add_surface_normals(
            view,
            sampled["ses_points"],
            sampled["normals"],
            length=normal_length,
            radius=normal_radius,
            color=normal_color,
            opacity=normal_opacity,
            max_normals=max_normals,
        )

    counts = ", ".join(
        f"support {support_count}: {count}"
        for support_count, count in sampled["support_count_counts"].items()
    )
    view.addLabel(
        (
            f"{molecule_name}: {len(sampled['ses_points'])} tiled analytic SES samples"
            f" ({counts})"
        ),
        {
            "position": {
                "x": float(molecule["coords"][:, 0].min() - 1.4),
                "y": float(molecule["coords"][:, 1].min() - 1.2),
                "z": float(molecule["coords"][:, 2].max() + 1.2),
            },
            "fontSize": 11,
            "fontColor": "#333333",
            "backgroundOpacity": 0,
        },
    )

    view.zoomTo()
    view.show()
    return view
