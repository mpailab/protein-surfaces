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


def sample_display_points(molecule, m, probe_radius):
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    atom_points = sample_atom_points(coords, radii, m)
    ses_points, valid_mask = _sample_projected_grid(coords, radii, m, probe_radius)
    finite_mask = torch.isfinite(atom_points).all(dim=-1) & torch.isfinite(ses_points).all(dim=-1)
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
    owner_indices = display_mask.nonzero(as_tuple=False)[:, 1].detach().cpu().numpy()
    return {
        "raw_atom_points": atom_points.reshape(-1, 3).detach().cpu().numpy(),
        "raw_owner_indices": raw_owner_indices,
        "atom_points": atom_points[display_mask].detach().cpu().numpy(),
        "ses_points": ses_points[display_mask].detach().cpu().numpy(),
        "owner_indices": owner_indices,
        "valid_fraction": display_mask.to(dtype=torch.float64).mean().item(),
    }


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
):
    """Sample analytic SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
    samples = _sample_analytic_samples(
        coords,
        radii,
        probe_radius,
        point_area=point_area,
        probe_density_scale=probe_density_scale,
        include_atom_weights=include_atom_weights,
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

    return {
        "ses_points": samples.points.detach().cpu().numpy(),
        "block_types": samples.block_types.detach().cpu().numpy(),
        "support_indices": samples.support_indices.detach().cpu().numpy(),
        "support_mask": samples.support_mask.detach().cpu().numpy(),
        "support_weights": samples.support_weights.detach().cpu().numpy(),
        "atom_weights": atom_weights,
        "block_type_counts": block_type_counts,
        "num_blocks": samples.blocks.num_blocks if samples.blocks is not None else 0,
        "device": str(device),
    }


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
):
    """Sample SDF level-set SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
    points, atom_features = sample_sdf_points(
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
        max_grid_points=max_grid_points,
    )
    support_counts = atom_features.sum(dim=1).to(dtype=torch.long)
    unique_counts, count_counts = support_counts.detach().cpu().unique(return_counts=True)
    support_count_counts = {
        int(support_count): int(count)
        for support_count, count in zip(unique_counts, count_counts)
    }

    return {
        "ses_points": points.detach().cpu().numpy(),
        "atom_features": atom_features.detach().cpu().numpy(),
        "support_counts": support_counts.detach().cpu().numpy(),
        "support_count_counts": support_count_counts,
        "device": str(device),
    }


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
):
    """Sample tiled analytic SES points and return NumPy arrays for visualization."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    coords = torch.as_tensor(molecule["coords"], dtype=dtype, device=device)
    radii = torch.as_tensor(molecule["radii"], dtype=dtype, device=device)
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

    return {
        "ses_points": samples.points.detach().cpu().numpy(),
        "atom_features": atom_features.detach().cpu().numpy(),
        "atom_weights": samples.atom_weights.detach().cpu().numpy(),
        "support_indices": samples.support_indices.detach().cpu().numpy(),
        "support_mask": samples.support_mask.detach().cpu().numpy(),
        "support_weights": samples.support_weights.detach().cpu().numpy(),
        "support_counts": support_counts.detach().cpu().numpy(),
        "support_count_counts": support_count_counts,
        "device": str(device),
    }


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


def add_bonds(view, molecule):
    coords = molecule["coords"]
    elements = molecule["elements"]
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            max_dist = 1.25 * (COVALENT_RADII[elements[i]] + COVALENT_RADII[elements[j]])
            if np.linalg.norm(coords[i] - coords[j]) <= max_dist:
                add_cylinder(view, coords[i], coords[j], radius=0.045, color="#777777", opacity=0.85)


def render_py3dmol_points(
    molecule_name="Methionine",
    m=120,
    probe_radius=1.4,
    atom_point_radius=0.035,
    ses_point_radius=0.045,
    projection_line_radius=0.012,
    show_unfiltered_atom_points=False,
    width=760,
    height=580,
):
    molecule = get_molecule(molecule_name)
    sampled = sample_display_points(molecule, m=m, probe_radius=probe_radius)

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

    for point, atom_index in zip(sampled["ses_points"], sampled["owner_indices"]):
        element = molecule["elements"][atom_index]
        add_sphere(view, point, ses_point_radius, ATOM_COLORS[element], opacity=0.74)

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
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

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
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

    for row, point in enumerate(sampled["ses_points"]):
        support_count = int(sampled["support_counts"][row])
        color = SDF_SUPPORT_COLORS.get(min(support_count, 3), "#555555")
        if color_by == "dominant_atom":
            active_atoms = np.nonzero(sampled["atom_features"][row])[0]
            if active_atoms.size > 0:
                atom_index = int(active_atoms[0])
                color = ATOM_COLORS[molecule["elements"][atom_index]]
        add_sphere(view, point, ses_point_radius, color, opacity=0.76)

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
    )

    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor("white")
    add_bonds(view, molecule)

    for element, center, radius in zip(molecule["elements"], molecule["coords"], molecule["radii"]):
        add_sphere(view, center, radius, ATOM_COLORS[element], opacity=0.34)

    for row, point in enumerate(sampled["ses_points"]):
        support_count = int(sampled["support_counts"][row])
        color = SDF_SUPPORT_COLORS.get(min(support_count, 3), "#555555")
        if color_by == "dominant_atom":
            support_mask = sampled["support_mask"][row]
            if support_mask.any():
                support_indices = sampled["support_indices"][row][support_mask]
                support_weights = sampled["support_weights"][row][support_mask]
                atom_index = int(support_indices[int(np.argmax(support_weights))])
                color = ATOM_COLORS[molecule["elements"][atom_index]]
        add_sphere(view, point, ses_point_radius, color, opacity=0.76)

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
