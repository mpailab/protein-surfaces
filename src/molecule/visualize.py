"""py3Dmol helpers for SES molecule demos."""

from __future__ import annotations

import numpy as np
import py3Dmol
import torch

from molecule.examples import ATOM_COLORS, COVALENT_RADII, get_molecule
from ses import sample_atom_sphere_points, sample_ses_points


def sample_display_points(molecule, m, probe_radius):
    coords = torch.as_tensor(molecule["coords"], dtype=torch.float64)
    radii = torch.as_tensor(molecule["radii"], dtype=torch.float64)

    atom_points = sample_atom_sphere_points(coords, radii, m)
    ses_points, valid_mask = sample_ses_points(coords, radii, m, probe_radius)
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
