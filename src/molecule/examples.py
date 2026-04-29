"""Small molecule examples used by the SES notebook demos."""

from __future__ import annotations

import numpy as np


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
}

ATOM_COLORS = {
    "H": "#8ecae6",
    "C": "#3d405b",
    "N": "#3a86ff",
    "O": "#e63946",
    "S": "#f2c94c",
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
}


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def _with_radii(molecule):
    molecule["radii"] = np.array([VDW_RADII[element] for element in molecule["elements"]])
    return molecule


def make_methionine():
    elements = [
        "N", "C", "C", "O", "O", "C", "C", "S", "C",
        "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H",
    ]
    coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.450, 0.000, 0.000],
            [2.150, 1.250, 0.000],
            [1.640, 2.330, 0.000],
            [3.370, 1.220, 0.000],
            [2.250, -1.200, 0.000],
            [3.780, -1.080, 0.000],
            [4.460, -2.720, 0.000],
            [6.180, -2.460, 0.000],
            [-0.340, 0.540, 0.810],
            [-0.340, 0.540, -0.810],
            [1.700, 0.020, 1.040],
            [2.010, -2.160, 0.440],
            [2.010, -1.260, -1.030],
            [4.160, -0.500, 0.840],
            [4.160, -0.500, -0.840],
            [6.520, -1.430, 0.000],
            [6.520, -2.960, 0.880],
            [6.520, -2.960, -0.880],
            [3.790, 1.970, 0.000],
        ],
        dtype=float,
    )
    return {"name": "Methionine", "elements": elements, "coords": coords}


def make_hexane():
    carbon_coords = np.array(
        [
            [0.000, 0.000, 0.000],
            [1.540, 0.000, 0.000],
            [2.310, 1.334, 0.000],
            [3.850, 1.334, 0.000],
            [4.620, 0.000, 0.000],
            [6.160, 0.000, 0.000],
        ],
        dtype=float,
    )
    carbon_hydrogen_bond = 1.09
    z_axis = np.array([0.0, 0.0, 1.0])

    elements = ["C"] * len(carbon_coords)
    coords = [coord.copy() for coord in carbon_coords]

    for atom_index, carbon in enumerate(carbon_coords):
        neighbor_dirs = []
        if atom_index > 0:
            neighbor_dirs.append(_unit(carbon_coords[atom_index - 1] - carbon))
        if atom_index + 1 < len(carbon_coords):
            neighbor_dirs.append(_unit(carbon_coords[atom_index + 1] - carbon))

        if len(neighbor_dirs) == 1:
            bond_dir = neighbor_dirs[0]
            side_axis = np.cross(bond_dir, z_axis)
            if np.linalg.norm(side_axis) < 1e-12:
                side_axis = np.array([0.0, 1.0, 0.0])
            side_axis = _unit(side_axis)
            top_axis = _unit(np.cross(side_axis, bond_dir))
            hydrogen_dirs = [
                _unit(
                    (-1.0 / 3.0) * bond_dir
                    + np.sqrt(8.0 / 9.0)
                    * (np.cos(angle) * side_axis + np.sin(angle) * top_axis)
                )
                for angle in np.linspace(0, 2 * np.pi, 3, endpoint=False)
            ]
        else:
            open_side = -(neighbor_dirs[0] + neighbor_dirs[1])
            if np.linalg.norm(open_side) < 1e-12:
                open_side = np.cross(neighbor_dirs[0], z_axis)
            open_side = _unit(open_side)
            hydrogen_dirs = [
                _unit(0.55 * open_side + 0.83 * z_axis),
                _unit(0.55 * open_side - 0.83 * z_axis),
            ]

        for direction in hydrogen_dirs:
            elements.append("H")
            coords.append(carbon + carbon_hydrogen_bond * direction)

    return {"name": "C6H14", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_benzene():
    carbon_radius = 1.39
    hydrogen_offset = 1.09
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    carbon_coords = np.column_stack(
        [carbon_radius * np.cos(angles), carbon_radius * np.sin(angles), np.zeros_like(angles)]
    )
    hydrogen_coords = np.column_stack(
        [
            (carbon_radius + hydrogen_offset) * np.cos(angles),
            (carbon_radius + hydrogen_offset) * np.sin(angles),
            np.zeros_like(angles),
        ]
    )
    return {
        "name": "Benzene",
        "elements": ["C"] * 6 + ["H"] * 6,
        "coords": np.vstack([carbon_coords, hydrogen_coords]),
    }


def make_naphthalene():
    bond = 1.40
    half_width = np.sqrt(3.0) * bond / 2.0
    base_ring = np.array(
        [
            [half_width, 0.5 * bond, 0.0],
            [0.0, bond, 0.0],
            [-half_width, 0.5 * bond, 0.0],
            [-half_width, -0.5 * bond, 0.0],
            [0.0, -bond, 0.0],
            [half_width, -0.5 * bond, 0.0],
        ],
        dtype=float,
    )
    right_center = np.array([2.0 * half_width, 0.0, 0.0])
    right_ring = base_ring + right_center
    carbon_coords = np.vstack([base_ring, right_ring[[0, 1, 4, 5]]])
    hydrogen_specs = [
        (base_ring[1], np.array([0.0, 0.0, 0.0])),
        (base_ring[2], np.array([0.0, 0.0, 0.0])),
        (base_ring[3], np.array([0.0, 0.0, 0.0])),
        (base_ring[4], np.array([0.0, 0.0, 0.0])),
        (right_ring[0], right_center),
        (right_ring[1], right_center),
        (right_ring[4], right_center),
        (right_ring[5], right_center),
    ]
    hydrogen_coords = []
    for carbon, center in hydrogen_specs:
        direction = _unit(carbon - center)
        hydrogen_coords.append(carbon + 1.09 * direction)

    return {
        "name": "Naphthalene",
        "elements": ["C"] * 10 + ["H"] * 8,
        "coords": np.vstack([carbon_coords, np.array(hydrogen_coords, dtype=float)]),
    }


def make_cyclohexane_chair():
    ring_radius = 1.34
    z_amp = 0.38
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    carbon_coords = np.column_stack(
        [ring_radius * np.cos(angles), ring_radius * np.sin(angles), z_amp * (-1.0) ** np.arange(6)]
    )

    hydrogen_coords = []
    for carbon in carbon_coords:
        radial = _unit(np.array([carbon[0], carbon[1], 0.0]))
        z_sign = 1.0 if carbon[2] > 0 else -1.0
        axial = np.array([0.0, 0.0, z_sign])
        equatorial = _unit(radial + np.array([0.0, 0.0, -0.25 * z_sign]))
        hydrogen_coords.extend([carbon + 1.09 * axial, carbon + 1.09 * equatorial])

    return {
        "name": "Cyclohexane chair",
        "elements": ["C"] * 6 + ["H"] * 12,
        "coords": np.vstack([carbon_coords, np.array(hydrogen_coords, dtype=float)]),
    }


def make_glucose_chair():
    ring_coords = np.array(
        [
            [1.25, 0.00, 0.35],
            [0.62, 1.08, -0.35],
            [-0.62, 1.08, 0.35],
            [-1.25, 0.00, -0.35],
            [-0.62, -1.08, 0.35],
            [0.62, -1.08, -0.35],
        ],
        dtype=float,
    )
    elements = ["O", "C", "C", "C", "C", "C"]
    coords = [coord.copy() for coord in ring_coords]
    carbon_h_coords = []
    hydroxyl_oxygen_coords = []
    hydroxyl_h_coords = []
    for ring_index in range(1, 5):
        carbon = ring_coords[ring_index]
        radial = _unit(np.array([carbon[0], carbon[1], 0.0]))
        z_sign = 1.0 if ring_index % 2 else -1.0
        oxygen_dir = _unit(radial + np.array([0.0, 0.0, 0.35 * z_sign]))
        hydrogen_dir = _unit(-0.45 * radial + np.array([0.0, 0.0, 0.9 * z_sign]))
        oxygen = carbon + 1.43 * oxygen_dir
        hydroxyl_oxygen_coords.append(oxygen)
        hydroxyl_h_coords.append(oxygen + 0.96 * oxygen_dir)
        carbon_h_coords.append(carbon + 1.09 * hydrogen_dir)

    c5 = ring_coords[5]
    c5_radial = _unit(np.array([c5[0], c5[1], 0.0]))
    c6_dir = _unit(c5_radial + np.array([0.0, 0.0, -0.15]))
    c6 = c5 + 1.52 * c6_dir
    o6_dir = _unit(c5_radial + np.array([0.0, 0.0, 0.75]))
    o6 = c6 + 1.43 * o6_dir
    c6_h_dir_a = _unit(np.array([-c5_radial[1], c5_radial[0], 0.45]))
    c6_h_dir_b = _unit(np.array([c5_radial[1], -c5_radial[0], -0.45]))
    c5_h_dir = _unit(-0.45 * c5_radial + np.array([0.0, 0.0, -0.9]))

    elements.extend(["O"] * 5)
    coords.extend(hydroxyl_oxygen_coords + [o6])
    elements.extend(["C"])
    coords.append(c6)
    elements.extend(["H"] * 12)
    coords.extend(
        carbon_h_coords
        + [c5 + 1.09 * c5_h_dir, c6 + 1.09 * c6_h_dir_a, c6 + 1.09 * c6_h_dir_b]
        + hydroxyl_h_coords
        + [o6 + 0.96 * o6_dir]
    )
    return {"name": "Glucose chair", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_crown_ether_cavity():
    ring_size = 18
    ring_radius = 4.08
    angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    elements = []
    coords = []

    for index, angle in enumerate(angles):
        elements.append("O" if index % 3 == 0 else "C")
        coords.append(
            np.array(
                [
                    ring_radius * np.cos(angle),
                    ring_radius * np.sin(angle),
                    0.28 * np.sin(3 * angle),
                ],
                dtype=float,
            )
        )

    heavy_elements = list(elements)
    heavy_coords = [coord.copy() for coord in coords]
    for index, (element, coord) in enumerate(zip(heavy_elements, heavy_coords)):
        if element != "C":
            continue
        angle = angles[index]
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        z_sign = 1.0 if index % 2 == 0 else -1.0
        hydrogen_dirs = [
            _unit(0.85 * radial + 0.25 * tangent + np.array([0.0, 0.0, 0.45 * z_sign])),
            _unit(0.85 * radial - 0.25 * tangent + np.array([0.0, 0.0, -0.45 * z_sign])),
        ]
        for direction in hydrogen_dirs:
            elements.append("H")
            coords.append(coord + 1.09 * direction)

    return {"name": "18-Crown-6 cavity", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_porphyrin_pore():
    ring_size = 20
    ring_radius = 4.35
    angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    elements = []
    coords = []

    for index, angle in enumerate(angles):
        elements.append("N" if index % 5 == 0 else "C")
        coords.append(
            np.array(
                [
                    ring_radius * np.cos(angle),
                    ring_radius * np.sin(angle),
                    0.08 * np.sin(4 * angle),
                ],
                dtype=float,
            )
        )

    for element, coord, angle in zip(list(elements), list(coords), angles):
        if element != "C":
            continue
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        elements.append("H")
        coords.append(coord + 1.09 * radial)

    return {"name": "Porphyrin pore", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_calixarene_bowl():
    elements = []
    coords = []
    panel_radius = 2.65
    benzene_radius = 1.39
    z_axis = np.array([0.0, 0.0, 1.0])

    for panel_index in range(4):
        phi = panel_index * 0.5 * np.pi
        radial = np.array([np.cos(phi), np.sin(phi), 0.0])
        tangent = np.array([-np.sin(phi), np.cos(phi), 0.0])
        tilted_vertical = _unit(z_axis + 0.45 * radial)
        center = panel_radius * radial + np.array([0.0, 0.0, 0.25])
        ring_coords = []

        for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            coord = center + benzene_radius * (
                np.cos(angle) * tangent + np.sin(angle) * tilted_vertical
            )
            elements.append("C")
            coords.append(coord)
            ring_coords.append(coord)

        for coord in ring_coords:
            elements.append("H")
            coords.append(coord + 1.09 * _unit(radial + 0.12 * coord[2] * z_axis))

    for bridge_index in range(4):
        phi = (bridge_index + 0.5) * 0.5 * np.pi
        radial = np.array([np.cos(phi), np.sin(phi), 0.0])
        tangent = np.array([-np.sin(phi), np.cos(phi), 0.0])
        bridge = 1.55 * radial + np.array([0.0, 0.0, -1.35])
        elements.append("C")
        coords.append(bridge)
        for direction in [
            _unit(-0.35 * radial + 0.55 * tangent - 0.75 * z_axis),
            _unit(-0.35 * radial - 0.55 * tangent - 0.75 * z_axis),
        ]:
            elements.append("H")
            coords.append(bridge + 1.09 * direction)

    return {"name": "Calixarene bowl", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_cryptand_cage():
    ring_size = 12
    ring_radius = 3.00
    half_height = 1.35
    angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    elements = []
    coords = []

    for z in [-half_height, half_height]:
        z_sign = 1.0 if z > 0 else -1.0
        for index, angle in enumerate(angles):
            element = "O" if index % 4 == 0 else "C"
            coord = np.array([ring_radius * np.cos(angle), ring_radius * np.sin(angle), z], dtype=float)
            elements.append(element)
            coords.append(coord)
            if element == "C":
                radial = np.array([np.cos(angle), np.sin(angle), 0.0])
                elements.append("H")
                coords.append(coord + 1.09 * _unit(radial + 0.25 * z_sign * np.array([0.0, 0.0, 1.0])))

    for index in [2, 6, 10]:
        angle = angles[index]
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        bridge = ring_radius * radial
        elements.append("C")
        coords.append(bridge)
        for direction in [_unit(radial + 0.55 * tangent), _unit(radial - 0.55 * tangent)]:
            elements.append("H")
            coords.append(bridge + 1.09 * direction)

    return {"name": "Cryptand cage", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_octahedral_probe_cage():
    coords = np.array(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ],
        dtype=float,
    )
    return {"name": "Octahedral probe cage", "elements": ["C"] * 6, "coords": coords}


def make_closed_carbon_shell():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = []
    for a in [-1.0, 1.0]:
        for b in [-1.0, 1.0]:
            vertices.extend([[0.0, a, b * phi], [a, b * phi, 0.0], [a * phi, 0.0, b]])
    shell = np.array(vertices, dtype=float)
    shell = 3.25 * shell / np.linalg.norm(shell, axis=1, keepdims=True)
    hydrogens = [coord + 1.09 * _unit(coord) for coord in shell]
    return {
        "name": "Closed carbon shell",
        "elements": ["C"] * len(shell) + ["H"] * len(hydrogens),
        "coords": np.vstack([shell, np.array(hydrogens, dtype=float)]),
    }


def make_sealed_nanotube_capsule():
    elements = []
    coords = []
    ring_size = 12
    ring_radius = 3.35
    ring_zs = [-2.15, 0.0, 2.15]
    angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    for z in ring_zs:
        for angle in angles:
            coord = np.array([ring_radius * np.cos(angle), ring_radius * np.sin(angle), z], dtype=float)
            elements.append("C")
            coords.append(coord)
            elements.append("H")
            coords.append(coord + 1.09 * _unit(np.array([coord[0], coord[1], 0.25 * np.sign(z)])))

    for z in [-3.85, 3.85]:
        elements.append("C")
        coords.append(np.array([0.0, 0.0, z], dtype=float))
        for angle in angles[::3]:
            coord = np.array([1.55 * np.cos(angle), 1.55 * np.sin(angle), z], dtype=float)
            elements.append("C")
            coords.append(coord)
            elements.append("H")
            coords.append(coord + 1.09 * _unit(np.array([np.cos(angle), np.sin(angle), np.sign(z)])))

    return {"name": "Sealed nanotube capsule", "elements": elements, "coords": np.array(coords, dtype=float)}


def make_double_chamber_cage():
    elements = []
    coords = []
    ring_size = 10
    angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
    for chamber_center_x in [-2.35, 2.35]:
        for x_offset, radius in [(-1.25, 2.65), (0.0, 3.05), (1.25, 2.65)]:
            x = chamber_center_x + x_offset
            for angle in angles:
                coord = np.array([x, radius * np.cos(angle), radius * np.sin(angle)], dtype=float)
                elements.append("C")
                coords.append(coord)
                elements.append("H")
                coords.append(coord + 1.09 * _unit(np.array([0.25 * np.sign(x), coord[1], coord[2]])))

    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        coord = np.array([0.0, 1.75 * np.cos(angle), 1.75 * np.sin(angle)], dtype=float)
        elements.append("N")
        coords.append(coord)

    return {"name": "Double chamber cage", "elements": elements, "coords": np.array(coords, dtype=float)}


tetrahedron = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    dtype=float,
)
tetrahedron = tetrahedron / np.linalg.norm(tetrahedron, axis=1, keepdims=True)

MOLECULES = [
    {
        "name": "H2O",
        "elements": ["O", "H", "H"],
        "coords": np.array(
            [[0.0000, 0.0000, 0.0000], [0.9572, 0.0000, 0.0000], [-0.2390, 0.9270, 0.0000]],
            dtype=float,
        ),
    },
    {
        "name": "CO2",
        "elements": ["O", "C", "O"],
        "coords": np.array([[-1.1600, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000], [1.1600, 0.0000, 0.0000]], dtype=float),
    },
    {
        "name": "NH3",
        "elements": ["N", "H", "H", "H"],
        "coords": np.array(
            [[0.0000, 0.0000, 0.0000], [0.9400, 0.0000, -0.3600], [-0.4700, 0.8141, -0.3600], [-0.4700, -0.8141, -0.3600]],
            dtype=float,
        ),
    },
    {
        "name": "CH4",
        "elements": ["C", "H", "H", "H", "H"],
        "coords": np.vstack([np.zeros((1, 3)), 1.09 * tetrahedron]),
    },
    {
        "name": "Ethanol",
        "elements": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        "coords": np.array(
            [
                [0.000, 0.000, 0.000],
                [1.520, 0.000, 0.000],
                [2.120, 1.210, 0.000],
                [-0.540, 0.930, 0.000],
                [-0.540, -0.465, 0.805],
                [-0.540, -0.465, -0.805],
                [1.910, -0.520, 0.890],
                [1.910, -0.520, -0.890],
                [2.950, 1.100, 0.000],
            ],
            dtype=float,
        ),
    },
    make_benzene(),
    make_naphthalene(),
    make_cyclohexane_chair(),
    make_glucose_chair(),
    make_crown_ether_cavity(),
    make_porphyrin_pore(),
    make_calixarene_bowl(),
    make_cryptand_cage(),
    make_octahedral_probe_cage(),
    make_closed_carbon_shell(),
    make_sealed_nanotube_capsule(),
    make_double_chamber_cage(),
    make_hexane(),
    make_methionine(),
]
MOLECULES = [_with_radii(molecule) for molecule in MOLECULES]
MOLECULE_NAMES = tuple(molecule["name"] for molecule in MOLECULES)


def get_molecule(name):
    for molecule in MOLECULES:
        if molecule["name"] == name:
            return molecule
    available = ", ".join(MOLECULE_NAMES)
    raise ValueError(f"Unknown molecule {name!r}. Available molecules: {available}")
