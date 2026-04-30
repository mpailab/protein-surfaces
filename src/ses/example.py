"""Examples for the two SES point-sampling interfaces.

The package exposes two complementary ways to scatter points on a
solvent-excluded surface:

* ``sample_projected_points`` starts from points on atom spheres and projects
  them onto the SES.
* ``sample_analytic_points`` samples explicit analytic SES blocks: atom contact
  patches, pair toroidal patches and fixed-probe reentrant patches.
"""

from __future__ import annotations

import torch

from . import sample_analytic_points, sample_projected_points


def water_atoms(
    dtype: torch.dtype = torch.float64,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Return a tiny water-like molecule used by the examples below."""

    atom_types = ("O", "H1", "H2")

    # Coordinates are stored in Angstroms.  The molecule is intentionally small
    # so examples run quickly on CPU and are easy to inspect in a debugger.
    atom_coords = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=dtype,
    )

    # These are simple van der Waals radii for oxygen and hydrogens.  The
    # solvent probe radius is passed separately to the sampling functions.
    atom_radii = torch.tensor([1.52, 1.20, 1.20], dtype=dtype)
    return atom_types, atom_coords, atom_radii


def projection_example() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample SES points with projection and return atom-binding features."""

    atom_types, atom_coords, atom_radii = water_atoms()

    # The public projection interface returns a flat SES point cloud.  Passing
    # ``include_atom_features=True`` asks it to also return a dense feature
    # matrix with one column per input atom.
    points, atom_features = sample_projected_points(
        atom_coords,
        atom_radii,
        m=128,
        probe_radius=1.4,
        include_atom_features=True,
    )

    # ``atom_features[row, col]`` is one when ``points[row]`` is tied to
    # ``atom_types[col]``.  Projection starts from one owner atom, so each row is
    # one-hot: exactly one atom column is active.
    assert atom_features.shape == (points.shape[0], len(atom_types))
    assert torch.all(atom_features.sum(dim=1) == 1)
    return points, atom_features


def analytic_example() -> tuple[torch.Tensor, torch.Tensor]:
    """Sample SES points analytically and return atom-binding features."""

    atom_types, atom_coords, atom_radii = water_atoms()

    # The analytic interface returns the same public shape convention as the
    # projection interface: flat coordinates, and optionally dense atom-binding
    # features.  ``point_area`` controls density directly: smaller values create
    # more points.
    points, atom_features = sample_analytic_points(
        atom_coords,
        atom_radii,
        probe_radius=1.4,
        point_area=0.25,
        include_atom_features=True,
        max_grid_points=200_000,
    )

    # Analytic SES points may be supported by more than one atom.  A contact
    # patch row has one active column, a pair-torus row has two active columns,
    # and fixed-probe reentrant rows can have three or more active columns.
    assert atom_features.shape == (points.shape[0], len(atom_types))
    assert torch.all(atom_features.sum(dim=1) >= 1)
    return points, atom_features


def first_bindings(
    atom_features: torch.Tensor,
    atom_types: tuple[str, ...],
    count: int = 5,
) -> list[tuple[str, ...]]:
    """Decode the first feature rows into readable atom-type bindings."""

    bindings: list[tuple[str, ...]] = []
    for row in atom_features[:count]:
        active_columns = row.nonzero(as_tuple=False).reshape(-1).tolist()
        bindings.append(tuple(atom_types[index] for index in active_columns))
    return bindings


if __name__ == "__main__":
    atom_types, _, _ = water_atoms()
    projected_points, projected_features = projection_example()
    analytic_points, analytic_features = analytic_example()

    print(f"Projection sampler produced {projected_points.shape[0]} points")
    print(f"Projection feature shape: {tuple(projected_features.shape)}")
    print(f"Projection first bindings: {first_bindings(projected_features, atom_types)}")
    print(f"Analytic sampler produced {analytic_points.shape[0]} points")
    print(f"Analytic feature shape: {tuple(analytic_features.shape)}")
    print(f"Analytic first bindings: {first_bindings(analytic_features, atom_types)}")
