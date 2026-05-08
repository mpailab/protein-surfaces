# protein-surfaces
Machine learning solutions for protein-surfaces analysis

## SES point sampling

The `src/ses` package provides four interfaces for scattering points on a
solvent-excluded surface (SES). All four accept atom coordinates, atom radii
and a solvent probe radius. They return a flat tensor of SES point coordinates,
and can optionally return atom-binding features, outward normals and a sparse
surface adjacency matrix.

```python
import torch

from ses import (
    sample_analytic_points,
    sample_projected_points,
    sample_sdf_points,
    sample_tiled_analytic_points,
)

atom_coords = torch.tensor(
    [
        [0.0000, 0.0000, 0.0000],
        [0.9572, 0.0000, 0.0000],
        [-0.2390, 0.9270, 0.0000],
    ],
    dtype=torch.float64,
)
atom_radii = torch.tensor([1.52, 1.20, 1.20], dtype=torch.float64)
probe_radius = 1.4
```

`sample_projected_points` starts from points sampled on atom spheres and
projects visible points onto the SES. It is the simpler interface when density
is controlled by a fixed number of starting points per atom.

```python
(
    projected_points,
    projected_features,
    projected_normals,
    projected_adjacency,
) = sample_projected_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=probe_radius,
    include_atom_features=True,
    include_normals=True,
    include_adjacency=True,
)

assert projected_points.shape[1] == 3
assert projected_features.shape == (projected_points.shape[0], atom_coords.shape[0])
assert projected_normals.shape == projected_points.shape
assert projected_adjacency.shape == (projected_points.shape[0], projected_points.shape[0])
```

`sample_analytic_points` samples analytic SES blocks: atom contact patches,
pair toroidal patches and fixed-probe reentrant patches. It is the preferred
interface when density should be controlled by approximate surface area.

```python
analytic_points, analytic_features = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=probe_radius,
    point_area=0.25,
    include_atom_features=True,
)

assert analytic_points.shape[1] == 3
assert analytic_features.shape == (analytic_points.shape[0], atom_coords.shape[0])
```

`sample_sdf_points` follows the level-set surface generation idea used by
dMaSIF: seed probe-center candidates around atoms, project them onto a smooth
signed-distance level set, reject inaccessible centers and shift them back by
one probe radius to the SES.

```python
sdf_points, sdf_features = sample_sdf_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=probe_radius,
    smoothness=0.3,
    include_atom_features=True,
)

assert sdf_points.shape[1] == 3
assert sdf_features.shape == (sdf_points.shape[0], atom_coords.shape[0])
```

The feature tensor has one column per input atom. Projection features are
one-hot because every point originates from one owner atom. Analytic features
are multi-hot because pair and reentrant patches can be supported by multiple
atoms. SDF features are binary supports derived from the smooth SDF ownership
weights.

Pass `include_normals=True` to any sampler to append a `(num_points, 3)` unit
normal tensor. Pass `include_adjacency=True` to append a sparse COO matrix whose
edges connect local samples from the same SES patch or from adjacent patch
families. Edge weights are Euclidean by default; use `adjacency_weight="geodesic"`
for a local geodesic approximation. Optional outputs are ordered as
`(points, atom_features, normals, adjacency)` when all are requested.

See `src/ses/README.md` and `src/ses/example.py` for the full interface
description and larger examples.
