# SES Point Sampling

This package contains point-sampling interfaces for the solvent-excluded
surface (SES) of a molecule. The SES is the boundary traced by a spherical
solvent probe rolling over atom van der Waals spheres. The package is written
around PyTorch tensors so sampled points can be used directly in downstream
geometry, visualization and machine-learning pipelines.

The package-level public API intentionally exposes four high-level samplers:

```python
from ses import (
    sample_analytic_points,
    sample_projected_points,
    sample_sdf_points,
    sample_tiled_analytic_points,
)
```

All four interfaces share the same basic input and output convention:

- `atom_coords`: tensor with shape `(num_atoms, 3)`.
- `atom_radii`: tensor with one radius per atom.
- `probe_radius`: positive solvent probe radius.
- `points`: returned tensor with shape `(num_points, 3)`.
- `atom_features`: optional dense tensor with shape `(num_points, num_atoms)`.
- `normals`: optional outward SES normal tensor with shape `(num_points, 3)`.

All coordinates and radii are expected to use the same length unit. The examples
below use Angstroms.

## Atom-Binding Features

Pass `include_atom_features=True` when the sampled point cloud needs an explicit
binding back to source atoms.

```python
points, atom_features = sample_projected_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=1.4,
    include_atom_features=True,
)
```

`atom_features[row, col]` is active when `points[row]` is bound to
`atom_coords[col]`.

Projection features are one-hot. A projected point starts from one owner atom
sphere, so every feature row has exactly one active atom column.

Analytic features are multi-hot. An atom contact patch has one active atom, a
pair torus patch has two active atoms, and a fixed-probe reentrant patch can
have three or more active atoms.

SDF features are binary supports derived from smooth SDF ownership weights. The
strongest atom is always active, and neighboring atoms become active when their
smooth ownership weight passes the feature threshold.

```python
active_atom_indices = atom_features[row].nonzero(as_tuple=False).reshape(-1)
```

## Surface Normals

Pass `include_normals=True` when downstream code needs a unit normal for every
returned SES point.

```python
points, normals = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
    point_area=0.5,
    include_normals=True,
)
```

When both optional outputs are requested, the tuple order is
`(points, atom_features, normals)`.

## Projection Sampler

`sample_projected_points` is the simpler SES sampler:

```python
points = sample_projected_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=1.4,
)
```

With features:

```python
points, atom_features = sample_projected_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=1.4,
    include_atom_features=True,
)
```

### Idea

The projection sampler works in four stages:

1. It places `m` deterministic Fibonacci-lattice points on every atom sphere.
2. It rejects source points that are inside another atom.
3. It estimates a feasible rolling-probe center for each remaining point using
   local one-, two- or three-atom contact geometry.
4. It keeps only points whose probe centers are accessible from the exterior
   component of probe-center space.

The returned coordinates are flattened into `(num_points, 3)`. When features
are requested, the feature row records the owner atom that produced the source
sample before projection.

### When To Use It

Use the projection sampler when:

- you want a quick deterministic point cloud;
- density can be controlled as points per atom through `m`;
- one-hot ownership features are enough for the downstream task.

The sampler is intentionally direct, but point density can vary after
projection because the starting distribution is atom-sphere based rather than
area based on final SES patches.

## Analytic Sampler

`sample_analytic_points` samples explicit SES surface blocks:

```python
points = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
    point_area=0.5,
)
```

With features:

```python
points, atom_features = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
    point_area=0.5,
    include_atom_features=True,
)
```

### Idea

The analytic sampler represents the SES as three families of blocks:

1. Atom contact patches, where the SES lies on an atom sphere.
2. Pair toroidal patches, where the probe touches two atoms and its center
   travels around the intersection circle of two expanded atom spheres.
3. Fixed-probe reentrant patches, where the probe touches at least three atoms
   and the SES lies on the probe sphere itself.

The sampler first discovers which blocks are exterior. This discrete discovery
step uses expanded atom spheres with radius `atom_radius + probe_radius`, local
intersection tests and an exterior reachability check. It then samples the
selected blocks with deterministic area-aware rules. The continuous point
coordinates are rebuilt from the original atom tensors, which keeps useful
gradient paths for differentiable workflows.

### Density Controls

`point_area` is the main density parameter. Smaller values produce more points.
For example, `point_area=0.1` is denser than `point_area=1.0`.

`oversample_factor` creates extra candidate samples before exterior filtering.
Values above `1.0` can help preserve density around clipped patches.

`probe_density_scale` only affects fixed-probe reentrant patches. It is useful
when small multi-atom reentrant regions need more visual or numerical coverage.

### When To Use It

Use the analytic sampler when:

- density should follow approximate SES surface area;
- atom support should be multi-hot for pair and reentrant patches;
- you need explicit treatment of contact, toroidal and reentrant surface
  regions.

The analytic sampler usually gives more meaningful atom-support metadata than
projection, but block discovery is more involved.

## SDF Level-Set Sampler

`sample_sdf_points` samples a smooth signed-distance-function level set in
probe-center space:

```python
points = sample_sdf_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=1.4,
    smoothness=0.3,
)
```

With features:

```python
points, atom_features = sample_sdf_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=1.4,
    smoothness=0.3,
    include_atom_features=True,
)
```

### Idea

This interface follows the level-set sampling strategy used by dMaSIF, adapted
to this package's SES convention:

1. It places `m` deterministic Fibonacci-lattice candidates on every atom
   sphere expanded by `probe_radius`.
2. It iteratively projects those probe-center candidates onto the zero level
   set of a smooth minimum of signed distances to expanded atom spheres.
3. It keeps only finite, feasible centers that are accessible from the exterior
   probe-center component.
4. It shifts surviving centers inward by one `probe_radius` along the SDF
   normal to produce SES points.

`smoothness` controls how sharply the soft minimum approximates the hard union
of expanded atom spheres. Smaller values hug the analytic surface more closely;
larger values smooth crowded regions more aggressively. `subsample_spacing`
optionally applies a cubic-grid average to nearby probe centers before the
final normal shift.

### When To Use It

Use the SDF sampler when:

- you want a differentiable-geometry inspired level-set point cloud;
- density should be controlled by deterministic seeds per atom, like
  projection, with optional grid subsampling;
- approximate multi-atom support features from a smooth SDF are useful.

## Complete Example

```python
import torch

from ses import sample_analytic_points, sample_projected_points, sample_sdf_points

atom_types = ("O", "H1", "H2")
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

projected_points, projected_features = sample_projected_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=probe_radius,
    include_atom_features=True,
)

analytic_points, analytic_features = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=probe_radius,
    point_area=0.25,
    include_atom_features=True,
)

sdf_points, sdf_features = sample_sdf_points(
    atom_coords,
    atom_radii,
    m=128,
    probe_radius=probe_radius,
    smoothness=0.3,
    include_atom_features=True,
)

def decode_bindings(atom_features: torch.Tensor) -> list[tuple[str, ...]]:
    bindings = []
    for row in atom_features:
        active_columns = row.nonzero(as_tuple=False).reshape(-1).tolist()
        bindings.append(tuple(atom_types[index] for index in active_columns))
    return bindings

print(projected_points.shape)
print(projected_features.shape)
print(decode_bindings(projected_features[:5]))

print(analytic_points.shape)
print(analytic_features.shape)
print(decode_bindings(analytic_features[:5]))

print(sdf_points.shape)
print(sdf_features.shape)
print(decode_bindings(sdf_features[:5]))
```

## Reusing Analytic Blocks

For repeated analytic sampling of the same molecule, block discovery can be
done once and reused at different densities. Import this advanced interface
from `ses.analytic` rather than from package root:

```python
from ses.analytic import build_analytic_blocks
from ses import sample_analytic_points

blocks, _ = build_analytic_blocks(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
)

coarse_points = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
    point_area=1.0,
    blocks=blocks,
)

dense_points, dense_features = sample_analytic_points(
    atom_coords,
    atom_radii,
    probe_radius=1.4,
    point_area=0.2,
    blocks=blocks,
    include_atom_features=True,
)
```

`blocks` stores detached topology: atom patch indices, pair support indices and
fixed-probe support metadata. The point coordinates are still regenerated from
the provided `atom_coords` and `atom_radii`.

## Choosing An Interface

| Interface | Density parameter | Atom features | Best fit |
| --- | --- | --- | --- |
| `sample_projected_points` | `m`, points per atom | One-hot owner atom | Quick deterministic SES clouds |
| `sample_analytic_points` | `point_area`, approximate area per point | Multi-hot support atoms | Area-aware SES sampling and support metadata |
| `sample_sdf_points` | `m`, level-set seeds per atom | Binary smooth-SDF supports | Smooth level-set SES clouds |
| `sample_tiled_analytic_points` | `point_area`, approximate area per point | Multi-hot support atoms | Faster approximate analytic clouds for larger molecules |

The examples in `src/ses/example.py` are executable and show the core
interfaces with comments focused on atom-type binding.
