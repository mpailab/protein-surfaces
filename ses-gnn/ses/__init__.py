"""Public interfaces for solvent-excluded surface point sampling.

The package-level API intentionally exposes only the high-level SES point
samplers.  All return a flat coordinate tensor and can also return dense
atom-assignment features with ``include_atom_features=True`` and outward surface
normals with ``include_normals=True``.  They can also return sparse SES surface
adjacency matrices with ``include_adjacency=True``.  Lower-level helpers, block
metadata and debugging utilities live in implementation modules and should be
imported from those modules directly when needed.
"""

from .analytic import sample_analytic_points
from .projection import sample_projected_points
from .sdf import sample_sdf_points
from .tiled_analytic import sample_tiled_analytic_points

__all__ = [
    "sample_analytic_points",
    "sample_projected_points",
    "sample_sdf_points",
    "sample_tiled_analytic_points",
]
