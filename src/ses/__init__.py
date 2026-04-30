"""Public interfaces for solvent-excluded surface point sampling.

The package-level API intentionally exposes only the two high-level SES point
samplers.  Both return a flat coordinate tensor and can also return dense
atom-assignment features with ``include_atom_features=True``.  Lower-level
helpers, block metadata and debugging utilities live in ``ses.projection`` and
``ses.analytic`` and should be imported from those modules directly when needed.
"""

from .analytic import sample_analytic_points
from .projection import sample_projected_points

__all__ = [
    "sample_analytic_points",
    "sample_projected_points",
]
