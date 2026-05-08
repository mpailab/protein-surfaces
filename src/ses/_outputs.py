"""Shared return-value formatting for public SES samplers."""

from __future__ import annotations

from typing import Optional, Union

import torch


SamplerOutput = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def _format_sample_outputs(
    points: torch.Tensor,
    *,
    atom_features: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
) -> SamplerOutput:
    """Return public sampler outputs while preserving existing tuple order."""

    if atom_features is None:
        if normals is None:
            return points
        return points, normals
    if normals is None:
        return points, atom_features
    return points, atom_features, normals
