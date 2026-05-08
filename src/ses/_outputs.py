"""Shared return-value formatting for public SES samplers."""

from __future__ import annotations

from typing import Optional, Union

import torch


SamplerOutput = Union[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]


def _format_sample_outputs(
    points: torch.Tensor,
    *,
    atom_features: Optional[torch.Tensor] = None,
    normals: Optional[torch.Tensor] = None,
    adjacency: Optional[torch.Tensor] = None,
) -> SamplerOutput:
    """Return public sampler outputs while preserving existing tuple order."""

    outputs: list[torch.Tensor] = [points]
    if atom_features is not None:
        outputs.append(atom_features)
    if normals is not None:
        outputs.append(normals)
    if adjacency is not None:
        outputs.append(adjacency)
    if len(outputs) == 1:
        return points
    return tuple(outputs)  # type: ignore[return-value]
