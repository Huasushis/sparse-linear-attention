"""Optional ReplaySSM exercise: keep the algebra visible before writing a kernel."""

from __future__ import annotations

import torch


def output_only_from_checkpoint(
    checkpoint: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    decay_buffer: torch.Tensor,
    delta_buffer: torch.Tensor,
    q_t: torch.Tensor,
) -> torch.Tensor:
    """Return one output without constructing a ``[B,H,Dk,Dv]`` replay state.

    Shapes are ``checkpoint=[B,H,Dk,Dv]``, ``k_buffer=[B,H,L,Dk]``,
    ``v_buffer=[B,H,L,Dv]``, ``decay/delta=[B,H,L]`` and ``q_t=[B,H,Dk]``.
    TODO: derive the suffix decay weights and use ``v_j * (k_j @ q_t)``.
    """
    raise NotImplementedError("TODO: implement the output-only reassociation")
