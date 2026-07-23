"""Lab 2 exercise: one token update for causal kernelized linear attention."""

from __future__ import annotations

import torch


def update_state(
    state: torch.Tensor,
    normalizer: torch.Tensor,
    phi_k_t: torch.Tensor,
    v_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return S_t and z_t from S_{t-1}, z_{t-1}.

    Shapes: state=[B,H,D_k,D_v], normalizer=[B,H,D_k],
    phi_k_t=[B,H,D_k], v_t=[B,H,D_v].
    """
    # TODO: add the outer product phi_k_t ⊗ v_t and update normalizer.
    raise NotImplementedError("TODO: implement the recurrent linear-attention update")
