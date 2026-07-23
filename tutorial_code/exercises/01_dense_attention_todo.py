"""Lab 1 exercise: replace each TODO after first reading the reference."""

from __future__ import annotations

import math
import torch


def causal_mask(query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
    # TODO 1: build a bool [T_q, T_k] mask.  Hint: decode has T_q=1, T_k>1.
    raise NotImplementedError("TODO 1: implement causal_mask")


def dense_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # TODO 2: form scores [B,H,T_q,T_k] with the correct scale.
    # TODO 3: apply the mask, softmax over keys, then multiply by V.
    raise NotImplementedError("TODO 2/3: implement dense causal attention")
