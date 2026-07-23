"""Correctness-first dense attention for the accompanying tutorial.

The functions in this file intentionally materialize score matrices or use Python
loops. They are an oracle for reasoning and testing, not a replacement for
PyTorch SDPA or FlashAttention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def causal_mask(query_length: int, key_length: int, *, device: torch.device) -> torch.Tensor:
    """Return an ``[T_q, T_k]`` mask suitable for both prefill and decode.

    If ``T_q < T_k`` we treat the queries as the final ``T_q`` positions.  This
    is the useful convention for decode: a single new query may see all cached
    keys, rather than only key 0.
    """
    if query_length <= 0 or key_length <= 0:
        raise ValueError("query_length and key_length must be positive")
    query_positions = torch.arange(query_length, device=device) + key_length - query_length
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def masked_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Apply a boolean visibility mask then softmax over the final dimension.

    This reference rejects rows with no visible key. Returning a uniform row
    after filling with a finite minimum would silently attend to masked values.
    """
    if allowed.dtype is not torch.bool:
        raise TypeError("allowed must be a boolean tensor: True means visible")
    try:
        visible = torch.broadcast_to(allowed, scores.shape)
    except RuntimeError as exc:
        raise ValueError("allowed must be broadcastable to scores") from exc
    if not bool(visible.any(dim=-1).all()):
        raise ValueError("every query row must be allowed to attend to at least one key")
    return torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    allowed: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    return_weights: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """A transparent dense attention reference.

    Args:
        q, k: ``[B, H, T_q/T_k, D_k]``.
        v: ``[B, H, T_k, D_v]``.
        causal: apply the decoder's no-future-token rule.
        allowed: optional boolean mask broadcastable to ``[B,H,T_q,T_k]``.

    ``return_weights`` is useful for checking the mask but should not be used
    in a high-performance path because it materializes ``T_q × T_k`` values.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, T, D]")
    batch, heads, t_q, d_k = q.shape
    if k.shape[:2] != (batch, heads) or v.shape[:2] != (batch, heads):
        raise ValueError("q, k and v must agree on batch and head dimensions")
    if k.shape[-1] != d_k or v.shape[-2] != k.shape[-2]:
        raise ValueError("key dimension and key/value sequence lengths must agree")

    t_k = k.shape[-2]
    if scale is None:
        scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    visibility = torch.ones((t_q, t_k), dtype=torch.bool, device=q.device)
    if causal:
        visibility &= causal_mask(t_q, t_k, device=q.device)
    if allowed is not None:
        visibility = visibility & allowed.to(device=q.device, dtype=torch.bool)

    weights = masked_softmax(scores, visibility)
    output = torch.matmul(weights, v)
    return (output, weights) if return_weights else output


def online_attention_single_query(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute one unmasked attention row with the online-softmax recurrence.

    This is deliberately a one-query educational implementation.  It mirrors
    the numerically important ``m, l, acc`` idea in FlashAttention without
    claiming to be a GPU kernel.
    """
    if q.ndim != 1 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("expected q=[D], k=[T,D], v=[T,D_v]")
    if k.shape[0] != v.shape[0] or k.shape[1] != q.shape[0]:
        raise ValueError("incompatible q, k, v shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if scale is None:
        scale = q.numel() ** -0.5

    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_work = q.to(work_dtype)
    m = torch.tensor(float("-inf"), device=q.device, dtype=work_dtype)
    l = torch.zeros((), device=q.device, dtype=work_dtype)
    acc = torch.zeros(v.shape[-1], device=q.device, dtype=work_dtype)

    for start in range(0, k.shape[0], block_size):
        k_block = k[start:start + block_size].to(work_dtype)
        v_block = v[start:start + block_size].to(work_dtype)
        scores = (k_block @ q_work) * scale
        m_next = torch.maximum(m, scores.max())
        old_rescale = torch.exp(m - m_next)
        p = torch.exp(scores - m_next)
        l = l * old_rescale + p.sum()
        acc = acc * old_rescale + p @ v_block
        m = m_next
    return (acc / l).to(v.dtype)
