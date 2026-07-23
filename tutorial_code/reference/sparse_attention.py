"""Correctness-first structured sparse attention examples."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def _global_positions(length: int, global_tokens: Iterable[int]) -> tuple[int, ...]:
    positions = tuple(sorted(set(global_tokens)))
    if any(position < 0 or position >= length for position in positions):
        raise ValueError("global token positions must be in [0, length)")
    return positions


def sliding_window_causal_mask(
    length: int,
    *,
    window_size: int,
    global_tokens: Iterable[int] = (),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a ``[T,T]`` local-plus-global causal visibility mask.

    ``window_size`` includes the current token. A global query can read all
    history; every later query can read a global key. Future keys remain hidden.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    globals_ = _global_positions(length, global_tokens)
    query = torch.arange(length, device=device).unsqueeze(1)
    key = torch.arange(length, device=device).unsqueeze(0)
    causal = key <= query
    local = (query - key) < window_size
    allowed = causal & local
    if globals_:
        global_index = torch.tensor(globals_, device=device)
        global_query = (query == global_index).any(dim=-1, keepdim=True)
        global_key = (key == global_index.unsqueeze(1)).any(dim=0, keepdim=True)
        allowed = causal & (local | global_query | global_key)
    return allowed


def _selected_keys(
    query_position: int,
    *,
    window_size: int,
    global_tokens: tuple[int, ...],
) -> list[int]:
    if query_position in global_tokens:
        return list(range(query_position + 1))
    local_start = max(0, query_position - window_size + 1)
    selected = set(range(local_start, query_position + 1))
    selected.update(position for position in global_tokens if position <= query_position)
    return sorted(selected)


def sliding_window_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    window_size: int,
    global_tokens: Iterable[int] = (),
) -> torch.Tensor:
    """Compute local-plus-global causal attention without a dense score matrix.

    The Python loop makes the selected key set explicit. It is a semantic
    oracle, not a performant sparse GPU kernel.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B,H,T,D]")
    if q.shape != k.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k must match and v must agree on [B,H,T]")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    length = q.shape[2]
    globals_ = _global_positions(length, global_tokens)
    output = torch.empty(*q.shape[:3], v.shape[-1], dtype=v.dtype, device=v.device)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for query_position in range(length):
        selected = _selected_keys(
            query_position,
            window_size=window_size,
            global_tokens=globals_,
        )
        key_block = k[:, :, selected]
        value_block = v[:, :, selected]
        scores = torch.einsum("bhd,bhnd->bhn", q[:, :, query_position], key_block) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[:, :, query_position] = torch.einsum("bhn,bhnv->bhv", probabilities, value_block)
    return output


def mask_as_text(mask: torch.Tensor) -> str:
    """Render a small boolean mask with ``#`` for work and ``.`` for skip."""
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a rank-2 boolean tensor")
    if max(mask.shape) > 64:
        raise ValueError("text rendering is limited to masks no larger than 64x64")
    return "\n".join("".join("#" if value else "." for value in row.tolist()) for row in mask.cpu())
