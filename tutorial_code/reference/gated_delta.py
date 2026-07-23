"""Small educational GDN/KDA state updates.

This file isolates the recurrence shared by Gated DeltaNet and Kimi Delta
Attention.  It is not the complete FLA implementation, a DPLR chunk kernel, or
the complete Kimi Linear architecture.
"""

from __future__ import annotations

import math

import torch


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q,k,v must have shapes [B,T,H,D_k], [B,T,H,D_k], [B,T,H,D_v]")
    if v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("v and beta must agree on [B,T,H]")


def gated_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    return_state_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """GDN-like delta update with one scalar log-decay per head and token.

    When requested, the trace has shape ``[B,T,H,D_k,D_v]`` and stores the
    post-update state for every token.  It is useful for locating the first
    step where two supposedly equivalent recurrences diverge.
    """
    _validate(q, k, v, beta)
    if log_decay.shape != q.shape[:3]:
        raise ValueError("GDN log_decay must have shape [B,T,H]")
    dtype = q.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v, beta, log_decay = [x.to(work_dtype) for x in (q, k, v, beta, log_decay)]
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=q.dtype)
    output = torch.empty(batch, length, heads, value_dim, device=q.device, dtype=q.dtype)
    state_trace = []
    scale = key_dim ** -0.5
    for t in range(length):
        state = state * log_decay[:, t].exp()[..., None, None]
        retrieved = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        delta_value = beta[:, t, :, None] * (v[:, t] - retrieved)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta_value)
        output[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state)
        if return_state_trace:
            state_trace.append(state.clone())
    result = output.to(dtype)
    if return_state_trace:
        return result, torch.stack(state_trace, dim=1).to(dtype)
    return result


def kimi_delta_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    *,
    return_state_trace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """KDA-like delta update with one log-decay for every key channel.

    With ``log_decay[..., d] == scalar_log_decay`` for every ``d`` this reduces
    exactly to :func:`gated_delta_recurrent` in this same-head teaching setup.
    """
    _validate(q, k, v, beta)
    if log_decay.shape != q.shape:
        raise ValueError("KDA log_decay must have shape [B,T,H,D_k]")
    dtype = q.dtype
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    q, k, v, beta, log_decay = [x.to(work_dtype) for x in (q, k, v, beta, log_decay)]
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, key_dim, value_dim, device=q.device, dtype=q.dtype)
    output = torch.empty(batch, length, heads, value_dim, device=q.device, dtype=q.dtype)
    state_trace = []
    scale = key_dim ** -0.5
    for t in range(length):
        state = state * log_decay[:, t][..., None].exp()
        retrieved = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        delta_value = beta[:, t, :, None] * (v[:, t] - retrieved)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta_value)
        output[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state)
        if return_state_trace:
            state_trace.append(state.clone())
    result = output.to(dtype)
    if return_state_trace:
        return result, torch.stack(state_trace, dim=1).to(dtype)
    return result
