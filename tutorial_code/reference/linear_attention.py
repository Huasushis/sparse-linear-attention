"""Three correctness-first views of causal kernelized linear attention."""

from __future__ import annotations

import torch


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """A positive feature map used only for a small teaching example."""
    return torch.nn.functional.elu(x) + 1.0


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q,k,v with shape [B,H,T,D]")
    if q.shape != k.shape:
        raise ValueError("q and k must have identical shapes in this tutorial operator")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must agree with q/k on [B,H,T]")


def causal_linear_attention_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Prefix-sum form of causal kernelized linear attention.

    It computes ``S_t = sum_{i<=t} phi(k_i) v_i^T`` and ``z_t =
    sum_{i<=t} phi(k_i)`` for all ``t`` in parallel tensor operations.
    """
    _validate(q, k, v)
    phi_q, phi_k = elu_feature_map(q), elu_feature_map(k)
    kv = torch.einsum("bhtk,bhtv->bhtkv", phi_k, v)
    state_prefix = kv.cumsum(dim=2)
    normalizer_prefix = phi_k.cumsum(dim=2)
    numerator = torch.einsum("bhtk,bhtkv->bhtv", phi_q, state_prefix)
    denominator = torch.einsum("bhtk,bhtk->bht", phi_q, normalizer_prefix).unsqueeze(-1)
    return numerator / (denominator + eps)


def causal_linear_attention_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """The same operator, written as a token-by-token recurrent state update."""
    _validate(q, k, v)
    phi_q, phi_k = elu_feature_map(q), elu_feature_map(k)
    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, key_dim, value_dim, dtype=q.dtype, device=q.device)
    normalizer = torch.zeros(batch, heads, key_dim, dtype=q.dtype, device=q.device)
    output = torch.empty(batch, heads, length, value_dim, dtype=q.dtype, device=q.device)
    for t in range(length):
        state = state + torch.einsum("bhk,bhv->bhkv", phi_k[:, :, t], v[:, :, t])
        normalizer = normalizer + phi_k[:, :, t]
        numerator = torch.einsum("bhk,bhkv->bhv", phi_q[:, :, t], state)
        denominator = torch.einsum("bhk,bhk->bh", phi_q[:, :, t], normalizer).unsqueeze(-1)
        output[:, :, t] = numerator / (denominator + eps)
    return output


def causal_linear_attention_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Chunked recurrent form that explicitly carries state between chunks.

    The inner loop is intentionally simple.  Real kernels replace it with block
    algebra and parallel scans; this version makes the state boundary visible.
    """
    _validate(q, k, v)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    phi_q, phi_k = elu_feature_map(q), elu_feature_map(k)
    batch, heads, length, key_dim = q.shape
    value_dim = v.shape[-1]
    state = torch.zeros(batch, heads, key_dim, value_dim, dtype=q.dtype, device=q.device)
    normalizer = torch.zeros(batch, heads, key_dim, dtype=q.dtype, device=q.device)
    output = torch.empty(batch, heads, length, value_dim, dtype=q.dtype, device=q.device)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        for t in range(start, end):
            state = state + torch.einsum("bhk,bhv->bhkv", phi_k[:, :, t], v[:, :, t])
            normalizer = normalizer + phi_k[:, :, t]
            output[:, :, t] = torch.einsum("bhk,bhkv->bhv", phi_q[:, :, t], state) / (
                torch.einsum("bhk,bhk->bh", phi_q[:, :, t], normalizer).unsqueeze(-1) + eps
            )
    return output
