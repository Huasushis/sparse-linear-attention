"""A tiny, correctness-first ReplaySSM teaching reference.

This is deliberately not the vLLM/Triton implementation.  It exposes the
same algebra on one small tensor contract so that a learner can compare a
usual recurrent update with an output-only replay and a periodic flush.
"""

from __future__ import annotations

import torch


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    delta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q and k must be [B,H,T,Dk], and v must be [B,H,T,Dv]")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError("v must agree with q/k on [B,H,T]")
    if decay.shape != q.shape[:3] or delta.shape != q.shape[:3]:
        raise ValueError("decay and delta must be [B,H,T]")
    if initial_state is not None and initial_state.shape != (*q.shape[:2], q.shape[-1], v.shape[-1]):
        raise ValueError("initial_state must be [B,H,Dk,Dv]")


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.float64 else torch.float32


def mamba2_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    delta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    return_state_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference recurrence in the tutorial orientation ``S:[B,H,Dk,Dv]``.

    The update is ``S_t = decay_t * S_(t-1) + delta_t * k_t v_t^T`` and the
    output is ``S_t^T q_t``.  The state trace is optional and exists only to
    make the first divergent token easy to inspect.
    """
    _validate(q, k, v, decay, delta, initial_state)
    dtype = q.dtype
    work = _work_dtype(dtype)
    q, k, v, decay, delta = [x.to(work) for x in (q, k, v, decay, delta)]
    if initial_state is None:
        state = torch.zeros(
            q.shape[0], q.shape[1], q.shape[-1], v.shape[-1], device=q.device, dtype=work
        )
    else:
        state = initial_state.to(work).clone()

    output = torch.empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], device=q.device, dtype=work)
    trace: list[torch.Tensor] = []
    for t in range(q.shape[2]):
        state = decay[:, :, t, None, None] * state
        state = state + delta[:, :, t, None, None] * torch.einsum(
            "bhk,bhv->bhkv", k[:, :, t], v[:, :, t]
        )
        output[:, :, t] = torch.einsum("bhkv,bhk->bhv", state, q[:, :, t])
        if return_state_trace:
            trace.append(state.clone())

    output = output.to(dtype)
    final_state = state.to(dtype)
    if return_state_trace:
        return output, final_state, torch.stack(trace, dim=2).to(dtype)
    return output, final_state

def _suffix_decay(decay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return product(decay) and product(decay[j+1:]) for a short buffer."""
    batch, heads, length = decay.shape
    suffix = torch.ones_like(decay)
    running = torch.ones(batch, heads, device=decay.device, dtype=decay.dtype)
    for j in range(length - 1, -1, -1):
        suffix[:, :, j] = running
        running = decay[:, :, j] * running
    return running, suffix


def _reconstruct_state(
    checkpoint: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    decay_buffer: torch.Tensor,
    delta_buffer: torch.Tensor,
) -> torch.Tensor:
    total_decay, suffix = _suffix_decay(decay_buffer)
    state = total_decay[:, :, None, None] * checkpoint
    weights = delta_buffer * suffix
    state = state + torch.einsum(
        "bhl,bhlk,bhlv->bhkv", weights, k_buffer, v_buffer
    )
    return state


def _output_only(
    checkpoint: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    decay_buffer: torch.Tensor,
    delta_buffer: torch.Tensor,
    q_t: torch.Tensor,
) -> torch.Tensor:
    """Read an output by reassociating ``(k v^T)^T q`` as ``v (k^T q)``."""
    total_decay, suffix = _suffix_decay(decay_buffer)
    result = total_decay[:, :, None] * torch.einsum("bhkv,bhk->bhv", checkpoint, q_t)
    key_query = torch.einsum("bhlk,bhk->bhl", k_buffer, q_t)
    result = result + torch.einsum(
        "bhl,bhlv->bhv", key_query, (delta_buffer * suffix)[:, :, :, None] * v_buffer
    )
    return result


def replayssm_output_only_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    delta: torch.Tensor,
    *,
    buffer_len: int = 8,
    initial_state: torch.Tensor | None = None,
    return_state_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay recent inputs and flush a checkpoint when ``buffer_len`` is hit.

    The implementation uses Python lists because the buffer is intentionally
    tiny and readable.  The output path itself never materializes a state; a
    state is reconstructed only for a flush (and for the optional debug trace).
    """
    _validate(q, k, v, decay, delta, initial_state)
    if buffer_len <= 0:
        raise ValueError("buffer_len must be positive")

    dtype = q.dtype
    work = _work_dtype(dtype)
    q, k, v, decay, delta = [x.to(work) for x in (q, k, v, decay, delta)]
    if initial_state is None:
        checkpoint = torch.zeros(
            q.shape[0], q.shape[1], q.shape[-1], v.shape[-1], device=q.device, dtype=work
        )
    else:
        checkpoint = initial_state.to(work).clone()

    k_items: list[torch.Tensor] = []
    v_items: list[torch.Tensor] = []
    decay_items: list[torch.Tensor] = []
    delta_items: list[torch.Tensor] = []
    output = torch.empty(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], device=q.device, dtype=work)
    trace: list[torch.Tensor] = []

    for t in range(q.shape[2]):
        k_items.append(k[:, :, t])
        v_items.append(v[:, :, t])
        decay_items.append(decay[:, :, t])
        delta_items.append(delta[:, :, t])
        k_buffer = torch.stack(k_items, dim=2)
        v_buffer = torch.stack(v_items, dim=2)
        decay_buffer = torch.stack(decay_items, dim=2)
        delta_buffer = torch.stack(delta_items, dim=2)
        output[:, :, t] = _output_only(
            checkpoint, k_buffer, v_buffer, decay_buffer, delta_buffer, q[:, :, t]
        )

        current_state = _reconstruct_state(
            checkpoint, k_buffer, v_buffer, decay_buffer, delta_buffer
        )
        if return_state_trace:
            trace.append(current_state.clone())
        if len(k_items) == buffer_len:
            checkpoint = current_state
            k_items.clear()
            v_items.clear()
            decay_items.clear()
            delta_items.clear()

    final_state = checkpoint
    if k_items:
        final_state = _reconstruct_state(
            checkpoint,
            torch.stack(k_items, dim=2),
            torch.stack(v_items, dim=2),
            torch.stack(decay_items, dim=2),
            torch.stack(delta_items, dim=2),
        )
    output = output.to(dtype)
    final_state = final_state.to(dtype)
    if return_state_trace:
        return output, final_state, torch.stack(trace, dim=2).to(dtype)
    return output, final_state
