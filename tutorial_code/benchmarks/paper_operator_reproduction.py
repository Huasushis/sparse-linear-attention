"""Large operator sweeps following the configurations used in the papers.

This script is intentionally separate from the tutorial smoke benchmarks.  It
reproduces three published operator studies on one allocated CUDA GPU:

* DeltaNet: recurrent versus chunkwise execution with model width 2048 and
  ``batch * sequence_length = 16384`` (the setting of DeltaNet Figure 1).
* Kimi Linear: KDA versus the generalized DPLR delta rule with ``B=1``,
  ``H=16``, ``D=128`` and lengths from 2K to 64K (Kimi Linear Figure 2).
* NSA: selected-block attention versus dense SDPA with the paper's efficiency
  dimensions, including forward and forward+backward measurements.

The script records raw JSON.  It does not silently shrink failed shapes: CUDA
OOMs and unsupported backend combinations become explicit result rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
import triton


TensorTuple = tuple[torch.Tensor, ...]


def git_commit(path: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@dataclass
class OperatorCase:
    """A fresh set of inputs and the operator call made from those inputs."""

    inputs: TensorTuple
    call: Callable[[], torch.Tensor]


def _memory_once(fn: Callable[[], object]) -> dict[str, float]:
    clear_cuda()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return {
        "input_allocated_mib": baseline / 2**20,
        "peak_allocated_mib": peak / 2**20,
        "peak_extra_mib": max(0, peak - baseline) / 2**20,
    }


def _timing(fn: Callable[[], object], warmup_ms: int, rep_ms: int) -> dict[str, float]:
    # The first invocation triggers Triton compilation/autotuning and is not timed.
    result = fn()
    torch.cuda.synchronize()
    del result
    p50, p20, p80 = triton.testing.do_bench(
        fn,
        quantiles=[0.5, 0.2, 0.8],
        warmup=max(1, warmup_ms),
        rep=max(1, rep_ms),
    )
    return {"median_ms": p50, "p20_ms": p20, "p80_ms": p80}


def benchmark_method(
    *,
    study: str,
    method: str,
    shape: dict[str, int],
    mode: str,
    make_case: Callable[[bool], OperatorCase],
    warmup_ms: int,
    rep_ms: int,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "study": study,
        "method": method,
        "mode": mode,
        "shape": shape,
        **(metadata or {}),
    }
    case: OperatorCase | None = None
    try:
        requires_grad = mode == "fwdbwd"
        case = make_case(requires_grad)
        if requires_grad:
            with torch.enable_grad():
                probe = case.call()
            grad_output = torch.randn_like(probe)
            del probe

            def measured() -> object:
                output = case.call()
                return torch.autograd.grad(
                    output,
                    case.inputs,
                    grad_outputs=grad_output,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

            context = nullcontext()
        else:

            def measured() -> object:
                return case.call()

            context = torch.inference_mode()

        with context:
            row["memory"] = _memory_once(measured)
            row.update(_timing(measured, warmup_ms, rep_ms))
        row["status"] = "ok"
    except torch.OutOfMemoryError as exc:
        row.update(status="oom", error=str(exc))
        torch.cuda.empty_cache()
    except Exception as exc:  # keep the rest of a 64K sweep usable
        row.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=8),
        )
    finally:
        del case
        clear_cuda()
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


def _leaf(tensor: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    return tensor.detach().requires_grad_(requires_grad)


def delta_case(
    method: str,
    *,
    batch: int,
    length: int,
    heads: int,
    dim: int,
    requires_grad: bool,
) -> OperatorCase:
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

    dtype = torch.bfloat16
    q = _leaf(torch.randn(batch, length, heads, dim, device="cuda", dtype=dtype), requires_grad)
    k = _leaf(torch.randn_like(q), requires_grad)
    v = _leaf(torch.randn_like(q), requires_grad)
    beta = _leaf(torch.randn(batch, length, heads, device="cuda", dtype=dtype).sigmoid(), requires_grad)
    inputs = (q, k, v, beta)
    if method == "chunk_delta":
        call = lambda: chunk_delta_rule(
            q,
            k,
            v,
            beta,
            use_qk_l2norm_in_kernel=True,
            chunk_size=64,
        )[0]
    elif method == "fused_recurrent_delta":
        call = lambda: fused_recurrent_delta_rule(
            q,
            k,
            v,
            beta,
            use_qk_l2norm_in_kernel=True,
        )[0]
    else:
        raise ValueError(method)
    return OperatorCase(inputs=inputs, call=call)


def run_delta(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model_dim = 2048
    total_tokens = 16384
    for dim in (64, 128, 256):
        heads = model_dim // dim
        for length in (512, 1024, 2048, 4096, 8192, 16384):
            batch = total_tokens // length
            shape = {"B": batch, "T": length, "H": heads, "D": dim}
            for method in ("fused_recurrent_delta", "chunk_delta"):
                for mode in args.modes:
                    rows.append(
                        benchmark_method(
                            study="deltanet_figure1",
                            method=method,
                            shape=shape,
                            mode=mode,
                            make_case=lambda grad, m=method, b=batch, t=length, h=heads, d=dim: delta_case(
                                m,
                                batch=b,
                                length=t,
                                heads=h,
                                dim=d,
                                requires_grad=grad,
                            ),
                            warmup_ms=args.warmup_ms,
                            rep_ms=args.rep_ms,
                            metadata={
                                "fixed_model_dim": model_dim,
                                "fixed_batch_times_length": total_tokens,
                                "chunk_size": 64,
                            },
                        )
                    )
    return rows


def kda_case(
    *,
    length: int,
    heads: int,
    dim: int,
    requires_grad: bool,
) -> OperatorCase:
    from fla.ops.kda import chunk_kda

    dtype = torch.bfloat16
    shape = (1, length, heads, dim)
    q = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    k = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    v = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    g = _leaf(F.logsigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)).clamp_min(-5), requires_grad)
    beta = _leaf(torch.randn(1, length, heads, device="cuda", dtype=dtype).sigmoid(), requires_grad)
    inputs = (q, k, v, g, beta)
    call = lambda: chunk_kda(
        q,
        k,
        v,
        g=g,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        lower_bound=-5,
        chunk_size=64,
    )[0]
    return OperatorCase(inputs=inputs, call=call)


def dplr_case(
    *,
    length: int,
    heads: int,
    dim: int,
    requires_grad: bool,
) -> OperatorCase:
    from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

    dtype = torch.bfloat16
    shape = (1, length, heads, dim)
    q = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    k = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    v = _leaf(torch.randn(shape, device="cuda", dtype=dtype), requires_grad)
    a_value = F.normalize(torch.rand(shape, device="cuda", dtype=dtype), p=2, dim=-1)
    a = _leaf(a_value, requires_grad)
    b = _leaf(-a_value, requires_grad)
    gk = _leaf(F.logsigmoid(torch.randn(shape, device="cuda", dtype=torch.float32)).clamp_min(-5), requires_grad)
    inputs = (q, k, v, a, b, gk)
    call = lambda: chunk_dplr_delta_rule(
        q,
        k,
        v,
        a=a,
        b=b,
        gk=gk,
        safe_gate=True,
        chunk_size=16,
    )[0]
    return OperatorCase(inputs=inputs, call=call)


def run_kimi(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    heads, dim = 16, 128
    for length in (2048, 4096, 8192, 16384, 32768, 65536):
        shape = {"B": 1, "T": length, "H": heads, "D": dim}
        for method, factory in (("chunk_dplr", dplr_case), ("chunk_kda", kda_case)):
            for mode in args.modes:
                rows.append(
                    benchmark_method(
                        study="kimi_linear_figure2",
                        method=method,
                        shape=shape,
                        mode=mode,
                        make_case=lambda grad, f=factory, t=length: f(
                            length=t,
                            heads=heads,
                            dim=dim,
                            requires_grad=grad,
                        ),
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        metadata={
                            "gate": "channel-wise log decay clamped to [-5, 0)",
                            "chunk_size": 64 if method == "chunk_kda" else 16,
                        },
                    )
                )
    return rows


def paper_block_layout(
    length: int,
    *,
    kv_heads: int,
    block_size: int,
    selected_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sorted, scattered, causal blocks without a quadratic score tensor."""

    current = torch.arange(length, device="cuda", dtype=torch.long) // block_size
    slots = torch.arange(selected_blocks, device="cuda", dtype=torch.long)
    # Once enough history exists, sample uniformly across it.  Early tokens use
    # every available block and pad the remaining slots with -1.
    denom = max(1, selected_blocks - 1)
    indices = (current[:, None] * slots[None, :]) // denom
    counts = torch.minimum(current + 1, torch.full_like(current, selected_blocks))
    indices = indices.masked_fill(slots[None, :] >= counts[:, None], -1)
    indices = indices[None, :, None, :].expand(1, length, kv_heads, selected_blocks).contiguous()
    counts = counts[None, :, None].expand(1, length, kv_heads).contiguous()
    return indices, counts


def nsa_selected_case(
    *,
    length: int,
    q_heads: int,
    kv_heads: int,
    key_dim: int,
    value_dim: int,
    block_size: int,
    selected_blocks: int,
    requires_grad: bool,
    include_selector: bool,
) -> OperatorCase:
    from fla.ops.nsa import parallel_nsa

    dtype = torch.bfloat16
    q = _leaf(torch.randn(1, length, q_heads, key_dim, device="cuda", dtype=dtype), requires_grad)
    k = _leaf(torch.randn(1, length, kv_heads, key_dim, device="cuda", dtype=dtype), requires_grad)
    v = _leaf(torch.randn(1, length, kv_heads, value_dim, device="cuda", dtype=dtype), requires_grad)
    inputs: TensorTuple = (q, k, v)
    if include_selector:
        g_cmp = torch.full((1, length, q_heads), 0.5, device="cuda", dtype=dtype)
        g_slc = torch.full_like(g_cmp, 0.5)
        call = lambda: parallel_nsa(
            q,
            k,
            v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            block_indices=None,
            block_counts=selected_blocks,
            block_size=block_size,
        )
    else:
        indices, counts = paper_block_layout(
            length,
            kv_heads=kv_heads,
            block_size=block_size,
            selected_blocks=selected_blocks,
        )
        call = lambda: parallel_nsa(
            q,
            k,
            v,
            block_indices=indices,
            block_counts=counts,
            block_size=block_size,
        )
    return OperatorCase(inputs=inputs, call=call)


def dense_gqa_case(
    *,
    length: int,
    q_heads: int,
    kv_heads: int,
    key_dim: int,
    value_dim: int,
    requires_grad: bool,
) -> OperatorCase:
    dtype = torch.bfloat16
    q = _leaf(torch.randn(1, q_heads, length, key_dim, device="cuda", dtype=dtype), requires_grad)
    k = _leaf(torch.randn(1, kv_heads, length, key_dim, device="cuda", dtype=dtype), requires_grad)
    v = _leaf(torch.randn(1, kv_heads, length, value_dim, device="cuda", dtype=dtype), requires_grad)
    inputs = (q, k, v)

    def call() -> torch.Tensor:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                enable_gqa=True,
            )

    return OperatorCase(inputs=inputs, call=call)


def run_nsa(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # NSA Section 5 efficiency configuration: four KV groups, sixteen query
    # heads per group, dk=192, dv=128, selected block size/count 64/16.
    kv_heads, heads_per_group = 4, 16
    q_heads = kv_heads * heads_per_group
    key_dim, value_dim = args.nsa_key_dim, args.nsa_value_dim
    block_size, selected_blocks = 64, 16
    for length in args.nsa_lengths:
        shape = {
            "B": 1,
            "T": length,
            "Hq": q_heads,
            "Hkv": kv_heads,
            "Dk": key_dim,
            "Dv": value_dim,
        }
        factories: Iterable[tuple[str, Callable[[bool], OperatorCase]]] = (
            (
                "dense_sdpa_flash_gqa",
                lambda grad, t=length: dense_gqa_case(
                    length=t,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    requires_grad=grad,
                ),
            ),
            (
                "nsa_selected_kernel",
                lambda grad, t=length: nsa_selected_case(
                    length=t,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    block_size=block_size,
                    selected_blocks=selected_blocks,
                    requires_grad=grad,
                    include_selector=False,
                ),
            ),
            (
                "nsa_compression_topk_selection",
                lambda grad, t=length: nsa_selected_case(
                    length=t,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    key_dim=key_dim,
                    value_dim=value_dim,
                    block_size=block_size,
                    selected_blocks=selected_blocks,
                    requires_grad=grad,
                    include_selector=True,
                ),
            ),
        )
        for method, factory in factories:
            for mode in args.modes:
                rows.append(
                    benchmark_method(
                        study="nsa_figure6_fla_supported_dimensions",
                        method=method,
                        shape=shape,
                        mode=mode,
                        make_case=factory,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                        metadata={
                            "selected_block_size": block_size,
                            "selected_block_count": selected_blocks,
                            "window_size_in_paper": 512,
                            "paper_key_dim": 192,
                            "paper_value_dim": 128,
                        },
                    )
                )
    return rows


def environment() -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    fla_repo = os.environ.get("FLA_REPO", "")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_mib": props.total_memory / 2**20,
        "tutorial_commit": git_commit(),
        "fla_commit": git_commit(fla_repo) if fla_repo else None,
        "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("delta", "kimi", "nsa"), required=True)
    parser.add_argument("--modes", choices=("fwd", "fwdbwd"), nargs="+", default=("fwd", "fwdbwd"))
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--nsa-key-dim", type=int, default=128)
    parser.add_argument("--nsa-value-dim", type=int, default=128)
    parser.add_argument("--nsa-lengths", type=int, nargs="+", default=(8192, 16384, 32768, 65536))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("submit this benchmark inside a Slurm GPU allocation")
    torch.manual_seed(20260828)
    torch.cuda.manual_seed_all(20260828)
    os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

    runner = {"delta": run_delta, "kimi": run_kimi, "nsa": run_nsa}[args.study]
    result = {
        "environment": environment(),
        "methodology": {
            "dtype": "torch.bfloat16 (gates requiring stable accumulation use float32)",
            "timing": "triton.testing.do_bench; quantiles p20/p50/p80; inputs outside timed region",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "fwdbwd": "fresh forward followed by torch.autograd.grad for every timed iteration",
            "memory": "peak torch allocator bytes above already-materialized operator inputs",
        },
        "rows": runner(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
