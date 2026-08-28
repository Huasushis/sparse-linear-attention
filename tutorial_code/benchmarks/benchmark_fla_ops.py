"""Reproduce small FLA GDN/KDA and NSA operator results on one allocated GPU.

The suite deliberately separates operators with different semantics.  GDN and
scalarized KDA are compared directly because their recurrences coincide under
the recorded contract.  NSA results report both a preselected block-sparse
kernel and a compression/top-k path; neither is labelled equivalent to full
dense attention.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def git_commit(path: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def time_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "p10_ms": percentile(samples, 0.10),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "samples_ms": samples,
        "peak_allocated_delta_mib": max(
            0, torch.cuda.max_memory_allocated() - baseline_memory
        ) / 2**20,
    }


def scalarized_gdn_kda_inputs(
    length: int,
    *,
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    shape = (batch, length, heads, key_dim)
    q = F.normalize(torch.randn(shape, device="cuda", dtype=dtype), p=2, dim=-1)
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(batch, length, heads, value_dim, device="cuda", dtype=dtype)
    beta = torch.sigmoid(torch.randn(batch, length, heads, device="cuda", dtype=dtype))
    scalar_gate = F.logsigmoid(torch.randn(batch, length, heads, device="cuda", dtype=dtype))
    channel_gate = scalar_gate.unsqueeze(-1).expand_as(q).contiguous()
    return q, k, v, beta, scalar_gate, channel_gate


def recent_block_indices(
    length: int,
    *,
    block_size: int,
    selected_blocks: int,
) -> torch.Tensor:
    query_blocks = torch.arange(length, device="cuda") // block_size
    offsets = torch.arange(selected_blocks - 1, -1, -1, device="cuda")
    # Negative block ids are deliberately retained: the FLA kernel treats them
    # as skipped entries, matching the naive reference's mask.
    indices = query_blocks[:, None] - offsets[None, :]
    return indices[None, :, None, :].to(torch.int32)


def logical_recent_block_density(length: int, block_size: int, selected_blocks: int) -> float:
    selected = 0
    for query in range(length):
        current_block = query // block_size
        first_block = max(0, current_block - selected_blocks + 1)
        selected += query - first_block * block_size + 1
    dense_causal = length * (length + 1) // 2
    return selected / dense_causal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--nsa-block-size", type=int, default=64)
    parser.add_argument("--nsa-selected-blocks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a Slurm GPU allocation")
    if any(length <= 0 for length in args.seq_lens):
        raise ValueError("all sequence lengths must be positive")

    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.nsa import naive_nsa, parallel_nsa

    torch.manual_seed(20260828)
    torch.cuda.manual_seed_all(20260828)
    dtype = torch.bfloat16
    fla_repo = os.environ.get("FLA_REPO", "")
    report: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "tutorial_commit": git_commit(),
            "fla_commit": git_commit(fla_repo) if fla_repo else None,
        },
        "contract": {
            "dtype": str(dtype),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "gdn_kda": "L2-normalized q/k; KDA channel log-decay is a broadcast GDN scalar log-decay",
            "nsa": "selected-only path uses precomputed recent causal blocks; selector cost excluded",
            "warning": "dense, linear-state, and sparse operators do not share mathematical semantics",
        },
    }

    # Correctness: the transparent recurrence degeneration and one small NSA
    # selected-block oracle are checked before performance sweeps.
    q, k, v, beta, g, gk = scalarized_gdn_kda_inputs(
        64,
        batch=args.batch,
        heads=args.heads,
        key_dim=args.head_dim,
        value_dim=args.head_dim,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        gdn_o, gdn_s = fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, output_final_state=True
        )
        kda_o, kda_s = fused_recurrent_kda(
            q, k, v, g=gk, beta=beta, output_final_state=True
        )
    gdn_kda_error = {
        "output_max_abs": (gdn_o.float() - kda_o.float()).abs().max().item(),
        "state_max_abs": (gdn_s.float() - kda_s.float()).abs().max().item(),
    }

    nsa_t = 128
    hq, hkv = 16, 1
    qn = torch.randn(args.batch, nsa_t, hq, args.head_dim, device="cuda", dtype=dtype)
    kn = torch.randn(args.batch, nsa_t, hkv, args.head_dim, device="cuda", dtype=dtype)
    vn = torch.randn_like(kn)
    indices = recent_block_indices(
        nsa_t,
        block_size=args.nsa_block_size,
        selected_blocks=min(args.nsa_selected_blocks, 2),
    )
    with torch.inference_mode():
        nsa_reference = naive_nsa(
            qn,
            kn,
            vn,
            block_indices=indices,
            block_counts=indices.shape[-1],
            block_size=args.nsa_block_size,
        )
        nsa_kernel = parallel_nsa(
            qn,
            kn,
            vn,
            block_indices=indices,
            block_counts=indices.shape[-1],
            block_size=args.nsa_block_size,
        )
    report["correctness"] = {
        "scalarized_kda_vs_gdn": gdn_kda_error,
        "nsa_selected_kernel_vs_naive_max_abs": (
            nsa_kernel.float() - nsa_reference.float()
        ).abs().max().item(),
    }

    linear_results: list[dict[str, object]] = []
    with torch.inference_mode():
        for length in args.seq_lens:
            q, k, v, beta, g, gk = scalarized_gdn_kda_inputs(
                length,
                batch=args.batch,
                heads=args.heads,
                key_dim=args.head_dim,
                value_dim=args.head_dim,
                dtype=dtype,
            )
            methods = {
                "chunk_gdn": lambda: chunk_gated_delta_rule(q, k, v, g=g, beta=beta),
                "chunk_kda_scalar_gate": lambda: chunk_kda(q, k, v, g=gk, beta=beta),
                "fused_recurrent_gdn": lambda: fused_recurrent_gated_delta_rule(
                    q, k, v, g=g, beta=beta
                ),
                "fused_recurrent_kda_scalar_gate": lambda: fused_recurrent_kda(
                    q, k, v, g=gk, beta=beta
                ),
            }
            row = {
                "shape": {
                    "B": args.batch,
                    "T": length,
                    "H": args.heads,
                    "K": args.head_dim,
                    "V": args.head_dim,
                },
                "methods": {
                    name: time_cuda(fn, warmup=args.warmup, repeats=args.repeats)
                    for name, fn in methods.items()
                },
            }
            linear_results.append(row)

        # Single-token recurrent decode from an already materialized fixed-size state.
        q, k, v, beta, g, gk = scalarized_gdn_kda_inputs(
            1,
            batch=args.batch,
            heads=args.heads,
            key_dim=args.head_dim,
            value_dim=args.head_dim,
            dtype=dtype,
        )
        initial_state = torch.randn(
            args.batch,
            args.heads,
            args.head_dim,
            args.head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        decode_methods = {
            "fused_recurrent_gdn": lambda: fused_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=True
            ),
            "fused_recurrent_kda_scalar_gate": lambda: fused_recurrent_kda(
                q, k, v, g=gk, beta=beta, initial_state=initial_state, output_final_state=True
            ),
        }
        report["linear_decode"] = {
            "shape": {
                "B": args.batch,
                "T": 1,
                "H": args.heads,
                "K": args.head_dim,
                "V": args.head_dim,
                "state_dtype": str(initial_state.dtype),
            },
            "methods": {
                name: time_cuda(fn, warmup=args.warmup, repeats=args.repeats)
                for name, fn in decode_methods.items()
            },
        }
    report["linear_prefill"] = linear_results

    sparse_results: list[dict[str, object]] = []
    with torch.inference_mode():
        for length in args.seq_lens:
            q = torch.randn(args.batch, length, hq, args.head_dim, device="cuda", dtype=dtype)
            k = torch.randn(args.batch, length, hkv, args.head_dim, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            indices = recent_block_indices(
                length,
                block_size=args.nsa_block_size,
                selected_blocks=args.nsa_selected_blocks,
            )
            gate_shape = (args.batch, length, hq)
            g_cmp = torch.full(gate_shape, 0.5, device="cuda", dtype=dtype)
            g_slc = torch.full_like(g_cmp, 0.5)
            q_sdpa = q.transpose(1, 2).contiguous()
            k_sdpa = k.expand(args.batch, length, hq, args.head_dim).transpose(1, 2).contiguous()
            v_sdpa = v.expand_as(k_sdpa.transpose(1, 2)).transpose(1, 2).contiguous()
            methods = {
                "torch_sdpa_full_causal": lambda: F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, is_causal=True
                ),
                "nsa_selected_fixed_blocks": lambda: parallel_nsa(
                    q,
                    k,
                    v,
                    block_indices=indices,
                    block_counts=args.nsa_selected_blocks,
                    block_size=args.nsa_block_size,
                ),
                "nsa_compression_topk_plus_selected": lambda: parallel_nsa(
                    q,
                    k,
                    v,
                    g_cmp=g_cmp,
                    g_slc=g_slc,
                    block_indices=None,
                    block_counts=args.nsa_selected_blocks,
                    block_size=args.nsa_block_size,
                ),
            }
            sparse_results.append({
                "shape": {
                    "B": args.batch,
                    "T": length,
                    "HQ": hq,
                    "H_kv": hkv,
                    "D": args.head_dim,
                },
                "block_size": args.nsa_block_size,
                "selected_blocks": args.nsa_selected_blocks,
                "logical_selected_density_vs_dense_causal": logical_recent_block_density(
                    length, args.nsa_block_size, args.nsa_selected_blocks
                ),
                "methods": {
                    name: time_cuda(fn, warmup=args.warmup, repeats=args.repeats)
                    for name, fn in methods.items()
                },
            })
    report["sparse_prefill"] = sparse_results

    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
