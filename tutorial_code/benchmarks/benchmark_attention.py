"""Explicit prefill/decode timing for one attention operator family at a time."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F

from tutorial_code.reference.dense_attention import scaled_dot_product_attention
from tutorial_code.reference.linear_attention import causal_linear_attention_parallel, elu_feature_map


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: List[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def time_callable(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples: List[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start_time) * 1e3)
    return {
        "p10_ms": percentile(samples, .10),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, .90),
        "samples_ms": samples,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(device: torch.device) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_mib": properties.total_memory / 2**20,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }
    return metadata


def linear_decode_from_state(
    query: torch.Tensor,
    state: torch.Tensor,
    normalizer: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Read one query from a precomputed kernelized-linear-attention state."""
    phi_query = elu_feature_map(query)
    numerator = torch.einsum("bhk,bhkv->bhv", phi_query, state)
    denominator = torch.einsum("bhk,bhk->bh", phi_query, normalizer).unsqueeze(-1)
    return numerator / (denominator + eps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["dense", "linear"], default="dense")
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256, help="prefill length or decode cache length")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, help="optional JSON artifact path")
    args = parser.parse_args()

    if args.seq_len <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("seq-len and repeats must be positive; warmup must be non-negative")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("CPU uses float32 in this teaching benchmark; requested dtype is ignored.")
        dtype = torch.float32

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    query_length = args.seq_len if args.mode == "prefill" else 1
    key_length = args.seq_len
    q = torch.randn(args.batch, args.heads, query_length, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, key_length, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)

    methods: Dict[str, Callable[[], torch.Tensor]] = {}
    if args.operator == "dense":
        methods["reference_dense"] = lambda: scaled_dot_product_attention(q, k, v, causal=True)
        if hasattr(F, "scaled_dot_product_attention"):
            # During single-token decode every cached key is in the past, so no
            # triangular mask is needed. PyTorch's rectangular causal alignment
            # is backend/version dependent and is not used as the contract here.
            methods["torch_sdpa"] = lambda: F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=args.mode == "prefill",
            )
    elif args.mode == "prefill":
        methods["reference_linear_parallel"] = lambda: causal_linear_attention_parallel(q, k, v)
    else:
        phi_k = elu_feature_map(k)
        state = torch.einsum("bhtk,bhtv->bhkv", phi_k, v)
        normalizer = phi_k.sum(dim=2)
        methods["reference_linear_state_decode"] = lambda: linear_decode_from_state(q[:, :, 0], state, normalizer)

    results: Dict[str, Dict[str, object]] = {}
    for name, fn in methods.items():
        with torch.inference_mode():
            if device.type == "cuda":
                synchronize(device)
                baseline_memory = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            timing = time_callable(fn, device, args.warmup, args.repeats)
            if device.type == "cuda":
                timing["peak_allocated_delta_mib"] = max(
                    0,
                    torch.cuda.max_memory_allocated(device) - baseline_memory,
                ) / 2**20
        results[name] = timing

    report = {
        "contract": {
            "operator_family": args.operator,
            "mode": args.mode,
            "semantic_warning": "dense softmax and kernelized linear attention are different operators",
        },
        "environment": environment_metadata(device),
        "shape": {
            "B": args.batch,
            "H": args.heads,
            "T_q": query_length,
            "T_kv": key_length,
            "D": args.head_dim,
        },
        "dtype": str(dtype),
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
