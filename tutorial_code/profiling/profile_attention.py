"""Profile one fixed attention workload or expose it to an external profiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from tutorial_code.reference.dense_attention import scaled_dot_product_attention


def marked_call(name: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Run one call inside an NVTX push/pop range for Nsight filtering."""
    torch.cuda.nvtx.range_push(name)
    try:
        return fn()
    finally:
        torch.cuda.nvtx.range_pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["torch_sdpa", "reference_dense"], default="torch_sdpa")
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trace", type=Path, help="write a PyTorch Chrome trace; omit for nsys/ncu target mode")
    parser.add_argument("--with-stack", action="store_true", help="record Python stacks at substantial extra overhead")
    args = parser.parse_args()

    if args.batch <= 0 or args.heads <= 0 or args.seq_len <= 0 or args.head_dim <= 0:
        raise ValueError("batch, heads, seq-len and head-dim must be positive")
    if args.warmup < 0 or args.steps <= 0:
        raise ValueError("warmup must be non-negative and steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("this profiling target requires a CUDA allocation")

    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    query_length = args.seq_len if args.mode == "prefill" else 1
    shape_q = (args.batch, args.heads, query_length, args.head_dim)
    shape_kv = (args.batch, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape_q, device=device, dtype=dtype)
    k = torch.randn(shape_kv, device=device, dtype=dtype)
    v = torch.randn_like(k)

    if args.operator == "torch_sdpa":
        fn = lambda: F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=args.mode == "prefill",
        )
    else:
        fn = lambda: scaled_dot_product_attention(q, k, v, causal=True)

    metadata = {
        "operator": args.operator,
        "mode": args.mode,
        "shape": {"q": shape_q, "k": shape_kv, "v": shape_kv},
        "dtype": str(dtype),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "profiled_steps": args.steps,
        "warning": "profiler time includes instrumentation overhead; use the benchmark CLI for latency",
    }
    print(json.dumps(metadata, indent=2))

    with torch.inference_mode():
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)

        if args.trace is None:
            for _ in range(args.steps):
                output = marked_call("sla_attention_step", fn)
            torch.cuda.synchronize(device)
            print(f"external profiler target completed; output_shape={tuple(output.shape)}")
            return

        args.trace.parent.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=args.with_stack,
            acc_events=True,
        ) as prof:
            for _ in range(args.steps):
                with record_function("attention_step"):
                    output = marked_call("sla_attention_step", fn)
        torch.cuda.synchronize(device)
        # The cluster may clean a node-local temporary directory while a long
        # profile is running.  Recreate it immediately before Kineto opens the
        # output file instead of relying only on the pre-profile mkdir above.
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total",
            row_limit=15,
        ))
        print(f"trace={args.trace.resolve()}")
        print(f"output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
