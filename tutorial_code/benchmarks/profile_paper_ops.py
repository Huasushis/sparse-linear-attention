"""Profile representative paper-scale DeltaNet, KDA, and NSA operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from tutorial_code.benchmarks.paper_operator_reproduction import (
    delta_case,
    kda_case,
    nsa_selected_case,
)


def make_profile_case(name: str):
    if name == "delta":
        return delta_case(
            "chunk_delta",
            batch=4,
            length=4096,
            heads=16,
            dim=128,
            requires_grad=True,
        ), {"B": 4, "T": 4096, "H": 16, "D": 128}
    if name == "kda":
        return kda_case(
            length=16384,
            heads=16,
            dim=128,
            requires_grad=True,
        ), {"B": 1, "T": 16384, "H": 16, "D": 128}
    if name == "nsa":
        return nsa_selected_case(
            length=16384,
            q_heads=64,
            kv_heads=4,
            key_dim=128,
            value_dim=128,
            block_size=64,
            selected_blocks=16,
            requires_grad=True,
            include_selector=False,
        ), {
            "B": 1,
            "T": 16384,
            "Hq": 64,
            "Hkv": 4,
            "Dk": 128,
            "Dv": 128,
            "block_size": 64,
            "selected_blocks": 16,
        }
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("delta", "kda", "nsa"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("submit this profiler inside a Slurm GPU allocation")

    case, shape = make_profile_case(args.op)
    probe = case.call()
    grad_output = torch.randn_like(probe)
    del probe

    def step():
        output = case.call()
        return torch.autograd.grad(
            output,
            case.inputs,
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
        )

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(f"paper_{args.op}_fwdbwd"):
            result = step()
        torch.cuda.synchronize()
        del result

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))
    events = []
    for event in prof.key_averages(group_by_input_shape=True):
        self_device = getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        device_total = getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", 0.0),
        )
        events.append(
            {
                "name": event.key,
                "count": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "cpu_total_us": event.cpu_time_total,
                "self_device_us": self_device,
                "device_total_us": device_total,
                "cpu_memory_bytes": event.cpu_memory_usage,
                "device_memory_bytes": getattr(event, "device_memory_usage", 0),
                "input_shapes": event.input_shapes,
            }
        )
    events.sort(key=lambda item: item["self_device_us"], reverse=True)
    report = {
        "op": args.op,
        "shape": shape,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "events_by_self_device_time": events[:40],
        "trace": str(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
