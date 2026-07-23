"""Compare FLA GDN and scalarized KDA recurrent kernels on one small shape."""

from __future__ import annotations

import argparse
import json

import torch

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule, naive_recurrent_gated_delta_rule
from fla.ops.kda import fused_recurrent_kda


def max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--key-dim", type=int, default=32)
    parser.add_argument("--value-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=107)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FLA operator probe requires a CUDA allocation")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = (args.batch, args.seq_len, args.heads, args.key_dim)
    q = torch.randn(shape, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(
        args.batch,
        args.seq_len,
        args.heads,
        args.value_dim,
        device=device,
        dtype=torch.float32,
    )
    beta = torch.sigmoid(torch.randn(args.batch, args.seq_len, args.heads, device=device))
    scalar_log_decay = -torch.rand(args.batch, args.seq_len, args.heads, device=device)
    channel_log_decay = scalar_log_decay.unsqueeze(-1).expand_as(q).contiguous()

    with torch.inference_mode():
        reference_output, reference_state = naive_recurrent_gated_delta_rule(
            q,
            k,
            v,
            beta,
            scalar_log_decay,
            output_final_state=True,
        )
        gdn_output, gdn_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=scalar_log_decay,
            beta=beta,
            output_final_state=True,
        )
        kda_output, kda_state = fused_recurrent_kda(
            q,
            k,
            v,
            g=channel_log_decay,
            beta=beta,
            output_final_state=True,
        )
    torch.cuda.synchronize()

    errors = {
        "gdn_output_vs_naive": max_error(gdn_output, reference_output),
        "gdn_state_vs_naive": max_error(gdn_state, reference_state),
        "scalarized_kda_output_vs_naive_gdn": max_error(kda_output, reference_output),
        "scalarized_kda_state_vs_naive_gdn": max_error(kda_state, reference_state),
        "gdn_output_vs_scalarized_kda": max_error(gdn_output, kda_output),
        "gdn_state_vs_scalarized_kda": max_error(gdn_state, kda_state),
    }
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "shape": vars(args),
        "max_absolute_error": errors,
    }
    print(json.dumps(report, indent=2))

    torch.testing.assert_close(gdn_output, reference_output, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(gdn_state, reference_state, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_output, reference_output, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_state, reference_state, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    main()
