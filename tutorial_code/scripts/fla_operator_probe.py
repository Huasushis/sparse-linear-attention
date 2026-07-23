"""Compare FLA GDN and scalarized KDA recurrent kernels on one small shape."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule, naive_recurrent_gated_delta_rule
from fla.ops.kda import fused_recurrent_kda, naive_recurrent_kda


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
    # FLA's operator tests normalize q/k before comparing recurrent kernels.
    # Without this contract, an unstable random recurrence can make a relative
    # tolerance look acceptable while its absolute error is several units.
    q = F.normalize(torch.randn(shape, device=device, dtype=torch.float32), p=2, dim=-1)
    k = F.normalize(torch.randn_like(q), p=2, dim=-1)
    v = torch.randn(
        args.batch,
        args.seq_len,
        args.heads,
        args.value_dim,
        device=device,
        dtype=torch.float32,
    )
    beta = torch.sigmoid(torch.randn(args.batch, args.seq_len, args.heads, device=device))
    scalar_log_decay = F.logsigmoid(
        torch.randn(args.batch, args.seq_len, args.heads, device=device, dtype=torch.float32),
    )
    channel_log_decay = scalar_log_decay.unsqueeze(-1).expand_as(q).contiguous()

    with torch.inference_mode():
        gdn_reference_output, gdn_reference_state = naive_recurrent_gated_delta_rule(
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
        kda_reference_output, kda_reference_state = naive_recurrent_kda(
            q,
            k,
            v,
            g=channel_log_decay,
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
        "naive_kda_output_vs_naive_gdn": max_error(kda_reference_output, gdn_reference_output),
        "naive_kda_state_vs_naive_gdn": max_error(kda_reference_state, gdn_reference_state),
        "fused_gdn_output_vs_naive_gdn": max_error(gdn_output, gdn_reference_output),
        "fused_gdn_state_vs_naive_gdn": max_error(gdn_state, gdn_reference_state),
        "fused_kda_output_vs_naive_kda": max_error(kda_output, kda_reference_output),
        "fused_kda_state_vs_naive_kda": max_error(kda_state, kda_reference_state),
        "gdn_output_vs_scalarized_kda": max_error(gdn_output, kda_output),
        "gdn_state_vs_scalarized_kda": max_error(gdn_state, kda_state),
    }
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "shape": vars(args),
        "contract": {
            "qk_l2_normalized": True,
            "gate_space": "log decay",
            "kda_gate": "GDN scalar gate broadcast over key channels",
        },
        "max_absolute_error": errors,
    }
    print(json.dumps(report, indent=2))

    # First verify the algebraic degeneration in the two transparent loops,
    # then allow the small accumulation-order error of each Triton kernel.
    torch.testing.assert_close(kda_reference_output, gdn_reference_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(kda_reference_state, gdn_reference_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gdn_output, gdn_reference_output, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(gdn_state, gdn_reference_state, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_output, kda_reference_output, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_state, kda_reference_state, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_output, gdn_output, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(kda_state, gdn_state, atol=3e-3, rtol=3e-3)


if __name__ == "__main__":
    main()
