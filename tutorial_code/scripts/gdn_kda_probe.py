"""Demonstrate the scalar-gate degeneration between tutorial GDN and KDA."""

from __future__ import annotations

import torch

from tutorial_code.reference.gated_delta import gated_delta_recurrent, kimi_delta_recurrent


torch.manual_seed(7)
B, T, H, D_K, D_V = 2, 7, 3, 5, 4
q = torch.randn(B, T, H, D_K, dtype=torch.float64)
k = torch.randn_like(q)
v = torch.randn(B, T, H, D_V, dtype=torch.float64)
beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64))
scalar_gate = -torch.rand(B, T, H, dtype=torch.float64)

gdn, gdn_states = gated_delta_recurrent(q, k, v, beta, scalar_gate, return_state_trace=True)
kda_same_gate, kda_same_states = kimi_delta_recurrent(
    q,
    k,
    v,
    beta,
    scalar_gate.unsqueeze(-1).expand_as(q),
    return_state_trace=True,
)
kda_channel_gate = kimi_delta_recurrent(q, k, v, beta, scalar_gate.unsqueeze(-1) - .15 * torch.rand_like(q))

print("max |GDN state - KDA state|             =", (gdn_states - kda_same_states).abs().max().item())
print("max |GDN - KDA(broadcast scalar gate)| =", (gdn - kda_same_gate).abs().max().item())
print("max |GDN - KDA(per-channel gate)|       =", (gdn - kda_channel_gate).abs().max().item())
print("Interpretation: the first two equalities verify the shared recurrence; the third line shows the extra channel-wise freedom.")
