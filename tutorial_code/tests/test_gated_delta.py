import torch

from tutorial_code.reference.gated_delta import gated_delta_recurrent, kimi_delta_recurrent


def test_broadcast_channel_gate_reduces_kda_to_gdn():
    torch.manual_seed(3)
    q = torch.randn(1, 6, 2, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 6, 2, 3, dtype=torch.float64)
    beta = torch.sigmoid(torch.randn(1, 6, 2, dtype=torch.float64))
    scalar_gate = -torch.rand(1, 6, 2, dtype=torch.float64)
    gdn, gdn_states = gated_delta_recurrent(q, k, v, beta, scalar_gate, return_state_trace=True)
    kda, kda_states = kimi_delta_recurrent(
        q,
        k,
        v,
        beta,
        scalar_gate.unsqueeze(-1).expand_as(q),
        return_state_trace=True,
    )
    assert gdn_states.shape == (1, 6, 2, 4, 3)
    torch.testing.assert_close(gdn_states, kda_states, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(gdn, kda, atol=1e-10, rtol=1e-10)
