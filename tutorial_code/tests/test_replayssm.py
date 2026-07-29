import torch

from tutorial_code.reference.replayssm import (
    mamba2_recurrent_reference,
    replayssm_output_only_reference,
)


def test_replayssm_matches_recurrent_outputs_and_states():
    torch.manual_seed(7)
    q = torch.randn(1, 1, 11, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 11, 4, dtype=torch.float64)
    decay = torch.sigmoid(torch.randn(1, 1, 11, dtype=torch.float64))
    delta = torch.sigmoid(torch.randn(1, 1, 11, dtype=torch.float64))

    recurrent = mamba2_recurrent_reference(q, k, v, decay, delta, return_state_trace=True)
    replay = replayssm_output_only_reference(
        q, k, v, decay, delta, buffer_len=4, return_state_trace=True
    )
    torch.testing.assert_close(recurrent[0], replay[0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(recurrent[1], replay[1], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(recurrent[2], replay[2], atol=1e-10, rtol=1e-10)


def test_replayssm_buffer_len_does_not_change_semantics():
    torch.manual_seed(8)
    q = torch.randn(1, 1, 9, 2, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 9, 2, dtype=torch.float64)
    decay = torch.full((1, 1, 9), 0.9, dtype=torch.float64)
    delta = torch.full((1, 1, 9), 0.7, dtype=torch.float64)

    short = replayssm_output_only_reference(q, k, v, decay, delta, buffer_len=2)
    long = replayssm_output_only_reference(q, k, v, decay, delta, buffer_len=8)
    torch.testing.assert_close(short[0], long[0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(short[1], long[1], atol=1e-10, rtol=1e-10)
