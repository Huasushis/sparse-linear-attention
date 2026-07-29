import importlib

import torch


student = importlib.import_module("tutorial_code.exercises.05_replayssm_todo")


def test_student_output_only_matches_explicit_small_reference():
    torch.manual_seed(107)
    checkpoint = torch.randn(1, 1, 3, 4, dtype=torch.float64)
    k_buffer = torch.randn(1, 1, 3, 3, dtype=torch.float64)
    v_buffer = torch.randn(1, 1, 3, 4, dtype=torch.float64)
    decay_buffer = torch.tensor([[[0.8, 0.9, 0.7]]], dtype=torch.float64)
    delta_buffer = torch.tensor([[[0.6, 0.5, 0.4]]], dtype=torch.float64)
    q_t = torch.randn(1, 1, 3, dtype=torch.float64)

    state = checkpoint.clone()
    for j in range(k_buffer.shape[2]):
        state = decay_buffer[:, :, j, None, None] * state
        state = state + delta_buffer[:, :, j, None, None] * torch.einsum(
            "bhk,bhv->bhkv", k_buffer[:, :, j], v_buffer[:, :, j]
        )
    expected = torch.einsum("bhkv,bhk->bhv", state, q_t)

    actual = student.output_only_from_checkpoint(
        checkpoint, k_buffer, v_buffer, decay_buffer, delta_buffer, q_t
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
