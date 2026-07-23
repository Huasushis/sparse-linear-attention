import importlib

import torch


student = importlib.import_module("tutorial_code.exercises.02_linear_attention_todo")


def test_student_state_update_matches_outer_product_definition():
    torch.manual_seed(102)
    state = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    normalizer = torch.randn(2, 3, 4, dtype=torch.float64)
    phi_k = torch.randn(2, 3, 4, dtype=torch.float64)
    value = torch.randn(2, 3, 5, dtype=torch.float64)

    actual_state, actual_normalizer = student.update_state(state, normalizer, phi_k, value)
    expected_state = state + torch.einsum("bhk,bhv->bhkv", phi_k, value)
    expected_normalizer = normalizer + phi_k
    torch.testing.assert_close(actual_state, expected_state, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual_normalizer, expected_normalizer, atol=1e-10, rtol=1e-10)
