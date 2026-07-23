import importlib

import torch

from tutorial_code.reference.dense_attention import scaled_dot_product_attention


student = importlib.import_module("tutorial_code.exercises.01_dense_attention_todo")


def test_student_decode_mask_sees_cached_keys():
    mask = student.causal_mask(1, 5, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 5)
    assert bool(mask.all())


def test_student_dense_attention_matches_reference():
    torch.manual_seed(101)
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    expected = scaled_dot_product_attention(q, k, v)
    actual = student.dense_causal_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
