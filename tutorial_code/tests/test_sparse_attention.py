import torch

from tutorial_code.reference.dense_attention import scaled_dot_product_attention
from tutorial_code.reference.sparse_attention import (
    mask_as_text,
    sliding_window_attention_reference,
    sliding_window_causal_mask,
)


def test_sliding_global_mask_contract():
    mask = sliding_window_causal_mask(6, window_size=2, global_tokens=(2,))
    assert not bool(torch.triu(mask, diagonal=1).any())
    assert torch.equal(mask[2], torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool))
    assert bool(mask[5, 2])
    assert not bool(mask[5, 1])


def test_sparse_loop_matches_masked_dense_oracle():
    torch.manual_seed(103)
    q = torch.randn(2, 3, 7, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 7, 5, dtype=torch.float64)
    mask = sliding_window_causal_mask(7, window_size=3, global_tokens=(0, 4))
    expected = scaled_dot_product_attention(q, k, v, causal=False, allowed=mask)
    actual = sliding_window_attention_reference(q, k, v, window_size=3, global_tokens=(0, 4))
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_mask_text_is_stable_for_notes():
    mask = sliding_window_causal_mask(4, window_size=2)
    assert mask_as_text(mask) == "#...\n##..\n.##.\n..##"
