import pytest
import torch

from tutorial_code.reference.dense_attention import (
    causal_mask,
    masked_softmax,
    online_attention_single_query,
    scaled_dot_product_attention,
)


def test_decode_mask_can_see_all_cached_keys():
    mask = causal_mask(1, 5, device=torch.device("cpu"))
    assert mask.shape == (1, 5)
    assert mask.all()


def test_future_tokens_do_not_change_earlier_causal_output():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 3, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(1, 1, 4, 2, dtype=torch.float64)
    baseline = scaled_dot_product_attention(q, k, v)
    k_changed, v_changed = k.clone(), v.clone()
    k_changed[..., 3, :] += 100
    v_changed[..., 3, :] += 100
    changed = scaled_dot_product_attention(q, k_changed, v_changed)
    torch.testing.assert_close(baseline[..., :3, :], changed[..., :3, :], atol=1e-10, rtol=1e-10)


def test_online_single_query_matches_dense_attention():
    torch.manual_seed(1)
    q = torch.randn(5, dtype=torch.float64)
    k = torch.randn(11, 5, dtype=torch.float64)
    v = torch.randn(11, 4, dtype=torch.float64)
    online = online_attention_single_query(q, k, v, block_size=3)
    dense = scaled_dot_product_attention(q[None, None, None], k[None, None], v[None, None], causal=False)[0, 0, 0]
    torch.testing.assert_close(online, dense, atol=1e-10, rtol=1e-10)


def test_masked_softmax_rejects_fully_masked_row():
    scores = torch.zeros(2, 3, dtype=torch.float64)
    allowed = torch.tensor([[True, False, False], [False, False, False]])
    with pytest.raises(ValueError, match="at least one key"):
        masked_softmax(scores, allowed)
