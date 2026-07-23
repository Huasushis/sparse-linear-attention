import torch

from tutorial_code.reference.linear_attention import (
    causal_linear_attention_chunkwise,
    causal_linear_attention_parallel,
    causal_linear_attention_recurrent,
)


def test_parallel_recurrent_and_chunkwise_agree():
    torch.manual_seed(2)
    q = torch.randn(2, 3, 9, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 9, 5, dtype=torch.float64)
    parallel = causal_linear_attention_parallel(q, k, v)
    recurrent = causal_linear_attention_recurrent(q, k, v)
    chunkwise = causal_linear_attention_chunkwise(q, k, v, chunk_size=4)
    torch.testing.assert_close(parallel, recurrent, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(parallel, chunkwise, atol=1e-10, rtol=1e-10)
