import pytest
import torch


triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


from tutorial_code.kernels.vector_add_triton import vector_add


@pytest.mark.parametrize("length", [0, 1, 257, 4099])
def test_triton_vector_add_matches_torch(length):
    torch.manual_seed(105)
    x = torch.randn(length, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(vector_add(x, y), x + y)
