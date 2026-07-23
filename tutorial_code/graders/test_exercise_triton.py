import importlib

import pytest
import torch


triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
student = importlib.import_module("tutorial_code.exercises.04_triton_vector_add_todo")


@pytest.mark.parametrize("length", [1, 257, 4099])
def test_student_triton_add_handles_tail_mask(length):
    torch.manual_seed(106)
    x = torch.randn(length, device="cuda")
    y = torch.randn_like(x)
    torch.testing.assert_close(student.vector_add(x, y), x + y)
