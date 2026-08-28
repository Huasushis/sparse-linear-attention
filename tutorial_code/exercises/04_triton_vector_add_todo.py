"""Lab 5 exercise: finish the store in a minimal Triton kernel."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        # TODO: store x + y at output_ptr + offsets, using the same boundary mask.
        out = x + y
        tl.store(output_ptr + offsets, out, mask=mask)

else:
    _vector_add_kernel = None


def vector_add(x: torch.Tensor, y: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("x and y must have identical shape, dtype, and device")
    if x.device.type != "cuda" or not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("the exercise expects contiguous CUDA tensors")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    output = torch.empty_like(x)
    if x.numel() == 0:
        return output
    grid = (triton.cdiv(x.numel(), block_size),)
    _vector_add_kernel[grid](x, y, output, x.numel(), BLOCK_SIZE=block_size)
    return output
