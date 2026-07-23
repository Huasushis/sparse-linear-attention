import importlib

import torch


student = importlib.import_module("tutorial_code.exercises.03_sparse_attention_todo")


def test_student_mask_is_causal_and_local():
    mask = student.sliding_window_causal_mask(6, window_size=2, global_tokens=())
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask.cpu(), expected)


def test_student_global_token_has_two_way_causal_visibility():
    mask = student.sliding_window_causal_mask(6, window_size=2, global_tokens=(2,))
    assert torch.equal(mask[2], torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool))
    assert bool(mask[5, 2])
    assert not bool(mask[1, 2])
