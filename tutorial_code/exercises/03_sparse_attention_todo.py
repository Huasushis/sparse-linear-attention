"""Lab 7 exercise: implement a local-plus-global causal mask."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def sliding_window_causal_mask(
    length: int,
    *,
    window_size: int,
    global_tokens: Iterable[int] = (),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return ``[T,T]`` where True means that a query may read a key.

    Contract:
    - no query may read a future key;
    - the local window includes the current token;
    - a global query reads all history;
    - later queries may read every earlier global key.
    """
    # TODO: validate arguments, construct causal/local masks, then add global visibility.
    raise NotImplementedError("TODO: implement sliding_window_causal_mask")
