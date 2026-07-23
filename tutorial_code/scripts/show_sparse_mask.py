"""Print a small local-plus-global causal mask for notes and debugging."""

from __future__ import annotations

import argparse

from tutorial_code.reference.sparse_attention import mask_as_text, sliding_window_causal_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--global-token", type=int, action="append", default=[])
    args = parser.parse_args()
    mask = sliding_window_causal_mask(
        args.length,
        window_size=args.window,
        global_tokens=args.global_token,
    )
    print(mask_as_text(mask))
    print(f"visible={int(mask.sum())}/{mask.numel()} density={mask.float().mean().item():.4f}")


if __name__ == "__main__":
    main()
