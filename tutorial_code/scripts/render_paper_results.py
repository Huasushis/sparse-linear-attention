"""Render the compact figures embedded in the final Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "64": "#4C78A8",
    "128": "#F58518",
    "256": "#54A24B",
    "gdn": "#4C78A8",
    "kda": "#E45756",
    "dplr": "#72B7B2",
    "dense": "#4C78A8",
    "selected": "#E45756",
    "selector": "#F2CF5B",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def length_labels(lengths: list[int]) -> list[str]:
    return [f"{length // 1024}K" if length >= 1024 else str(length) for length in lengths]


def render_delta(data: dict, output: Path) -> None:
    rows = [row for row in data["rows"] if row["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, mode, title in zip(axes, ("fwd", "fwdbwd"), ("Forward", "Forward + backward")):
        for dim in (64, 128, 256):
            points = []
            lengths = sorted({row["shape"]["T"] for row in rows if row["shape"]["D"] == dim})
            for length in lengths:
                selected = {
                    row["method"]: row
                    for row in rows
                    if row["shape"]["D"] == dim
                    and row["shape"]["T"] == length
                    and row["mode"] == mode
                }
                recurrent = selected["fused_recurrent_delta"]["median_ms"]
                chunk = selected["chunk_delta"]["median_ms"]
                points.append(recurrent / chunk)
            ax.plot(
                lengths,
                points,
                marker="o",
                linewidth=2,
                label=f"head dim {dim}",
                color=COLORS[str(dim)],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels(length_labels(lengths))
        ax.set_title(title)
        style_axis(ax, xlabel="Sequence length", ylabel="chunkwise speedup over recurrent")
        ax.legend(frameon=False)
    fig.suptitle("DeltaNet: paper-style fixed-token sweep on A100 80G")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_kimi(data: dict, output: Path) -> None:
    rows = [row for row in data["rows"] if row["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = (("chunk_dplr", "DPLR", COLORS["dplr"]), ("chunk_kda", "KDA", COLORS["kda"]))
    for ax, mode, title in zip(axes, ("fwd", "fwdbwd"), ("Forward", "Forward + backward")):
        for method, label, color in labels:
            selected = sorted(
                (row for row in rows if row["method"] == method and row["mode"] == mode),
                key=lambda row: row["shape"]["T"],
            )
            ax.plot(
                [row["shape"]["T"] for row in selected],
                [row["median_ms"] for row in selected],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
            lengths = [row["shape"]["T"] for row in selected]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels(length_labels(lengths))
        ax.set_title(title)
        style_axis(ax, xlabel="Sequence length", ylabel="Median latency (ms)")
        ax.legend(frameon=False)
    fig.suptitle("Kimi Linear Figure 2 configuration: DPLR vs KDA on A100 80G")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_mqar(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model in data["models"]:
        points = [point for point in model["history"] if "validation_accuracy" in point]
        architecture = model["architecture"]
        ax.plot(
            [point["step"] for point in points],
            [point["validation_accuracy"] for point in points],
            marker="o",
            linewidth=2,
            label=architecture.upper(),
            color=COLORS[architecture],
        )
    ax.set_ylim(-0.03, 1.03)
    style_axis(ax, xlabel="Training step", ylabel="Held-out MQAR accuracy")
    ax.legend(frameon=False)
    ax.set_title("Two-layer MQAR training (T=512, 128 pairs, 64 queries)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_nsa(data: dict, output: Path) -> None:
    rows = [row for row in data["rows"] if row["status"] == "ok"]
    labels = (
        ("dense_sdpa_flash_gqa", "Dense SDPA", COLORS["dense"]),
        ("nsa_selected_kernel", "NSA selected", COLORS["selected"]),
        ("nsa_compression_topk_selection", "NSA + selector", COLORS["selector"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, mode, title in zip(axes, ("fwd", "fwdbwd"), ("Forward", "Forward + backward")):
        for method, label, color in labels:
            selected = sorted(
                (row for row in rows if row["method"] == method and row["mode"] == mode),
                key=lambda row: row["shape"]["T"],
            )
            if not selected:
                continue
            ax.plot(
                [row["shape"]["T"] for row in selected],
                [row["median_ms"] for row in selected],
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
            lengths = [row["shape"]["T"] for row in selected]
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(lengths)
        ax.set_xticklabels(length_labels(lengths))
        ax.set_title(title)
        style_axis(ax, xlabel="Sequence length", ylabel="Median latency (ms)")
        ax.legend(frameon=False)
    fig.suptitle("NSA long-sequence sweep (Hq/Hkv=64/4, 16 selected blocks)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--kimi", type=Path, required=True)
    parser.add_argument("--mqar", type=Path, required=True)
    parser.add_argument("--nsa", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_delta(load(args.delta), args.output_dir / "deltanet-speedup.png")
    render_kimi(load(args.kimi), args.output_dir / "kimi-dplr-kda.png")
    render_mqar(load(args.mqar), args.output_dir / "mqar-gdn-kda.png")
    if args.nsa is not None:
        render_nsa(load(args.nsa), args.output_dir / "nsa-long-sequence.png")


if __name__ == "__main__":
    main()
