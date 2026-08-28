"""Train small GDN and KDA language models on multi-query associative recall.

The Kimi Linear paper evaluates two-layer, two-head, head-dimension-128 models
on MQAR.  This experiment keeps that mixer configuration and trains both
models on exactly the same procedurally generated batches.  It is a controlled
algorithm-effect reproduction: unlike an operator benchmark, the output is a
learning curve and held-out retrieval accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def git_commit(path: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class DataConfig:
    sequence_length: int = 512
    key_vocab: int = 256
    value_vocab: int = 256
    noise_vocab: int = 64
    num_pairs: int = 128
    num_queries: int = 64

    @property
    def separator(self) -> int:
        return self.key_vocab + self.value_vocab

    @property
    def noise_start(self) -> int:
        return self.separator + 1

    @property
    def vocab_size(self) -> int:
        return self.noise_start + self.noise_vocab

    @property
    def noise_length(self) -> int:
        used = 2 * self.num_pairs + 1 + 2 * self.num_queries
        return self.sequence_length - used

    def validate(self) -> None:
        if self.num_pairs > self.key_vocab:
            raise ValueError("num_pairs must not exceed key_vocab because keys are unique")
        if self.num_queries > self.num_pairs:
            raise ValueError("num_queries must not exceed num_pairs")
        if self.noise_length < 0:
            raise ValueError("sequence_length is too small for the requested pairs and queries")


def make_mqar_batch(
    config: DataConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tokens, query positions, and the value following each query."""

    config.validate()
    generator = torch.Generator(device=device).manual_seed(seed)
    rows: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    answers: list[torch.Tensor] = []
    tail_start = 2 * config.num_pairs + config.noise_length + 1
    query_positions = tail_start + 2 * torch.arange(config.num_queries, device=device)
    for _ in range(batch_size):
        keys = torch.randperm(config.key_vocab, generator=generator, device=device)[: config.num_pairs]
        values = torch.randint(
            config.key_vocab,
            config.key_vocab + config.value_vocab,
            (config.num_pairs,),
            generator=generator,
            device=device,
        )
        query_indices = torch.randperm(config.num_pairs, generator=generator, device=device)[: config.num_queries]
        query_keys = keys[query_indices]
        query_values = values[query_indices]
        memory = torch.stack((keys, values), dim=-1).flatten()
        noise = torch.randint(
            config.noise_start,
            config.vocab_size,
            (config.noise_length,),
            generator=generator,
            device=device,
        )
        tail = torch.stack((query_keys, query_values), dim=-1).flatten()
        row = torch.cat(
            (
                memory,
                noise,
                torch.tensor([config.separator], device=device),
                tail,
            )
        )
        rows.append(row)
        positions.append(query_positions)
        answers.append(query_values)
    return torch.stack(rows), torch.stack(positions), torch.stack(answers)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_gate = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class RecallBlock(nn.Module):
    def __init__(self, mixer: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed, _, _ = self.mixer(self.norm1(x), use_cache=False)
        x = x + mixed
        return x + self.mlp(self.norm2(x))


class RecallModel(nn.Module):
    def __init__(
        self,
        *,
        architecture: str,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 2,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet
        from fla.layers.kda import KimiDeltaAttention

        if hidden_size != num_heads * head_dim:
            raise ValueError("this reproduction uses hidden_size = num_heads * head_dim")
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        blocks = []
        for layer_idx in range(num_layers):
            if architecture == "gdn":
                mixer = GatedDeltaNet(
                    hidden_size=hidden_size,
                    expand_v=1,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mode="chunk",
                    use_gate=True,
                    use_short_conv=True,
                    layer_idx=layer_idx,
                )
            elif architecture == "kda":
                mixer = KimiDeltaAttention(
                    hidden_size=hidden_size,
                    expand_v=1,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mode="chunk",
                    use_short_conv=True,
                    safe_gate=True,
                    lower_bound=-5,
                    layer_idx=layer_idx,
                )
            else:
                raise ValueError(architecture)
            blocks.append(RecallBlock(mixer, hidden_size))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def answer_logits(
    model: nn.Module,
    tokens: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    logits = model(tokens[:, :-1])
    batch = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
    return logits[batch, query_positions]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    config: DataConfig,
    *,
    batch_size: int,
    batches: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_answers = 0
    for index in range(batches):
        tokens, positions, answers = make_mqar_batch(
            config,
            batch_size=batch_size,
            seed=seed + index,
            device=device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = answer_logits(model, tokens, positions)
            loss = F.cross_entropy(logits.flatten(0, 1), answers.flatten())
        total_loss += loss.item()
        total_correct += (logits.argmax(-1) == answers).sum().item()
        total_answers += answers.numel()
    model.train()
    return {
        "loss": total_loss / batches,
        "accuracy": total_correct / total_answers,
    }


def train_one(
    architecture: str,
    config: DataConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    model = RecallModel(
        architecture=architecture,
        vocab_size=config.vocab_size,
        hidden_size=256,
        num_layers=2,
        num_heads=2,
        head_dim=128,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    parameter_count = sum(p.numel() for p in model.parameters())
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    tokens_seen = 0
    model.train()
    for step in range(1, args.steps + 1):
        tokens, positions, answers = make_mqar_batch(
            config,
            batch_size=args.batch_size,
            seed=args.data_seed + step,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = answer_logits(model, tokens, positions)
            loss = F.cross_entropy(logits.flatten(0, 1), answers.flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += tokens.numel()

        if step == 1 or step % args.log_every == 0:
            train_accuracy = (logits.detach().argmax(-1) == answers).float().mean().item()
            point: dict[str, float | int] = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": loss.item(),
                "train_accuracy": train_accuracy,
                "grad_norm": float(grad_norm),
                "elapsed_s": time.perf_counter() - start,
            }
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                validation = evaluate(
                    model,
                    config,
                    batch_size=args.eval_batch_size,
                    batches=args.eval_batches,
                    seed=args.validation_seed,
                    device=device,
                )
                point.update(
                    validation_loss=validation["loss"],
                    validation_accuracy=validation["accuracy"],
                )
            history.append(point)
            print(json.dumps({"architecture": architecture, **point}), flush=True)

    elapsed = time.perf_counter() - start
    final_validation = evaluate(
        model,
        config,
        batch_size=args.eval_batch_size,
        batches=args.eval_batches,
        seed=args.validation_seed,
        device=device,
    )
    return {
        "architecture": architecture,
        "parameter_count": parameter_count,
        "elapsed_s": elapsed,
        "tokens_seen": tokens_seen,
        "training_tokens_per_second": tokens_seen / elapsed,
        "final_validation": final_validation,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=("gdn", "kda"), default=("gdn", "kda"))
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--model-seed", type=int, default=20260828)
    parser.add_argument("--data-seed", type=int, default=1000000)
    parser.add_argument("--validation-seed", type=int, default=2000000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("submit this experiment inside a Slurm GPU allocation")
    if args.eval_every % args.log_every != 0:
        raise ValueError("eval_every must be divisible by log_every")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = DataConfig()
    config.validate()
    fla_repo = os.environ.get("FLA_REPO", "")
    result: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "tutorial_commit": git_commit(),
            "fla_commit": git_commit(fla_repo) if fla_repo else None,
        },
        "data": asdict(config),
        "training": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output", "architectures"}
        },
        "models": [],
    }
    for architecture in args.architectures:
        result["models"].append(train_one(architecture, config, args, device))
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
