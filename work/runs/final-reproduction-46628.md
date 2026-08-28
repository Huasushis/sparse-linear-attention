# 最终算子复现记录：Slurm job 46628

## 状态与版本

```text
date: 2026-08-28
git branch: study/sparse-linear-attention
tutorial commit: 4d086aa5936059bfe02cb89c842ed3f44c525bdb
FLA commit: d1ce07369d581813553f30a750af3b6b5f9af6a9
node: anode02
GPU: NVIDIA GeForce RTX 5090, 32607 MiB
driver: 580.173.02
Python: 3.12.13
PyTorch: 2.11.0+cu128
PyTorch CUDA runtime: 12.8
dtype: BF16
job stderr: empty
completion marker: result_dir=.../artifacts/final-46628
```

原始 JSON 和完整样本位于 107：

```text
~/sparse_linear/sparse-linear-attention/artifacts/final-46628/
~/sparse_linear/sparse-linear-attention/artifacts/sla-final-repro-46628.out
~/sparse_linear/sparse-linear-attention/artifacts/sla-final-repro-46628.err
```

这些文件在 `.gitignore` 的 `artifacts/` 中；本页只保存可审计摘要。复跑命令：

```bash
cd ~/sparse_linear/sparse-linear-attention
cd .
sbatch cluster/final-reproduction.sbatch
```

## 正确性

- 课程 tests + 已完成的 dense/Triton grader：`20 passed in 9.75s`。
- FLA FP32 probe，`B=1,T=64,H=2,Dk=Dv=32`，q/k L2-normalized：

| 对照 | output max abs | final-state max abs |
| --- | ---: | ---: |
| naive KDA vs naive GDN（scalarized gate） | 1.49e-8 | 5.96e-8 |
| fused GDN vs naive GDN | 2.98e-8 | 1.19e-7 |
| fused KDA vs naive KDA | 2.24e-8 | 5.96e-8 |
| fused GDN vs scalarized fused KDA | 2.98e-8 | 5.96e-8 |

- 独立 final-suite scalarization check（FP32）：output/state max abs 均为 `0.0`。
- FLA NSA selected kernel vs naive oracle（BF16，`T=128,HQ/Hkv=16/1,D=64`）：max abs
  `0.015625`。这是 BF16/不同 reduction 顺序下的算子误差记录，不是模型质量指标。

## Dense 与教学 linear

共同配置：`B=1,H=4,D=64`，warm-up 10，repeats 20，CUDA Event。下表为 p50 ms；括号内为
peak allocated delta MiB。

| mode/operator | T=512 | T=2048 | T=8192 |
| --- | ---: | ---: | ---: |
| explicit dense prefill | 0.2538 (14.38) | 0.2818 (108.13) | 4.8040 (1608.13) |
| torch SDPA prefill | 0.04859 (1.27) | 0.04653 (5.09) | 0.3709 (4.13) |
| explicit dense decode | 0.2233 (8.14) | 0.2279 (8.17) | 0.2243 (8.32) |
| torch SDPA decode | 0.03922 (0.004) | 0.03818 (0.010) | 0.03832 (0.034) |
| teaching linear parallel prefill | 0.2465 (41.38) | 1.4415 (141.16) | 6.0358 (540.25) |
| teaching linear state decode | 0.1094 (0.002) | 0.1095 (0.002) | 0.1068 (0.002) |

解释边界：

- explicit dense 与 SDPA 数学语义相同；`T=8192` 时 SDPA p50 约快 `12.95x`，显式路径
  的峰值临时分配约为 SDPA 的 `390x`，支持“不要物化 score/probability”的 IO 解释。
- teaching linear 使用 ELU+1 kernel feature，与 dense softmax 语义不同；其 parallel 实现显式
  materialize prefix state，是 correctness reference，不是 FLA 性能代表。
- 在这些小 batch/head shape 上，dense SDPA decode 仍约比 teaching fixed-state read 快
  `2.8x`。固定 state 复杂度没有自动战胜成熟 dense kernel。

## FLA GDN/KDA

共同配置：`B=1,H=4,K=V=64,BF16`，q/k L2-normalized；KDA 的 channel gate 是 GDN scalar
gate 的广播，因此两者语义相同。warm-up 5，repeats 20，CUDA Event，表中为 p50 ms。

| method | T=512 | T=2048 | T=8192 |
| --- | ---: | ---: | ---: |
| chunk GDN | 0.6061 | 0.5941 | 0.5922 |
| chunk KDA（scalar gate） | 0.7407 | 0.7169 | 0.7460 |
| fused recurrent GDN | 0.3621 | 1.1937 | 4.5224 |
| fused recurrent KDA（scalar gate） | 0.4044 | 1.3691 | 5.2234 |

单步 decode 从已有 FP32 state 读取：GDN `0.1142 ms`，scalarized KDA `0.1106 ms`；约 3%
差异处于这个小 workload 的固定开销/噪声范围，不据此宣称 KDA decode 更快。

观察：

- `T=512` 时 recurrent GDN 比 chunk GDN 快约 `1.67x`；
- `T=2048`、`8192` 时 chunk GDN 分别比 recurrent 快约 `2.01x`、`7.64x`；
- 复现了“短序列固定/chunk 开销占主导，长序列 chunk parallelism 开始摊薄”的 crossover；
- scalarized KDA chunk 比 GDN chunk 慢约 `21%--26%`，recurrent 慢约 `12%--16%`。这是
  当前通用 channel-gate kernel 的实现成本，不是 KDA 模型质量的代价换算。

## FLA NSA selected/block selector

共同配置：`B=1,HQ/Hkv=16/1,D=64,BF16`，block size 64，每 query 最多保留最近 4 个 block。
`nsa_selected_fixed_blocks` 排除 selector；`nsa_compression_topk_plus_selected` 包含 mean-pool
compression、top-k 和 selected path，但不含 sliding branch。表中为 p50 ms。

| T | 逻辑 selected density | full causal SDPA | fixed selected kernel | compression/top-k + selected |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 68.81% | 0.03725 | 0.3026 | 0.5833 |
| 2048 | 20.74% | 0.1218 | 0.3043 | 0.5827 |
| 8192 | 5.41% | 0.8200 | 0.3773 | 0.6427 |

观察：

- `T=512/2048` 时，即使逻辑工作减少，sparse kernel 仍比成熟 dense SDPA 慢；
- `T=8192` 时 fixed selected kernel 相对 full dense latency 约低 `2.17x`；包含 compression/
  top-k 后仅低约 `1.28x`；
- selector-included 路径相对 fixed-index kernel 慢约 `1.70--1.93x`；
- 这说明 density 不能预测 crossover，且漏报 selector 会显著夸大收益；
- sparse 与 full dense 输出语义不同，因此上述比值是执行成本上下文，不是“无损模型
  speedup”。模型质量需要训练/任务证据。

## 停止条件与未覆盖项

- 未下载大模型权重，未运行 Kimi 48B、NSA training、MInference/HiLS serving；
- 未安装 MoBA 额外依赖 `flash-attn`，不在已有环境中强行改变依赖；
- NSA 未测 sliding branch、backward、model gate 与端到端 serving；
- 5090 节点缺少兼容的 `nsys/ncu`，不报告 DRAM throughput/occupancy；
- 当前数据是单 GPU、单 batch、三个长度的 operator evidence，不外推为所有硬件。
