# Benchmark 规范：让“更快”成为可比较的话

## 先拆开要测的东西

| 层级 | 问题 | 常用指标 |
| --- | --- | --- |
| 正确性 | 算子是否算对？ | max/mean absolute error、relative error、gradient check |
| 算子性能 | 单次 attention/kernel 多快、占多少显存？ | latency p50/p10/p90、峰值显存、tokens/s |
| 模型推理 | 用户实际等多久？ | prefill latency / TTFT、decode tok/s、端到端 latency |
| 模型效果 | 加速有没有损害能力？ | loss/perplexity、长上下文任务、任务准确率 |

不要把不同层级的指标混成一个 speedup。尤其要把 **prefill**（整段 prompt 的 attention，通常受序列长度影响大）和 **decode**（每步生成一个 token，通常受 KV cache/带宽影响大）分开。

## 每张性能表必须带的元数据

```text
git commit: ...                 author implementation commit: ...
GPU / driver: ...               CUDA / PyTorch / Triton: ...
mode: forward | fwd+bwd | prefill | decode
shape: B=..., T=..., H=..., H_kv=..., D=...
dtype/layout/causal: ...        warmup / repeats / statistic: ...
baseline and command: ...
```

初期形状可先覆盖短、中、长序列，以及至少两组 head dimension。不要只挑一个对自己方法最有利的形状。

## 最小计时原则

1. 在 GPU 完成前同步；否则测到的只是 CPU 发射 kernel 的时间。
2. 先 warm-up，避免首次编译、内存分配和 cache 冷启动污染结果。
3. 重复多次，报告中位数；噪声较大时保留分位数。
4. 分配与初始化尽量放在计时区间外；如果方法的选择/mask 构建是运行时必要工作，则必须把它作为单独一列或包含在端到端时间中。
5. 先在很小形状验证正确性，再扩大到长序列；OOM 也应当记录为结果。

## 初期 baseline 顺序

1. 教学用 FP32 reference（只负责“对不对”）；
2. PyTorch `scaled_dot_product_attention`（框架基线）；
3. 可用的 FlashAttention / SDPA 高效 backend（exact 高性能基线）；
4. 你的 linear/sparse operator 或作者实现。

## 常见的无效比较

- 方法 A 用 BF16、方法 B 用 FP32；
- A 在 prefill、B 在 decode；
- A 不计 dynamic top-k/mask 构造，B 计了全部；
- 对 approximate attention 只报 kernel 时间，不报质量下降；
- 直接抄别人的 A100 或 GB200 表格当作自己机器的速度结论；
- 在不同 `T`、batch、head configuration 下比较“几倍加速”。

实验完成后使用[实验记录模板](templates/experiment-record.md)。
