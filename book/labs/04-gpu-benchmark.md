# Lab 4：做一张可信的算子 benchmark 表

## 目标

用统一的 warm-up、同步和统计方式，分别测 dense 与 kernelized-linear operator 的 prefill / decode。
两种 operator 数学语义不同，结果保存在两个表中，不能只凭延迟把它们排成“胜负榜”。

```bash
python -m tutorial_code.benchmarks.benchmark_attention --operator dense --mode prefill --device auto --seq-len 512 --repeats 30 --output artifacts/dense-prefill-512.json
python -m tutorial_code.benchmarks.benchmark_attention --operator dense --mode decode --device auto --seq-len 512 --repeats 30 --output artifacts/dense-decode-512.json
python -m tutorial_code.benchmarks.benchmark_attention --operator linear --mode prefill --device auto --seq-len 512 --repeats 30 --output artifacts/linear-prefill-512.json
python -m tutorial_code.benchmarks.benchmark_attention --operator linear --mode decode --device auto --seq-len 512 --repeats 30 --output artifacts/linear-decode-512.json
```

在 GPU job 内把 `--seq-len` 逐步改为 `512, 2048, 8192`，并记录是否 OOM。decode 的
`--seq-len` 表示已有 cache/state 对应的上下文长度，计时只覆盖一个新 query 的读取。不要
为了凑表格跳过 selector、mask 构造或同步。

## 需要记录

- `B,H,T,D,dtype,causal`；
- forward / prefill / decode 的哪一个；
- GPU、driver、PyTorch/CUDA、commit；
- warm-up、repeats、p50/p10/p90；
- 在每个方法中是否包含必要的额外工作。

JSON 中还保存每次原始 sample 与环境元数据。正式表格由这些 artifact 生成，不要手抄终端中
最好看的一个数字。

## 通过条件

你能解释一条“linear reference 比 SDPA 慢”的结果为何完全可能成立，也能指出它不等于线性模型没有价值。

时间表稳定后继续 [Lab 4B：从计时到 profiler 证据](04b-profiling.md)。先有 benchmark，
再用 profiler 解释，不把两者的时间混用。
