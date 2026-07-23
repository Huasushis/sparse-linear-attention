# Lab 0：环境与可复跑记录

## 目标

得到一份真实环境报告，而不是猜测自己的 CUDA、GPU 或 FLA 是否可用。

## 本地先做

```bash
python -m tutorial_code.scripts.environment_report
python -m pytest tutorial_code/tests -q
```

本地 CPU 通过测试完全有效：它验证数学和代码结构；它**不**代表 GPU 性能。

## 107 上再做

第一份作业先提交无依赖的 `cluster/discover-gpu.sbatch`，记录真实 GPU/driver；确认相匹配的 PyTorch 环境后，再提交 `cluster/slurm-smoke.sbatch`。短 smoke job 会记录：GPU 名称、driver、Python、PyTorch、CUDA、当前 Git commit、教学测试和一个小 benchmark。

## 交付物

把输出中以下字段写入实验记录：job id、GPU、driver、PyTorch/CUDA、Git commit、命令和日志路径。不要把整份大日志或模型 cache 提交到 Git。

## 通过条件

你能解释“为什么在 GPU job 内测环境”以及“为什么 CPU 上正确不等于 GPU 上快”。
