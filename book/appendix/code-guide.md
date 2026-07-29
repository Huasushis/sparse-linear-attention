# 教学代码目录说明

```text
tutorial_code/
├── reference/       # 可读、可测、但不追求高性能的正确实现
├── exercises/       # 留给你填写的 TODO
├── graders/         # 单独导入学生 TODO；未填写时预期失败
├── kernels/         # 课程自带的最小 Triton reference
├── tests/           # reference 回归测试；不代表 TODO 已完成
├── benchmarks/      # 统一 warm-up / synchronize / percentile 的小基准
├── profiling/       # PyTorch Profiler 与 nsys/ncu 共用的小目标程序
└── scripts/         # 环境报告、mask 与 GDN/KDA/FLA 探针
```

ReplaySSM 的教学入口也保持这条分层：

```text
tutorial_code/reference/replayssm.py       # recurrent 与 output-only/flush oracle
tutorial_code/exercises/05_replayssm_todo.py # 留给学习者的重结合 TODO
tutorial_code/tests/test_replayssm.py      # 小 shape 的等价性回归测试
tutorial_code/graders/test_exercise_replayssm.py # 完成 TODO 后的独立检查
```

这些文件只覆盖 `B=H=1` 的小算子，不是官方 vLLM fork 的替代品；先用它们确认公式和
buffer 语义，再决定是否阅读上游 Triton kernel。

## 最常用的命令

```bash
# 在仓库根目录；本地 CPU 也可运行
python -m pytest tutorial_code/tests -q

# 填完对应 TODO 后再运行；空白作业失败是预期行为
python -m pytest tutorial_code/graders/test_exercise_dense.py -q

# 查看你的 Python / PyTorch / GPU 环境
python -m tutorial_code.scripts.environment_report

# 证明“逐维 gate 退化为标量 gate 时，教学版 KDA 与 GDN 一致”
python -m tutorial_code.scripts.gdn_kda_probe

# CPU 或已分配 GPU 上的小 benchmark
python -m tutorial_code.benchmarks.benchmark_attention --operator dense --mode prefill --device auto --seq-len 512

# 仅在已分配 CUDA 的 Slurm job 内；生成小型 PyTorch trace
python -m tutorial_code.profiling.profile_attention --trace artifacts/attention-trace.json
```

`reference` 的慢循环是刻意保留的：它们是解释公式和验证 kernel 的 oracle，不是高性能实现。后续用 Triton/FLA 时，要始终先问“我相对的是哪个 reference、什么容差、哪个阶段？”
