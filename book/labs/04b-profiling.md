# Lab 4B：从计时到 profiler 证据

## 目标

同一个 `torch_sdpa` prefill workload 依次回答三个不同问题：

1. benchmark 已经告诉我们的稳态 latency 是什么；
2. PyTorch/系统时间线上实际发生了什么；
3. 若工具权限允许，一个关键 kernel 的硬件瓶颈证据是什么。

这个 Lab 不要求你把 profiler 输出“优化掉”。交付物是一条有证据的诊断链。

## 0. 先写假设

在运行前完成：

```text
workload: torch_sdpa / prefill / B=1,H=4,T=512,D=64 / BF16
已有 benchmark p50: __________________
我预计一次调用的 kernel 数: __________
我预计主要受限于: launch / bandwidth / compute / 不知道
能推翻我的观测: ______________________
```

“不知道”是合法答案，但仍要写什么观测会帮助你分类。

## 1. 提交默认工具探测与 PyTorch trace

环境 Lab 已完成后，在 107 教程 worktree 根目录：

```bash
cd ~/sparse_linear/sla-tutorial-run
cd .
sbatch cluster/profile-attention.sbatch
```

默认作业会：

- 打印 `nvidia-smi`、`nsys`、`ncu`、`compute-sanitizer`、`nvprof` 是否存在；
- 记录 GPU、driver、PyTorch/CUDA；
- 预热 10 次，只 profile 5 次固定 workload；
- 输出聚合表与 `artifacts/torch-profiler-$JOB_ID.json`。

trace 是诊断产物，不提交 Git。可在有图形界面的机器用 Chrome trace viewer 打开；不要把
包含私有模型/请求名称的 trace 上传到第三方网站。本 Lab 的随机张量 trace 不含 tensor
数值，但仍应养成先检查再分享的习惯。

### 聚合表阅读记录

```text
top CUDA item: _________________________
self CUDA time / CUDA total: ___________
call count: ____________________________
是否看到 allocation/memcpy: __________
profiler 时间能否替代 benchmark: 不能，因为 __________________
```

## 2. 用 NVTX 切出研究区间

打开 `tutorial_code/profiling/profile_attention.py`，找到 `sla_attention_step` 的 NVTX push/pop。
回答：warm-up 是否在这个 range 内？为什么外部 profiler 应只采 range 内的 kernel？

动态 sparse 实验以后应至少使用三个 range：

```text
selector
attention_kernel
scatter_merge
```

这让你能报告 kernel-only 和 end-to-end，而不是靠猜测拆时间。

## 3. Nsight Systems（工具存在时）

```bash
sbatch --export=ALL,RUN_TORCH_PROFILER=0,RUN_NSYS=1 \
  cluster/profile-attention.sbatch
```

作业生成 `.nsys-rep`。先用当前版本列出可用文本报告，再汇总 GPU kernel：

```bash
nsys stats --help-reports
profile_job=12345  # 替换成 sbatch 返回的真实 job id
nsys stats --report cuda_gpu_kern_sum "artifacts/nsys-attention-${profile_job}.nsys-rep"
```

记录：

```text
attention_step 中 kernel 数: __________
最长 kernel: __________________________
CPU launch 到 GPU 执行是否有大空洞: ___
是否有 H2D/D2H copy: _________________
一条时间线证据: ______________________
```

若节点没有 `nsys`，记录 `NOT FOUND` 即可；不要在计算作业中自行下载数 GB GUI 工具。

## 4. Nsight Compute（工具与权限都存在时）

只在 `nsys` 已定位关键 kernel 后执行：

```bash
sbatch --export=ALL,RUN_TORCH_PROFILER=0,RUN_NCU=1 \
  cluster/profile-attention.sbatch
```

模板先读取 `ncu --list-sets/--list-chips`，再用当前旧版可用的 `default` set、NVTX filter
与 `--launch-count 1`，避免 profile 整个进程。记录：

```text
被抓到的 kernel: ______________________
replay pass 数: ________________________
duration: ______________________________
compute throughput: ____________________
memory throughput: _____________________
achieved occupancy: ____________________
第一条有证据的瓶颈假设: ______________
还不能由这些 counter 证明的事: ________
```

若出现 performance-counter 权限错误，保存错误摘要并停止。这个结果说明当前队列不能做该层
profiling，不说明代码有 bug，也不授权普通用户修改系统设置。

!!! note "教师实跑：工具存在不等于兼容"
    2026-07-23 的 job `26199` 在 RTX 5090 节点发现 `ncu`，但版本是 2022.4.1；
    `ncu --list-chips` 只列到 Ada/Hopper，没有 Blackwell `gb202`。job `26200` 的首次尝试
    因旧 section/注入不兼容失败。修订后的脚本会先检查 chip list，在该节点明确输出
    `SKIPPED`。如果之后分到 A100，仍需重新运行预检并检查 counter 权限。该节点也没有
    `nsys`；这不是要求你自行安装系统工具，而是本次环境限制。

## 5. 做一次可证伪比较

将 operator 改成 `reference_dense`，保持 shape/dtype/mode 相同，再生成一个 PyTorch trace。
预测它相对 SDPA 会多出哪些 op/kernel，然后核对。注意两者数学结果相同，但 reference
显式构造 scores/mask，正适合观察 fusion 与中间量的差别。

```bash
sbatch --export=ALL,PROFILE_OPERATOR=reference_dense \
  cluster/profile-attention.sbatch
```

模板还接受 `PROFILE_MODE`、`PROFILE_SEQ_LEN`、`PROFILE_DTYPE` 与 `PROFILE_WARMUP`。一次只改
一个轴，并把实际值写进 run note；不要同时改 shape 和 operator 后声称差异来自 fusion。

## 交付物

在你的 run note 中提交小摘要，不提交 `.json/.nsys-rep/.ncu-rep`：

```text
commit / job / GPU / tool version
workload contract
未插桩 benchmark p50
trace 中 kernel 数与最长项
一条 timeline/counter 证据
原假设是否保留
下一次只改哪个变量
```

## 通过条件

你能解释 CUDA Event、PyTorch Profiler、`nsys`、`ncu` 各自回答什么；能从同一 workload
得到一条“时间 -> 时间线 -> kernel”证据链；不会把 profiler 插桩后的时间抄进主性能表。
