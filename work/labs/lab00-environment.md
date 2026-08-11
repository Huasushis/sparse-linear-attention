# Lab 0 作业：环境与可复跑记录

> 按网站 Lab 0 操作后填写；不要预先猜测 GPU 节点环境。

## 代码版本

- 日期：2026-08-11
- 分支：`study/sparse-linear-attention`
- commit：`bc18fc08f1faeaa7a1bd2262284f8127222f11fe`
- 工作树是否干净：是；提交 Slurm 作业前 `git status --short --branch` 无文件改动。

## 本地最小检查

- Python：3.9.7，Windows 10；PyTorch 1.12.1+cpu，无可用 CUDA 设备。
- 执行的命令：

  ```powershell
  python -m tutorial_code.scripts.environment_report
  python -m pytest tutorial_code/tests -q
  ```

- 结果摘要：`11 passed, 1 skipped in 1.80s`；跳过项是本地 CPU 环境无法执行的 GPU/Triton 测试。环境报告显示 `torch_cuda=null`、`cuda_available=false`、`torch_device_count=0`。

## 107 GPU 作业

- Slurm job id：环境发现 `36201`；smoke test 与小型 benchmark `36217`。
- 节点：`anode17`，分区 `Students`，单节点、单 GPU；smoke job 使用 4 个 CPU core，时限 20 分钟。
- GPU 与显存：NVIDIA A100-SXM4-80GB，`nvidia-smi` 报告 81920 MiB；PyTorch 报告 81151.75 MiB，compute capability 8.0。
- driver：580.159.03；`nvidia-smi` 显示其支持的 CUDA 版本为 13.0。
- Python / PyTorch / CUDA / Triton：计算环境 Python 3.12.13，PyTorch 2.11.0+cu128，PyTorch CUDA runtime 12.8，Triton 3.6.0。发现作业中的系统 Python 为 3.14.6，因此实际测试使用独立环境 `~/sparse_linear/.envs/sla-tutorial-py312`。
- 执行的命令：

  ```bash
  cd ~/sparse_linear/sparse-linear-attention
  cd .
  sbatch cluster/discover-gpu.sbatch
  sbatch cluster/slurm-smoke.sbatch
  ```

- 结果摘要：
  - job `36201` 在计算节点记录了 GPU、driver、系统 Python 和 Git commit；stderr 为空。
  - job `36217` 的 correctness tests 为 `15 passed in 11.41s`；stderr 为空。
  - GDN/KDA probe 中，共享 scalar-gate recurrence 的两项最大误差均为 0；per-channel gate 对照误差为 0.6637051611063987，表明 KDA 具有额外的逐通道自由度。
  - dense prefill：`B=1,H=4,T_q=T_kv=512,D=64`，BF16，warm-up 10 次、测量 20 次；reference dense/torch SDPA 的 p50 分别为 0.252672/0.039552 ms。
  - dense decode：`B=1,H=4,T_q=1,T_kv=512,D=64`，BF16，warm-up 10 次、测量 20 次；reference dense/torch SDPA 的 p50 分别为 0.245776/0.036912 ms。
  - 原始 stdout/stderr 和 benchmark JSON 保存在远端忽略目录 `artifacts/`，没有提交 Git。

## 失败与处理

记录原始错误的关键行、原因判断和下一次只改变的一个变量。

- 本次两个作业均正常完成，没有产生 stderr。
- 集群登录/系统 Python 为 3.14.6，不作为课程运行环境；为避免依赖和 CUDA wheel 不匹配，smoke job 使用独立的 Python 3.12 Conda 环境。
- 107 的仓库位于 NFS；读取文件或判断文件缺失前先执行 `cd .` 刷新目录状态。

## 通过条件自检

- [x] 别人能根据本页找到 commit 和完整命令。
- [x] GPU 信息来自计算节点，而不是登录节点。
- [x] 大日志、环境目录和模型权重没有提交 Git。
