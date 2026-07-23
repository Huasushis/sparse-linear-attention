# Lab 0：把研究工作区变成可复跑的实验台

## 目标

本 lab 不训练模型、不追求性能。目标是确认“你知道自己在哪台机器上、用了什么软件、运行的是哪个版本的代码”，并留下一个可提交的小记录。

## 你的两种环境

| 环境 | 做什么 | 不做什么 |
| --- | --- | --- |
| 本地仓库 | 阅读、写笔记、编辑代码、小型 CPU 检查、Git | 假装本机 benchmark 能代表 GPU |
| `ssh 107` 的 Slurm 集群 | GPU 正确性、microbenchmark、小模型推理 | 在登录节点跑 GPU 工作；无记录地改代码 |

集群中项目目录是 `~/sparse_linear/`，其中已有本仓库和 `flash-linear-attention/`。先把每次运行的代码版本固定下来；同一文件不要在本地和远程同时编辑。

## 操作清单

1. 在本地仓库执行 `git status --short --branch`，确认要运行的改动已 commit 或明确记录为未提交。
2. 登录 `ssh 107` 后，查看 `~/job.sh`，把其中必要的 partition/account/module 约束作为本站点事实；不要照抄网上的 Slurm 参数。
3. 从 `~/sparse_linear/` 提交一个很短的 GPU job。默认申请不超过 4 CPU、1 GPU、4 小时；第一次 smoke test 只给很短时间。
4. **在 GPU job 内**记录 `nvidia-smi`、`python --version`、`torch.__version__`、`torch.version.cuda`、`git rev-parse HEAD`。登录节点的输出不能替代这一步。
5. 把不含敏感/大数据的环境摘要填入实验记录；原始长日志放在远程 `logs/` 或 `artifacts/`（已被 Git 忽略）。

## 环境策略

- 节点 GPU 可能是 A100 80G 或 5090，先检测再决定 PyTorch/CUDA 轮子；不要在没有分配 GPU 时猜版本。
- 先在独立 conda/venv 中安装，避免污染 base；安装命令和版本写入后续的 requirements/lock 文件。
- 对 FLA 等作者仓库，先跑其最小测试或 import，再碰 benchmark。若安装失败，先记录 driver/CUDA/Python/commit，不要连续换十个版本。
- 大模型权重、HF cache、checkpoint、dataset、profiler dump 一律不进 Git。需要在本地和远程同步代码时，先 commit/pull 并检查工作树，避免同步生成物和权重。

## 交付物与通过条件

创建一条实验记录，至少包含：作业号、GPU 名称、软件版本、当前 commit、提交命令、作业输出路径。能够解释为什么它属于“实验配置的一部分”，即通过本 lab。

下一步不是安装所有依赖，而是完成[第 1 章](../chapters/01-transformer-refresh.md)并选择一个 L1 dense reference 目标。
