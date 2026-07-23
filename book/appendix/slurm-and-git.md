# Slurm 与远程协作：让集群实验可追踪

## 107 的已知约束

根据你的作业模板，默认最多申请 4 CPU、1 GPU、最长 4 小时；账户/分区/QoS 为集群本地约束。GPU 节点可能是 A100 80G 或 5090，所以 **在 job 内** 检测 driver、CUDA、PyTorch，而不是在登录节点猜测。

集群链分成五个短作业，失败时能明确卡在哪一层：

| 脚本 | 作用 | 是否安装依赖 |
| --- | --- | --- |
| `discover-gpu.sbatch` | GPU/driver/Python/commit | 否 |
| `setup-env.sbatch` | 用 Conda 建独立 Python 3.12 prefix，安装匹配的 torch/Triton/FLA | 是 |
| `slurm-smoke.sbatch` | 教学测试、Triton、dense prefill/decode | 否 |
| `fla-smoke.sbatch` | FLA GDN/KDA 小 shape 正确性 probe | 否 |
| `profile-attention.sbatch` | 工具探测、PyTorch trace；按需尝试 nsys/ncu | 否 |

提交前在远程项目根目录执行：

```bash
mkdir -p artifacts
sbatch cluster/discover-gpu.sbatch
```

不要在 discovery 前把 CUDA wheel 写死。完整命令链见 [Lab 6](../labs/06-fla-on-107.md)。

!!! warning "不要在登录节点做 GPU 实验"
    登录节点只用于编辑、Git 和提交作业。任何 `torch.cuda`、FLA benchmark、模型加载，都放到 `sbatch` 分配的节点中。

## 107 的网络文件系统偶发不同步

107 的用户目录位于网络文件系统。若刚拉取/生成的文件没有出现、`git status` 与预期不符，
或程序报告一个明明已经修正的旧文件，先在当前目录执行：

```bash
cd .
```

然后重新运行最小只读检查，例如 `pwd`、`git status`、`ls` 或 `stat <file>`。只有刷新后仍能
复现，才继续排查 Git、Python import cache、环境或代码。不要因为一次目录视图陈旧就重复
clone、强制 reset 或删除环境；那会把 NFS 可见性问题扩大成真正的数据冲突。

## 本地与远程同步规则

1. 运行前：`git status` 必须干净，或者在实验记录中写明未提交改动；
2. 远程拉取/切换前：先提交本地笔记和代码，避免两个副本修改同一文件；
3. 权重、HF cache、dataset、profile dump 放远程目录，不进 Git；
4. 回传 Git 的只有脚本、config、小型 JSON/CSV 汇总、Markdown 结论；
5. 每张表必须写本仓库 commit、FLA/作者仓库 commit、job id。

推荐在 107 上保留原 clone 不动，用 worktree 跑教程分支：

```bash
cd ~/sparse_linear/sparse-linear-attention
git fetch origin codex/tutorial-book
git worktree add ~/sparse_linear/sla-tutorial-run origin/codex/tutorial-book
cd ~/sparse_linear/sla-tutorial-run
```

## 为什么不在第一天安装所有东西

不同 GPU 节点可能需要不同的 PyTorch/CUDA 组合。第一份 GPU job 只收集环境事实；第二次才在独立 conda/venv 内安装与节点相匹配的依赖。这样排错时能知道是代码、GPU、driver 还是 wheel 的问题。
