# Sparse & Linear Attention：研究教程、代码与复现实验台

这个仓库是一部可构建的网站式教程，不是一份一次性的论文清单。它从 Transformer 的张量形状、训练最小闭环和 GPU 性能模型开始，逐步走到 FlashAttention、linear attention、GDN/KDA、sparse attention、FLA 与可复跑 benchmark。

核心原则是：**先会解释，后会运行；先有 dense baseline，后谈加速；每个结论都留下可复核的证据。**

## 阅读入口

- 网站内容位于 [`book/`](book/)；导航配置在 [`mkdocs.yml`](mkdocs.yml)。
- 从 [`book/index.md`](book/index.md) 或 [`book/00-how-to-use.md`](book/00-how-to-use.md) 开始。
- 研究任务表、73 篇分级阅读图和长版模板仍保留在 [`study/`](study/)。
- 原始、已分类的 73 篇 BibTeX 位于 [`references/attention.bib`](references/attention.bib)。

## 本地构建在线书

文档依赖和模型/训练环境隔离；不要把 MkDocs 安装进未来的 CUDA/PyTorch 环境。

### Windows PowerShell

```powershell
python -m venv .venv-docs
.\.venv-docs\Scripts\python -m pip install -r requirements-docs.txt
.\.venv-docs\Scripts\python -m mkdocs serve
```

### Linux / 107

```bash
python -m venv .venv-docs
. .venv-docs/bin/activate
pip install -r requirements-docs.txt
mkdocs serve
```

打开终端显示的本地地址即可阅读。静态检查使用：

```bash
python -m mkdocs build --strict
```

## 先跑的代码

这些命令不需要 GPU；它们验证数学 reference 和教程中的 KDA/GDN 关系。

```bash
python -m pytest tutorial_code/tests -q
python -m tutorial_code.scripts.gdn_kda_probe
python -m tutorial_code.benchmarks.benchmark_attention --device cpu --seq-len 128
```

`tutorial_code/reference/` 是正确性 oracle，故意不追求快；`tutorial_code/exercises/` 留有可填写的 TODO；`tutorial_code/benchmarks/` 统一处理 warm-up、同步和分位数。

## 107 上的顺序

1. 先提交 `cluster/discover-gpu.sbatch`，在真实 GPU node 记录 GPU、driver 和基础 Python；
2. 根据该节点选择 PyTorch wheel，提交 `cluster/setup-env.sbatch` 创建独立 Python 3.12 环境；
3. 提交 `cluster/slurm-smoke.sbatch`，运行 reference、Triton 和 prefill/decode benchmark；
4. 提交 `cluster/fla-smoke.sbatch`，运行 FLA GDN/KDA operator probe；
5. 最后才进入模型级或现代 sparse 方法复现。

不要将模型权重、HF cache、数据集、checkpoint 或 profiler dump 提交 Git。提交的是代码、配置、短摘要表和结论。
