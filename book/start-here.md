# 第 0 章：学习控制台（从这里开始）

!!! abstract "以后只用这一页决定下一步"
    这页把**学习路线、任务、作业位置和通过条件**放在了一起。网站是只读教材，网页上的勾选状态不会保存；你的唯一阶段进度表是仓库中的 `work/progress.md`。

    当前先做 **P0**。不要同时配置 FLA、读 Kimi、跑长上下文模型或开始 Triton。

旧任务表中的“第 0 章”原先指向 GitHub 上的旧资料页，因此会离开网站。现在**本页就是第 0 章**，后续章节链接也都在站内打开。

[在当前教程分支查看个人进度表](https://github.com/Huasushis/sparse-linear-attention/blob/codex/tutorial-book/work/progress.md) · [打开第一份 P0 作业](https://github.com/Huasushis/sparse-linear-attention/blob/codex/tutorial-book/work/notes/p0-transformer-refresh.md)

这两个链接用于从网站定位文件；真正作答时编辑本地仓库中的同名文件。

## 你现在要做什么

按顺序完成下面四步；做到哪一步就停在哪一步，不必一次做完。

1. 重看 3Blue1Brown 的 [But what is a GPT?](https://www.3blue1brown.com/lessons/gpt) 与 [Attention in transformers](https://www.3blue1brown.com/lessons/attention)。看到不懂的符号时只记问题，不要同时展开十条支线。
2. 慢读[第 1 章：从张量理解 Transformer](part1/01-transformer-from-tensors.md)，自己写出 `Q`、`K`、`V`、score、probability 和 output 的形状。
3. 打开 [`work/notes/p0-transformer-refresh.md`](https://github.com/Huasushis/sparse-linear-attention/blob/codex/tutorial-book/work/notes/p0-transformer-refresh.md)，填写其中的检查题。它是你的**第一份作业**。
4. 通过 P0 条件后，在 `work/progress.md` 把对应任务从 `[ ]` 改为 `[x]`，再进入 P1。

!!! info "当前分支"
    教程目前在 [`codex/tutorial-book`](https://github.com/Huasushis/sparse-linear-attention/tree/codex/tutorial-book) 分支。学习期间在这个已检出的分支上编辑即可；不要根据旧链接切回内容尚不完整的 `main`。本站的课程导航全部使用相对链接，不依赖 GitHub 分支名。

## 作业到底放在哪里

| 你产生的内容 | 唯一位置 | 说明 |
| --- | --- | --- |
| 阶段进度与下一步 | `work/progress.md` | **阶段完成状态只在这里更新**；Lab 内清单只用于自检 |
| 概念题、推导、章节笔记 | `work/notes/` | P0 的第一份作业已经建好 |
| 代码练习 | `tutorial_code/exercises/` | 填 `TODO`；不要改 reference 来让 grader 通过 |
| 论文笔记 | `work/papers/` | 一篇论文一个 Markdown 文件 |
| Lab 记录与回答 | `work/labs/` | 命令、环境、结果、失败和解释写在同一份记录里 |
| 小型实验结果摘要 | `work/runs/` | 提交小型 Markdown/CSV/JSON；原始大日志留在 `artifacts/` |
| 最终调研报告草稿 | `work/report/draft.md` | 从 P1 起逐步生长，不等到最后再写 |

`book/` 是教材，`tutorial_code/reference/` 是教师参考实现，`study/runs/` 是教师预跑记录。它们不是你的答题区。模型权重、数据集、完整 profiler trace、checkpoint 和大日志不要提交 Git。

## 每次学习只做一个小闭环

1. 在 `work/progress.md` 看“当前阶段”和“下一步”。
2. 从本页对应阶段选**一项阅读 + 一项动手任务**，不要并行开多个阶段。
3. 把答案或实验记录写进上表指定位置。
4. 运行该 Lab 给出的测试或检查命令。
5. 满足“通过条件”后再勾选，写下新的“下一步”。

卡住时保留未通过状态，并在作业末尾写清楚：卡在哪一步、实际看到什么、已经试过什么。这样我可以按证据帮你定位，而不是猜。

## P0：Transformer 刷新与研究方法（现在）

**学什么：** attention 的数据流；“读过”“运行过”和“验证过一个主张”的区别。

**读：** [课程规则](00-how-to-use.md)中“我们怎样分工”与“三种阅读速度”；[第 1 章](part1/01-transformer-from-tensors.md)。

**做：** 重看 3Blue1Brown；填写 `work/notes/p0-transformer-refresh.md`；暂时不写 kernel。

**交到：** `work/notes/p0-transformer-refresh.md`，并在 `work/progress.md` 更新状态。

**通过条件：** 不看资料也能解释 `Q/K/V`、causal mask、prefill/decode 和 $O(T^2)$；能说明“理论计算量更小”不自动等于“GPU 实测更快”。

## P1：最小前置知识与可复跑环境

**学什么：** 最小训练闭环、RNN/state/scan、必要数学、GPU 性能模型，以及实验为什么必须记录环境。

**读：** [第 2 章：训练最小闭环](part1/02-training-minimum.md)、[第 3 章：RNN、状态与 scan](part1/03-rnn-state-and-scan.md)、[第 4 章：数学工具](part1/04-math-toolkit.md)、[第 5 章：GPU 性能心智模型](part2/05-gpu-mental-model.md)、[第 6 章：可信 benchmark](part2/06-benchmarking-gpu.md)。数学章节按需回看，不要求先背完。

**做：** 完成 [Lab 0：环境与可复跑记录](labs/00-environment.md)。先在本地跑最小测试，再在 107 的 GPU 作业中记录节点、GPU、driver、Python、PyTorch/CUDA、commit 和 job id。

**交到：** `work/labs/lab00-environment.md`；小型运行摘要放 `work/runs/`。

**通过条件：** 能画出一次 causal attention 的形状流；能说清 forward、loss、backward、optimizer 的职责；别人可以根据你的记录重跑同一条命令。

## P2：Dense attention 与性能地基

**学什么：** dense reference、online softmax、FlashAttention 的 IO 思路、Triton 入门，以及“计时”和“解释瓶颈”是两件事。

**读：** [第 7 章：Dense attention reference](part2/07-dense-attention.md)、[第 8 章：FlashAttention](part2/08-flashattention.md)、[第 9 章：第一段 Triton kernel](part2/09-triton-first-kernel.md)。

**做：** 依次完成 [Lab 1](labs/01-dense-attention.md)、[Lab 4：benchmark](labs/04-gpu-benchmark.md)、[Lab 4B：profiling](labs/04b-profiling.md) 和 [Lab 5：Triton](labs/05-triton.md)。代码 TODO 在 `tutorial_code/exercises/01_dense_attention_todo.py` 与 `04_triton_vector_add_todo.py`。

**交到：** `work/labs/` 中的 Lab 记录；汇总表放 `work/runs/`。原始 `nsys`、`ncu`、trace 文件放 `artifacts/`，不提交 Git。

**通过条件：** benchmark 表标明 GPU、软件版本、shape、dtype、warmup、repeats 和同步方式；另有一条 profiler 证据解释一个观察，且 prefill/decode 分开报告。

## P3：Linear attention 的算法主线

**学什么：** parallel、recurrent、chunkwise 三种形式；gating、delta rule、GDN 和 KDA 各自改变什么。

**读：** [第 10 章：Linear attention](part3/10-linear-attention.md)、[第 11 章：Gating、Delta rule 与 GDN](part3/11-gating-delta-gdn.md)、[第 12 章：Kimi KDA 与 GDN](part3/12-kimi-kda-vs-gdn.md)。论文精读顺序从[论文阅读图](appendix/paper-map.md)的 A0 项中取，不要求同时读完全部论文。

**做：** 完成 [Lab 2](labs/02-linear-attention.md) 与 [Lab 3](labs/03-gdn-kda.md)；填写 `tutorial_code/exercises/02_linear_attention_todo.py`。

**交到：** 算法推导写 `work/notes/`，论文证据写 `work/papers/`，Lab 结果写 `work/labs/`。

**通过条件：** 能画出 recurrent state 的形状和更新；能解释线性复杂度为何不保证更快；能把“GDN 与 KDA 几乎重合”拆成可检验的相同点与差异点。

## P4：Linear kernel 与 FLA

**学什么：** 一个 layer 如何调用 operator、dispatch 到 kernel，并由 tests 与 benchmark 约束；chunk、tile、state 和 buffer 如何落到实现。

**读：** [第 13 章：读 FLA](part3/13-fla-code-map.md)。

**做：** 完成 [Lab 6：107 上的 FLA smoke test](labs/06-fla-on-107.md)。只选 GLA 或 Gated DeltaNet 的一条路径，从 API 追到 kernel、test 和 benchmark；ReplaySSM 只是可选练习。

**交到：** 调用路径图和回答放 `work/notes/`；环境、job id、结果和失败放 `work/labs/` 或 `work/runs/`。

**通过条件：** 能指出实测覆盖 forward、backward、prefill 还是 decode；能说明比较双方的 shape、dtype 和状态语义是否一致。

## P5：Sparse attention 的算法主线

**学什么：** 固定结构、内容选择、KV-cache 选择和可训练 block sparse 的区别；选择器成本与稀疏率的关系。

**读：** [第 14 章：Sparse 方法分类](part4/14-sparse-taxonomy.md)，并从[论文阅读图](appendix/paper-map.md)按 Longformer/BigBird/Reformer → MInference → 一个现代候选的顺序取舍。HiLS-Attention 先做定位卡，不要求下载 7B 权重或复现超长上下文结论。

**做：** 完成 [Lab 7：Sparse mask 与 operator](labs/07-sparse-mask.md)，填写 `tutorial_code/exercises/03_sparse_attention_todo.py`；再从 NSA、MoBA、SpargeAttention 等候选中只选一个升级复现。

**交到：** mask 图和分类笔记放 `work/notes/`；论文七问卡放 `work/papers/`；Lab 结果放 `work/labs/`。

**通过条件：** 能明确区分稀疏率、实际 speedup、近似误差与模型任务质量，并能指出 selector 的开销是否被计入。

## P6：Sparse kernel、serving 与端到端边界

**学什么：** 稀疏语义如何真正跳过 GPU 工作；prefill/decode、operator/model-serving 两个拆分维度；KV cache 与调度的影响。

**读：** [第 15 章：Sparse kernel 与 serving](part4/15-sparse-kernels-serving.md)、[第 16 章：从论文到复现卡](part4/16-reading-and-reproduction-path.md)。

**做：** 对 P5 选中的一个方法做受控 L2/L3 复现；用 [Lab 4B](labs/04b-profiling.md) 找到至少一个瓶颈，并记录“证据 → 假设 → 下一步”。

**交到：** 复现卡放 `work/papers/` 或 `work/labs/`；可提交的小结果放 `work/runs/`；大 trace 仍放 `artifacts/`。

**通过条件：** 结果能区分算法、selector/index、kernel、KV cache 与框架调度时间；比较使用相同且明确的输入与环境。

## P7：形成调研与复现报告

**学什么：** 如何把阶段证据组织成有边界的结论，而不是论文摘要拼接。

**做：** 从 P1 起持续填写 `work/report/draft.md`；P7 主要负责统一术语、图表、baseline、配置和引用，删除无法由证据支持的句子。

**交到：** `work/report/draft.md`，以及它引用的 `work/papers/`、`work/labs/`、`work/runs/` 记录。

**通过条件：** 陌生读者可以从报告中的 commit、环境、命令和配置重跑关键表格；每个结论都说明适用的硬件、shape、模型、精度和推理阶段。

## 暂时不要做的事

- 不要按 74 篇论文的编号从头到尾平均用力；按[论文阅读图](appendix/paper-map.md)的层级取舍。
- 不要把“作者仓库 import 成功”记作复现完成。
- 不要在登录节点测 GPU，也不要把登录节点环境当作计算节点环境。
- 不要先训练大模型；当前目标是小张量正确性、算子测量和受控复现。
- 不要为了让测试通过修改 `tutorial_code/reference/` 或 grader。
