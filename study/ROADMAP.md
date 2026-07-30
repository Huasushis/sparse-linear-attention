# 总路线：先理解，再测量，再复现，再提出问题

> 本页保留课程设计的背景与取舍理由，不再作为个人任务入口。当前路线、站内阅读、交付物与通过条件统一见 `book/start-here.md`；个人状态只在 `work/progress.md` 勾选。

## 终点与优先级

最终目标是一个可复核的 sparse / linear attention 调研与复现报告，而不是 74 篇论文的逐篇摘要。主线优先级为：

```
前置概念 → dense / FlashAttention 基线 → linear 算法 → linear kernel
       → sparse 算法 → sparse kernel / serving → 统一 benchmark → 研究问题
```

其中 **算法（ALG）** 和 **kernel（KER）** 是深水区；系统、部署、量化、分布式与视觉/视频是“理解接口与结论边界”的广度区。每一关结束时写下一个小产物，最后自然拼成报告。

## P0：先学会提出和记录问题

**学什么：** 一篇方法论文由 problem、idea、algorithm、implementation、experiment、claim 组成；图表不是结论本身，必须知道它的设置。

**做什么：** 先用摘要、图 1、引言结尾、结论做 20 分钟的第一遍阅读；把不懂的词记录为问题，不要立刻查到无穷深。

**留下什么：** 第一张[论文笔记](templates/paper-note.md)、一张[复现卡](templates/reproduction-card.md)。

## P1：最小前置知识包

这一阶段不追求完整机器学习课程，只补后续阅读会反复用到的“最小闭环”。

| 模块 | 必须会回答的问题 | 暂时不必深入 |
| --- | --- | --- |
| Transformer | `QK^T`、softmax、mask、`PV` 的形状和作用是什么？ | 多模态、RLHF、复杂位置编码 |
| 训练 | loss 如何使参数得到梯度，optimizer 做什么？ | 从零训练数十亿参数模型 |
| RNN / 状态 | 什么信息被压到 state，递归为何难并行？ | RNN 历史全谱系 |
| GPU | 哪些数据在显存/片上，带宽与算力何时限制速度？ | 特定 Blackwell 指令细节 |
| 工程 | 环境、commit、seed、config 为何是实验的一部分？ | 大型分布式训练平台 |

**建议顺序：** 先看 3Blue1Brown 视频，再做[第 1 章](chapters/01-transformer-refresh.md)，再看 PyTorch autograd；最后才开始 Triton 和 GPU kernel。

## P2：Dense attention 与 FlashAttention 是地基

任何 sparse 或 linear 方法都必须有密集基线。精读 FlashAttention / FlashAttention-2 时，不要只记“更快”：

- 算法仍是 exact attention，改变的是访存和中间量是否 materialize；
- 用 online softmax 保持数值正确；
- 关注 IO、work partition、shape、dtype 对性能的影响；
- 分别测 forward、forward+backward、prefill 和 decode。

**Lab 目标：** reference correctness → PyTorch SDPA → 可用 FlashAttention 后端。该 lab 产出的 benchmark 表会在后续所有实验复用。

## P3：Linear attention 的算法主线

按这个顺序读，而不是直接跳 Kimi：

1. `Transformers are RNNs`：核特征与 recurrent state 的原点；
2. GLA：gating 如何提升表达能力且考虑硬件；
3. DeltaNet：delta rule 与按 chunk 并行；
4. Mamba-2/SSD：把 state-space / attention 的关系讲清；
5. Gated DeltaNet：组合上面的关键想法；
6. Kimi Linear：一个实际大模型架构的综合案例。

**每篇深读都回答五件事：**

1. 它压缩或维护的 state 是什么，尺寸随序列长度增长吗？
2. 它的公式相对 dense attention 改了什么？是 exact、近似，还是换了模型族？
3. 训练时怎么并行，推理时怎么更新 state？
4. 算法的理论复杂度和真实 kernel 的瓶颈分别是什么？
5. 作者用什么 quality、speed、memory、length 设置支持主张？

## P4：Linear attention 的 kernel 与 FLA

FLA 应当被当作“可观察的实现标本库”，不是黑盒库。建议沿着一条短路径走：

```
layer API → operator dispatch → kernel 变体 → unit test → benchmark script → 表格
```

先选 GLA 或 Gated DeltaNet 的一条路径。读实现时的产物不是“看懂每行 Triton”，而是一张 dataflow 图：输入/输出形状、block/chunk、state、是否有 backward、哪些 buffer 被读写。然后再看 Tiled Flash Linear Attention，理解 tile、layout、reduction、parallelism 如何对应到算法。

## P5：Sparse attention 的算法主线

把 sparse 方法放进四个篮子，避免只按年份背论文：

| 篮子 | 关键问题 | 代表工作 |
| --- | --- | --- |
| 固定结构掩码 | 哪些 token 永远可见？ | Sparse Transformer、Longformer、BigBird |
| 内容选择 / 路由 | 谁来决定要看谁？选择成本是多少？ | Reformer、Routing、Quest、MInference |
| KV cache / 推理选择 | decode 时保留哪些历史？ | H2O、StreamingLLM、SparQ、DuoAttention |
| 可训练、硬件对齐的 block sparse | mask 能否既有效又被高效执行？ | Native Sparse Attention、MoBA、SpargeAttention |

此阶段至少要亲手做一个 structured mask 的 reference，才能理解“稀疏率”不是 kernel speedup。

## P6：Sparse kernel、serving 与端到端效果

FlexAttention 和 FlashInfer 解释了把 attention variant 变成可运行 GPU 程序的接口；MInference、Native Sparse Attention、MoBA、SpargeAttention 是更贴近实际部署/模型的候选复现。一次完整实验必须拆为两层：

- **operator 层：** 固定 `B,T,H,D,dtype`，验证正确性、延迟、显存；
- **model / serving 层：** 区分 prefill / decode，报告 token/s、TTFT 或延迟，以及任务质量/近似误差。

先执行作者代码，固定 commit；再做最小 reference；最后改一个变量。这样即使没有完整训练预算，也能得到可靠结论。

## P7：报告自然生长的方式

报告不要最后才写。每完成一个阶段，在[报告模板](templates/report-outline.md)填一小节：

- P1 填背景与术语；
- P2 填 dense baseline 和性能方法；
- P3/P5 填分类和原理图；
- P4/P6 填实现、实验和局限；
- P7 只做统一、删减和论证边界。

## MLC 课程与 Triton 的取舍

导师给的 MLC 教程很有价值，但其可运行代码目标是 **Blackwell / TIRx / FlashAttention-4**；没有 Blackwell 时不应把“跑通 TIRx”当作前置门槛。

现在选读（概念可迁移）：GPU 执行模型、Kernel 性能从何而来、数据布局、GEMM 的分块思路、FlashAttention 的 online softmax / causal mask / GQA。

暂缓实操（强绑定 Blackwell）：TMA、`tcgen05.mma`、TMEM、mbarrier、Cluster Launch Control、FA-4 的特定 pipeline。

实践语言优先级：**PyTorch 参考实现 → Triton → 阅读 CUDA/FLA 内核 → 以后按硬件需要接触 TileLang / TIRx。** Triton 更适合作为目前的第一门 kernel DSL；但“先会 Triton”不等于要跳过 GPU 性能模型。

## 集群的使用边界

- 在提交作业后再探测节点 GPU、driver、CUDA、PyTorch 可用性；不能把登录节点信息当成 GPU 节点信息。
- 默认 4 CPU / 1 GPU / 4 小时适合 microbenchmark、正确性测试、小型预训练模型推理；不适合从零复现大模型训练。
- 每次远程实验开始前保证 Git 工作树干净，记录 commit；本地与 `~/sparse_linear` 同步时不要并行编辑同一文件。
- checkpoint、HF cache、数据集和 profiler dump 放在用户目录/作业目录，不提交 Git；提交配置、脚本和汇总结果。
