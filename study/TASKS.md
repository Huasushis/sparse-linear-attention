# 任务表（按关卡推进，无固定时间表）

状态说明：`[ ]` 未开始，`[-]` 进行中，`[x]` 已完成。只有在“通过条件”满足后再勾选，不以读了多少页作为完成标准。

## P0：定向与研究方法（现在）

- [ ] 阅读[第 0 章：怎样把学习变成研究证据](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/chapters/00-how-to-study.md)。
- [ ] 重看 3Blue1Brown 的 Transformer / attention 视频；完成[第 1 章](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/chapters/01-transformer-refresh.md)的 6 个检查题。
- [ ] 用[论文笔记模板](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/templates/paper-note.md)给 `dao2022flashattention` 或 `katharopoulos2020transformers` 写第一张“只含摘要、图 1、结论”的笔记。
- [ ] 读懂[复现规范](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/REPRODUCTION.md)中 L0--L4 的区别，并为第一张复现卡选一个 L1 目标。

**通过条件：** 你能不看资料解释 `Q/K/V`、causal mask、prefill/decode、`O(T^2)`，并能说出“算法快”和“GPU 实际跑得快”不是同一句话。

## P1：补齐最小前置知识

- [ ] 完成 dense attention 的形状推导和手算例子。
- [ ] 了解最小训练闭环：forward、loss、backward、optimizer；不要求先会训练大模型。
- [ ] 对比 RNN 的递归状态与 linear attention 的递归/scan 形式。
- [ ] 学会 GPU 的线程层次、显存层次、带宽、算力和同步的基本作用。
- [ ] 完成 [Lab 0：研究工作区](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/labs/lab00-research-workspace.md)；在一次 Slurm 作业中保存环境清单。

**通过条件：** 能画出一次 causal attention 的数据形状，说明训练与推理为什么需要的测试不同，并提交一份可复跑的环境记录。

## P2：建立 dense-attention 性能基线

- [ ] 精读 FlashAttention 与 FlashAttention-2（阅读图中的 A 级）。
- [ ] 实现一个仅用于正确性验证的 dense attention reference（L1）。
- [ ] 比较 reference、PyTorch SDPA 和可用的 FlashAttention 后端（L2）。
- [ ] 记录不同 `B, T, H, D, dtype` 下的延迟、显存和数值误差。
- [ ] 完成 Lab 4B：先用 PyTorch Profiler/`nsys` 解释一个 shape；若 `ncu` 权限可用，再只抓一个关键 kernel。

**通过条件：** 一张 benchmark 表能回答“哪个形状下谁快”，并注明 GPU、PyTorch/CUDA、
精度和同步方法；另有一条 profiler 证据支持“为什么”，且没有把插桩时间当主结果。

## P3：Linear attention 的算法主线

- [ ] 精读 `katharopoulos2020transformers`，写出 parallel 与 recurrent 两种形式的等价关系。
- [ ] 再从 RetNet、GLA、DeltaNet 学习 chunkwise：它如何在块内并行、在块间传递 state。
- [ ] 精读 GLA、DeltaNet、Gated DeltaNet 和 Kimi Linear；Mamba-2/SSD 在理解 GDN 后按需回补。
- [ ] 为每篇 A0 必读论文写一页笔记：状态是什么、如何并行、相对 dense 的代价、质量代价、作者如何测量。
- [ ] 完成一个小型 linear-attention operator 的 L1/L2 复现，不涉及完整预训练。

**通过条件：** 能用自己的图解释“线性复杂度”并不自动保证更快，以及 gating / delta rule 分别试图弥补什么能力缺口。

## P4：Linear-attention kernel 与 FLA

- [ ] 在隔离环境中运行 FLA 的最小测试和一个现有 benchmark；固定其 commit。
- [ ] 重点阅读 FLA 中 GLA / Gated DeltaNet 的 layer、ops、tests、benchmarks 的对应关系。
- [ ] 精读 Tiled Flash Linear Attention；略读 LASP/LASP-2，知道分布式序列并行解决的是什么。
- [ ] 用同一套形状对比 FLA 算子和 dense baseline；先记录差异，不急着优化。

**通过条件：** 可以追踪一个 FLA 算子从 layer 调用到 kernel / 测试 / benchmark，并能解释你测的到底是 forward、backward 还是 decode。

## P5：Sparse attention 的算法主线

- [ ] 对 Longformer、BigBird、Reformer 做结构化阅读；自己画出 local/global/random/LSH 等模式。
- [ ] 区分静态稀疏、内容自适应稀疏、KV-cache 选择和训练可学习稀疏。
- [ ] 精读 MInference；从 Native Sparse Attention、MoBA、SpargeAttention 中只选一个升级精读与复现。
- [ ] 对一个掩码模式做 L1 reference，并相对 dense attention 测数值/质量代理与速度。

**通过条件：** 能说明稀疏率、实际 speedup 和模型质量不是同一指标；能解释一个方法的选择开销是否吃掉收益。

## P6：Sparse kernel 与 serving

- [ ] 按选题精读 FlexAttention 或 FlashInfer；了解为何“任意 Python 掩码”不能直接等于高效 kernel。
- [ ] 选一个实际系统做小范围 L2/L3 复现：MInference、Native Sparse Attention、MoBA 或 SpargeAttention。
- [ ] 分开报告 prefill 和 decode；分开报告 operator microbenchmark 和端到端模型结果。
- [ ] 用 profiler 找到至少一个性能瓶颈，并写出“证据—假设—下一步”的记录。

**通过条件：** 实验记录能区分算法本身、掩码/索引开销、kernel、KV cache 与框架调度造成的时间。

## P7：形成调研与复现报告

- [ ] 按[报告模板](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/templates/report-outline.md)整理背景、分类、核心方法、实验和局限。
- [ ] 把每个图表关联到配置、commit 和实验记录。
- [ ] 写出 2--3 个可验证的后续 sparse-attention 性能优化问题，而不是泛泛地说“继续优化”。
- [ ] 请导师/同学追问一次：为什么选择这些 baseline、结果是否公平、结论边界在哪里。

**通过条件：** 一个陌生读者可根据仓库复跑关键表格，并准确知道你的结论适用于什么硬件、形状、模型和推理阶段。

## P8：后续可选支线

- [ ] 量化与 state reduction（分类 4）。
- [ ] 多 GPU / 序列并行（LASP、LASP-2）。
- [ ] Vision / Video Diffusion sparse attention（分类 9）。
- [ ] 针对观察到的瓶颈写 Triton kernel 原型。

这些支线在 P6 前不应抢占主线；它们需要更多模型、数据或多 GPU 条件。
