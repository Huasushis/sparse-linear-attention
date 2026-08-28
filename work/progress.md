# 我的学习进度

> 这是唯一需要勾选的个人任务表。完成某项且满足对应通过条件后，把 `[ ]` 改为 `[x]`。不要在网站或 `study/TASKS.md` 记录进度。

**当前阶段：** P2

**下一步：** 修复并完成 Lab 4B profiler 运行，记录一条可证伪的性能观察；随后完成 Lab 5。

**当前卡点：** Lab 4B job `40083` 已成功生成 profiler trace；待根据 kernel 表填写瓶颈解释与下一次单变量比较。

完整路线、站内阅读链接、交付位置和通过条件统一见网站的“第 0 章：学习控制台”。

## P0：Transformer 刷新与研究方法

- [x] 重看 3Blue1Brown 的 Transformer / attention 视频，记录不懂的符号。
- [x] 阅读 [`book/part1/01-transformer-from-tensors.md`](../book/part1/01-transformer-from-tensors.md)。
- [x] 独立填写 [`notes/p0-transformer-refresh.md`](notes/p0-transformer-refresh.md)。
- [x] 不看资料解释 `Q/K/V`、causal mask、prefill/decode、$O(T^2)$，以及“计算量更小”为什么不保证 GPU 更快。

## P1：最小前置知识与环境

- [x] 完成第 2--6 章的必修部分；数学章节按需回看。
- [x] 完成 [Lab 0](../book/labs/00-environment.md)，填写 [`labs/lab00-environment.md`](labs/lab00-environment.md)。
- [x] 在 107 的 GPU job 中记录 GPU、driver、软件版本、commit、job id 和命令。

## P2：Dense attention 与性能地基

- [x] 完成第 7--9 章与 Lab 1，独立填写 dense attention TODO。
- [x] 完成 Lab 4；分别测 prefill/decode，留下包含完整配置的 benchmark 表。
- [x] 完成 Lab 4B；用 profiler 证据解释一个观察。
- [x] 完成 Lab 5 的 Triton vector-add TODO 与 grader。

## P3：Linear attention 算法

- [ ] 完成第 10--12 章与 Lab 2/3，独立填写 linear attention TODO。
- [ ] 写出 parallel/recurrent/chunkwise 的关系和 state 形状。
- [ ] 把 GDN/KDA 的重合与差异写成可检验命题。

## P4：Linear kernel 与 FLA

- [ ] 完成第 13 章与 Lab 6，在 107 固定 FLA commit 跑最小 smoke test。
- [ ] 只选一个 FLA operator，从 layer 追到 kernel、test 和 benchmark。
- [ ] 分清测量的是 forward、backward、prefill 还是 decode。

## P5：Sparse attention 算法

- [ ] 完成第 14 章与 Lab 7，独立填写 sparse attention TODO。
- [ ] 给 HiLS-Attention 做定位卡，写出结论适用边界。
- [ ] 从 NSA、MoBA、SpargeAttention 等候选中只选一个升级复现。

## P6：Sparse kernel 与 serving

- [ ] 完成第 15--16 章，对 P5 候选做受控 L2/L3 复现。
- [ ] 区分 operator/model-serving 与 prefill/decode 结果。
- [ ] 用 profiler 区分 selector、kernel、KV cache 与调度开销。

## P7：调研与复现报告

- [ ] 持续填写 [`report/draft.md`](report/draft.md)，把每个图表关联到证据记录。
- [ ] 统一 baseline、硬件、shape、dtype、计时方法和结论边界。
- [ ] 写出 2--3 个可以通过后续实验验证的性能优化问题。
