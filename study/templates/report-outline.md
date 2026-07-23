# Sparse and Linear Attention 调研与复现报告（生长式提纲）

## 摘要

（最后写：问题、范围、最可靠的实验结论、边界。）

## 1. 背景与问题定义

- Transformer causal self-attention 与 `O(T²)` 的计算/访存背景。
- prefill 与 decode 的不同；为何 dense exact baseline 必不可少。

## 2. 分类框架

- Linear attention：kernel feature / recurrence / gating / delta rule / SSD / hybrid。
- Sparse attention：固定模式 / 内容选择 / KV cache / hardware-aligned block sparse。
- Kernel 与 serving：算法主张如何变成 GPU 上可测的程序。

## 3. 核心方法

每个深读方法统一写：问题、机制图、复杂度、GPU 映射、实验主张、局限。

## 4. 实现与实验方法

- 硬件与软件版本、shape matrix、正确性 reference。
- baseline、公平性、计时、warm-up、统计量。

## 5. 结果

- dense baseline；linear operator/FLA；sparse operator/serving。
- 分开报告正确性、性能、效果；分开 prefill/decode。

## 6. 讨论与局限

- 哪些结论只在某 GPU、dtype、形状、模型或阶段成立？
- 选择开销、kernel 覆盖率、质量变化、训练成本有何影响？

## 7. 后续研究问题

每个问题必须写成可测形式：假设、变量、baseline、指标、所需资源。

## 附录

论文表、复现卡、实验记录、命令与配置、失败案例。
