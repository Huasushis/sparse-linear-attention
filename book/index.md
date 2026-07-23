# Sparse & Linear Attention：从“看懂公式”到“跑出证据”

这是一本面向初学者、但不回避真实实现细节的研究教程。它的目标不是替你把 73 篇论文总结完，而是让你逐步获得三种能力：

1. 能从公式和张量形状解释一种 attention 方法；
2. 能区分算法上的省算力与 GPU 上实际更快；
3. 能把一个论文主张变成可复跑的正确性、性能或效果实验。

!!! tip "从哪里开始"
    先阅读[使用这本书](00-how-to-use.md)，然后完成第 1 章和 Lab 1。不要先安装 FLA、Triton 或下载大模型；那会让环境问题遮住真正要学的东西。

## 这本书的主线

```text
Transformer 张量与训练
          ↓
RNN / state / scan 与 GPU 性能模型
          ↓
Dense attention reference ──→ FlashAttention（exact baseline）
          ↓
Linear attention ──→ GLA / DeltaNet / GDN ──→ Kimi KDA
          ↓
Sparse masks / selectors ──→ kernel / serving / benchmark
          ↓
可复核的调研报告与后续优化问题
```

**先 dense，后 sparse / linear** 是刻意设计的：如果没有一个被验证且被计时的 dense baseline，就无法判断“加速”来自算法、GPU kernel、精度、输入形状，还是不公平的比较。

## 你会实际运行什么

仓库提供一组小而完整的教学代码，而不是要求你马上读百万行框架代码：

| 模块 | 你将运行/修改什么 | 它回答什么问题 |
| --- | --- | --- |
| `tutorial_code/reference/` | 正确但不追求快的 PyTorch 算子 | 公式究竟算出了什么？ |
| `tutorial_code/exercises/` | 留有 `TODO` 的小练习 | 你是否真的能把形状和公式写出来？ |
| `tutorial_code/tests/` | 自动正确性测试 | 改动后是否仍与 reference 一致？ |
| `tutorial_code/benchmarks/` | 可复跑的计时框架 | 哪个形状、阶段和精度下更快？ |
| `cluster/` | Slurm smoke-job 模板 | 如何在 107 的 GPU 节点上留下证据？ |

## 这本书不要求什么

- 不要求从零训练一个大语言模型；
- 不要求拥有 Blackwell GPU；
- 不要求先学完 CUDA、Triton、TVM、编译器；
- 不把“作者仓库 import 成功”当作复现成功。

你需要做的是：每到一个关卡，交出一个小而可检查的产物——一张形状图、一段 reference、一份测试结果、一张 benchmark 表，或一张论文复现卡。

## 当前研究重点

本书将 **algorithmic** 与 **kernel** 放在最深层：

- 算法层：state、kernel feature、gating、delta update、mask/selector；
- kernel 层：tile、布局、片上状态、访存、并行和数值稳定性；
- 系统层：prefill/decode、KV cache、batching 和 serving。

量化、分布式和视频生成不会被忽略，但在建立主线前只作为横向参照。
