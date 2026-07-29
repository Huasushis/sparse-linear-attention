# 第 16 章：从阅读到可复核复现——你的研究路线与交付物

你现在不需要“把 74 篇论文读完”，也不需要“复现一个 48B 模型”。真正的目标是逐步建立一种
研究能力：看到一个 attention 论文时，能把它拆成可验证的公式、代码、性能和质量主张；
看到一个速度数字时，能判断它是否回答了你的问题。

这章不给死时间表。每一关都有一个**通过条件**和一个小产物；通过再进入下一关，没通过就
回到最靠前的依赖补洞。这样既不会把自己推得太紧，也不会因为“还没看完全部”而停住。

## 学习目标

读完后，你应当能够：

1. 解释“复现”至少分哪些层，为什么 install 成功不等于复现；
2. 按当前基础选择一条线性主线和一条 sparse 主线；
3. 为 A/B/C 级论文分别安排慢读、重点略读和脉络浏览；
4. 从小 reference 到 cluster benchmark 逐级增加复杂度；
5. 把每次阅读/实验沉淀成报告可用的证据，而不是零散聊天记录；
6. 在需要帮助时提供足够上下文，让助教式反馈能真正定位问题。

## 16.1 复现不是一个按钮，而是一组可声明的主张

当别人说“我复现了某篇论文”，先追问他复现了哪一级：

| 等级 | 名称 | 你实际证明了什么 | 典型产物 |
| --- | --- | --- | --- |
| R0 | 读懂主张 | 知道问题、改动、证据和边界 | 四句论文卡 |
| R1 | 公式复现 | 小张量上的数学定义正确 | 手算 + independent reference |
| R2 | 算子复现 | 自己/作者实现与 reference、梯度一致 | 单测、误差表 |
| R3 | kernel 性能复现 | 特定 GPU/shape 上有可重跑速度证据 | benchmark 表/曲线 |
| R4 | layer/model 复现 | 模型接口、缓存、一个小质量指标可工作 | integration test、小评测 |
| R5 | 论文级结果 | 接近作者完整训练/服务设置与质量结果 | 完整配置、资源、表格 |

对现在的研究，最有价值的目标通常是 **R1--R3**，偶尔到 R4。R5 需要数据、训练 token、
模型规模、工程基础设施和预算；它不是大一入门阶段是否“理解 attention”的标准。

### 把复现卡写得诚实

不要写“复现 Kimi Linear”。更准确的写法可能是：

```text
R1：验证 GDN 是 KDA 的 scalar-gate special case；
R2：在 FLA commit <hash> 上通过 KDA recurrent/chunk 的指定测试；
R3：在 A100、BF16、B=..., T=... 上比较 KDA/GDN 的 forward latency；
未做：48B 模型训练、1.4T-token 质量与 RL 结果。
```

这种边界不是“做得少”，而是可被他人相信的研究记录。

## 16.2 两条主线，交替推进

你的重点是 algorithmic + kernel，因此不要按文献目录从头到底读。采用下面两条互相照亮的
主线：

```text
线性主线
dense/Flash baseline
  -> kernelized linear state
  -> GLA
  -> DeltaNet
  -> Gated DeltaNet
  -> KDA / Kimi Linear
  -> FLA operator/kernel

sparse 主线
dense/Flash baseline
  -> fixed structured mask
  -> dynamic selector
  -> block sparse kernel
  -> KV/serving
  -> MInference / NSA / MoBA / SpargeAttention
```

两条线共同拥有 dense reference、GPU 性能模型、benchmark 纪律和 Slurm 工作流。不要先在
其中一条读到尽头再开始另一条：线性 state 会帮助你理解“固定容量记忆”，sparse KV 会帮助
你理解“保留哪些历史”；二者的取舍在长上下文研究中经常并列出现。

## 16.3 关卡式路径：每步只新增一个困难

### Gate A：Transformer、训练与 GPU 词汇够用

**阅读：** 第 1--6 章；先重看 3Blue1Brown Transformer/attention 视频。

**实践：** [Lab 0：环境](../labs/00-environment.md)、[Lab 1：dense attention](../labs/01-dense-attention.md)。

**产物：** 一页形状图、一个 dense reference、一次正确的 GPU 计时。

**通过：** 能解释 `QK^T`、causal softmax、prefill/decode、HBM/片上复用与异步计时。

这一步不要求你会 Adam 推导或 CUDA 汇编；只要能在论文里认出训练、forward、backward 和
inference 指标分别在说什么。

### Gate B：先有可信 dense/Flash 基线

**阅读：** 第 7--9 章，慢读 FlashAttention/FlashAttention-2 的 online softmax 与 IO 动机。

**实践：** 用 `tutorial_code/reference/dense_attention.py`、练习和 benchmark 框架完成
dense correctness + timing；之后可进入[Lab 4：GPU benchmark](../labs/04-gpu-benchmark.md)。

**产物：** 一张注明 GPU、dtype、shape、mode 的 dense 表。

**通过：** 知道“exact dense”与“没有 materialize `T×T`”可以同时成立，且不会用 CPU
`time.time()` 给异步 GPU 计时。

### Gate C：线性 state 与递推

**阅读：** 第 10--12 章；慢读 `katharopoulos2020transformers`、`yang2024gated`、
`yang2024delta`、`yang2025gated`。

**实践：** [Lab 2：linear attention](../labs/02-linear-attention.md) 和
[Lab 3：GDN/KDA](../labs/03-gdn-kda.md)。

**产物：** linear state reference、GDN/KDA scalarization test、一次分块边界 state 对照。

**通过：** 你能同时写出 scalar GDN 与 vector-gate KDA，并用数值而非口头证明退化关系。

### Gate D：读懂一个真实 kernel 仓库

**阅读：** 第 13 章；选 FLA 中 **KDA 或 GDN 一个**路径，不要两条并行硬啃。

**实践：** [Lab 5：Triton](../labs/05-triton.md)，之后才做
[Lab 6：FLA 与 107](../labs/06-fla-on-107.md)。

**产物：** FLA dataflow card、选中 op 的 test 结果、一个小 operator benchmark。

**通过：** 能从 layer 找到 op/naive/chunk/test，能指出你的运行处在 R1、R2 还是 R3。

### Gate E：Sparse 的语义与内核分层

**阅读：** 第 14--15 章；慢读 Longformer/BigBird 之一、MInference，以及 NSA/MoBA/
SpargeAttention 三选一。

**实践：** 先完成 [Lab 7：sparse mask 与 operator](../labs/07-sparse-mask.md)；再选择一个真实
sparse kernel 项目做 R2/R3。

**产物：** mask 图、七问卡、L0--L3 benchmark 设计、一份选题理由。

**通过：** 能说清 selector 是否计时、是否 prefill/decode、block 粒度和 quality 证据。

每个 Gate 的产物都能直接进入报告；不是“先学习、最后才写报告”的两套工作。

## 16.4 A/B/C 论文要读到什么程度

你已有的 74 篇分类不应被扔掉。把它们转成三种阅读速度：

| 级别 | 目标 | 阅读动作 | 当前代表 |
| --- | --- | --- | --- |
| A：慢读 | 可讲清、可写 reference/复现卡 | 摘要/图/公式/算法/实现/实验设置都读；至少一项 R1/R2/R3 | FlashAttention、GLA、DeltaNet、GDN、Kimi、FlexAttention、FlashInfer、MInference、NSA、MoBA、SpargeAttention |
| B：重点略读 | 建立分类和比较能力 | 摘要、图 1、方法图、实验设置/结论；写七问卡 | Longformer、BigBird、StreamingLLM、H2O、QUEST、SparQ、LServe、SeerAttention、XAttention、AdaSplash、HiLS-Attention |
| C：脉络浏览 | 知道历史分支和边界 | 摘要 + 贡献 + 它属于哪类；不强求公式 | 早期 accelerator、其余 selector、量化与视频分支 |

一篇论文可从 B 升为 A。例如你若选择“dynamic block selector 的 kernel 开销”为课题，
MInference、MoBA、NSA、SpargeAttention 中相关的一篇就必须升为 A；其他依旧 B。
HiLS-Attention 同样遵循这个规则：先用第 14 章的教师示范建立位置；只有在你的选题落到
learned chunk selector、query packing 或 native sparse training，并跑通独立算子证据后才升 A1。

`B*` 是另一种标签：它表示工程博客、代码仓库或上游 RFC，不是论文等级。ReplaySSM
属于这一类。它值得读，因为它把 GDN/Mamba-2 的递推、HBM traffic、Triton、CUDA Graph
和 speculative serving 串成一条可验证的因果链；但它不应挤占 sparse mask 的主线，也不应
把作者在 H100/B300 上的数字写成你在 107 上已经复现的结果。

## 16.5 每篇论文的三遍读法

### 第一遍：20 分钟建立地图

只读 abstract、introduction 的问题段、图 1、方法总图、结论和实验表标题。回答四句：

```text
问题：旧方法在什么场景卡住？
改动：它改了 attention 的状态、mask、selector 还是访存？
代价：它可能引入什么近似、训练、索引或硬件条件？
证据：作者在什么模型/GPU/阶段上报告什么？
```

此时遇到不懂的符号只圈起来，不要把 20 分钟变成三小时百科搜索。

### 第二遍：慢读一个“因果链”

对 A 级论文，选一条完整链而非平均地看所有页：

```text
问题 -> 一个核心公式/算法 -> 一个实现机制 -> 一张关键实验图 -> 一个局限
```

例如读 KDA 时可选“scalar GDN gate -> diagonal gate -> constrained DPLR -> chunkwise
kernel -> KDA/GDN-H TPOT 图”；读 MInference 时可选“attention pattern -> head pattern
assignment -> index construction -> sparse prefill kernel -> 1M prefill 图”。

第二遍结束应能自己重画一张不含原论文图片的流程图，并标出每个张量/状态的形状。

### 第三遍：证据读，为复现服务

只有决定做 R1--R4 时才进入第三遍。此时专门定位：

- exact formula、初始化、mask/position encoding 和归一化；
- code release、commit、license、模型/权重是否可得；
- GPU、CUDA/PyTorch、dtype、batch、长度、warmup、计时边界；
- baseline 是否真的相同，selector/index 是否计时；
- 训练 token、data、optimizer、评测脚本和随机性；
- 作者没有报告的地方。

把信息填入仓库 `study/templates/` 中的 `paper-note.md` 与 `reproduction-card.md`；书站中
不直接跳转这些根目录模板。这里的目标不是写长摘要，而是让下周的你能知道这次到底运行了
什么。

## 16.6 研究笔记怎样自然长成报告

每个 Gate 结束后，给报告增加一小节，而不是留到最后：

| 完成的关卡 | 报告中增加什么 | 证据来源 |
| --- | --- | --- |
| A | 背景、术语、dense 为什么贵 | 形状图、复杂度推导 |
| B | exact dense/Flash baseline | benchmark 表、计时方法 |
| C | linear attention 分类与 GDN/KDA 原理 | 统一递推、scalarization test |
| D | kernel 实现路线 | FLA dataflow card、test/benchmark |
| E | sparse taxonomy 与选题 | mask 图、七问卡、L0--L3 计划 |
| 后续 R3/R4 | 实验结果、局限、失败分析 | config、日志、图表、commit |

原始报告框架在仓库 `study/templates/report-outline.md`。保留第一版的解释，即使它不成熟；
后续阅读后再添加“我原先以为 X，现在发现 Y，因为 Z”的修订。这种变化本身就是调研的
价值。

## 16.7 从教程进入你的第一个小研究题

完成 Gate E 后，不要再增加十个新论文方向。选一个问题，限制变量，做到一份有边界的
结论。以下不是命题，而是可选的起跑线：

| 方向 | 最小问题 | 首先读/跑什么 | 成功的最低证据 |
| --- | --- | --- | --- |
| KDA/GDN kernel | channel gate 的实际 overhead 在何种 shape 出现？ | FLA KDA/GDN，scalarization + op benchmark | 同 GPU 的曲线与误差表 |
| structured sparse | 哪种 block/window 密度能越过 dense baseline？ | fixed-mask reference、FlexAttention 或可用 kernel | density/latency crossover 图 |
| dynamic selector | selector 是否吞掉 sparse kernel 的收益？ | MInference/MoBA/HiLS 类流程的简化版本 | selector 与 attention 分项计时 |
| KV serving | 某 cache policy 如何改变 TPOT 与质量？ | StreamingLLM/H2O/QUEST 一条线 | 固定请求下的 TTFT/TPOT/质量表 |
| recurrent-state serving（可选） | 不写回完整 SSM state 是否能降低 decode 的 memory traffic？ | ReplaySSM：小 ring buffer + output-only/flush | toy correctness、buffer sweep、kernel/E2E 分项计时 |

先选择你能获得的代码、GPU 和模型权重所支持的一项。研究问题应由可验证资源约束，而不是
由论文标题最酷决定。

## 16.8 在 107 上做实验的最小闭环

集群不是“更快的本地电脑”，而是另一台需要记录状态的实验环境。你的闭环应该是：

```text
本地：小 reference / 测试通过
  -> commit（只提交代码、配置、文档）
  -> 107：拉取同一 commit
  -> 短 smoke job 确认 GPU/CUDA/PyTorch
  -> 一次小 benchmark
  -> 保存结果表和环境信息
  -> 提交总结结果，不提交权重/cache/profiler dump
```

默认 4 CPU、1 GPU、4 小时的作业配额非常适合 correctness test、microbenchmark、短推理和
小型 layer 实验；它不适合从零预训练大模型。具体 Slurm/Git 规则见
[集群与 Git 附录](../appendix/slurm-and-git.md)和[Lab 6](../labs/06-fla-on-107.md)。

特别注意本地仓库和 `~/sparse_linear/` 的远程 clone 不要同时编辑同一文件。远程机器上的
checkpoint、Hugging Face cache、编译产物和大数据只留在用户目录/作业目录；Git 只保存可复
跑的脚本、配置、小结果和文档。

## 16.9 出错时的诊断顺序

面对“结果不对/速度没有提升”，按下面顺序排查，效率最高：

1. **定义：** dense/sparse/linear 是否真的在计算同一个目标？mask、causal、position、
   normalization、更新前后 state 是否一致？
2. **小张量：** `B=H=1,T<=8` 是否能手算并与 reference 对齐？
3. **数值：** float64/float32 是否正确，BF16 容差是否有依据？
4. **集成：** projection、reshape、GQA、cache、output gate 是否改变了语义？
5. **环境：** commit、GPU、driver、CUDA、PyTorch、Triton/FLA 是否记录？
6. **计时：** 是否 warmup/synchronize，是否把 selector/JIT/数据搬运算入？
7. **性能模型：** 这是 compute、bandwidth、launch、load balance 还是 memory-bound 问题？

不要从第 7 步跳回第 1 步。很多“kernel 比论文慢”的根因是 dtype、shape、错误计时或
未通过 correctness，而不是 tile 参数不够神奇。

## 16.10 你如何把学习内容交给助教式反馈

当你读完一节/跑完一次实验后，直接提供下面信息，反馈会非常具体：

```text
我在做：第 __ 章 / 论文 __ / Lab __
我的四句理解：问题 __；改动 __；代价 __；证据 __
我卡住的对象：公式 / 张量形状 / 一段代码 / 一张图 / 报错 / 速度结果
我已经检查：__
我希望得到：提示、逐步讲解、答案核对、代码审阅、还是实验设计建议？
```

对于代码问题，附上最小可复现片段、完整 traceback、命令、shape/dtype/device 和预期结果；
对于论文问题，附页码/公式号/截图或原句。这样我可以像课程助教一样指出前置缺口、让你
先自己补一个关键步骤，再核对你的推理，而不是替你把作业答案写完。

## 16.11 第一份阶段性研究包

完成前述关卡后，你应当拥有下面这些小而完整的东西：

```text
1. Transformer/dense/Flash 的一页背景说明
2. 一张 dense benchmark 表（有环境与方法）
3. 一张 Linear/GLA/Delta/GDN/KDA 的统一递推图
4. 一个通过测试的 GDN/KDA reference 或 scalarization test
5. 一张 FLA dataflow card 与一次 op benchmark
6. 一张 sparse taxonomy 图和一张固定 block mask 图
7. 一篇 A 级 sparse 论文的七问卡
8. 一个可证伪的后续优化问题 + L0--L3 实验设计
```

这些比“读过 74 篇”更有价值：它们能被导师检查、能被未来的你复跑，也能自然拼成调研与
复现报告的初稿。

## 常见误区

**误区 1：先学完所有前置再开始。**
前置知识永远学不完。用 Gate 的通过条件决定何时继续；遇到真正阻塞再回补。

**误区 2：论文笔记等于中文翻译。**
好的笔记写问题、改动、代价、证据和你的疑问；不是逐段转述 abstract。

**误区 3：只跑作者脚本。**
作者脚本是重要证据，但不能取代你的 independent reference、公平性检查和配置记录。

**误区 4：一次改多个变量。**
性能实验里同时改 block size、dtype、batch、selector 和 kernel 后，即使更快也不知道原因。

**误区 5：失败说明不适合研究。**
一个明确记录的失败（例如 selector 成本超过收益）本身就是结果；未记录的失败才没有价值。

## 最后练习：写下你的第一步

不要安排日期，只写下下一次学习要完成的一个 Gate 内动作。例如：

> 我会看完 3Blue1Brown 的 attention 部分，用 `B=1,H=1,T=4,D=2` 手画 `QK^T` 和 causal
> mask，完成 Lab 1 的 reference 测试，并写四句我的理解。

完成后保留你的原话，再回来对照本书。若你能指出哪一句不准确以及为什么，说明你已经在
真正学习，而不是被动收集名词。

## 通过条件

本书第一阶段完成的标志不是“全书读完”，而是你能：

- 为自己当前工作声明一个 R0--R3 级别及其未做边界；
- 选择一条 A 级线性论文和一条 A 级 sparse 论文；
- 在本地完成一个 reference，再用 107 的小作业复跑一个明确 benchmark；
- 用四句论文卡和七问卡说明你要研究的下一个问题；
- 带着日志、图、公式或代码片段来讨论，而不需要假装自己已经全懂。

到这里，规划已变成一门可以一步步完成的课程。下一次从 Gate A 的一个小动作开始即可。
