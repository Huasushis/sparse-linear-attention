# 第 15 章：Sparse Kernel 与 Serving——“少算”怎样变成真实延迟

第 14 章已经定义了谁可以看谁。本章讨论更难的第二半：把这个 mask 变成真正运行在 GPU
上的程序，并在 LLM serving 中测到可解释的收益。

一句话概括本章：

> 一个 sparse 方法要同时赢过 dense baseline，必须让 **选择** 足够便宜、**保留配对**足够
> 规则、**GPU kernel**足够饱和，并且在 **prefill/decode** 的真实请求形态下仍保持质量。

这也是你以后做 sparse attention 性能优化时最常用的检查表。

## 学习目标

读完后，你应当能够：

1. 解释 block sparse 为何通常比任意 token-level sparsity 更容易加速；
2. 区分 mask 语义、sparse operator 和 serving system 三层；
3. 区分 prefill 与 decode 在稀疏化上的不同瓶颈；
4. 设计 dense reference、masked reference、sparse kernel 的三级正确性/性能对照；
5. 正确使用 TTFT、TPOT/ITL、token/s、显存等 serving 指标；
6. 将 FlexAttention、FlashInfer、MInference、NSA、MoBA、SpargeAttention 放到同一系统图中。

## 15.1 Dense FlashAttention 是 sparse 研究的地基，不是对手的稻草人

FlashAttention 类方法已经能对 dense softmax attention 做 IO-aware、fused、分块的 exact
计算。它可能不物化完整 attention matrix，却仍计算所有允许的 dense `QK` 和 `PV`。

所以 sparse kernel 不能把“与三行 PyTorch 的 dense reference 比较”当作最终胜利；至少还
需要一个可信的 dense SDPA/FlashAttention baseline。比较对象应随任务变化：

| 你要证明什么 | 合理的对照 |
| --- | --- |
| mask 语义正确 | masked dense PyTorch reference |
| sparse kernel 值得存在 | 同设备的优化 dense SDPA/FlashAttention |
| long prefill 系统收益 | 同模型/权重/请求的 dense prefill serving |
| decode cache 策略收益 | 同 batch/历史长度/调度器的 dense KV serving |

若 sparse 输出本来就不是 dense output，不能只报 `allclose`。还应报告任务质量、近似误差，
或把它明确称为“一个不同的 sparse 模型”。

## 15.2 三层模型：同一个方法可能跨三层

```text
语义层（algorithm）
  M / A(i)：哪些 q-k 配对可见？是否近似？是否需要训练？
        ↓
算子层（operator / kernel）
  indices、block layout、QK/softmax/PV、gather、backward 如何在 GPU 上执行？
        ↓
系统层（serving）
  KV cache、paged layout、batching、scheduler、prefill/decode、请求异质性如何处理？
```

这三层经常被一张“speedup”图混在一起。看图前先问它测的是哪一层：

- 把 `Q,K,V` 随机张量送进 kernel，多半是 **operator** benchmark；
- 处理一条 prompt 并报告 latency，多半是 **prefill system** benchmark；
- 连续生成 token、报告 TPOT/ITL，多半是 **decode serving** benchmark；
- 报 LongBench/RULER/困惑度，多半是 **模型/质量** evidence。

一篇好的研究可以覆盖三层，但你必须分别记录每层的设置和结论。

## 15.3 GPU 为什么不喜欢任意稀疏

GPU 喜欢成块、连续、可预测的工作；任意 token 级 mask 往往带来：

- 每行不同长度，warp 工作量不均衡；
- 索引跳跃，K/V 读取不 coalesced；
- 每个保留元素工作太少，难以发挥 Tensor Core；
- metadata/index 本身占带宽；
- softmax reduction 的行长度不规则，归一化和同步困难。

把 token 划为 `Bq×Bk` 的 attention block 后，可以让一个 GPU program/warp group 处理
一个或少数完整块：

```text
query block r  ->  保留的 key block ids [j1, j2, ...]
                   |         |
                   v         v
              [Bq,Bk] score tiles -> online softmax -> [Bq,Dv] output tile
```

block 比 token 粗，会牺牲部分理想 sparsity；但它能换来连续访问、规整 GEMM 和更好的负载
均衡。研究中的“hardware-aligned”通常就在这个折中上做文章。

### 一个实用性能公式

对于 block sparse attention，先不要数浮点数，先问：

$$
\text{time} \approx \max(\text{compute time},\;\text{HBM traffic time})
                  + \text{index/launch/sync overhead}.
$$

保留块太少时，计算减少但 launch/metadata 占比上升；保留块太多时，又接近 dense。最佳点
依赖 GPU、dtype、batch、head dimension、block size 和 mask 分布，无法只由 density 决定。

## 15.4 Prefill 与 decode：两种不同的 sparse 问题

| 阶段 | 输入形态 | 常见主瓶颈 | sparse 设计关注点 | 应报告指标 |
| --- | --- | --- | --- | --- |
| prefill | 整段 prompt 的许多 query 同时到来 | `T²` work、显存 IO、矩阵乘利用率 | 每 head/block 的模式、selector、块内并行 | latency、TTFT、throughput、质量 |
| decode | 每请求每层通常只有一个新 query | 长 KV cache 的带宽、请求异质性、batching | cache 保留/检索、paged layout、GQA、调度 | TPOT/ITL、token/s、p50/p95、显存 |

MInference 主要是 long-context prefill 的例子：先为 head 分配/构造稀疏模式和索引，再用
GPU sparse kernel 加速。StreamingLLM、H2O、QUEST、SparQ 等则更接近 decode/KV policy。
NSA、MoBA 可能跨训练与 inference，但每个阶段的结果仍应分开读。

**TTFT（time to first token）**通常包含请求排队、prefill 及系统开销；**TPOT/ITL
（time per output token / inter-token latency）**描述生成阶段每 token 的间隔。两者不能
互相替代。只报“tokens/s”也不足够，必须说明是 aggregate throughput 还是单请求速度。

## 15.5 从 mask 到 kernel：五个执行问题

### 1. mask 从哪里来

固定 mask 可在编译时/运行前生成；动态 mask 需要 selector。selector 的输出应是紧凑的
block metadata，而不是一个巨大、密集 `[T,T]` bool tensor，否则可能先吃掉内存与时间。

### 2. K/V 怎样布局

block id 如何映射到 cache page、连续内存或压缩存储？若每个被选 block 都要进行随机
gather，理论少算可能变成带宽随机读。GQA/MQA 会改变每个 key/value 被多少 query head
共享，也应在 layout 设计中显式处理。

### 3. softmax 怎样保持正确

对一个 query block，kernel 常以 online softmax 维护 running max 与 running sum，避免
物化所有 score。对于 sparse 模型，running reduction 只遍历允许 block；对于近似 dense
方法，它得到的是被选集合上的近似归一化。两种语义要分开写。

### 4. 如何均衡工作

若部分 query block 有 2 个 key block，另一些有 64 个，同一个 grid 的工作会很不均衡。
可通过 bucket、split/reduce、动态调度或固定预算缓解，但每一种都会增加 metadata/sync。

### 5. backward 怎么办

训练可用 sparse attention 不能只展示 forward。必须问 mask 是否可微、selector 怎样得到
梯度、saved tensors 多大、backward 有没有同样高效的 sparse kernel。NSA、SeerAttention、
AdaSplash-2 一类工作尤其要从这个角度阅读。

## 15.6 三个系统案例放到同一张图上

### FlexAttention：从“写一个变体”到“生成 fused attention”

FlexAttention 的问题是，attention 的 score/mask 变体组合很多，手写每个 fused kernel 不
可持续。它提供 compiler-driven 的编程模型，把一些 score 或 mask 修改表达为高层代码，再
生成优化 attention kernel。对你最有用的价值不是立刻把它当作 sparse 方法，而是学习：

```text
语义（score/mask modification） -> block-level schedule -> fused implementation
```

它适合成为固定 window/block mask 的第一个“语义正确后再看 kernel”实验入口。

### FlashInfer：从 op 到 LLM serving engine

FlashInfer 的摘要强调 block-sparse/composable KV-cache format、JIT attention template 和
对动态请求的 load-balanced scheduling。它提醒我们：一个很快的 attention kernel 若不能
处理 paged cache、variable length、batching 和服务框架接口，未必能给最终用户降低 latency。

因此读 FlashInfer 时重点问它如何处理“异质请求”和“静态 CUDA Graph 约束”，而不只是记
某个 kernel 加速百分比。

### MInference / NSA / MoBA / SpargeAttention：四种研究取舍

| 方法 | 从哪层切入 | 你应该重点审查 |
| --- | --- | --- |
| MInference | 预训练后 long prefill 的动态模式与 index | pattern/head 分配、index 构造、A100 设置 |
| NSA | 可训练、分层动态 sparse + hardware alignment | block/压缩设计、fwd/bwd/decode 是否都报告 |
| MoBA | attention block router | router 的选择单位、top-k/block layout、模型质量 |
| SpargeAttention | online filter + softmax-aware skip | filter 本身成本、误差、基础量化 kernel 的影响 |

它们都可能被称为“sparse attention”，但需要的复现证据完全不同。

## 15.7 一个可靠的 benchmark 阶梯

不要从端到端大模型开始。用同一 shape 逐层加复杂度：

```text
L0  dense definition reference
L1  masked dense reference（验证 sparse 语义）
L2  fixed-block sparse operator（验证 kernel 与真实跳过）
L3  dynamic selector + sparse operator（计入 selector/index）
L4  layer/model integration（记录质量或近似误差）
L5  serving prefill/decode（记录请求与 scheduler）
```

每升一级都保留上一级的输入种子、输出误差与计时配置。这样 L4/L5 失败时，仍能回到 L2
判断问题出在算法、kernel、缓存还是系统调度。

### 每一级的最少记录

| 级别 | 正确性 | 性能 | 质量/系统 |
| --- | --- | --- | --- |
| L0/L1 | max/mean error、mask 可视化 | 可不计时 | 定义是否改变 |
| L2 | 对 L1 的误差/梯度 | latency、GB/s、显存、density | block 分布 |
| L3 | selector 的 retained-set 差异 | selector 与 kernel 分项时间 | proxy recall/attention mass |
| L4 | layer 输出/回归测试 | fwd 或 fwd+bwd | 小任务/困惑度 |
| L5 | 固定请求回归 | TTFT、TPOT、token/s、p50/p95 | cache、batch、scheduler |

如果论文跳过某一层，把它标为证据缺口，而不是替作者补结论。

## 15.8 你的第一个 sparse-kernel 研究问题应怎样缩小

“优化 sparse attention”太大，先把它缩成一个可证伪问题。下面三种都适合作为后续选择：

1. **固定结构问题：** 在 A100 上，什么 block size/window density 才开始超过 dense
   FlashAttention？是否受 head dimension 与 batch 影响？
2. **selector 问题：** 一个低成本 block selector 的时间、保留 recall 和最终 sparse-kernel
   speedup 如何权衡？selector 会不会吃掉收益？
3. **serving 问题：** 在固定模型/请求分布下，稀疏 KV policy 降低 TPOT 的同时，TTFT、
   quality 和 tail latency 怎么变化？

不要一开始同时变 mask、block size、dtype、batch、模型、cache format 和 scheduler。先固定
除一个变量外的所有配置，再画一条曲线。性能研究最可贵的能力是排除解释，不是一次运行
得到很大的数字。

## 15.9 实验公平性清单

把这张清单放在每一份 benchmark 表旁边：

```text
[ ] dense 与 sparse 是否使用同一模型权重、同一输入和同一随机种子？
[ ] dtype、GQA/MQA、causal flag、head dimension 是否相同？
[ ] selector/index 的时间是否计入主结果？
[ ] warmup、同步、重复次数、统计量是否相同？
[ ] 测的是 forward、fwd+bwd、prefill 还是 decode？
[ ] 稀疏输出是精确 sparse-model 结果，还是近似 dense 结果？
[ ] 是否报告实际 density/每行 block 数分布，而不只报告目标 top-k？
[ ] quality/误差是否与速度表对应同一个配置？
[ ] 显存、KV cache、编译/JIT 时间是否单独说明？
```

“同一 GPU”不是公平比较的全部条件。若 dense 使用 BF16 FlashAttention、sparse 使用 FP16
且 selector 不计时，表面上的 speedup 没有可解释性。

## 常见误区

**误区 1：mask 小就不需要 FlashAttention。**
即便只保留一部分 block，在线 softmax、tile、融合和访问重排仍然决定性能；sparse kernel
不是在 dense 代码外包一个 `if`。

**误区 2：只要在 op benchmark 快，serving 就快。**
端到端还包含 KV cache/page table、请求 batching、调度、模型其他层和网络/队列。

**误区 3：把 TTFT 和 TPOT 加成一个平均数。**
它们对应不同用户体验和不同瓶颈，合成数字会掩盖回归。

**误区 4：不报告 selector。**
动态 sparse 的 selector 是方法的一部分。除非明确研究的是“给定 oracle mask 的 kernel”，
否则不能从主时间中剔除。

**误区 5：只看平均延迟。**
真实 serving 还关心 p50/p95/p99、请求长度分布和 batch 策略；不均衡 block mask 常影响 tail
latency。

**误区 6：把 layout 优化当作纯工程细节。**
对于 sparse attention，layout 决定某个算法主张是否能在硬件上实现，因此它本身就是研究问题。

## 练习

### 练习 15.1：为 fixed block sparse 写实验卡

任选 `Bq=Bk=64` 或另一个有理由的 block size。写出 dense baseline、masked reference、
sparse kernel 需要各比较什么；明确哪个阶段先不做动态 selector。

### 练习 15.2：拆一张速度图

找 MInference、NSA、MoBA 或 SpargeAttention 中一张速度图。将横轴、纵轴、硬件、shape、
batch、阶段、baseline、是否含 selector、质量条件逐项抄出。任何看不到的信息都标记为
“未报告/待查”，不要根据论文名猜。

### 练习 15.3：serving 指标诊断

假设一个方法将 TPOT 从 10 ms 降到 6 ms，但 TTFT 从 1.0 s 升到 1.3 s。列出至少三种可能
原因，并写出各自需要的测量（selector、prefill、cache、batch/scheduler 等）。

## 通过条件

进入最后的研究路线章前，你应当能：

- 画出算法、operator、serving 的三层图；
- 用一句话解释 block sparsity 的硬件动机；
- 为 prefill/decode 分别选择正确指标；
- 设计 L0--L3 benchmark 阶梯，不把 selector 从动态 sparse 主结果中删掉；
- 说出你最想研究的一个可证伪 sparse-kernel 问题。

## 本章文献锚点

- `dao2022flashattention`、`dao2024flashattention2`：exact dense IO-aware baseline；
- `dong2025flexattention`：由高层 attention 变体生成 fused kernel；
- `ye2025flashinfer`：KV cache format、JIT template 与 serving scheduling；
- `jiang2024minference`、`yuan2025native`、`lu2025moba`、`zhang2025spargeattention`：当前
  sparse 算法与 kernel/serving 的重点案例。

完整条目见仓库根目录的 `references/attention.bib`。
