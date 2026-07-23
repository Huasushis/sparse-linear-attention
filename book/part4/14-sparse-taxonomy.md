# 第 14 章：Sparse Attention 的地图——先问“删了谁”，再问“真的少算了吗”

Linear attention 用固定大小 state 压缩历史；sparse attention 走另一条路：仍然保留
attention 的 query-key-value 结构，但让每个 query **只看一部分 key**。这句话很简单，
却包含两个完全不同的问题：

1. **算法问题：** 哪些配对应该保留？谁来决定？会损失什么？
2. **系统问题：** 保留的配对能否被 GPU 真正高效执行？选择和索引本身花了多少时间？

如果只说“稀疏率 90%”，这两个问题一个也没有回答。本章给 73 篇文献建立一个可操作的
分类，不要求你把每篇都精读或复现。

## 学习目标

读完后，你应当能够：

1. 用 mask、density、block 和 selector 描述一种 sparse attention；
2. 区分固定结构、内容路由、KV cache、可训练 block sparse 和通用 sparse kernel；
3. 区分 prefill sparse 与 decode/KV sparse；
4. 解释“理论少算”为什么不等于“GPU 更快”；
5. 把你的 73 篇文献放入阅读优先级，而不是按年份背摘要；
6. 为一个新 sparse 论文写出它的选择成本和执行成本。

## 14.1 从一个 mask 开始

设 dense attention 的 score 是

$$
L_{ij}=q_i^\top k_j/\sqrt D.
$$

sparse attention 引入可见性集合 `A(i)`，或者二值 mask `M`：

$$
M_{ij}=1 \Leftrightarrow j\in A(i).
$$

在 causal 场景中，输出仍可写成

$$
o_i=\sum_{j\in A(i),j\le i}
\operatorname{softmax}_{j\in A(i),j\le i}(L_{ij})v_j.
$$

关键在于 softmax 的归一化集合也变了。若先完整计算所有 `L_ij`，再把 90% 权重设为零，
通常没有省下主要计算；真正的 sparse kernel 必须避免生成被排除配对。

### 三个必须写出的量

| 名称 | 记号/例子 | 为什么重要 |
| --- | --- | --- |
| density | `|A(i)| / T` | 单个 query 实际保留多少 key |
| sparsity | `1-density` | 便于表达删去比例，但不能单独预测速度 |
| granularity | token / block / head / layer | 决定索引、访存和 Tensor Core 是否友好 |

例如“每个 query 看最近 2048 个 token”是固定 window；“每个 query 选 top-16 个
128-token block”是动态 block sparse。两者即使 density 相同，GPU 数据访问也可能完全不同。

## 14.2 总成本不是 `nnz(M)` 一项

对于一个动态 sparse 方法，端到端代价更诚实的分解是：

$$
C_{total}=C_{select}+C_{metadata}+C_{gather}+C_{sparse\_attention}
          +C_{scatter/sync}+C_{quality\_repair}.
$$

其中：

- `C_select`：判定哪些 key/block 重要；若它先近似或完整算了 `QK^T`，节省可能消失；
- `C_metadata`：mask、block index、分页表或 routing 结果的存储/传递；
- `C_gather`：不连续 K/V 的读取、重排或 cache miss；
- `C_sparse_attention`：对保留项的真正 score/softmax/PV；
- `C_scatter/sync`：跨 block、跨 head 或跨请求的归并；
- `C_quality_repair`：为维持质量而增加的 dense layer、重算、训练或蒸馏。

所以在报告中不要写“因为 `O(T²)` 变为 `O(Tk)`，所以加速 `T/k` 倍”。正确写法是：
先给算法的 nominal work，再报告具体实现中 selector、layout、硬件和测量的实际结果。

## 14.3 五个主家族

下面的分类按“谁决定可见性”和“在哪个阶段生效”组织，而不是按发表年份。一个工作可以
跨两类；那时应把它的两个角色都写出来。

| 家族 | `A(i)` 如何决定 | 常见阶段 | 代表文献（Bib key） | 初学者第一遍要看 |
| --- | --- | --- | --- | --- |
| 固定结构 mask | window、stride、global token、dilated pattern | 训练 + prefill | `child2019sparse`、`beltagy2020longformer`、`zaheer2020bigbird`、`ainslie2020etc`、`ding2023longnet` | mask 图、连通性、复杂度 |
| 内容路由/近似选择 | LSH、聚类、routing、采样、动态 score | 多为训练/prefill | `kitaev2020reformer`、`tay2020sparse`、`roy2021routing`、`pagliardini2023dynamic` | selector 是否本身昂贵、是否可微 |
| inference/KV 选择 | 保留 sink/heavy hitter/retrieval KV 或按 query 筛 | decode 或 long prefill | `xiao2024streamingllm`、`zhang2023h2o`、`tang2024quest`、`ribar2024sparq`、`xiao2025duoattention` | cache 少了什么、质量损失在哪 |
| 可训练、硬件对齐的 block sparse | 分层压缩/动态 block router/attention gate | 训练 + inference | `yuan2025native`、`lu2025moba`、`gao2025seerattention`、`xu2025xattention` | block shape、训练信号、kernel 接口 |
| 通用/adaptive sparse kernel | 在线过滤、adaptive mask、softmax-aware skip | 常见于 inference，也可能可训练 | `zhang2025spargeattention`、`goncalves2025adasplash` | filter 成本、误差、真正 skip 的矩阵乘 |

视觉/视频分支（`wei2023sparsifiner`、`liu2025fpsattention`、`hu2026dfsattn` 等）先作为
“动态细粒度 sparsity 更难落到硬件”的横向参照；当前 LLM sparse/linear 主线尚不要求深入。

## 14.4 固定结构：最适合建立 mask 直觉

固定结构方法不看当前输入内容就能知道 `A(i)`。这使它们最容易画图、实现并 benchmark：

```text
sliding window:      每个 query 看附近 W 个位置
global token:        少数特殊位置可看全局/被全局看见
strided/dilated:     每隔若干位置连接一次，扩大感受野
block pattern:       整块 token 同时保留或丢弃
```

Longformer/BigBird 是第一批应看的论文，不是因为它们现在一定最强，而是因为它们强迫你
分清：mask 的**图结构**、理论可达性、模型任务质量和 GPU kernel 是四件事。固定 mask
也能通过 FlexAttention 一类接口变为 fused attention；这将在下一章说明。

第一遍只需完成一件事：用 `T=16` 画出一个 causal window+global-token mask，数出每行
保留的 key 数，再写出它为何不是所有行都同样密。

## 14.5 动态选择：强在适应性，难在选择成本

内容路由方法试图让模型/selector 根据 q、k、历史或低成本 proxy 决定重要位置。它们通常
比固定 window 更有表达力，但会引入关键问题：

> 如果知道谁重要本身就需要看完所有候选，节省从哪里来？

常见回答包括：LSH/聚类缩小候选集、低秩 key/proxy、两阶段筛选、token/block router，或
训练一个 gate。读论文时应专门画两条箭头：

```text
q/k  -> selector/proxy -> retained block indices
q/k/v + retained indices -> sparse attention kernel -> output
```

selector 的误选会带来质量损失；过于保守则 sparse ratio 不足。这里没有免费午餐，所以
报告应同时写 selection recall/attention mass 与端到端质量/延迟。

## 14.6 KV cache/serving 选择：与“训练一个 sparse 模型”不同

decode 时新 query 很少、历史 K/V 很长，瓶颈往往是读取 cache 的带宽。此时方法可能不改
训练好的模型权重，只在 inference 决定保留、迁移或检索哪些 cache 条目：

- StreamingLLM 关注 attention sink 与滑动 cache；
- H2O 关注 heavy-hitter token 的保留；
- QUEST、SparQ、Loki 等研究 query-aware/proxy/低秩选择；
- RetrievalAttention、HiP、DuoAttention、FlexPrefill 等从 retrieval、层次筛选或
  head 分工切入。

这些工作与“每个训练层都使用固定 sparse mask”不要混为一谈。它们最应报告的是：

```text
decode 的 batch、历史长度、每步读取字节、cache policy、TPOT/ITL、任务质量
```

而不是只报告训练时的 `T²` FLOPs。

## 14.7 硬件对齐的 trainable block sparse：当前重点候选

如果你后续重点是 sparse attention 性能优化，这一类最值得深入。它不满足于“attention
权重看起来稀疏”，而要求 mask 的粒度、block size、选择算法和 GPU kernel 一起设计。

### MInference：不改预训练的长 prefill 路线

`jiang2024minference` 从长上下文 attention 中识别 A-shape、Vertical-Slash、Block-Sparse
等模式，离线为 head 选择模式、在线建索引，并在 GPU kernel 中执行。这是理解
“pattern discovery -> index construction -> sparse kernel -> prefill latency”的好案例。
它的主张针对 long-context **prefill**；不要把其速度数字迁移到 decode。

### Native Sparse Attention（NSA）：训练与硬件一起考虑

`yuan2025native` 的摘要明确把动态分层稀疏、粗粒度压缩、细粒度选择、hardware-aligned
design 和 end-to-end training 放在一起。它适合作为“算法和 kernel 不能拆开”的重点精读
候选：读的时候必须同时记录 block 的组成、训练时的梯度路径、forward/backward/decode
各自的结果。

### MoBA：把 MoE 的路由思想放入 attention block

`lu2025moba` 以 block 为选择单元，让路由决定注意哪些 block。它对你的主线有两个价值：
一个是 router 的算法问题，另一个是 block routing 如何和实际 LLM long-context 服务连接。
不要把它简化为“top-k token attention”；它选择的是 block，粒度决定了 kernel 可行性。

### SpargeAttention：两阶段 online filter

`zhang2025spargeattention` 以两阶段在线过滤和 softmax-aware skip 为核心，目标是无额外
训练地加速多种模型 inference。它很适合做“filter 近似误差与端到端速度如何共同报告”的
案例，但需额外核对所依赖的量化/基础 attention backend 与目标模型是否一致。

## 14.8 先读哪一些，而不是 73 篇全读

当前阶段建议采用一条主线和两条横向线：

```text
主线：Longformer/BigBird -> MInference -> NSA 或 MoBA -> SpargeAttention
kernel 横线：FlashAttention -> FlexAttention -> FlashInfer
serving 横线：StreamingLLM/H2O -> QUEST 或 SparQ -> LServe
```

每篇的阅读级别如下：

- **慢读（A）**：MInference、NSA、MoBA、SpargeAttention、FlexAttention、FlashInfer；
- **重点略读（B）**：Longformer、BigBird、StreamingLLM、H2O、QUEST、SparQ、LServe、
  SeerAttention、XAttention、AdaSplash；
- **脉络浏览（C）**：更早的 accelerator、其余 selector、量化/视频分支和后续版本。

这不是“论文重要性排名”，而是与你现在的算法+kernel 目标相匹配的学习顺序。

## 14.9 一个新 sparse 论文的七问卡

以后读到任何新方法，先在 20 分钟内填写：

1. **阶段：** 训练、prefill、decode，还是三者？
2. **可见性：** `A(i)` 是固定、按 head、按输入，还是按 query 动态？
3. **粒度：** token、page、block、head，block shape 是多少？
4. **selector：** 用什么 proxy/route 得到 mask，复杂度与误选代价是什么？
5. **执行：** 是否真的避免了被丢弃的 `QK`/`PV`，K/V 怎样访问？
6. **质量：** 无训练可用、需要微调，还是从预训练就改变模型？用什么任务评估？
7. **证据：** 与哪个 dense baseline、什么 GPU/dtype/shape 比，报告什么指标？

若其中两项不能回答，不是你读得慢，而是论文/代码/图表还没被定位到。先标为“待核对”，
不要脑补。

## 14.10 最小实践：固定 block mask 的 reference

最适合作为第一份 sparse 代码的不是 top-k router，而是固定 block/window mask：

1. 从 dense causal reference 开始；
2. 写一个函数 `allowed(q_pos, k_pos) -> bool`；
3. 用它构造 `[T,T]` boolean mask 并在 softmax 前应用；
4. 用 `T=16` 打印/画出 mask；
5. 对比 dense 与 sparse 输出的误差、保留比例和 CPU/GPU 时间；
6. 明确注明该 reference **仍可能先 materialize dense scores**，所以它验证语义，不能证明
   sparse kernel 加速。

这一步听起来朴素，却能避免后面把“结构掩码正确”与“未计算被掩位置”混为一谈。

现在完成 [Lab 7：从 mask 语义走到真正的 sparse operator](../labs/07-sparse-mask.md)。仓库提供
masked-dense oracle、只 gather 被选 K/V 的 Python reference、mask 可视化和独立 TODO grader。

## 常见误区

**误区 1：sparse attention 都是近似。**
若模型定义本来就只允许固定 mask 中的边，稀疏 kernel 可以精确执行这个 sparse 模型；它
只是相对 full attention 改变了模型结构。另一些方法才是对 full attention 的 inference
近似。两者都应说明基准是什么。

**误区 2：attention map 稀疏，就能跳过计算。**
softmax 后许多值很小不等于 kernel 在计算前已知道它们小。选择器必须足够便宜且准确。

**误区 3：token-level 更稀疏就一定更快。**
极细粒度的不规则索引可能让 gather 和 metadata 吃掉收益；block-level 较粗却更贴合 Tensor
Core/连续访存，反而更快。

**误区 4：prefill 的 sparse 方法自然加速 decode。**
prefill 有很多 query 可并行，decode 常只有一个新 query 且带宽受限；二者必须分别测。

**误区 5：一个优秀的 long-context 分数就证明 kernel 高效。**
质量和性能是两张证据表。高质量可以来自更保守的 mask；高性能也可能来自不可接受的近似。

## 练习

### 练习 14.1：给五篇论文贴“二级标签”

从上述主线选 Longformer、MInference、NSA、MoBA、SpargeAttention。每篇都写：

```text
阶段 / 选择单位 / selector 或固定规则 / 是否训练 / 预期 kernel 难点
```

如果一个工作跨两个类别，保留两个标签并说明原因；不要强行塞进唯一格子。

### 练习 14.2：算密度，不算口号

设 `T=8192`，causal sliding window `W=512`，另有 `G=8` 个 global token。估算中间位置和
首尾位置的可见 key 数，给出平均 density 的近似。说明为什么它和“严格每行 512 个 token”
不是同一件事。

### 练习 14.3：selector 审计

任选一个动态 selector，画出它生成 index 前实际读过什么。若它需要读全部 K，说明其带宽
是否可能仍是瓶颈；写出一个能推翻你判断的 profiler/benchmark 证据。

## 通过条件

进入下一章前，你必须能够：

- 用 `A(i)` 和 block 粒度定义一个 sparse 方法；
- 把 MInference、NSA、MoBA、SpargeAttention 放到正确的阶段/训练标签中；
- 解释 sparse ratio 不足以预测 GPU speedup；
- 区分语义 reference 与真正 sparse kernel；
- 为自己准备精读的第一篇 sparse 论文完成七问卡。

## 本章文献锚点

本章的代表键包括 `child2019sparse`、`beltagy2020longformer`、`zaheer2020bigbird`、
`kitaev2020reformer`、`pagliardini2023dynamic`、`xiao2024streamingllm`、`zhang2023h2o`、
`jiang2024minference`、`yuan2025native`、`lu2025moba`、`zhang2025spargeattention`、
`dong2025flexattention` 与 `ye2025flashinfer`。完整来源见仓库根目录的
`references/attention.bib`。

下一章会把第二个问题放大：同样的 mask，为什么可能在一种 kernel 上很快、在另一种实现
上反而更慢？
