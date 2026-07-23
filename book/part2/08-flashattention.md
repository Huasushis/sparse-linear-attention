# 第 8 章：FlashAttention 为什么既 exact 又省显存

上一章的 dense reference 做了三件大事：生成 `S=QK^T`，把 S 写入显存；读取 S 做
softmax，把 P 写入显存；再读取 P 计算 `PV`。FlashAttention 的出发点是：**最终只需要
O，不需要让完整 S/P 在 HBM 中存在。**

它没有删掉 query-key 对，也没有把 softmax 换成近似核函数。它重新安排计算顺序，使
tile 在片上存储中完成 score、softmax 状态更新和 value 聚合。这是一种 IO-aware 的 exact
attention 算法。

## 学习目标

读完后，你应当能够：

1. 解释标准 dense attention 的 HBM 往返发生在哪里；
2. 推导 online softmax 的 `(m,l,u)` 分块更新公式；
3. 根据伪代码说清 Q tile、KV tile 和输出状态的生命周期；
4. 区分 FLOPs 复杂度、额外显存复杂度和 IO 复杂度；
5. 解释 backward 为什么选择重算 S/P；
6. 概括 FlashAttention-2 改进的是工作划分，而不是另一种 attention 数学定义；
7. 判断哪些场景不应期待明显加速。

## 8.1 先纠正名字带来的三个误解

### 误解一：FlashAttention 是 sparse attention

不是。对相同 Q/K/V、scale、mask 和精度契约，它计算 dense softmax attention 的相同
数学结果，只存在正常浮点运算重排造成的舍入差异。

### 误解二：它把 $O(T^2)$ 计算变成了 $O(T)$

不是。它仍计算所有允许的 query-key 点积，forward 的主 FLOPs 仍约为
$4BHT_qT_kD$。它把不必要的中间量 HBM 流量和额外存储从二次量级降下来。

### 误解三：它只是一个更快的 softmax

也不准确。关键是把 `QK^T -> mask -> softmax -> PV` 融合为按 tile 流动的数据路径，
其中 online softmax 是保证跨 KV tile 仍得到 exact 结果的数学工具。

## 8.2 标准实现的 IO 路径

先忽略 batch/head，对 `Q,K,V: [T,D]`：

```text
kernel/GEMM 1: read Q,K  -> write S[T,T] to HBM
softmax:      read S    -> write P[T,T] to HBM
kernel/GEMM 2: read P,V -> write O[T,D] to HBM
```

即使 GEMM 本身很快，`S/P` 的二次规模读写仍可能成为瓶颈，还会占用训练 activation 显存。
“标准实现”是概念对照；成熟框架也可能已有融合，不能假设你调用两个 Python 函数就一定
真的物化了所有中间量。实验时要明确 backend。

FlashAttention 把 Q 切成行块，把 K/V 切成列块：

```text
Q_i: [B_r,D]
K_j: [B_c,D]
V_j: [B_c,D]
S_ij = Q_i K_j^T: [B_r,B_c]  # 只在片上短暂存在
```

对固定 Q tile，依次流过所有可见 KV tile，持续更新该 Q tile 的 softmax 统计量和输出。
完整 `[T,T]` S/P 不写回 HBM。

## 8.3 Online softmax：数学核心

先只看一个 query 行。它的 score 被切成若干块 $s^{(1)},s^{(2)},\ldots$，对应不同 KV tile。
处理到任意时刻，维护三个状态：

$$
m=\max(\text{已经看过的 scores}),
$$

$$
l=\sum_{j\in\text{已看}}e^{s_j-m},
$$

$$
u=\sum_{j\in\text{已看}}e^{s_j-m}v_j.
$$

其中 $u$ 是尚未除以 softmax 分母的 value 加权和，维度为 D。最终输出是：

$$
o=\frac{u}{l}.
$$

### 合并一个新 score block

新块的行最大值为：

$$
m_b=\max(s^{(b)}),\qquad m'=\max(m,m_b).
$$

旧状态以旧最大值 m 为基准。为了改成新基准 $m'$，旧的指数和加权和都乘：

$$
\alpha=e^{m-m'}.
$$

新块直接以 $m'$ 为基准。于是：

$$
l'=\alpha l+\sum_{j\in b}e^{s_j-m'},
$$

$$
u'=\alpha u+\sum_{j\in b}e^{s_j-m'}v_j.
$$

更新后令 `(m,l,u) <- (m',l',u')`。全部块结束时 `u/l` 与一次性 stable softmax 后乘 V
完全相同。

### 为什么必须缩放旧状态

假设旧块最大分数为 2，新块出现 10。旧状态中的指数原本以 2 为零点；若直接与以 10
为零点的新指数相加，就比较了不同单位。乘 $e^{2-10}$ 是把旧状态换算到新零点。

漏掉这个 rescale 是初学者实现 online softmax 最常见、也最隐蔽的 bug。小随机 logits
可能误差不明显，加入一个极大新 score 后立刻暴露。

## 8.4 从一行扩展到 Q tile

对 `B_r` 个 query 行，同时维护：

```text
m: [B_r]       每行当前最大值
l: [B_r]       每行当前指数和
u: [B_r,D]     每行未归一化输出
```

教学版 forward 伪代码如下：

```text
for each batch b, head h:
  for each Q block i:
    load Q_i                        # [Br,D]
    m = -inf                        # [Br]
    l = 0                           # [Br]
    u = 0                           # [Br,D]

    for each visible KV block j:
      load K_j, V_j                 # [Bc,D]
      S = Q_i @ K_j^T * scale       # [Br,Bc]，片上
      apply causal/padding mask to S

      m_block = rowmax(S)           # [Br]
      m_new = maximum(m, m_block)   # [Br]
      alpha = exp(m - m_new)        # [Br]
      P_tilde = exp(S - m_new[:,None])

      l = alpha * l + rowsum(P_tilde)
      u = alpha[:,None] * u + P_tilde @ V_j
      m = m_new

    O_i = u / l[:,None]
    write O_i
```

实际 kernel 会根据硬件、causal、head dimension、dtype 和 backward 需要改变循环方向、
layout 及状态表示，但正确性骨架就是上面的不变量。

### 一个必须亲手做的四分数例子

取 score `[1, 2 | 10, 9]`，V 为四个标量或二维向量。分别：

1. 一次性 stable softmax；
2. 先处理 `[1,2]`，再处理 `[10,9]`；
3. 检查第二块到来时旧 `l,u` 是否乘 `exp(2-10)`。

如果两种输出不一致，不要进入 kernel 代码。

## 8.5 Causal mask 怎样与 tile 相遇

对 causal self-attention，一个 Q tile 与 KV tile 可能是：

1. 完全在对角线左下方：整个 tile 可见，不需要逐元素 mask；
2. 与对角线相交：tile 内做逐元素 causal mask；
3. 完全在对角线右上方：整块跳过。

这比先算完整 S 再填 `-inf` 真正减少了无效工作。边界上最容易出错的是：非整 tile
长度、`T_q != T_k` 的对齐、padding 与 causal 同时存在。正确性测试必须覆盖三种 tile。

上面的 online 伪代码还隐含一个契约：每个 query 行至少有一个可见 key。若整行都被 mask，
会出现 `m=-inf` 与 `m_new=-inf`，于是 `exp(m-m_new)=exp(NaN)`。成熟 API/kernel 会规定
这种输入的行为或专门处理；教学实现应先拒绝它，或明确把该行定义为零输出，不能指望
`-inf` 的普通算术自动得到正确答案。

## 8.6 它究竟省了什么

### 计算量

允许的 query-key 对仍需做点积，主项仍是二次：

$$
O(BHT_qT_kD).
$$

甚至为了 backward 重算，某些算术会增加。FlashAttention 的成功正说明：**FLOPs 增加
一点，也可能因 HBM IO 大幅减少而更快。**

### 额外显存

不再保存完整 S/P 后，attention operator 的大中间量从 $O(BHT_qT_k)$ 消失。输出、QKV
和每行 log-sum-exp 等量随序列长度线性增长。注意这描述的是 attention 中间量，不是整个
Transformer 模型的全部显存。

### HBM IO

Q/K/V tile 会按循环安排从 HBM 搬到片上，并在片上尽量复用；S/P tile 在消费后丢弃。
精确 IO 次数取决于 tile 大小、片上存储容量、循环方向、cache 与实现版本。学习阶段正确
的结论是“避免 S/P 的大规模 HBM materialization，并用 tiling 增加复用”，不要简化成
“每个 Q/K/V 永远只读一次”。

## 8.7 Backward：重算为什么可能更便宜

朴素 autograd 为 backward 保存 P `[B,H,T,T]`。FlashAttention forward 可只保存 O 和每行
log-sum-exp（或等价状态）。Backward 再次加载 Q/K/V，在 tile 内重算：

```text
S_tile -> P_tile -> dV/dP/dS -> dQ/dK
```

P tile 用完即丢，不写成完整矩阵。这里做了更多计算，却节省大量 activation 存储和 HBM
流量。因为 GPU 的 tensor core 矩阵乘吞吐远高于片外数据往返成本，重算常常是划算交换。

不过 forward 快不保证 backward 同比例快。训练 benchmark 必须单独测：

- forward-only；
- backward-only（若能清晰隔离）；
- forward + backward。

Dropout 时还涉及随机数状态与重算一致性，不能把 inference-only kernel 当训练实现。

## 8.8 FlashAttention-2 改了什么

原始 FlashAttention 已减少 IO，但不一定把 GPU 吃满。FlashAttention-2 的核心仍是相同的
exact tiled attention，重点改善三件事：

1. **减少非矩阵乘 FLOPs。** GPU 对 tensor-core matmul 的吞吐远高于 exp、rescale 等
   非 matmul 运算，不能只看它们 FLOPs 占比小；
2. **增加 sequence 维并行。** 长序列常伴随小 batch/少 head，只按 batch 与 head 分 block
   可能没有足够工作；沿 query 序列分更多 thread block 可提高利用；
3. **改进 warp 间工作划分。** 减少通过 shared memory 交换中间结果和不必要同步。

论文在 A100 上报告相对第一版的显著提升，但这不是你本机实验的预定答案。具体 speedup
取决于版本、GPU、shape、dtype、causal 和 baseline。

### 那 FlashAttention-3/4 呢

后续版本继续利用新硬件的异步执行、低精度和 pipeline。当前应提取的可迁移问题是：

- 搬运与计算能否重叠？
- 不同执行单元的吞吐是否不对称？
- producer/consumer 怎样分工？
- layout 怎样减少通信？

Hopper/Blackwell 的具体指令、TMEM/TMA 协议和 FA-4 pipeline 暂不作为本阶段实现门槛。
没有对应硬件时，能读懂设计动机比抄一份无法验证的指令级代码更有价值。

## 8.9 为什么有时看不到加速

- **序列很短。** S/P 很小，launch、dispatch 和固定开销占主导；
- **单 token decode。** `T_q=1`，工作形态更像流式读取 KV cache，prefill kernel 的分块
  策略未必合适；
- **shape/backend 不支持。** 框架可能 fallback 到其他实现；
- **dtype 或 head dimension 不友好。** 无法走高效 tensor-core/layout 路径；
- **baseline 已经融合。** 你以为在比“朴素 dense”，其实 SDPA 已选择高效 backend；
- **计时错误。** 首次编译、同步、输入分配或 dropout 设置不一致；
- **并行量不足。** batch/head/sequence 分块产生的 program 数太少；
- **显存没成为瓶颈。** 小问题中省下的中间量没有转化为可观时间收益。

性能曲线比单点 speedup 更重要。理想实验会展示随着 T 增长，两个实现何时交叉、何时 OOM。

## 8.10 与 sparse/linear attention 的关系

FlashAttention 给后续研究两条重要教训：

1. 渐近 FLOPs 更低不保证 wall-clock 更快；实现必须适合 GPU 的 tile 和访存；
2. 算法与 kernel 应共同设计。一个稀疏模式若索引不规则、复用差，理论省下乘法也可能被
   gather 和调度开销抵消。

很多 linear attention 的 chunkwise 算法同样在解决“怎样把递归定义变成块矩阵运算”，
FLA 则把这些块算法落成 Triton/CUDA kernel。读 Kimi、GDN 或 sparse 方法时，应继续问：

```text
什么量不再 materialize？什么状态留在片上？tile 之间传什么？
理论减少了多少工作？新的索引、scan、state 更新又花多少？
```

## 8.11 建议的复现实验

第一轮不手写 FlashAttention kernel，而是比较三个层次：

1. 显式 FP32 dense reference，只跑小 shape 验证；
2. PyTorch SDPA，并控制或记录 backend；
3. 可用的 FlashAttention/SDPA 高效 backend。

实验至少包含：

```text
T = 128, 1K, 4K, 8K（资源允许再到 16K）
D = 64, 128
causal = false, true
mode = forward, forward+backward
dtype = bf16（reference 用 fp32）
```

记录 latency、峰值 allocated memory、误差和 OOM。暂时不要追求复现论文绝对数字；先复现
趋势：长序列下中间显存差异，以及高效 backend 的交叉点。

## 常见坑

- 把 online softmax 的 `u` 当成已归一化 O，却继续用未归一化更新式；
- 最大值更新后忘记 rescale 旧 `l/u`；
- 对 masked 元素先做 exp 再乘 0，导致 `inf * 0` 或 NaN；
- tile 边界 mask 与 causal mask 混淆；
- 认为没有 S/P 就没有 $T^2$ 计算；
- 只验证输出，不验证 backward；
- 将不同运算顺序产生的 BF16 差异误判为 exact 性被破坏；
- 把框架 API 名称当作实际 backend；
- 用 prefill 结论推断 decode；
- 抄论文硬件上的 speedup，不在当前环境重新测。

## 练习

### 练习 8.1：证明 online 更新

从 `l=sum(exp(s-m))`、`u=sum(exp(s-m)v)` 出发，推导合并新 block 时为什么旧状态乘
`exp(m-m_new)`。推导中不允许只写“为了稳定”。

### 练习 8.2：手算两个 block

完成 `[1,2 | 10,9]` 的一次性与 online softmax，并使用你指定的四个标量 V 比较输出。
然后故意删掉 rescale，记录错误有多大。

### 练习 8.3：画 tile 生命周期

对 `T=8,Br=2,Bc=4,D=2`，画 Q/KV tile 网格。标出 causal 情况中完整可见、对角线相交、
整块跳过三类，并列出一个 Q tile 在写 O 前保持的状态。

### 练习 8.4：预测性能曲线

在运行前画出显式 dense 与高效 backend 的 latency/peak-memory 随 T 变化的预期曲线，标记
你预计的交叉点和 OOM 点。实验后用数据修改，而不是删除错误预测。

### 练习 8.5：读一段 FA-2 摘要

把“减少非 matmul FLOPs、sequence 维并行、warp 工作划分”各用一句自己的话解释，并为
每项写一个 profiler 中可能支持或反驳它的证据。

## 通过条件

开始写 Triton kernel 前，你应当能够：

- 不看资料写出 `(m,l,u)` 初始化、更新和最终归一化；
- 明确说出 FlashAttention 保留了什么数学结果、没有降低什么复杂度；
- 画一张 tiled forward 数据流图，标出 HBM 与片上临时量；
- 完成 dense reference、SDPA、高效 backend 的 correctness 与 shape 扫描表；
- 对“为什么更快”给出 IO/并行证据，而不只复述“用了 tiling”；
- 解释为什么 Blackwell 专属 FA-4 细节目前不是继续学习 Triton 的前置条件。

## 延伸阅读

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [PyTorch scaled dot product attention 文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
