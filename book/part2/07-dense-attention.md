# 第 7 章：把 Dense Attention 当成一段张量程序

Dense attention 是后面一切比较的坐标原点。Sparse attention 决定哪些 query-key 对不算，
linear attention 改变计算结合顺序或模型形式，FlashAttention 改变 exact attention 的执行
方式。若 dense baseline 的形状、mask 或数值稳定性不清楚，后面的“加速”就没有可靠分母。

本章只讨论 attention operator，不展开 QKV 投影、残差和 MLP 的完整模型成本。

## 学习目标

读完后，你应当能够：

1. 从 `B,T,H,D` 写出每一步张量形状；
2. 用下标解释 score、softmax 和 value 聚合；
3. 区分 MHA、GQA、MQA，以及 self-attention、cross-attention；
4. 推导 dense attention 的主要 FLOPs 和中间张量空间；
5. 解释训练、prefill、decode 的计算形态为什么不同；
6. 写出一个易读的 FP32 reference，并设计边界测试。

## 7.1 先固定符号，不要只写一个 `N`

使用比教科书稍完整的形状：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `T_q` | query 序列长度 |
| `T_k` | key/value 序列长度 |
| `H_q` | query head 数 |
| `H_kv` | key/value head 数 |
| `D` | 每个 head 的维度 |

最简单的 multi-head attention（MHA）满足 `H_q = H_kv = H`：

```text
Q: [B, H, T_q, D]
K: [B, H, T_k, D]
V: [B, H, T_k, D]
O: [B, H, T_q, D]
```

Self-attention 常有 `T_q = T_k = T`。cross-attention 的 Q 来自一条序列，K/V 来自另一条
序列（例如 decoder 查询 encoder 表示），所以长度可以不同，且通常不使用 self-attention
的 causal 对角线。为了学习 kernel，从此刻开始不要省略 batch 和 head 维，因为它们决定
有多少独立工作可分给 GPU。

## 7.2 一行公式拆成四步

经典 scaled dot-product attention 是：

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right)V.
$$

这行公式容易让人误以为它是一个原子操作。实际可以拆为：

### 第一步：query 与所有 key 做点积

$$
S_{b,h,i,j}
=\frac{1}{\sqrt D}\sum_{d=1}^{D}Q_{b,h,i,d}K_{b,h,j,d}.
$$

形状：

```text
Q                 [B, H, T_q, D]
K.transpose(-1,-2)[B, H, D, T_k]
S = Q @ K^T       [B, H, T_q, T_k]
```

`S[b,h,i,j]` 表示第 `i` 个 query 与第 `j` 个 key 的匹配分数。最后两个维度不是“两个
feature 维”，而是所有 query-key 位置对。

为什么除以 $\sqrt D$？若 Q、K 各维大致独立、方差相近，点积方差会随 D 增长。缩放能
避免 logits 随 head dimension 变大而过早进入非常尖锐、梯度不友好的 softmax 区域。

### 第二步：加入 mask

mask 常见两类：

- causal mask：位置 `i` 不允许读取未来位置；
- padding/arbitrary mask：无效 token 或业务规则指定的位置不可见。

概念上，把不可见位置加上 $-\infty$：

$$
\tilde S_{b,h,i,j}=
\begin{cases}
S_{b,h,i,j}, & (i,j)\text{ 可见},\\
-\infty, & (i,j)\text{ 不可见}.
\end{cases}
$$

不要默认所有 API 的布尔 mask 语义相同。有的约定 `True=保留`，有的约定
`True=屏蔽`。必须用 3 到 4 个 token 的手算例子测试。

当 `T_q != T_k` 时，causal 对角线如何对齐也要明确。增量 decode 的 query 往往对应整段
KV 的最后位置，而不是矩阵左上角的第 0 行。本书的 right-aligned decode 契约是：

$$
j \le i + (T_k-T_q).
$$

因此 `T_q=1,T_k=5` 时唯一 query 可见 5 个 cached keys。框架 API 可能采用另一种矩形
对齐；不要凭 `tril()` 猜语义，应先用小矩阵验证。

### 第三步：沿 key 维做 softmax

$$
P_{b,h,i,j}
=\frac{\exp(\tilde S_{b,h,i,j})}
{\sum_{j'=1}^{T_k}\exp(\tilde S_{b,h,i,j'})}.
$$

形状保持 `[B,H,T_q,T_k]`。对固定 `b,h,i`：

```text
sum_j P[b,h,i,j] = 1
```

因此每个 query 都得到对全部可见 value 的一组权重。softmax 必须沿 `T_k` 维，而不是
head 或 feature 维。

### 第四步：加权汇总 value

$$
O_{b,h,i,d}=\sum_{j=1}^{T_k}P_{b,h,i,j}V_{b,h,j,d}.
$$

形状：

```text
P [B, H, T_q, T_k]
V [B, H, T_k, D]
O [B, H, T_q, D]
```

注意 score 使用 Q 和 K，输出内容来自 V。一个位置得到高分，只表示其 value 被赋予较大
权重。

## 7.3 一个教学用 reference

下面代码追求可读性，不追求性能：

```python
def dense_attention_reference(q, k, v, keep_mask=None):
    # q: [B,H,Tq,D], k/v: [B,H,Tk,D]
    d = q.shape[-1]
    scores = q.float() @ k.float().transpose(-1, -2)
    scores = scores / math.sqrt(d)

    if keep_mask is not None:
        # keep_mask 应可 broadcast 到 [B,H,Tq,Tk]
        scores = scores.masked_fill(~keep_mask, float("-inf"))

    prob = torch.softmax(scores, dim=-1)
    out = prob @ v.float()
    return out
```

第一批测试使用小整数或固定 seed，手工检查：

1. 输出形状；
2. causal 第一行只能读取第一个 value；
3. 若所有 score 相等，概率在可见位置均匀分布；
4. mask 掉一个位置后，其 value 无论多大都不影响输出；
5. 将 Q、K 同时按相同方式改变是否真保持预期，不要假设无关性质。

### 全 mask 行是一个真实边界

若某一行所有 score 都是 $-\infty$，softmax 会遇到未定义的 $0/0$，可能产生 NaN。不同
高效 kernel 对这种输入的契约可能不同。应在 API 层避免全 mask 行，或明确规定其输出，
不能把 NaN 当作“浮点误差”。

## 7.4 Softmax 的数值稳定性

直接计算 $e^{s_j}$ 可能溢出。利用 softmax 对整体平移不变：

$$
\operatorname{softmax}(s)_j
=\frac{e^{s_j-m}}{\sum_k e^{s_k-m}},
\qquad m=\max_k s_k.
$$

此时所有指数的输入不大于 0。常规实现至少进行：

```text
m = row_max(scores)
p_unnorm = exp(scores - m)
l = row_sum(p_unnorm)
p = p_unnorm / l
```

这里的 `m` 和 `l` 正是下一章 online softmax 的状态。FlashAttention 不是抛弃稳定
softmax，而是让它能分块更新。

低精度输入不意味着所有中间累加都必须低精度。高效 kernel 常使用 BF16/FP16 读取和
tensor core 乘法，但在 FP32 中做部分累加、max 和 sum。具体精度契约必须通过误差测试
核验。

## 7.5 MHA、GQA 与 MQA

若 `H_q = H_kv`，每个 query head 有独立的 K/V head，这是 MHA。

Grouped-query attention（GQA）使用更少的 K/V head：

```text
H_q = 16, H_kv = 4
每 4 个 query heads 共享一组 K/V head
```

这种均匀分组要求 `H_q % H_kv == 0`；否则每组大小不是整数，必须定义另一种显式映射。

可以用映射表示：

$$
g(h_q)=\left\lfloor\frac{h_q}{H_q/H_{kv}}\right\rfloor,
$$

然后在 score 和 value 聚合中使用 `K[b,g(hq),...]` 与 `V[b,g(hq),...]`。Multi-query
attention（MQA）是 `H_kv=1` 的极端情况。

GQA/MQA 的重要系统收益是减小 KV cache。不要在 reference 之外真的用 `repeat_interleave`
物化 K/V 到 `H_q` 个 head 再宣称测到了 GQA 性能；高效 kernel 应逻辑共享、物理不复制。

## 7.6 训练、prefill、decode 是三种形态

### 训练

训练通常对完整序列进行 causal self-attention：`T_q=T_k=T`，还需要 backward。为了求
梯度，朴素实现会保存 `P` 等中间量，显存压力很大。

若 `dO` 是上游梯度，忽略 mask/scale 的细节，可看到 backward 仍包含大矩阵：

$$
dV=P^\top dO,
\qquad dP=dOV^\top,
$$

$$
dS=P\odot\left(dP-\operatorname{rowsum}(dP\odot P)\right),
$$

$$
dQ=dSK,\qquad dK=dS^\top Q.
$$

这说明只优化 forward 不能代表训练速度。FlashAttention backward 会在 tile 内重算 S/P，
以计算换 HBM 读写和保存空间。

### Prefill

推理收到整段 prompt 时，所有 prompt token 一起计算，仍大致是 `T_q=T_k=T_prompt`。
这是矩阵乘并行度较高、长序列二次成本显著的阶段。prefill 结果会写入 KV cache。

### Decode

之后每次生成一个新 token，常见形状为：

```text
Q_new: [B, H_q, 1, D]
K_cache/V_cache: [B, H_kv, T_context, D]
```

此时不再重新计算历史 Q，但要读取历史 K/V。单步只有一个 query，矩阵很“瘦”，tensor
core 利用与并行度不同，常更受 KV cache 带宽、batch 和 launch 影响。FlashAttention 在
大 prefill 上很强，不代表相同实现一定主导单 token decode。

## 7.7 时间和空间从哪里来

先看 MHA，不计 QKV projection。`QK^T` 的乘加约为：

$$
2BH T_qT_kD\quad \text{FLOPs}.
$$

`PV` 同量级，因此 forward 主项：

$$
F_{attention}\approx4BH T_qT_kD.
$$

Self-attention 中 `T_q=T_k=T`，于是是 $O(BHT^2D)$。softmax 还做 max、exp、sum、divide，
渐近为 $O(BHT^2)$，但这些非矩阵运算的硬件吞吐特性与 tensor core matmul 不同，不能只因
FLOPs 较少就忽略其实际时间。

朴素实现物化：

```text
S: [B,H,T_q,T_k]
P: [B,H,T_q,T_k]
```

仅一个 BF16 矩阵的字节数约为：

$$
2BHT_qT_k\ \text{bytes}.
$$

例如 `B=1,H=32,T=8192` 时，一个 `[B,H,T,T]` BF16 张量约为：

$$
1\times32\times8192^2\times2 \approx 4\ \text{GiB}.
$$

S 和 P 若都存在就是约 8 GiB，还未计 QKV、输出、workspace 与 backward。这个例子解释
了为什么“只有 8K token”也可能迅速耗尽显存。

### causal 并不自动减半存储

数学上 causal 只需计算下三角，约省一半 score 工作。但若代码先生成完整 `T*T` 矩阵再
mask 上三角，物理显存和大部分计算并没有省。只有 kernel 在 tile/元素层真正跳过被 mask
区域时，才获得对应系统收益。

## 7.8 Dense、Flash、Sparse、Linear 的边界

| 名称 | 是否得到同一 exact dense 输出 | 主要改变什么 |
| --- | --- | --- |
| 朴素 dense | 是 | 显式物化 S/P |
| FlashAttention | 是（允许浮点舍入差异） | tiling、online softmax、IO 与工作划分 |
| sparse attention | 通常否 | 只计算部分 query-key 对，或选择 KV |
| linear attention | 通常不是同一 softmax operator | 核函数、递归 state、结合顺序或模型族 |

因此不能把 FlashAttention 称为“近似 attention”，也不能只因某个 linear attention 使用
矩阵乘就叫它 FlashAttention。

## 7.9 从 reference 到框架 baseline

建议保留三层：

1. **教学 reference**：显式 FP32 S/P，用于小 shape 正确性；
2. **PyTorch SDPA**：`scaled_dot_product_attention`，作为框架接口 baseline；
3. **指定高效 backend/作者实现**：作为性能 baseline。

SDPA 是统一接口，不是单一 kernel。框架可能根据硬件、dtype、shape、mask、dropout 和是否
需要梯度选择不同 backend。实验记录必须写明实际 backend 或至少写明选择约束，不能把
“调用了 SDPA”自动等同于“使用 FlashAttention”。

## 常见坑

- `softmax(dim=-2)`，把 query 维归一化；
- 忘记 $1/\sqrt D$，或重复缩放两次；
- mask 布尔语义写反；
- `T_q != T_k` 时 causal 对角线错位；
- GQA 用 `repeat` 物化 K/V，污染性能和显存结论；
- 把 `[B,T,H,D]` 传给期望 `[B,H,T,D]` 的 kernel；
- `transpose` 后张量 non-contiguous，却假设 stride 未变；
- BF16 reference 也使用相同低精度路径，掩盖误差；
- 只测试 `T` 为 128 的整数倍，漏掉边界 mask bug；
- 用 forward-only 数字代表训练。

## 练习

### 练习 7.1：逐项标形状

对 `B=2,T_q=3,T_k=5,H=4,D=8`，写出 Q、K、V、S、P、O 的形状，并解释
`P[1,2,0,4]` 的含义。

### 练习 7.2：三 token 手算

取 `B=H=D=1,T=3`，自行指定 Q/K/V 小整数，手算 causal attention 三行输出，再用
reference 验证。必须展示每行可见 key、减 max 后的指数和概率。

### 练习 7.3：复杂度表

固定 `B=1,H=16,D=64`，对 `T=1K,2K,4K,8K` 计算 forward 主项 FLOPs 和单个 BF16
S 矩阵大小。观察 T 翻倍时两者如何变化。

### 练习 7.4：prefill 与 decode

对 4K prompt 后生成第一个 token，分别写 prefill 和首个 decode step 的 Q/K/V 形状。
解释为什么 decode 的理论 FLOPs 降低很多，却不一定按同倍数变快。

### 练习 7.5：GQA 不物化

为 `H_q=8,H_kv=2` 写出 query head 到 KV head 的映射。实现教学 reference 时可先用循环，
再与 `repeat_interleave` 版本比正确性；性能表中解释为什么后者不能作高效 baseline。

## 通过条件

进入 FlashAttention 前，你需要交付：

- 一页形状推导，能从公式走到 `[B,H,T_q,T_k]`；
- 一个 FP32 dense reference 和至少 6 个小 shape 测试的结果记录；
- 一张 `T` 扫描表，包含 FLOPs、S/P 理论空间及实测 OOM；
- 用自己的话说明 exact、sparse、linear 分别改变了什么；
- 能解释“causal mask 存在”为什么不等于朴素实现自动省一半显存。
