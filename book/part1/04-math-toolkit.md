# 第 4 章：给 attention 研究准备的数学工具箱

这不是一本“把线性代数、概率论和 CUDA 全部重讲一遍”的章节。它只收集你接下来读
attention、scan、FlashAttention、稀疏选择和 kernel 代码时会反复遇到、且最容易卡住的
工具。用法不是从头背到尾，而是在论文出现一个公式时回到相应小节核对。

本章的首要技能不是计算快，而是**不让符号骗过你**：每一个乘法先问形状，每一个求和
先问沿哪一维，每一个“更快”先问比较的是 FLOPs、字节还是端到端时间。

## 学习目标

读完后，你应当能够：

1. 用形状和下标同时检查向量、矩阵、张量运算；
2. 区分内积、外积、矩阵乘与逐元素乘，并在 attention 公式中定位它们；
3. 解释 broadcast、transpose、reshape、contiguous 的数学含义和工程含义；
4. 手算稳定 softmax/log-sum-exp，并理解 online softmax 如何合并块；
5. 解释结合律、非交换性和 scan 的关系；
6. 用“对角加低秩（DPLR）”理解一类高效状态转移；
7. 正确报告数值误差、复杂度、访存量和 benchmark 边界。

[第 1 章](01-transformer-from-tensors.md)给出 attention 数据流，[第 2 章](02-training-minimum.md)
给出 loss/backward，[第 3 章](03-rnn-state-and-scan.md)给出状态与 scan。本章把它们所用
的数学语言统一起来。

## 4.1 读公式的第一步：形状先于直觉

看到一行公式，先在每个符号旁写形状，再看文字解释。比如：

```text
Q: [B,H,T,D]
K: [B,H,T,D]
S = Q K^T / sqrt(D)
```

这里的 `K^T` 不可能是“把所有四维完全倒过来”。在 attention 语境中，它通常表示最后
两维转置：

```text
K.transpose(-2,-1): [B,H,D,T]
S:                   [B,H,T,T]
```

下标写法消除歧义：

```text
S[b,h,i,j] = sum_d Q[b,h,i,d] K[b,h,j,d] / sqrt(D)
```

一个实用规则：

> 出现在同一求和符号里的下标必须在各项中匹配；未被求和的下标会留在输出中。

上例中 `d` 被求和，剩下 `b,h,i,j`，正好对应 `[B,H,T,T]`。若论文省略了下标，自己
补出来往往比反复读文字更快。

### 一份常用符号表

| 符号 | 常见含义 | 常见形状 |
| --- | --- | --- |
| `B` | batch size | 标量 |
| `T`、`T_q`、`T_k` | 序列/query/key 长度 | 标量 |
| `H`、`H_q`、`H_kv` | attention head 数 | 标量 |
| `C` | model width | 标量 |
| `D`、`D_k`、`D_v` | 每个 head 的 key/value 宽度 | 标量 |
| `R` | feature/state rank 或 feature dimension | 标量 |
| `Q,K,V` | query/key/value | 通常 `[B,H,T,D]` |
| `S` | score matrix 或 state，必须看上下文 | `[B,H,T,T]` 或 `[B,H,R,D]` |
| `P` | attention probability | `[B,H,T,T]` |
| `m,l` | online softmax 的最大值/归一化统计 | 常按 query row 保存 |

`S` 尤其危险：在 dense attention 里常是 score，在 linear-attention 论文里常是 state。
不要因为字母相同就假定形状相同。

## 4.2 四种最常见的乘法

令 `a,b in R^D`，`A in R^[M,N]`，`B in R^[N,P]`。

### 内积（dot product）

```text
a^T b = sum_d a_d b_d      # scalar
```

attention score `q_i^T k_j` 就是内积。它将两个 `D` 维向量压成一个标量相似度。

### 外积（outer product）

```text
a b^T                      # [D,D]
(a b^T)[i,j] = a_i b_j
```

线性 attention 的 `phi(k) v^T` 是外积。它不是点积：它把两个向量扩成一个矩阵，因此
可被累计成 state。

### 矩阵乘（matrix multiplication）

```text
C = A B                    # [M,P]
C[i,p] = sum_n A[i,n] B[n,p]
```

矩阵乘把中间维 `N` 消掉。`QK^T` 消掉 `D`；`PV` 消掉 key 位置 `j`。

### 逐元素乘（Hadamard product）

```text
c = a ⊙ b                  # [D]
c_d = a_d b_d
```

gate 通常是逐元素乘或按某一维 broadcast 的缩放。`a⊙b` 不能随意写成矩阵乘；二者的
输出形状、含义和计算成本都不同。

### 一个形状速查例子

```text
q: [D]       k: [D]       v: [D_v]
q^T k:       []           # scalar
k v^T:       [D,D_v]      # outer product
q^T (k v^T): [D_v]        # 先左乘 state 的读出
```

这正好是[第 3 章](03-rnn-state-and-scan.md)中 linear-attention state 的局部结构。

## 4.3 Tensor 轴、broadcast 与 Einstein 求和

真实代码有 batch/head 维后，手写许多 `transpose` 很容易出错。`einsum` 是把下标公式
直接写进代码的好工具。

```python
# q, k: [B,H,T,D]
scores = torch.einsum("bhtd,bhsd->bhts", q, k)

# probs: [B,H,T,S]，v: [B,H,S,D]
out = torch.einsum("bhts,bhsd->bhtd", probs, v)
```

字母在两个输入都出现、但输出没有出现时，会被求和。上例第一个表达式消去 `d`，第二个
表达式消去 key 位置 `s`。对照[第 1 章](01-transformer-from-tensors.md)的下标公式读一遍，
你就能检查普通 `@` 被复杂 batch 维隐藏的含义。

### broadcast 不是复制千万次数据

设：

```text
scores: [B,H,T,T]
mask:   [T,T]
```

写 `scores.masked_fill(mask, -inf)` 时，框架在逻辑上把 mask 看成 `[1,1,T,T]`，并沿 batch
和 head 维广播。它通常不真的创建 `B*H` 份 mask，但运算语义等价于每个 batch/head 都
使用同一张 mask。

常见广播规则从右向左对齐：两个维度相等，或其中一个为 1，才可兼容。比如 `[B,T,1]`
和 `[B,1,D]` 可以相乘得到 `[B,T,D]`；`[B,T]` 与 `[B,D]` 在 `T != D` 时则不能凭直觉
广播成功。

### `reshape`、`view`、`transpose` 的区别

- `view`：只改变形状解释，要求当前 stride 与目标形状兼容；不兼容时会直接报错；
- `reshape`：优先返回 view，但 stride 不兼容时允许创建 contiguous copy；
- `transpose/permute`：改变轴的逻辑顺序；
- `contiguous`：在需要时重新安排内存，让某种轴顺序连续；
- `clone`：复制数值，通常不是 shape 操作。

因此 `reshape` 与 `view` 得到的数值可以相同，却不能据此断言二者都“零拷贝”。数学上，
给 tensor 增删一个长度为 1 的轴或重排轴不改变元素；工程上，它可能改变 stride
和后续 kernel 是否能连续读写。不要把“某行只是 transpose”理解成一定没有成本。具体
成本依赖于后续是否需要 materialize contiguous copy。

## 4.4 softmax、log-sum-exp 与数值稳定性

softmax 对向量 `x` 的定义为：

```text
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

直接计算 `exp(x)` 会在大 logit 时溢出。令 `m=max_i x_i`：

```text
softmax(x)_i = exp(x_i-m) / sum_j exp(x_j-m)
```

因为分子分母都乘了 `exp(-m)`，数学结果不变。交叉熵中常见的：

```text
logsumexp(x) = log(sum_i exp(x_i))
             = m + log(sum_i exp(x_i-m))
```

也是同一技巧。

### 为什么“先 softmax 再分块相加”不行

softmax 的分母依赖整行所有 key。若将 key 分成 A、B 两块，各自单独 softmax 再直接拼接，
每一块都把自己的权重归一到 1，整体不再正确。FlashAttention 的重要思想是在不存储整行
的前提下维护足够统计量，正确合并块。

### online softmax 的块合并

对一个 query row，块 A 的稳定统计为：

```text
m_A = max(x in A)
l_A = sum_(i in A) exp(x_i-m_A)
u_A = sum_(i in A) exp(x_i-m_A) v_i     # 向量
```

块 B 类似。合并时令 `m=max(m_A,m_B)`：

```text
l = exp(m_A-m) l_A + exp(m_B-m) l_B
u = exp(m_A-m) u_A + exp(m_B-m) u_B
output = u / l
```

这里 `m` 是标量，`l` 是正标量，`u` 的形状与 value 向量相同。你可以把 A/B 看成任意两个
相邻 key tile；反复合并等价于对整行 softmax。它正是 FlashAttention 能一块一块读取 K/V
而仍保持 exact attention 的数学基础之一。

### 手算 online softmax

令块 A 的 score/value 为 `(0, 2)`、`(log 2, 6)`，块 B 为 `(0, 10)`，每个 value 都是
scalar。先求：

```text
m_A = log 2,  l_A = 1/2 + 1 = 3/2,  u_A = 1/2*2 + 1*6 = 7
m_B = 0,      l_B = 1,              u_B = 10
```

合并 `m=log 2`，所以：

```text
l = 3/2 + (1/2)*1 = 2
u = 7   + (1/2)*10 = 12
output = 6
```

直接对原始 scores `[0,log2,0]` 做 softmax，权重是 `[1/4,1/2,1/4]`，输出同样是
`1/4*2+1/2*6+1/4*10=6`。两种顺序结果一致；区别在于块算法不需要把全部 score/probability
写到高带宽显存。

## 4.5 结合律、交换律与不能随便改括号的地方

阅读算法改写时，需要区分三条性质：

| 性质 | 例子 | 对计算重排的含义 |
| --- | --- | --- |
| 结合律 | `(a+b)+c=a+(b+c)` | 可改变括号/归约树 |
| 交换律 | `a+b=b+a` | 可改变顺序（浮点中仍有数值差异） |
| 分配律 | `a(b+c)=ab+ac` | 可把计算展开/提取公因子 |

矩阵乘满足结合律，但一般**不满足交换律**：`AB` 通常不等于 `BA`，甚至形状也可能不同。
因此：

```text
(Q K^T) V = Q (K^T V)
```

在不含 softmax 时可以依据结合律重排，这正是某些线性/核方法得到 state 的起点；但：

```text
softmax(Q K^T) V != Q (softmax(K^T V))
```

softmax 夹在中间，不能被随意穿过去。看到“attention 可以线性化”的推导时，第一问应是：
作者怎样处理/替换了 softmax？

在[第 3 章](03-rnn-state-and-scan.md)中，scan 可行是因为仿射状态转移的**函数复合**满足
结合律。它同样不允许交换 token 顺序；只允许用不同的括号方式组合保持原顺序的变换。

## 4.6 对角、低秩与 DPLR：给 Kimi/SSD 的预览

一个一般的 `D x D` 状态转移矩阵 `A` 乘向量的成本约为 `O(D^2)`，存储也为 `O(D^2)`。
若它有结构，计算可以便宜得多。

### 对角矩阵

```text
A = diag(a)
A x = a ⊙ x
```

只需 `O(D)`。这可理解为每个状态通道独立衰减/放大，是许多 gate/状态空间模型的基本
结构，但单纯对角可能表达力有限。

### 低秩矩阵

若：

```text
A = U V^T
U: [D,r], V: [D,r], r << D
```

则：

```text
A x = U (V^T x)
```

成本约 `O(D*r)`，而不是 `O(D^2)`。这里 `r` 叫 rank 上限；它不是“只有 r 个非零元素”。

### Diagonal-Plus-Low-Rank（DPLR）

把两者合在一起：

```text
A = diag(a) + U V^T
A x = a ⊙ x + U(V^T x)
```

DPLR 同时提供逐通道的直接控制和少量跨通道耦合。Kimi Linear 报告的 KDA chunkwise
algorithm 使用了专门的 DPLR transition 表示，以降低状态转移组合/计算成本。此刻不用
硬背其具体 kernel；看到 DPLR 时，只要先检查：对角部分的维度是什么、低秩 `r` 是多少、
它是在 token 内更新还是 chunk 间合并、比较对象是否是一般稠密矩阵。

一个重要边界：`A=diag(a)+UV^T` 的乘向量快，不自动说明多个这样的矩阵相乘后仍保持
同样低 rank。实际算法往往利用更具体的结构、分块方式或重新参数化；不要擅自把“DPLR
单步”推广成“任意长序列的所有组合都免费”。

## 4.7 Mask、稀疏集合与 block sparse

causal mask 可以写成布尔集合：

```text
M(i,j) = 1 if j <= i else 0
```

稠密 attention 默认对所有 `(i,j)` 形成 score，再将 `M=0` 的位置排除。sparse attention
则定义一个更小的可见集合 `N(i)`：

```text
o_i = sum_(j in N(i)) p_(i,j) v_j
```

关键不只在 `|N(i)|` 小，还在集合怎样表示：

- **固定结构**：sliding window、global token、dilated pattern，容易按块布局；
- **block sparse**：选择 `[block_q, block_k]` 对，索引与 tile 对齐；
- **动态选择**：不同 query 选择不同 key，可能有评分、排序、gather 和不规则访问成本；
- **KV eviction**：集合随 decode 更新，选择策略本身也需要时间和状态。

“稀疏率”只描述被保留的数学配对比例。若先算 dense score 再根据它选 top-k，或不规则
gather 使带宽很差，端到端加速可能很小。以后读 Native Sparse Attention、MoBA、MInference、
SpargeAttention 时，要同时画出数学 mask 和实际 block/index 数据结构。

## 4.8 浮点数：为什么正确结果不总是逐位相等

实数加法满足结合律，浮点加法受舍入限制通常不严格满足：

```text
(a + b) + c may differ from a + (b + c)
```

把长归约从串行顺序改成并行树、从一个 tile 改成另一个 tile，都可能改变最后几位。这不
必然是 bug。正确性测试通常比较：

```text
absolute_error = max(abs(test - reference))
relative_error = max(abs(test-reference) / (abs(reference)+tiny))
```

并以 `allclose(atol, rtol)` 等规则判断。选择阈值时要报告 dtype、shape、随机输入分布和
reference 实现，不能只写“误差很小”。

### 常见 dtype 的直觉

| dtype | 直觉用途 | 后续实验的提醒 |
| --- | --- | --- |
| fp32 | 较稳的 reference/accumulation | 不代表生产最快路径 |
| bf16 | 动态范围大、现代训练常用 | 尾数较粗，误差容忍度需更宽 |
| fp16 | 节省显存、吞吐常高 | 动态范围较小，易有溢出/下溢风险 |
| fp8 | 更激进的低精度路径 | 不是初次复现的起点 |

同一算法在 fp32 与 bf16 的性能结论可能不同；同一 kernel 在不同 GPU 代际也可能不同。
因此不能把某篇 A100 fp16 图直接当作 5090 或其他节点上的承诺。

## 4.9 复杂度、FLOPs、字节与屋顶线直觉

Big-O 描述变量足够大时的增长趋势，不描述所有实际速度。以每 head 的 dense attention
为例：

```text
QK^T:  about T_q * T_k * D multiply-add work
PV:    about T_q * T_k * D_v multiply-add work
```

若 `T_q=T_k=T` 且 `D_v≈D`，就是 `O(T^2D)`。但实际运行还要读写 Q/K/V、输出、临时
统计量和可能的 `[T,T]` score/probability。

一个粗略但有用的性能视角是算术强度：

```text
arithmetic intensity = useful FLOPs / bytes moved from/to main memory
```

- 算术强度高、计算足够大时，可能接近算力受限；
- 算术强度低或访问不连续时，可能受内存带宽限制；
- 太小的工作量还可能受 kernel launch、调度和同步限制。

FlashAttention 的主要贡献可理解为降低 HBM 上不必要的 `T x T` 中间读写，提高有效算术
强度，而不是改变 dense attention 的 `O(T^2D)` 数学配对。稀疏方法希望减少配对，但也
需要让剩余工作具有足够规则的访问；线性方法希望降低随 `T` 的增长，但状态 update/scan
也要实现为高吞吐块计算。

### 不要只报告一个“速度倍数”

每次 benchmark 至少固定并报告：

```text
GPU 与驱动/CUDA
软件版本与 commit
dtype、B、T、H_q、H_kv、D
causal / non-causal，dropout，forward 或 backward
warmup 次数、测量次数、同步方式
比较的 baseline 与是否调用优化库
正确性误差与峰值显存测量口径
```

吞吐、延迟、显存三者不可混用：`tokens/s` 可能包含不同 batch；平均 latency 可能掩盖
长尾；峰值 allocated 和 reserved memory 也不是同一指标。详细的实验记录格式会在后续
实验部分统一。

## 4.10 代码阅读练习：用 `einsum` 写出两次归约

给定：

```python
# q, k, v: [B, H, T, D]
scores = torch.einsum("bhtd,bhsd->bhts", q, k)
probs = torch.softmax(scores, dim=-1)
out = torch.einsum("bhts,bhsd->bhtd", probs, v)
```

完成：

1. 对 `scores` 和 `out` 分别写出一个完整下标公式；
2. 为什么第一个 `einsum` 的输出包含两个序列字母 `t,s`，第二个只保留 `t`？
3. 若 `v` 的最后一维改为 `D_v=128`、而 `q/k` 的最后一维为 `D_k=64`，第二个式子怎样
   改，输出形状是什么？
4. 若 `scores` 是 `[B,H,T,T]`，causal mask 的 `-inf` 应加在哪一维之前的 softmax？
5. 用一个 `T=3` 的例子说明为何 `softmax(dim=-2)` 与 `softmax(dim=-1)` 语义不同。

核对：

```text
scores[b,h,t,s] = sum_d q[b,h,t,d] k[b,h,s,d]
out[b,h,t,d] = sum_s probs[b,h,t,s] v[b,h,s,d]
```

若 value 宽度独立，写作 `v:[B,H,S,D_v]`，输出为 `[B,H,T,D_v]`。causal attention
应对 key 轴 `s` 做 softmax，也就是最后一维。

## 4.11 手算练习：DPLR 与 shape 检查

令：

```text
a = [2,3]
U = [[1], [2]]        # [2,1]
V = [[4], [5]]        # [2,1]
x = [1,2]
A = diag(a) + U V^T
```

1. 不显式写完整 `A`，按 `A x = a⊙x + U(V^T x)` 求结果；
2. 再写出完整 `A`，验证结果一致；
3. `U`、`V` 的 rank 上限是多少？它们的外积为何仍是 `[2,2]`？
4. 若 `D=4096,r=8`，比较存储一个一般 `D x D` 矩阵与存储 `a,U,V` 的元素数量级；
5. 这是否足以证明长序列中的所有 transition composition 都是 `O(D*r)`？为什么？

答案要点：`V^T x=14`，所以 `a⊙x=[2,6]`，`U*14=[14,28]`，结果 `[16,34]`。
完整矩阵为 `[[6,5],[8,13]]`。外积 rank 至多 1；最后一问答案是否定的，组合后的结构
是否保持可处理形式取决于具体算法。

## 4.12 研究中最容易犯的数学/测量误区

**误区 1：维度相同就能相乘。**
矩阵乘要求左矩阵最后一维等于右矩阵倒数第二维；逐元素乘要求可 broadcast。写形状能立刻
抓住大部分错误。

**误区 2：`K^T V` 与 `V K^T` 差不多。**
它们一般形状、语义均不同。前者可能是 `[D,D_v]` 的 state，后者是 `[D_v,D]`。

**误区 3：softmax 可以自由穿过矩阵乘。**
不行。只有在作者明确替换 kernel/normalizer 或利用特殊结构时，才可能有新的重排。

**误区 4：浮点 allclose 就证明算法相同。**
还要检查 mask、边界、不同长度、dtype、随机 seed、backward 和极端输入。

**误区 5：稀疏率等于速度提升。**
稀疏的选择、索引、gather、负载不均和 block 对齐都可能吞掉收益。

**误区 6：复杂度中的 `O(T)` 忽略 `D` 和常数。**
linear state 若为 `D x D`，在实际 `D` 上可能很重；而高效 dense GEMM 的常数可能很低。

**误区 7：数学上合法的结合重排必然 bitwise 相等。**
浮点归约顺序不同会产生微小差异，应以合理容差和数值稳定性分析判断。

## 4.13 把工具箱用于后续阅读的固定流程

遇到任意一篇 sparse/linear attention 论文或 FLA operator，按下面顺序处理：

1. **重命名符号。** 把作者的 `n,d,m,r` 翻译成自己的 `B,H,T,D,R`，写所有形状；
2. **展开一项。** 将关键矩阵式展开成下标求和，确认谁被归约；
3. **找状态或 mask。** 问历史是完整 KV、固定 state，还是选择的子集；
4. **找可重排处。** 作者用了结合律、低秩、分块、近似还是新 normalizer？
5. **找数据移动。** 哪些张量是 `[T,T]`、随 `T` 增长，哪些能够留在片上；
6. **找误差边界。** exact/近似/重新训练的模型族分别是什么意思；
7. **找实验口径。** dtype、shape、GPU、forward/backward/prefill/decode 和 baseline 是否可比。

每读一篇 A 级论文，至少把第 1–4 步写进论文笔记。每读一个 kernel，至少把第 5–7 步写
进实验记录。这样数学不是独立作业，而是后续调研报告的证据链。

## 本章小结

attention 研究最常用的数学不复杂，但要求严格：按下标检查张量收缩，按数值稳定方式
处理 softmax，按结合律而非想象重排递归，按结构分析对角/低秩状态转移，按 FLOPs 与
字节共同判断性能。FlashAttention 的 online softmax、linear attention 的外积 state、
Kimi/SSD 的结构化 transition、sparse attention 的 mask 都可以放进这套工具箱。

你的目标不是见到公式就立即推完证明，而是能明确说出：这行公式在算什么、它的形状和
复杂度是什么、它为何可能/不可能在 GPU 上变快、它与标准 dense attention 的差异来自
哪里。

## 通过条件

不看正文，完成以下任务，才算通过本章：

- 给出 `Q:[B,H,T,D]`、`K:[B,H,S,D]` 时，写出 score 和 output 的形状及下标式；
- 区分内积、外积、矩阵乘、逐元素 gate，并各举一个 attention 中的例子；
- 用 `m,l,u` 正确合并两个 online-softmax 块；
- 解释为什么普通 softmax attention 不能直接从 `(QK^T)V` 改成 `Q(K^TV)`；
- 用一句话说明 DPLR 的计算优势和一个不能过度推广的边界；
- 对一份 benchmark 列出至少八项必要元数据；
- 对一个新公式能先写形状和求和下标，再讨论“它是否更快”。

## 延伸材料

- Stanford CS336, [Basics: Tensors, Einstein notation, and numerical precision](https://stanford-cs336.github.io/spring2025/)：
  作为按需查阅的数学/系统补充。
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)：完成本章后优先读 online
  softmax 与 IO-aware 算法部分，再看性能图。
- MLC, [Modern GPU Programming for ML Systems（中文）](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/)：
  现在优先读执行模型、数据布局、GEMM 分块与 softmax；Blackwell 专属章节留到有对应硬件
  或已有前面基础后再读。
