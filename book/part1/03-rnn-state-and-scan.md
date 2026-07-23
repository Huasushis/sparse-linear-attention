# 第 3 章：RNN、状态与 scan——走进 linear attention 的桥

很多 linear attention 论文有一种初看很反直觉的表述：它既像 Transformer，又可以像 RNN
一样维护状态；它训练时又能并行，推理时却递归更新。这个章节的任务，就是把这三个
看似矛盾的说法拆开。

先给出一句最重要的话：**“递归定义”不等于“只能一个 token 接一个 token 地执行”。**
能否并行取决于状态更新能否组合成一个结合的摘要（scan），以及怎样在 GPU 上安排这些
摘要。这个区别正是 linear-attention algorithm 和 kernel 之间的接口。

## 学习目标

读完后，你应当能够：

1. 用状态 `h_t` 写出最简单 RNN 的前向计算，并说明其因果性；
2. 用 prefix sum 解释 scan 的含义，知道“结合律”为什么重要；
3. 写出核特征 linear attention 的 state、更新和读出公式，并标出形状；
4. 区分 dense KV cache 与线性方法的固定大小 state；
5. 解释 recurrent、parallel 和 chunkwise 三种执行视角；
6. 读懂 GLA、DeltaNet、Gated DeltaNet、Kimi Delta Attention 属于同一状态更新谱系的原因；
7. 不把 `O(T)` 的算法复杂度误读成“任何 GPU、任何 shape 都一定更快”。

建议先完成[第 1 章](01-transformer-from-tensors.md)的 attention 形状练习。本章会多次
使用 `B,H,T,D`，也会在最后接回[第 2 章](02-training-minimum.md)的 backward 问题。

!!! info "数学按需预读"
    如果“外积”“结合律”“转置后形状”还不熟，先看第 4 章的 4.2--4.5，再回来继续。
    不需要先学完整门线性代数。

## 3.1 什么是“状态”

考虑按时间读取输入 `x_1, x_2, ..., x_T`。一个最简单的循环模型写作：

```text
h_t = f(h_(t-1), x_t)
o_t = g(h_t)
```

`h_t` 叫 hidden state。它是到时刻 `t` 为止历史信息的压缩摘要；`o_t` 是此时输出。
若只从过去状态和当前输入计算 `h_t`，就天然满足因果性：`o_t` 不会依赖未来 `x_(t+1)`。

### 一个小得不能再小的例子

```text
h_t = 0.8 * h_(t-1) + x_t,  h_0 = 0
```

给定 `x=[1,2,3]`：

```text
h_1 = 1
h_2 = 0.8*1 + 2 = 2.8
h_3 = 0.8*2.8 + 3 = 5.24
```

系数 0.8 使较早的信息逐渐衰减，因此可被看作一种简单的 forget gate。真实模型的
state 往往是向量或矩阵，gate 可以依赖当前 token，但这个标量例子已经包含了后文的
两个关键：**把历史压进 state**，以及**用更新规则控制保留/遗忘**。

### RNN 不是“过时技术”的同义词

早期 RNN、LSTM/GRU 和现代 state-space/linear-attention 模型都使用状态递归，但状态
的结构、训练并行方式、数值稳定性和硬件实现差别很大。把后者简单叫作“旧 RNN”会掩盖
真正需要研究的算法与 kernel 创新。

## 3.2 从 prefix sum 学 scan

先暂时不谈神经网络。给定：

```text
x = [3, 1, 4, 1]
```

prefix sum（前缀和）输出：

```text
[3, 4, 8, 9]
```

递归写法是 `s_t = s_(t-1) + x_t`，看起来必须依次计算。然而加法满足结合律：

```text
(a + b) + c = a + (b + c)
```

所以可以先让不同线程或不同块各自求局部和，再把块摘要组合。例如把四个数分为
`[3,1]` 和 `[4,1]`：局部前缀分别为 `[3,4]`、`[4,5]`；第二块只需加上第一块总和 4，
就得到 `[8,9]`。

这类“给出每个位置之前的归约结果”的操作叫 **scan** 或 **prefix scan**。sum 是最熟悉的
scan，但不是唯一一个。

### affine recurrence 也能组合

把前面的状态更新稍微一般化：

```text
h_t = A_t h_(t-1) + b_t
```

此处 `A_t` 可以是标量、逐元素 gate、矩阵，`b_t` 是当前 token 注入的量。每一步可看成
一个变换对 `(A_t,b_t)`。若先做变换 1，再做变换 2，合成变换是：

```text
(A_2, b_2) compose (A_1, b_1)
  = (A_2 A_1, A_2 b_1 + b_2)
```

它仍然满足结合律，因为矩阵乘法本身满足结合律。于是可以对“变换摘要”做 scan，不必把
每一个状态严格串行到底。若 `A_t` 是逐元素 gate，则乘法是逐元素乘；若 `A_t` 有特殊
低秩/对角结构，组合可以更便宜。这是理解 chunkwise algorithms、SSD 和 Kimi KDA 中
状态转移矩阵时应抓住的骨架。

注意：可结合不表示实现自动高效。矩阵摘要可能很大，组合也可能昂贵；实际 kernel 还要
处理 tile、寄存器、共享内存、读写和数值误差。这正是后续性能研究的空间。

## 3.3 标准 softmax attention 为什么难以直接压成固定 state

对单个 head 的因果 softmax attention：

```text
o_t = sum_(j<=t) softmax_j(q_t^T k_j / sqrt(D)) * v_j
```

每个新的 `q_t` 都会用不同的权重重新评价所有历史 key。一般情况下，若要完全精确地
回答任意未来 query，似乎需要保留所有过去的 `k_j,v_j`，也就是 KV cache 的长度随 `T`
增长。

如果相似度可以写成有限维特征内积：

```text
similarity(q,k) approximately phi(q)^T phi(k)
```

那么求和次序可以改写。这里“approximately”很重要：softmax kernel 通常不能用一个很
小的有限维 `phi` 在任意输入上精确表示。有的线性注意力用随机/正特征近似它；有的干脆
采用不同的 normalizer 或状态更新规则，得到一种新的模型族，而非原 softmax 的严格
等价物。

## 3.4 核特征 linear attention：状态从哪里来

令：

```text
phi(q_t), phi(k_t): [R]        # R 是 feature dimension
v_t:               [D_v]
```

定义两个前缀状态：

```text
S_t = sum_(j<=t) phi(k_j) v_j^T     # [R, D_v]
z_t = sum_(j<=t) phi(k_j)           # [R]
```

给定当前 query，输出为：

```text
numerator_t   = phi(q_t)^T S_t      # [D_v]
denominator_t = phi(q_t)^T z_t      # scalar
o_t = numerator_t / (denominator_t + eps)
```

把它解释为“归一化加权平均”时，通常要求 `phi(q)` 与 `phi(k)` 非负；否则 denominator
可能为零或为负。教学代码使用 `ELU(x)+1` 来满足这一点。`eps` 只处理数值边界，不能把
任意带符号 feature map 自动变成概率权重。

把分子展开即可验证：

```text
phi(q_t)^T S_t
= sum_(j<=t) phi(q_t)^T phi(k_j) * v_j
```

这正是把“对每个历史位置重新做乘法”改成“先把历史汇总为状态，再读取状态”。更新也只
需要当前 token：

```text
S_t = S_(t-1) + phi(k_t) v_t^T
z_t = z_(t-1) + phi(k_t)
```

`phi(k_t) v_t^T` 是外积，形状 `[R,1] @ [1,D_v] -> [R,D_v]`。这类随 token 更新的
矩阵状态常被称为 **fast-weight state**。

### 加上 batch 和 heads 后的形状

若每个 head 独立、所有 head 使用同一 `R,D_v`，则：

```text
K-feature: [B, H, T, R]
V:         [B, H, T, D_v]
S:         [B, H, R, D_v]
z:         [B, H, R]
```

state 不随 `T` 增长，但不意味着它很小：当 `R` 与 `D_v` 都接近 head dimension 时，
`S` 约有 `D^2` 元素。对长 decode，`D^2` state 可能比 `T*D` KV cache 有利；对短序列或
高维 state，常数和算子形状可能并不占优。

### 一个小型手算状态

令 `R=2, D_v=1`，初始 `S_0=[0,0]^T, z_0=[0,0]^T`。第一个 token：

```text
phi(k_1)=[1,2],  v_1=3
```

则 `S_1=[3,6]^T, z_1=[1,2]^T`。第二个 token：

```text
phi(k_2)=[2,1],  v_2=4
```

则：

```text
S_2 = [3,6]^T + [2,1]^T*4 = [11,10]^T
z_2 = [1,2]^T + [2,1]^T   = [3,3]^T
```

若 `phi(q_2)=[1,1]`，输出是 `(1*11+1*10)/(1*3+1*3)=21/6=3.5`。你应该能直接将它
展开成对两个 value 的加权平均来核对。

## 3.5 Gate：为什么只做累加还不够

上面的状态把所有历史同等累加。真实序列经常需要忘掉不相关内容，或针对不同特征以
不同速度遗忘。一个示意性的 gated state 可以写为：

```text
S_t = alpha_t ⊙ S_(t-1) + k_t v_t^T
```

其中 `alpha_t` 的取值常在 0 与 1 附近，`⊙` 表示逐元素/按行广播的缩放。具体论文可能
让 gate 作用在 state 的某一个维度、使用 log-space 参数化，或同时有 input gate；此公式
只表达共同思想：**当前 token 决定旧 state 有多少能留下来。**

这与[第 1 章](01-transformer-from-tensors.md)的 dense attention 不同。dense attention
由当前 query 在读出时为所有历史位置临时打分；gated linear/state 方法在写入时就压缩
历史。因此它的效率和表达能力都来自同一选择：状态是有限容量的。

## 3.6 Delta rule：不仅写入，还尝试纠正旧记忆

另一类重要更新不是简单加法，而是朝“让当前 key 读出当前 value”的方向修正状态。为了
贴近部分 DeltaNet 论文，这里把前面的 state 转置后书写：

| 约定 | state 形状 | 读出 | 二者关系 |
| --- | --- | --- | --- |
| 本章 3.4 的 kernel-feature 写法 | `[D_k,D_v]` | `q_t^T S_t` | `S_feature` |
| 本节的 delta-rule 写法 | `[D_v,D_k]` | `S_t q_t` | `S_delta = S_feature^T` |

这只是朝向不同，不是突然换了另一种记忆。采用 delta-rule 约定：

```text
S_t: [D_v, D_k]
o_t = S_t q_t
```

一个 delta-rule 的示意式为：

```text
S_t = S_(t-1) - beta_t * (S_(t-1) k_t - v_t) k_t^T
```

其中：

```text
k_t: [D_k]
v_t: [D_v]
S_(t-1) k_t - v_t: [D_v]
(S_(t-1) k_t - v_t) k_t^T: [D_v, D_k]
```

如果当前 state 用 `k_t` 读出的结果与 `v_t` 有误差，更新就沿误差方向修正。`beta_t` 控制
步长。不同论文在 key 归一化、state 朝向、gate 和更新顺序上会有细节差异，阅读时不应
仅凭一眼相似就宣称公式相同；但“误差驱动的 fast-weight 写入”是它们共同的逻辑。

## 3.7 GLA、DeltaNet、Gated DeltaNet 与 Kimi：怎样建立正确的家谱

下面的表不是严格历史谱系，而是第一遍阅读时的概念地图：

| 工作 | 首先要抓的东西 | 与下一项的关系 |
| --- | --- | --- |
| `Transformers are RNNs` / linear transformers | 核特征、前缀 state、recurrent 读出 | 提供“attention 可以状态化”的起点 |
| Gated Linear Attention (GLA) | data-dependent forget gate、chunkwise 计算 | 说明 gate 与硬件并行如何共同设计 |
| DeltaNet | delta-rule / fast-weight 修正 | 不只累加，改为有误差的状态更新 |
| Gated DeltaNet (GDN) | gate 与 delta rule 的结合 | 当前线性注意力主线的重要节点 |
| Kimi Delta Attention (KDA) | 更细粒度 gate、适配硬件的 chunkwise 状态转移 | Kimi Linear 中使用的线性模块 |

导师说“Kimi 和 GDN 几乎重合”，在初学阶段最有用的理解是：**不要把 Kimi Linear 当作
必须从零学习的一条独立线性 attention 家族。** Kimi 的 KDA 明确建立在 Gated DeltaNet
方向上，并扩展了更细粒度的 gating；Kimi Linear 又在模型层面采用 KDA 与 MLA 的混合。
因此推荐顺序是：先读 GLA 的状态/gate 与 chunkwise 思想，再读 DeltaNet，接着精读
Gated DeltaNet，最后把 Kimi 作为“工业级混合架构和实现案例”比较其新增点。

“几乎重合”不应被误写成“公式、训练设置、模型架构、kernel 完全一样”。正式报告中应
逐项核对：state 公式、gate 粒度、chunk algorithm、模型混合比例、训练 token、硬件和
benchmark，才可以说某个具体层面相同或不同。

## 3.8 recurrent、parallel、chunkwise 是同一计算的三种观察角度

### Recurrent view：最适合理解 decode

本书的 causal linear-attention reference 采用 **write-then-read**：先把当前 `(k_t,v_t)`
写入，再由 `q_t` 读出，因此求和范围是 `j<=t`：

```text
for t in range(T):
    state = update(state, k[t], v[t], gates[t])
    output[t] = read(state, q[t])
```

另一种合法 API 会 read-then-write，对应 `j<t` 或“state 表示进入 token t 之前的历史”。
两者可通过移动一次更新互相转换，但实现、公式和测试必须选定同一契约；不能把 off-by-one
当作无关细节。

它直接说明为什么推理时不必保留完整 KV cache（取决于具体方法），但若训练也按这个
Python loop 做，GPU 会因大量小操作和时间依赖而利用率很差。

### Parallel view：最适合理解训练

若更新可表示成可结合的摘要，就能用 scan 一次性计算许多前缀状态。例如纯加法 state
可做 prefix sum；affine/gated state 可做仿射变换的 scan。这里的“并行”不是魔法地消除
全部依赖，而是改变组合树的形状，以更多临时计算/存储换较短的关键路径。

### Chunkwise view：最适合理解实际 kernel

将长度 `T` 切成大小 `C` 的 chunk：

```text
[0 ... C-1] [C ... 2C-1] [2C ... 3C-1] ...
```

一个 chunk 内，kernel 可用矩阵乘法或三角形式高效计算局部 token 间贡献；chunk 与
chunk 之间只传递较小的 state 或状态转移摘要。粗略地说：

```text
输出 = chunk 内局部项 + 进入该 chunk 的历史 state 贡献
```

chunk 过小会产生很多状态传递与 launch/调度开销；过大则局部计算可能接近需要避免的
大三角矩阵，且降低并行灵活性。最优块大小取决于 `D`、dtype、GPU、具体更新公式和是否
需要 backward，所以不能从算法复杂度直接猜出。

这就是阅读 FLA 时最应该画的图：一个 tile/chunk 接收哪些 `Q/K/V/gate/state`，在块内
累计什么，写出哪些 output 和最后 state，而不是试图第一天读懂每个 Triton 语法细节。

## 3.9 scan 的一个可读伪代码

下面没有高效实现，只展示“块摘要”需要表达什么。仍用标量仿射更新
`h <- a*h + b`：

```python
# 第 i 个 token 表示一个变换 (a[i], b[i])
def compose(later, earlier):
    a2, b2 = later
    a1, b1 = earlier
    return a2 * a1, a2 * b1 + b2

# 一个 chunk 的 summary：把任意入站 h 映射成出站 h
summary = (1.0, 0.0)  # identity: h -> h
for a_t, b_t in chunk:
    summary = compose((a_t, b_t), summary)
```

若 `summary=(A,B)`，任何 entering state `h_in` 出块后都是 `A*h_in+B`。不同 chunk 的
summary 可以再 compose；这就是能做 scan 的根本。真实 linear-attention kernel 的 state
可能是矩阵，`A` 可能含对角/低秩结构，但你在代码里寻找的仍是同一对象：**局部块怎样
摘要为可传递的 transition**。

## 3.10 算法“线性”与硬件“快”之间还隔着什么

即便 state 不随 `T` 增长，也不能写“linear attention 一定比 FlashAttention 快”。至少要
问：

1. 每个 token 更新 `S` 的计算是 `O(D)`、`O(D^2)` 还是其他？
2. 训练时用了真正并行的 scan/chunkwise kernel，还是逐 token Python loop？
3. state 是否放得进寄存器/共享内存，是否频繁读写 HBM？
4. shape 是短 prefill、长 prefill 还是单 token decode？
5. dtype、head dimension、batch 和硬件是什么？
6. 相比对象是朴素 attention、PyTorch SDPA，还是优化好的 FlashAttention？
7. 是否包含 backward，数值误差和模型质量是否可接受？

尤其在短序列训练中，FlashAttention 的 dense GEMM-like 工作负载可能非常适合 GPU；线性
方法的状态更新/scan 反而可能缺少足够算术强度或并行度。研究的价值不在于预设赢家，
而在于在明确条件下测量这组权衡。

## 3.11 代码阅读练习：把循环翻译成公式

阅读下面教学代码。它实现的是未归一化的、单 head、纯累加状态，不是完整生产级 linear
attention：

```python
state = torch.zeros(R, Dv)
normalizer = torch.zeros(R)
outputs = []

for t in range(T):
    kt = phi(k[t])                      # [R]
    qt = phi(q[t])                      # [R]
    state = state + torch.outer(kt, v[t])
    normalizer = normalizer + kt
    numerator = qt @ state              # [Dv]
    denom = qt @ normalizer + eps       # scalar
    outputs.append(numerator / denom)
```

请完成：

1. 将 `state`、`normalizer`、`numerator` 写成对 `j<=t` 的求和；
2. 若把 `state` 更新移到输出之后，模型读到的是 `j<t` 还是 `j<=t`？这和 causal 的
   “是否包含当前位置”有什么关系？
3. `torch.outer(kt,v[t])` 的形状为什么不是 `[R]`？
4. 为使其支持 batch 和 heads，state 应增加哪些维度？
5. 指出至少两个使此代码不适合做 GPU benchmark 的原因。

核对要点：第 2 问改变了 self-token 是否可见；第 4 问 state 可写成 `[B,H,R,Dv]`；第 5
问至少包括 Python 循环、每步小算子 launch、没有并行 scan、未处理 dtype/layout/backward。

## 3.12 手算练习：仿射摘要的结合律

设两个 token 的状态更新分别为：

```text
f_1(h) = 0.5h + 1
f_2(h) = 0.2h + 3
```

1. 将 `f_2(f_1(h))` 写成 `A*h+B`；
2. 若第三步是 `f_3(h)=h-2`，分别计算 `(f_3 compose f_2) compose f_1` 与
   `f_3 compose (f_2 compose f_1)`；
3. 从 `h_0=0` 出发，按 token 顺序求 `h_1,h_2,h_3`，核对合成摘要；
4. 说明为什么可以先在块内求 `(A,B)`，再对块做 scan。

答案：`f_2(f_1(h))=0.1h+3.2`；加入第三步后都是 `0.1h+1.2`；从零出发状态为 `1,3.2,1.2`。
关键不是交换 token 顺序，而是用不同括号组合保持原有顺序的函数复合。

## 3.13 常见误区

**误区 1：状态大小不随 `T` 增长，就表示信息完全无损。**
有限 state 是压缩；表达能力和长程记忆是否足够，必须由任务/训练实验检验。

**误区 2：linear attention 都是 softmax attention 的精确重排。**
多数不是。要查核特征近似、normalizer、gate 和 state update，而不是只看“linear”标签。

**误区 3：有递归就无法训练并行。**
若状态变换可结合，可用 scan/chunkwise；但能并行不代表实现自动高效。

**误区 4：scan 可以任意重排 token。**
不可以。结合律允许改变括号，不允许改变因果顺序。

**误区 5：Kimi 是与 GDN 无关的另一套基础理论。**
更有效的学习方式是把 KDA 放入 gated delta-rule 家族，先理解共同 state，再核对 Kimi
在 gate 粒度、DPLR chunk algorithm、MLA 混合和系统实现上的新增点。

**误区 6：`O(T)` 必胜 `O(T^2)`。**
渐近量级不含 GPU 利用率、state 维度、选择/scan 开销、序列长度和质量代价。

## 3.14 本章怎样接到论文、FLA 与后续实验

现在读一篇 linear-attention 论文，不妨先填这张“状态卡”：

```text
输入与输出：Q/K/V/gate 的形状是什么？
state：      具体是什么张量，形状是否随 T 增长？
write：      update(state, current token) 的公式是什么？
read：       output 如何从 state 和 current query 得到？
并行：       recurrence 如何转成 scan 或 chunkwise？
训练：       backward 需要什么 saved state/summary？
kernel：     block/tile 传递什么，读写哪里？
证据：       quality、forward、backward、prefill、decode 分别测了什么？
```

对 GLA/Gated DeltaNet/Kimi，先完成这一张卡再看性能图；它能防止你把算法章节与 CUDA/Triton
细节混成一团。对 FLA 代码，先从已有实现的一条 operator 路径画出这张卡，然后才尝试
修改一个维度、block size 或 layout。对 sparse attention，同样的状态思维也有用：很多
KV eviction 方法的“state”是被保留的 token 子集，而不是固定矩阵；这会带来不同的选择
成本与精度风险。

## 本章小结

状态模型把因果历史压进 `h_t` 或 fast-weight matrix。核特征 linear attention 通过把
历史 `key-value` 外积累计为 `S_t`，让当前 query 读取固定形状的 state；gate 和 delta rule
进一步控制遗忘与纠错。递归公式可以借助结合的变换摘要做 parallel scan，实际 kernel
通常采用 chunkwise 形式，在局部矩阵计算和跨块状态传递之间折中。

GLA、DeltaNet、Gated DeltaNet 与 Kimi KDA 应作为一条连续主线阅读。Kimi 的价值在于把
该主线带到大模型混合架构与工程实现，但并不要求初学者先跳过前面的共同基础。

## 通过条件

不看正文，完成以下任务，才算通过本章：

- 写出 `h_t=A_t h_(t-1)+b_t` 的两个相邻变换如何合成；
- 用自己的话解释 scan 允许改变什么、不允许改变什么；
- 写出 `S_t,z_t` 的形状、更新和 linear-attention 读出公式；
- 解释 KV cache 与固定 state 各自怎样随 `T` 变化；
- 区分 recurrent、parallel、chunkwise view，并为每一种说出适用场景；
- 用“共同主线 + 逐项核对差异”的方式解释 Kimi 与 Gated DeltaNet 的关系；
- 列出至少四个使某个 linear kernel 可能没有 dense baseline 快的原因。

## 延伸材料

- Katharopoulos et al., [Transformers are RNNs](https://arxiv.org/abs/2006.16236)：优先读
  recurrent formulation 与 causal 实验，不必一开始追全部近似理论。
- Smith et al., [Parallel Prefix Scan](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)：
  作为 scan 的硬件直觉材料；代码年代较早，概念仍适用。
- FLA, [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)：后续沿一条
  GLA/Gated DeltaNet operator 路径阅读，不把仓库当作第一本教材。
- Kimi Team, [Kimi Linear](https://github.com/MoonshotAI/Kimi-Linear)：等完成 GDN 主线后，用其
  KDA、DPLR chunkwise algorithm 与 MLA 混合做对照阅读。
