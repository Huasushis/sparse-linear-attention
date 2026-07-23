# 第 1 章：沿着张量走一遍 Transformer

这一章不从“Transformer 是一种神经网络架构”开始，而从一批整数怎样变成下一
个 token 的概率开始。后面读 sparse attention、linear attention 和 GPU kernel 时，
几乎每一个争论最终都会落回三个问题：输入输出是什么形状，哪些中间量真的被算出
来了，以及它们被存在哪里。现在把这条数据流走扎实，之后才不会只记住论文里的彩色
方框。

本章默认你已经看过一次 3Blue1Brown 的 Transformer/attention 视频。视频适合建立
画面感；本章负责把画面落到张量、下标和可检查的计算上。

## 学习目标

读完后，你应当能够：

1. 从 token id 开始，说清 embedding、Transformer block、logits 各自的形状；
2. 不依赖图示，写出 causal multi-head self-attention 的完整前向计算；
3. 用下标解释 `QK^T` 的两个序列维分别代表什么；
4. 区分训练、prefill 和 decode 时 attention 的计算形态；
5. 解释 dense、Flash、sparse、linear 四个词分别改变了什么，没有改变什么；
6. 看到一段 PyTorch attention 代码时，能标注每一行的输入输出形状。

这里暂时不要求你会训练模型。loss、反向传播和 AdamW 会在[第 2 章](02-training-minimum.md)
补齐。

## 1.1 一组贯穿全章的形状

先固定一组很小的符号。真实模型的数字会大得多，但计算关系完全一样。

| 符号 | 含义 | 本章示例 |
| --- | --- | ---: |
| `B` | batch size，一次处理几条序列 | 2 |
| `T` | 当前序列长度 | 4 |
| `C` | 模型宽度，常写作 `d_model` | 8 |
| `H` | query head 数 | 2 |
| `D` | 每个 head 的维度，常写作 `d_head` | 4 |
| `V` | 词表大小 | 32,000 |
| `L` | Transformer block 数 | 暂不固定 |

这里恰好有 `C = H * D = 8`。许多常规模型也满足这个关系，但不要把它当成矩阵乘法
的自然定律；它是模型设计者选出来的配置。

一批输入 token id 的形状是

```text
token_ids: [B, T] = [2, 4]
```

每个元素只是一个整数索引，例如 `314`。它不是 314 维向量，也不表示“这个词有
314 个单位的意义”。embedding table 是一个可学习矩阵：

```text
E: [V, C] = [32000, 8]
X = E[token_ids]
X: [B, T, C] = [2, 4, 8]
```

`X[b, t, :]` 才是第 `b` 条序列、第 `t` 个位置的表示。后文把最后一维 `C` 称为
**特征维**，把 `T` 称为**序列维**。能持续分清这两类维度，是阅读 attention 代码
最重要的基本功之一。

### 位置从哪里来

仅查 embedding 表时，同一个 token 在不同位置会得到同一个向量。因此模型还需要
位置信息。经典 Transformer 把位置编码加到 `X` 上；现代语言模型常把 RoPE 作用到
每层的 `Q` 和 `K` 上。现在只需记住两点：

- 位置处理不会凭空增加一个序列维，张量仍保持原有形状；
- RoPE 会改变 query/key 的匹配分数，但不直接旋转 value。

后面复现论文时必须记录位置编码设置。一个 attention 变体在短序列上正确，并不代表
换一种 RoPE scaling 后仍能在长上下文上保持质量。

## 1.2 一个 Transformer block 在做什么

现代 decoder-only 模型的一层常可概括为 pre-norm 形式：

```text
U = X + Attention(Norm(X))
Y = U + MLP(Norm(U))
```

输入输出都是 `[B, T, C]`。这件事看似平凡，却很重要：多层 block 可以堆叠，是因为
每层遵守相同的形状接口。

- **Norm** 在每个 token 的特征维上做归一化，通常不混合不同位置；
- **Attention** 让一个位置读取其他可见位置的信息；
- **MLP** 分别处理每个位置，通常先把特征维扩宽，再投影回 `C`；
- **Residual connection** 把模块输出加回输入，要求两者形状相同。

因此，只有 attention 在这一层中显式混合序列位置。后面所有 sparse/linear attention
工作，主要都是替换或重写这个模块，而不是把整个 Transformer 都推倒重来。

## 1.3 从 `X` 得到 `Q`、`K`、`V`

先忽略 Norm。对输入 `X in R^[B,T,C]` 做三个可学习线性投影：

```text
Q_flat = X W_Q
K_flat = X W_K
V_flat = X W_V
```

若 query、key、value 都有 `H` 个 head，三个权重都可看作 `[C, H*D]`，于是投影结果
都是 `[B, T, H*D]`。接着只做 reshape 和换轴：

```text
[B, T, H*D] -> [B, T, H, D] -> [B, H, T, D]
```

所以本章示例中：

```text
Q, K, V: [2, 2, 4, 4]
```

不要把 `reshape` 理解成“又学了一次变换”。它没有新参数，通常只是重新解释元素的
组织方式。换轴则可能影响内存是否连续，后面写 kernel 时会变得重要，但数学上只是
改变下标顺序。

### Q、K、V 的直觉要有边界

常见说法是：query 表示“我想找什么”，key 表示“我能怎样被匹配”，value 表示
“匹配后取走的信息”。这是有帮助的直觉，但它们不是数据库中由人写好的查询、键和值，
而是训练学到的向量。判断一个实现是否正确，最终要回到公式和形状，而不能只靠比喻。

## 1.4 `QK^T` 到底乘了什么

对固定的 batch `b` 和 head `h`，取出：

```text
Q[b,h,:,:]: [T, D]
K[b,h,:,:]: [T, D]
```

第二个矩阵在最后两维转置后是 `[D, T]`，相乘得到 `[T, T]`。完整写法为：

```text
S = Q @ K.transpose(-2, -1) / sqrt(D)
S: [B, H, T_query, T_key] = [2, 2, 4, 4]
```

用下标写得更清楚：

```text
S[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d] / sqrt(D)
```

这里行下标 `i` 是“哪个 query 正在读取”，列下标 `j` 是“它正在给哪个 key
打分”。两个维度都恰好等于 `T`，但语义不同。后面做 cross-attention 时它们甚至可以
有不同长度。

为什么除以 `sqrt(D)`？如果向量分量的尺度相近，`D` 个乘积相加会让点积分布随 `D`
变宽。过大的分数会让 softmax 很快饱和，梯度变得不友好。缩放不是为了改变形状，
而是控制数值尺度。

## 1.5 causal mask 和 softmax

自回归语言模型不能让位置 `i` 看到未来 `j > i`。一个长度为 4 的可见性矩阵是：

```text
        key j ->  0  1  2  3
query i
       0          1  0  0  0
       1          1  1  0  0
       2          1  1  1  0
       3          1  1  1  1
```

实现上通常在 softmax 前把不可见分数设成负无穷：

```text
S_masked[i,j] = -inf,  if j > i
P = softmax(S_masked, dim=-1)
```

softmax 沿最后的 key 维做，因此每个 `(b,h,i)` 对应的一整行权重和为 1：

```text
P[b,h,i,j] = exp(S[b,h,i,j]) / sum_k exp(S[b,h,i,k])
```

被 mask 的位置指数为 0。然后用这些权重加权 value：

```text
O_head = P @ V
O_head: [B, H, T, D]
```

下标形式是：

```text
O_head[b,h,i,d] = sum_j P[b,h,i,j] * V[b,h,j,d]
```

注意两次求和的对象不同：打分时沿特征 `d` 求和，聚合 value 时沿历史位置 `j`
求和。

### 一个可以手算的单 head 例子

假设某个 query 对三个可见位置的**缩放后**分数为 `[0, ln 2, -inf]`，三个 scalar
value 为 `[10, 20, 100]`。那么：

```text
exp(scores) = [1, 2, 0]
P           = [1/3, 2/3, 0]
output      = 1/3*10 + 2/3*20 + 0*100 = 50/3
```

未来位置的 value 即使非常大也没有贡献，因为 mask 在 softmax 前将它排除。

## 1.6 合并多个 head

每个 head 得到 `[B,T,D]` 的结果。把 head 维重新排到序列维旁并拼接：

```text
[B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
```

再乘输出投影 `W_O in R^[H*D,C]`：

```text
O = concat(heads) W_O
O: [B, T, C]
```

这就恢复了 block 的形状接口，可以与残差 `X` 相加。

多个 head 并不保证人能给每个 head 一个稳定的自然语言标签。更可靠的理解是：它们有
不同的投影参数，可以在不同子空间中形成不同的匹配与聚合。

### MHA、GQA 和 MQA

上面描述的是 multi-head attention（MHA）：query、key、value 的 head 数相同。推理
系统常使用 grouped-query attention（GQA）或 multi-query attention（MQA）：多个
query head 共享较少的 key/value head。

设 query head 数为 `H_q`，KV head 数为 `H_kv`：

```text
Q: [B, H_q,  T, D]
K: [B, H_kv, T, D]
V: [B, H_kv, T, D]
```

- MHA：`H_kv = H_q`；
- GQA：`1 < H_kv < H_q`，每组 query head 共享一对 K/V；
- MQA：`H_kv = 1`。

这会显著改变 KV cache 大小和 decode 带宽。以后比较 kernel 时，`H_q` 与 `H_kv`
必须分别记录；只写“head 数 32”是不够的。

## 1.7 从最后一层表示到下一个 token

经过 `L` 层后，张量仍是 `[B,T,C]`。最后一个线性层把每个位置投影到整个词表：

```text
logits = X_final W_vocab
W_vocab: [C, V]
logits:  [B, T, V]
```

`logits[b,t,:]` 是模型在读到位置 `t` 及其之前内容后，对“下一个 token”的未归一化
分数。训练时会在所有位置上与右移一位的答案计算 loss；生成时通常只取当前最后一个
位置，再通过 greedy、sampling 等策略选出一个 token。具体训练闭环见[第 2 章](02-training-minimum.md)。

## 1.8 训练、prefill、decode 不是同一种计时对象

公式相同不代表运行形态相同。

### 训练

一批完整序列一起进入模型，所有位置都要产生 loss，而且还要保留或重算反向传播需要
的信息。论文若报告 training throughput，必须确认它是否包括 backward 和 optimizer
step。

### Prefill

推理开始时，prompt 的 `T` 个 token 一起通过模型。attention 很像一个带 causal mask
的大矩阵计算，并为之后生成建立 K/V cache。prefill 常更容易利用 GPU 的大规模并行。

### Decode

之后每步只新增一个 token。新 query 的长度是 1，但它要读取所有历史 K/V：

```text
Q_new:   [B, H_q,  1, D]
K_cache: [B, H_kv, T, D]
V_cache: [B, H_kv, T, D]
```

每生成一个 token，cache 长度增长 1。对每一层，忽略对齐与元数据时，K/V cache 字节数
约为：

```text
2 * B * H_kv * T * D * bytes_per_element
^ K和V
```

例如 `B=1, H_kv=8, T=32768, D=128, bf16=2 bytes`，仅一层约为 128 MiB；32 层约为
4 GiB。GQA/MQA 为什么对长上下文推理重要，由此就能看出来。

decode 常受读取 cache 的内存带宽限制，而不是只受乘法数量限制。后面看到“理论 FLOPs
更少却没有更快”的结果时，首先检查它是不是仍读取了大量不连续的 K/V。

## 1.9 为什么 dense attention 在长序列上昂贵

对每个 head，`QK^T` 和 `PV` 都涉及约 `T^2 D` 量级的乘加。所有 batch/head 合起来是
`O(B H T^2 D)`。分数或概率矩阵的逻辑形状为 `[B,H,T,T]`，即 `O(BHT^2)`。

当 `T` 从 4k 增加到 16k 时，线性部分变成 4 倍，二次部分变成 16 倍。这里要分清：

- **数学工作量**：需要形成所有 query-key 配对；
- **中间量存储/访存**：是否真的把整个 `T x T` 矩阵写回显存；
- **端到端时间**：还受到硬件、dtype、shape、编译与其他模块影响。

这一区分引出后续三条主线：

| 方法 | 数学上改了什么 | 实现上追求什么 |
| --- | --- | --- |
| FlashAttention | 不改 softmax attention 的结果，是 exact dense attention | 分块与 online softmax，减少大矩阵的显存读写 |
| Sparse attention | 只保留某些 `i,j` 配对，或先选择再精算 | 让保留的结构适合真正高效执行 |
| Linear attention | 改变/因式分解 attention，使历史可压入固定大小 state | 用 recurrent 或 chunkwise kernel 避免二次计算 |

一个常见误解是把 FlashAttention 也叫“近似 attention”。它没有因为不保存完整概率矩阵
就变成近似；在给定精度的数值误差范围内，它计算的仍是同一个 dense softmax 公式。

另一个常见误解是“mask 掉 90% 就自然快 10 倍”。如果实现仍先算完整 `T x T` 分数再
mask，数学上的稀疏没有变成执行上的稀疏。后续 sparse kernel 章节会一直追问：选择成本
是多少，索引是否规则，实际读了多少字节。

## 1.10 代码阅读：逐行标形状

下面是教学用实现，不追求性能。先不要运行，拿纸在每行右边写形状，再对照注释。

```python
import math
import torch

def causal_attention(x, wq, wk, wv, wo, n_heads):
    # x: [B, T, C]
    B, T, C = x.shape
    D = C // n_heads

    q = (x @ wq).view(B, T, n_heads, D).transpose(1, 2)
    k = (x @ wk).view(B, T, n_heads, D).transpose(1, 2)
    v = (x @ wv).view(B, T, n_heads, D).transpose(1, 2)

    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    blocked = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    scores = scores.masked_fill(blocked, float("-inf"))
    probs = torch.softmax(scores, dim=-1)

    heads = probs @ v
    merged = heads.transpose(1, 2).contiguous().view(B, T, C)
    return merged @ wo
```

阅读时回答：

1. `blocked[i,j]` 何时为真？为什么是 `triu(1)` 而不是 `tril()`？
2. 二维 `[T,T]` mask 为什么可以作用到四维 `[B,H,T,T]` scores？这利用了什么规则？
3. 如果把 `softmax(dim=-1)` 错写成 `dim=-2`，归一化的是谁？
4. `transpose` 后为什么常接 `contiguous()` 再 `view`？数学结果与内存布局分别发生了什么？
5. 这段代码会显式创建哪些 `T x T` 张量？它为什么不是 FlashAttention？

你不需要现在解释 `contiguous()` 的底层实现；只要知道换轴后的逻辑顺序与底层连续存储
可能不同，而某些 view/内核要求连续布局。GPU 篇会重新展开。

## 1.11 手算练习

### 练习 A：只看形状

设 `B=3, T=128, C=512, H=8, D=64`。

1. `Q` reshape 并换轴后的形状是什么？
2. `QK^T` 的形状是什么？
3. 合并 heads 后、乘 `W_O` 前的形状是什么？
4. 若使用 `H_kv=2` 的 GQA，K/V 的形状是什么？

### 练习 B：真正算一行 attention

设 `D=2`，当前 query 为 `q=[1,1]`，三个 key 为：

```text
k0=[1,0], k1=[0,1], k2=[1,1]
```

value 为 scalar：`v0=2, v1=4, v2=10`。

1. 先算除以 `sqrt(2)` 后的三个 score；
2. 若当前是位置 1，应用 causal mask 后哪个 key 不可见？
3. 写出输出的精确表达式，不必把指数算成小数；
4. 若错误地先对三个位置做 softmax、再把未来概率改成 0 而不重新归一化，输出有何不同？

### 核对答案

练习 A：`[3,8,128,64]`；`[3,8,128,128]`；`[3,128,512]`；K/V 为
`[3,2,128,64]`。

练习 B：三个分数为 `[1/sqrt(2), 1/sqrt(2), sqrt(2)]`；位置 1 看不到 `k2`；前两个
可见分数相同，所以正确输出为 `(2+4)/2=3`。先纳入未来位置再把概率清零，会使保留概率
和小于 1，得到的结果小于 3；这不等价于 causal softmax。

## 1.12 常见误区

**误区 1：token、位置和特征是同一种维。**
token 位置沿 `T` 排列；每个位置的表示沿 `C` 展开。attention 主要混合位置，线性投影
主要混合特征。

**误区 2：`K^T` 是把整个四维张量随便转置。**
代码中的 `transpose(-2,-1)` 只交换最后的序列维和 head 特征维，batch/head 维不动。

**误区 3：softmax 后再乘 mask 也一样。**
一般不一样。mask 必须参与归一化；否则可见位置的概率和不再为 1。

**误区 4：attention 矩阵就是模型的“解释”。**
它是一次前向中的中间权重，不自动构成可靠的因果解释，也不能单独代表模型全部行为。

**误区 5：复杂度低就必然在 GPU 上快。**
渐近复杂度忽略常数、并行度、访存和选择开销。算法论文与 kernel 论文必须分层阅读。

**误区 6：prefill benchmark 能代表 decode。**
前者是很多 query 一起计算，后者常是一个 query 读取长 cache；瓶颈可能完全不同。

## 1.13 这一章怎样接到后面的研究

- 读 FlashAttention 时，把注意力放在它如何不物化完整 `P`，同时保持同一公式；
- 读 sparse attention 时，把方法翻译成“每个 query 保留哪些 key，以及怎么存这个 mask”；
- 读 linear attention 时，追问 `sum_j` 能否借助结合律先汇总成 state；
- 读 FLA kernel 时，为每个 API 写出 `B,H,T,D`、dtype、causal、state 和 backward；
- 做 benchmark 时，永远注明是 forward、forward+backward、prefill 还是 decode。

特别是 Kimi Linear/Gated DeltaNet 一线，论文公式可能不再出现显式的 `[T,T]` softmax
矩阵，但它们仍要回答同一个问题：位置 `t` 如何从历史中读信息。理解这种 state 形式的
桥梁是[第 3 章](03-rnn-state-and-scan.md)。

## 本章小结

Transformer 的主干不是一团神秘模块，而是一条可追踪的张量管线：`[B,T]` 的 token id
经 embedding 成为 `[B,T,C]`，每层 attention 把它投影成多头 `Q/K/V`，形成按 key 维
归一化的因果权重并聚合 value，再回到 `[B,T,C]`。dense attention 的核心压力来自所有
query-key 对；Flash、sparse 和 linear 分别从访存重排、配对删减、状态化重写三个方向
处理它。

真正应该带走的不是一串字母，而是形状纪律：每看到一个公式或代码变量，就问它的每个
维度代表什么、在哪一维求和、是否随序列增长、是否真的写入显存。

## 通过条件

不看正文，完成以下任务，才算通过本章：

- 在 5 分钟内从 `X:[B,T,C]` 写到 attention 输出 `[B,T,C]`，所有中间形状正确；
- 用一句下标公式解释 `S[b,h,i,j]` 和 `O[b,h,i,d]`；
- 解释 mask 为什么在 softmax 前应用；
- 用自己的话区分训练、prefill、decode，并说明 KV cache 存什么；
- 解释“FlashAttention 是 exact，而 sparse/linear 往往改变计算结构”这句话；
- 独立做对练习 A/B，并把错误原因写进学习笔记，而不只抄答案。

## 延伸材料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：先看架构图和
  3.2 节，不必现在追完整训练细节。
- PyTorch, [Scaled Dot Product Attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：
  后续实验会用它作为可靠 dense 基线。
- Dive into Deep Learning, [Multi-Head Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)：
  适合补充可执行的形状演示。
