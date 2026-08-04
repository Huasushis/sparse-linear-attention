# P0 作业：Transformer 与 attention 刷新

> 请用自己的话填写。暂时不追求术语漂亮；空白和明确的问题比复制资料更有价值。

## 1. 观看记录

- 观看的资源与章节：3Blue1Brown 的 Transformer / attention 视频 https://www.3blue1brown.com/lessons/attention/
- 日期：2026.8.3
- 回看后仍不熟悉的三个问题：
  - decode 时为什么只需要新的 Q，却要保留历史 K/V？
  - torch 怎么写？
  - 常见的浮点标准有哪些？

## 2. 一次 attention 的形状流

设 batch 为 `B`，序列长度为 `T`，head 数为 `H`，每个 head 维度为 `D`。不要看教程，填写下表；如果采用了不同的张量排列，请先声明。

| 对象 | 我写出的形状 | 它表达什么 |
| --- | --- | --- |
| 输入 hidden states | `[B, T, H*D]` | 每个 token 当前的隐藏表示 |
| `Q` | `[B, H, T, D]` | 每个位置发出的查询，表示它想寻找什么信息 |
| `K` | `[B, H, T, D]` | 每个位置用于和 query 匹配的特征 |
| `V` | `[B, H, T, D]` | 匹配后实际被加权汇聚的内容 |
| attention scores | `[B, H, T, T]` | 每个 query 与所有 key 的未归一化相关性分数 |
| causal mask 后的 scores | `[B, H, T, T]` | 屏蔽当前位置之后的未来 token 后的分数 |
| softmax probabilities | `[B, H, T, T]` | 沿 key/token 维度归一化后的权重 |
| attention output | `[B, H, T, D]`，合并 heads 后为 `[B, T, H*D]` | 每个 query 根据概率对所有 `V` 加权求和的结果；它本身不是 residual |

## 3. 六个检查题

1. `Q`、`K`、`V` 各自参与什么比较或汇聚？

`Q` 表示当前位置想查询什么，`K` 表示每个位置提供什么匹配特征，二者通过 `QK^T` 得到相关性分数；softmax 后的分数再对 `V` 加权求和，汇聚实际内容。整体写成 `softmax(QK^T / sqrt(D))V`。

2. 为什么计算 score 时需要转置 `K` 的最后两个维度？

因为是Q的每一行和K的每一行两两点积，所以要转置K才能做矩阵乘法。

3. 为什么通常除以 `sqrt(D)`？你暂时不会证明也可以先写直觉。

直观地假设 `Q` 和 `K` 的各维元素独立且方差为 1，一个点积是 `D` 项乘积之和，其方差会随 `D` 增长。除以 `sqrt(D)` 可以让 logits 的尺度更稳定，避免 softmax 因输入绝对值过大而变得过于尖锐。

4. causal mask 禁止了哪些位置？它应在 softmax 之前还是之后应用？

禁止了未来的位置，在softmax之前。

5. prefill 和单步 decode 的输入形状与要复用的历史信息有什么不同？

prefill 是一次性输入整个 prompt，并为其中所有 token 计算 `Q/K/V`。单步 decode 每次只新增一个 token，会产生一个新的 `Q/K/V`；此前 token 的 `K/V` 不会改变，因此可以保存在 KV cache 中。新的 `Q` 仍需查询全部历史 `K`，KV cache 也会随生成长度增长。

6. 标准 dense attention 的哪一个中间对象随 `T^2` 增长？为什么换成线性或稀疏算法后仍不一定在 GPU 上更快？

QK^T，因为还需要考虑到常数，访存效率，并行度，选择开销等。

## 4. 用四句话留下研究笔记

- **问题：** dense attention 的 score 矩阵随序列长度平方增长。
- **改动：** 可以改变计算顺序、避免保存完整 score，或者只计算部分 token 对。
- **代价：** 新算法可能损失表达能力，也可能引入额外的选择、访存或 kernel 调度开销。
- **证据：** 逻辑上的 `QK^T` 形状为 `[B,H,T,T]`；但实际速度还必须通过固定硬件、shape、dtype 和计时方法的实验验证。

## 5. 仍然不懂的地方

写成可以回答的具体问题。例如“`softmax` 的输入维度为什么是最后一维”，不要只写“attention 不懂”。

GQA 有什么用，为什么需要 GQA？
