# 第 1 章：重看 Transformer——带着后续研究需要的问题

## 学习目标

看完 3Blue1Brown 的 Transformer / attention 视频后，你应能解释下面的计算。暂时不要求理解大模型训练技巧、RLHF 或复杂位置编码。

设输入为 `X ∈ R[B, T, d_model]`：

```text
Q = X W_Q,  K = X W_K,  V = X W_V
S = Q K^T / sqrt(d_head)             # [B, H, T, T]
P = softmax(S + causal_mask)
O = P V
```

- `B`：batch size；`T`：序列长度；`H`：注意力头数；`d_head`：每头维度。
- `Q` 问“当前位置要找什么”；`K` 描述“每个位置可被怎样匹配”；`V` 是被聚合的信息。它们是有用的直觉，不是物理实体。
- causal mask 让位置 `t` 看不到未来 `j > t`，所以语言模型生成时没有偷看答案。

## 为什么会慢

`S` 的每个 head 都是 `T × T`。因此标准 attention 随长度大致需要 `O(T² d_head)` 的算术和 `O(T²)` 级别的注意力矩阵（具体是否 materialize 取决于实现）。长上下文时，慢不只来自 FLOPs，也常来自读写显存中的大矩阵。

这正是三条路线出现的背景：

| 路线 | 改什么 | 一个直觉 |
| --- | --- | --- |
| FlashAttention | 不改 exact 公式，重排读写与中间量 | 少把大 `T×T` 矩阵写回慢显存 |
| Linear attention | 改写/近似注意力，使状态不随 `T` 增长 | 把历史压缩进 state |
| Sparse attention | 只计算/保留部分 `q-k` 关系 | 每个 token 不必看所有历史 |

所以这三者不能简单排成“谁取代谁”：FlashAttention 是 exact dense baseline；linear 与 sparse 常在模型能力、近似/结构限制和实际 kernel 效率之间做取舍。

## Prefill 与 decode

- **Prefill：** prompt 的许多 token 一起计算，通常 `T` 大、矩阵计算和 attention pattern 是重点。
- **Decode：** 每次只生成一个新 token，需要读过去的 KV cache；带宽、cache 管理、batching 常成为重点。

同一方法可能 prefill 很快但 decode 不快，或反过来。之后所有 benchmark 都必须写清自己测的是哪一个。

## 只需知道的训练闭环

一次训练迭代是：forward 得到 logits → 与正确下一个 token 算 loss → backward 算各参数的梯度 → optimizer（如 Adam）更新参数。attention 的 kernel 论文常分开讨论 forward 和 backward；而推理/serving 论文通常只测 prefill/decode。现在不要求你手推 Adam，但要能辨认论文在测哪个阶段。

## 检查题（写在自己的笔记里）

1. 若 `Q` 是 `[B,H,T,d_head]`，为什么 `QK^T` 是 `[B,H,T,T]`？
2. causal mask 把哪些位置设为不可见？
3. 为什么 `softmax(QK^T)V` 的括号不能随意交换？
4. `T` 从 4k 增加到 16k 时，dense attention 的 `T²` 部分理论上变为多少倍？
5. FlashAttention 为什么仍然是 dense / exact attention？
6. prefill 和 decode 分别更像“整张表一起算”和“拿一个新 query 查历史”的哪一种？为什么？

## 第一个可写入报告的段落

完成后，用自己的话写 100--200 字：

> 标准 causal self-attention 在长上下文中遇到的主要计算/访存问题是什么？Flash、linear、sparse 三类方法分别改变了什么？本文后续报告为什么仍需要 exact dense baseline？

保留最初版本。后续读完 FlashAttention 后再回来看，会很直观地看到理解如何变深。
