# 记号与形状：以后所有章节共用的语言

## 常用维度

| 记号 | 含义 | 常见数量级 | 备注 |
| --- | --- | --- | --- |
| `B` | batch size | 1--数十 | 一次并行处理多少条序列 |
| `T` | sequence length | 128--1M | 本书不再用 `L` 表示长度，避免和层数混淆 |
| `N_layers` | Transformer 层数 | 数十--数百 | 第 1 章偶尔简称为“层数” |
| `N_vocab` | 词表大小 | 数万--数十万 | logits 的最后一维 |
| `H` | 教学例子中的统一 head 数 | 8--数百 | 仅在 Q/K/V head 数相同时使用 |
| `H_q` | query head 数 | 8--数百 | GQA/MQA 中通常多于 KV heads |
| `H_kv` | key/value head 数 | 1--`H_q` | 必须满足 `H_q % H_kv == 0` |
| `D_k` | query/key head dimension | 64--256 | 简单 MHA 中也写作 `D` |
| `D_v` | value head dimension | 64--256 | `V` 专门保留给 value 张量 |

本书默认 dense attention 的布局是 `Q,K ∈ R[B,H,T,D_k]`、`V ∈ R[B,H,T,D_v]`。
FLA 与论文可能写成 `[B,T,H,D]`；**布局变了不是数学变了**，但写 kernel 时会影响
连续访存和操作顺序。GQA/MQA 需要把统一的 `H` 展开写成 `H_q` 与 `H_kv`。

## 三个容易混淆的 K

1. `K` 张量：key；
2. 论文偶尔用 `K` 表示每个 head 的 key dimension；本书优先写 `D_k`；
3. `KDA`：Kimi Delta Attention 的缩写。

写笔记时尽量给维度使用 `D_k`、`D_v`，给张量使用粗体或上下文区分。

## 本书的“准确”用词

- **exact**：在同样的浮点运算语义/容差下，计算的是原 attention 公式；FlashAttention 属于这里。
- **等价重写**：换了计算顺序，数学结果相同；parallel/recurrent/chunkwise 可能是这种关系。
- **近似**：故意改变结果以节省成本；随机特征或 sampling 常在这里。
- **换模型族**：linear state model 不必逐项近似 softmax attention；它的质量需要模型级证据。
- **速度**：必须附带硬件、输入形状、dtype、测量阶段；没有这些限定的“快 N 倍”不完整。
