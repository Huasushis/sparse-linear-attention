# 第 11 章：从累加记忆到 Gated DeltaNet

上一章的最小 state update 是不断外积相加：

$$
S_t=S_{t-1}+k_tv_t^\top.
$$

它容易并行化，也很容易理解，但有两个直观问题：

1. **不会主动忘记。** 无关信息会一直留在有限 state 中；
2. **不会先擦再写。** 同一个或相近的 key 再出现时，新旧 value 会叠在一起。

GLA 给第一个问题加入 gate；DeltaNet 给第二个问题加入 delta rule；Gated DeltaNet
（GDN）把两者组合。本章的任务就是把这三个改动放在同一张图上。

## 学习目标

读完后，你应当能够：

1. 用“忘记”和“定向覆盖”解释 gate 与 delta rule 的不同作用；
2. 在统一形状下写出 Linear Attention、GLA、DeltaNet、Mamba2 简化式和 GDN；
3. 从 delta update 推导出 `(I-βkk^T)` 的 rank-1 transition；
4. 解释 GDN 为什么比逐 token RNN 更有机会做 chunkwise 训练；
5. 为 GDN 写一个慢但清楚的 reference，并设计三个正确性测试；
6. 看懂论文中的“更强记忆”与“更快 kernel”是两条不同证据链。

## 11.1 先统一论文中容易打架的转置

不同论文有时把 state 写成 `[Dv,Dk]`，输出 `S_t q_t`；有时写成 `[Dk,Dv]`，输出
`S_t^T q_t`。两者只是互相转置。为了和 Kimi KDA 便于比较，本书固定：

```text
k_t, q_t: [Dk]
v_t:      [Dv]
S_t:      [Dk, Dv]
o_t = S_t^T q_t: [Dv]
```

后文所有式子都使用这个方向。若你打开 Gated DeltaNet 原论文，看到 state 乘法写在右边，
先把整条式子转置再比较，不要立即判定公式不同。

## 11.2 一个最小例子：为什么只加不够

设 key 已经 L2 归一化，`k=[1,0]`，state 初始为零。第一次写入 `v_old=[1,2]`：

$$
S_1 = k v_{old}^\top =
\begin{bmatrix}1&2\\0&0\end{bmatrix}.
$$

若同一 key 随后对应新 value `v_new=[5,8]`，普通累加得到：

$$
S_2 = k(v_{old}+v_{new})^\top,
$$

查询该 key 时读到的是两者之和，而不是新值。这不是实现 bug；它正是 additive fast
weight update 的行为。有限 state 中出现多个相似 key 时，干扰会更复杂。

Delta rule 的出发点是：先问 state 对当前 key 已经预测了什么，再只写入**误差**。

## 11.3 DeltaNet：用预测误差更新

定义旧 state 对当前 key 的读取：

$$
\hat v_t = S_{t-1}^\top k_t.
$$

delta update 为

$$
S_t = S_{t-1} + \beta_t k_t(v_t-\hat v_t)^\top,
\qquad \beta_t\in[0,1].
$$

`β_t` 控制这次纠正走多远。展开括号：

$$
\begin{aligned}
S_t
&=S_{t-1}+\beta_tk_tv_t^\top
  -\beta_tk_tk_t^\top S_{t-1}\\
&=(I-\beta_tk_tk_t^\top)S_{t-1}
  +\beta_tk_tv_t^\top.
\end{aligned}
$$

`k_t k_t^T` 是 rank-1 矩阵，因此 transition `I-β_t k_t k_t^T` 主要沿当前 key
方向修改 state。若 `||k_t||=1` 且 `β_t=1`，再次沿 `k_t` 查询时：

$$
S_t^\top k_t=v_t.
$$

你可以把它理解为“沿这个 key 的方向先擦除旧预测，再写新 target”。但不要把这句话扩大
成“DeltaNet 能无损保存无限多键值对”。不同 key 不正交、state 维度有限、低精度和模型
参数化都会造成干扰。

### 对照前面的手算例子

仍取 `k=[1,0]`、`β=1`。第二次更新前，`S_1^T k=v_old`，所以写入误差恰为
`v_new-v_old`。更新后 `S_2^T k=v_new`，而不是 `v_old+v_new`。这就是 delta 的名字所指：
存储目标与当前预测之间的差。

## 11.4 Gate：主动控制记忆寿命

另一条路线不首先处理“同 key 覆盖”，而让 state 在写入前衰减。

### Mamba2 风格的简化标量 gate

忽略具体参数化，最简形式为：

$$
S_t=\alpha_tS_{t-1}+k_tv_t^\top,
\qquad \alpha_t\in(0,1).
$$

`α_t` 小，表示这一时刻快速清除历史；`α_t` 接近 1，表示较长保留。这里的 `α_t` 是
每个 head/时间步的标量，所以整个 key 维用同一衰减率。

### GLA 的通道级 gate

Gated Linear Attention（GLA）把 gate 细化到 key channel：

$$
S_t=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}+k_tv_t^\top,
\qquad \boldsymbol\alpha_t\in(0,1)^{D_k}.
$$

不同 key 特征维可以有不同记忆寿命。GLA 原论文还讨论了更一般的二维 gate，最终采用上述
沿 key 维的低秩参数化，以兼顾表达力、state 大小和训练效率。它的重要性不只是“多了
一个 sigmoid”，而是证明细粒度、数据依赖的 gate 仍可做硬件友好的 chunkwise 重写。

### Gate 与 delta 解决的不是同一件事

| 机制 | 它观察什么 | 主要动作 | 直观问题 |
| --- | --- | --- | --- |
| gate `α` | 当前输入产生的遗忘率 | 广泛缩放旧 state | 哪些历史现在应被忘掉？ |
| delta `β,k` | 当前 key 下的旧预测误差 | 沿 key 方向纠正 state | 这个 key 对应的旧值怎样被覆盖？ |

因此二者不是互相替代的技巧。GDN 的贡献正是把它们组合，并给出可训练的 chunkwise 算法。

## 11.5 Gated DeltaNet：先遗忘，再按误差写入

GDN 的递推可以分成两行：

$$
\bar S_t=\alpha_t S_{t-1},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

合并为 transition 形式：

$$
S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}
    +\beta_tk_tv_t^\top.
$$

这里 `α_t` 与 `β_t` 在每个 head、每个时间步上是标量；`k_t`、`v_t` 仍是向量。
由于 `α_t` 是标量，它和 rank-1 transition 可交换。下一章 KDA 把它换成对角矩阵后，
乘法顺序就必须写清楚。

这条式子包含四个退化情形，适合做单元测试：

| 设置 | 递推退化为 | 应检查什么 |
| --- | --- | --- |
| `α=1, β=0` | state 不变 | 输出只来自旧 state |
| `α=1` | DeltaNet | 与 delta reference 一致 |
| `β=0` | 纯衰减 | 多步后 state 按 `α` 累乘 |
| `S_0=0, β=1` | 第一步写入 `kv^T` | 第一 token 可手算 |

### 为什么论文说 gate 与 delta 互补

GDN 论文用合成 recall 任务给出的解释是：

- 只有 decay 时，旧信息可能在需要保留时过快消失；
- 只有 delta 时，state 缺少快速清空无关内容的机制，有限容量更容易碰撞；
- gated delta 既能选择性遗忘，又能对某个 key 做定向更新。

这是一种由实验支持的机制解释，不是数学定理。你的报告应写“作者在何种任务和规模上
观察到”，而不是写成“GDN 对所有长上下文任务都更强”。

## 11.6 从递推到 chunkwise：WY 表示在做什么

若按公式逐 token 更新，`S_t` 依赖 `S_{t-1}`，训练时 GPU 很难同时处理整个序列。
DeltaNet/GDN 的关键工程问题，是压缩一段 rank-1 transition 的乘积：

$$
\prod_{i=1}^{C}(I-\beta_i k_i k_i^\top).
$$

论文使用 WY representation 把一个 chunk 内的一系列 rank-1 更新组织成紧凑矩阵形式，
从而将大量工作转化为矩阵乘；GDN 再把一段标量 gate 的累计乘积合入这个表示。第一遍不必
背完整推导，但要理解算法结构：

```text
输入 chunk 的 Q/K/V/α/β
          |
          |  形成累计 gate 与 WY 辅助量
          v
块内并行输出  +  上一块 boundary state
          |
          v
更新下一块 boundary state
```

这使得串行依赖从“每个 token 一次”降到“每个 chunk 一次”，块内主要工作可以使用大块
矩阵乘。它不消除因果依赖，也不保证任意 chunk size 都快。

### 为什么输入依赖 gate 仍可并行

现代 GDN 的 gate 由当前输入投影产生，而不依赖前一个 hidden state。因而一整段
`α_t,β_t` 可以先并行算出，再做累计乘积/scan。如果 gate 本身依赖 `S_{t-1}` 的任意
非线性函数，重写会困难得多。

## 11.7 模型 block 不等于一条递推式

GDN 论文的 token mixer 还包含：

- 线性投影得到 q/k/v；
- short convolution 与 SiLU；
- q/k 的 L2 normalization；
- `α,β` 的参数化；
- head-wise normalization、output gate 和输出投影；
- 残差、norm、MLP，以及可选的 hybrid 结构。

因此复现要标层级：

1. **operator reference**：只验证 gated delta recurrence；
2. **layer**：加入投影、卷积、norm、output gate；
3. **model architecture**：规定层如何堆叠或与 attention/Mamba2 混合；
4. **training recipe**：数据、token 数、优化器和评测。

把第 1 层跑对，不能证明整模型论文结论；但它是进入 kernel 的正确第一步。反过来，下载
整模型跑出一段文本，也不能证明你理解了 operator。

## 11.8 一个应当亲手完成的 reference

下面的伪代码只描述单 head、单 batch；实际练习再向量化 batch/head：

```python
S = zeros(Dk, Dv)
for t in range(T):
    S_bar = alpha[t] * S
    prediction = S_bar.T @ k[t]             # [Dv]
    residual = v[t] - prediction            # [Dv]
    S = S_bar + beta[t] * outer(k[t], residual)
    out[t] = S.T @ q[t]
```

建议建立三组测试：

1. **展开等价性**：上面 residual 写法与 transition 矩阵写法相同；
2. **退化等价性**：`α=1` 时与 DeltaNet reference 相同；
3. **分块等价性**：在 token `C-1` 保存 state，从 `C` 继续运行，输出应与整段循环一致。

测试先用 float64 小张量；再用 FP32/BF16 观察容差，而不要反过来用很宽的容差掩盖公式
错误。梯度检查应在 forward 通过后再做。

## 11.9 GPU 与 benchmark 问题

GDN 的实际效率至少取决于：

- `Dk×Dv` state 是否适合当前 GPU 的 tile 和片上容量；
- chunk size、head 数、batch 是否提供足够并行；
- q/k normalization、gate、WY 辅助计算能否融合；
- forward 与 backward 是否都优化；
- decode 使用 recurrent kernel，还是误用训练用 chunk kernel；
- 与 dense baseline 比较时是否使用同样 dtype、shape 和计时边界。

论文报告 GDN 与 DeltaNet 训练吞吐接近、但比 Mamba2 多一些 transition 开销；这类结果只能
在论文给定的硬件与形状下解读。你在 A100/5090 上的任务不是“复述那个数字”，而是记录
自己的 crossover：从什么长度开始，哪个 implementation 更快，状态/显存变化如何。

## 常见误区

**误区 1：gate 小就是“删除某个 token”。**
标量 gate 缩放整个 head 的 state；GLA/KDA 的通道 gate缩放特征通道。它们都不是直接从
KV cache 中删除一个离散 token。

**误区 2：delta rule 等价于 attention softmax。**
它是不同的 state update，可从 fast-weight/online-learning 角度理解，但不计算标准 softmax
attention 的同一结果。

**误区 3：`I-βkk^T` 是一个需要每步显式创建的大矩阵。**
reference 可以这样写以核对公式；高效实现利用 rank-1 结构，不应每步物化 identity 和
transition。

**误区 4：GDN 就是 GDN-H。**
GDN 可以指 operator 或纯 Gated DeltaNet 模型；带 `-H` 的名字通常指某个 hybrid
configuration。具体组成必须回到该论文/代码配置，下一章会专门处理。

## 练习

### 练习 11.1：一步覆盖

取 `Dk=2,Dv=1,k=[1,0],S_0=[[3],[4]],v=[10],β=1`。分别计算普通累加和 delta
update 后的 state 与 `S^T k`。说明第二个 key 方向发生了什么。

### 练习 11.2：gate 与 delta 的消融

设计四组固定随机输入：Linear、只有 gate、只有 delta、gated delta。不要先比较速度；先
画 `||S_t||_F` 与一个重复 key 的读取误差随 `t` 的变化。写下哪些现象来自你的构造，不能
推广为语言模型质量结论。

### 练习 11.3：读论文只看三处

精读 `yang2025gated` 的递推式、chunkwise algorithm 小节和 throughput 图。各写四句：
问题、改动、代价、证据。第一遍可跳过 appendix 中完整 backward 推导。

## 通过条件

进入 KDA 前，你必须能：

- 从 residual 形式独立展开出 `(I-βkk^T)`；
- 解释 `α` 与 `β` 分别控制什么，而不是统称为“两个 gate”；
- 画出 Linear/GLA/DeltaNet/GDN 四条 update 的差异；
- 写出并通过三类 reference 测试；
- 用一句话解释 WY/chunkwise 的任务，但不假装已经会推导所有 backward。

## 本章文献锚点

- `yang2024gated`：GLA 与硬件友好的细粒度 gate；
- `yang2024delta`：DeltaNet 的序列并行与 WY 表示；
- `dao2024transformers`：Mamba2/SSD 与线性 attention 的联系；
- `yang2025gated`：gating 与 delta rule 的组合及 GDN hybrid。

条目均在仓库根目录的 `references/attention.bib` 中。下一章将用完全相同的记号回答
导师的提醒：**Kimi 与 GDN 到底在哪一层“几乎重合”，又在哪一层绝不能画等号？**
