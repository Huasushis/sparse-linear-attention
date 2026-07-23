# 第 12 章：Kimi、KDA 与 GDN 到底有多像

导师补充的“**Kimi 和 GDN 几乎重合**”是非常有价值的提示，但必须先问：这里的
“Kimi”指整套 Kimi Linear 模型，还是核心算子 KDA？这里的“GDN”指 gated delta
递推、原论文的纯模型，还是实验表中的 GDN-H？

先给结论：

> 在 **token-mixing operator 的递推骨架**上，KDA 是 Gated DeltaNet 的细粒度 gate
> 扩展；把 KDA 的每个通道 gate 设成相同标量时，可以退化到 GDN 的递推。这个层面说
> “几乎重合”很准确。
>
> 但 **Kimi Linear 整体架构**还包括 KDA/MLA 的层间混合、MoE、位置编码选择和完整训练
> 配方，不能写成“Kimi Linear 就是 GDN”。

本章不会用命名游戏回避相似性，也不会为了强调新颖性夸大差异。我们用同一记号逐项比较，
并把可验证的事实、作者主张和仍需查看配置的部分分开。

## 学习目标

读完后，你应当能够：

1. 区分 KDA operator、Kimi Linear architecture、GDN operator 和 GDN-H baseline；
2. 写出 GDN 与 KDA 的唯一核心递推差异；
3. 用一个数值测试证明“通道 gate 全相等时 KDA 退化为 GDN”；
4. 解释 constrained DPLR 与 chunkwise kernel 在这里扮演什么角色；
5. 正确解读论文中 Kimi Linear、GDN-H、MLA 的速度和质量比较；
6. 知道初学阶段该复现 operator，而不是试图重训 48B MoE。

## 12.1 先拆开四个名字

| 名字 | 所在层级 | 本章采用的准确含义 |
| --- | --- | --- |
| gated delta rule | 数学递推 | scalar forget gate + delta update |
| Gated DeltaNet / GDN | operator 或模型族 | 使用 gated delta rule 的 token mixer；原论文也构建纯模型和 hybrid |
| Kimi Delta Attention / KDA | operator + kernel | 把 GDN 的 scalar decay 换为 channel-wise diagonal decay，并给出专用 chunkwise 算法 |
| Kimi Linear | 整体模型架构 | 以 3:1 交错 KDA 与 MLA，并使用 MoE 等完整 backbone 设计 |

还有一个容易混淆的名字：**GDN-H**。Kimi Linear 报告将其定义为实验中的“hybrid
Gated DeltaNet baseline”，并说明它与 Kimi Linear 共用架构、参数量和训练设置以公平
比较。这个标签不应未经核对就等同于 GDN 原论文中的 `GatedDeltaNet-H1` 或 `H2`：原论文
的 H1/H2 分别组合 GDN+sliding-window attention，以及 Mamba2+GDN+sliding-window
attention。名字相近，不代表配置相同。

因此报告中第一次出现 `GDN-H` 时，应给出具体来源和配置，而不是把 `-H` 当成一个通用
算法后缀。

## 12.2 共同骨架：先忘记，再做 delta update

继续使用上一章的 state 方向：

```text
S_t: [Dk, Dv]
k_t, q_t: [Dk]
v_t: [Dv]
o_t = S_t^T q_t
```

### GDN：每个 head 一个 scalar decay

$$
\bar S_t=\alpha_tS_{t-1},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top,
$$

等价地，

$$
S_t=(I-\beta_tk_tk_t^\top)(\alpha_tI)S_{t-1}
    +\beta_tk_tv_t^\top.
$$

`α_t`、`β_t` 对一个 head 的当前 token 是标量。

### KDA：每个 key channel 一个 decay

KDA 把 `α_t I` 改为对角矩阵：

$$
\bar S_t=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1},
\qquad \boldsymbol\alpha_t\in(0,1)^{D_k},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

展开为 Kimi 报告采用的 transition：

$$
S_t=(I-\beta_tk_tk_t^\top)
    \operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}
    +\beta_tk_tv_t^\top.
$$

注意乘法顺序。对 scalar `α_t`，它和任何矩阵可交换；对 diagonal
`Diag(α_t)`，一般不能把它随意移到 `(I-βkk^T)` 左边。

### “几乎重合”的严格版本

若设置

$$
\boldsymbol\alpha_t=\alpha_t\mathbf 1,
$$

则 `Diag(α_t)=α_tI`，KDA recurrence **精确退化为** GDN recurrence（假设 q/k/v、
`β`、初始 state 和输出定义一致）。所以二者不是只有模糊的思想相似；在 operator 数学上，
GDN 是 KDA 的一个受限 gate 情形。

“几乎”而非“完全”的部分在于：KDA 的 gate 参数化、chunkwise 算法、输出门，以及它被
放入 Kimi Linear 整体模型时的其他设计，不都等于原始 GDN 配置。

## 12.3 共同部分与差异，逐项核对

| 维度 | GDN | KDA | 该差异意味着什么 |
| --- | --- | --- | --- |
| state | 每 head 的 `[Dk,Dv]` 矩阵 | 相同 | recurrent decode 都是固定 state |
| 写入 | `β k(v-pred)^T` | 相同骨架 | 都是 delta rule，不是普通外积累加 |
| forget gate | 每 head/时间步 scalar `α` | 每 key channel/时间步 vector `α` | KDA 可给各通道不同寿命 |
| transition | scalar-gated rank-1 | diagonal-plus-tied-rank-1 | KDA 属于受约束的 DPLR 结构 |
| 并行训练 | gated WY/chunkwise | KDA 专用 chunkwise | 都追求块内 GEMM，但辅助量与数值处理不同 |
| q/k/v 前处理 | projection、short conv、SiLU，q/k L2 norm | 相近的 projection、short conv、Swish，q/k L2 norm | 外围 block 高度同源 |
| 原论文 output gate | GDN 使用其 block 参数化 | KDA 报告采用 sigmoid output gate | 比较时要看具体 baseline 是否对齐 |
| 整体模型 | 纯 GDN 或论文定义的 H1/H2 hybrid | Kimi Linear 中 KDA 与 MLA 3:1 交错 | 这是架构层差异，不是递推式差异 |

Kimi 报告还注明，其公平比较中的 GDN-H 也采用报告选择的 sigmoid output gate；因此不能
把表中 GDN-H 简单理解为“从 GDN 官方仓库原样拿来的默认模型”。公平 baseline 经常会
对齐 backbone 与外围组件，只替换被研究的核心机制，这是合理做法，也要求读者把配置写清。

## 12.4 为什么 KDA 被写成 constrained DPLR

一般 diagonal-plus-low-rank transition 可写成：

$$
A_t=D_t-a_tb_t^\top.
$$

它比纯对角 transition 表达力更强，但一般形式的 chunkwise 计算与数值稳定处理可能昂贵。
KDA 的 transition 是

$$
A_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\boldsymbol\alpha_t)
   =\operatorname{Diag}(\boldsymbol\alpha_t)
    -\beta_tk_t(k_t^\top\operatorname{Diag}(\boldsymbol\alpha_t)).
$$

它确实是 diagonal-plus-rank-1，但 low-rank 两侧都被当前 key 与 gate 约束，而不是任意
`a_t,b_t`。这种约束保留了经典 delta residual 的解释，同时让作者可以针对其结构减少
chunk 内辅助计算。

这部分需要分成两个主张：

1. **算法主张**：通道级 decay 比 scalar decay 更细致；
2. **kernel 主张**：这种受约束 DPLR 可用专门 chunkwise 表示高效计算。

第一条要由模型消融和质量任务支持；第二条要由同 shape、同 dtype 的 operator benchmark
支持。只看一条曲线不能同时证明两者。

## 12.5 Kimi Linear 比 KDA 多了什么

Kimi Linear 报告中的整体 backbone 不是“连续堆 KDA”这么简单。核心结构包括：

```text
3 个 KDA token-mixing layer
          ↓
1 个全局 MLA token-mixing layer
          ↓
按层重复；token mixer 后接 MoE channel mixer
```

报告采用 3:1 的 KDA:MLA 比例。KDA 层的 recurrent state 不随上下文长度增加；周期性的
MLA 层仍提供全局 full-attention 路径并维护相应 KV cache。由此，“最多减少约 75% KV
cache”是该层比例和报告配置下的结果，不是 KDA operator 自身对任意模型的固定比例。

报告还讨论了：

- 基于 Moonlight 风格的 MoE backbone；
- KDA/MLA 的混合层安排；
- global MLA 层采用 NoPE 的位置处理选择；
- 48B 总参数、3B 激活参数的实验模型与相同训练 recipe；
- 预训练、SFT、长上下文和 RL 阶段的比较。

这些都属于 **model / training** 层。你可以复现 KDA operator 的正确性和速度，却不能据此
声称复现了 Kimi Linear 的模型质量。

## 12.6 怎样解读 Kimi 报告中的 GDN-H 对照

报告给出三类 baseline：full-attention MLA、hybrid GDN-H、Kimi Linear。可安全读出的
信息包括：

- 它们在报告的 48B 设置下对齐了架构、参数量和训练设置；
- Kimi Linear 与 GDN-H 的 prefill/TPOT 曲线在图中非常接近，说明更细 gate 在该实现和
  测试形状下没有造成明显额外延迟；
- Kimi Linear 在报告的多项短/长上下文评测上总体高于 GDN-H，但并非每一单项都获胜；
- 这些是单一研究中的 matched comparison，不能直接推广到任意规模、GPU 或训练配方。

不能从图中推出的结论包括：

- “KDA 和 GDN kernel 在所有 shape 上一样快”；
- “质量差全部由 channel-wise gate 单独造成”；
- “原 GDN-H1/H2 的质量就是 Kimi 表中的 GDN-H”；
- “跑通 KDA kernel 就复现了 1.4T-token 训练结论”。

在报告中，GDN-H 的确切 layer schedule、配置字段和实现 commit 应在正式复现时从发布配置
再核对一次。论文正文的“same architecture”足以支持公平比较的意图，但研究记录仍应保存
实际配置，而不是只记缩写。

## 12.7 最有价值的复现：一个 scalarization test

第一项实验不需要大模型。它检验本章最关键的数学关系：

```python
# 相同 q, k, v, beta, S0
alpha_scalar = rand(B, T, H)          # GDN
alpha_vector = alpha_scalar[..., None].expand(B, T, H, Dk)  # KDA

out_gdn, state_gdn = gdn_reference(..., alpha_scalar)
out_kda, state_kda = kda_reference(..., alpha_vector)

assert_close(out_gdn, out_kda)
assert_close(state_gdn, state_kda)
```

为了让失败有诊断价值：

1. 先用 `B=H=1,T=4,Dk=3,Dv=2,float64`；
2. 同时比较每一步 state，而不只比较最终输出；
3. 检查 gate 的广播轴与 transition 乘法顺序；
4. reference 相同后，再比较 FLA 的 recurrent/chunk implementations；
5. 最后才换 BF16 并放宽到由误差统计支持的容差。

若这项测试失败，最常见原因不是论文矛盾，而是 state 转置、gate 广播维、是否先 decay、
或某个实现额外使用了 normalization/output gate。

## 12.8 第二项实验：证明细粒度 gate 确实“多了一种行为”

构造 `Dk=2` 的 toy case，让第一个通道需要长期保留、第二个通道需要快速清空：

```text
KDA alpha = [0.99, 0.10]
GDN alpha = 一个标量，只能在两种需求之间折中
```

固定 q/k/v 后画两条通道上的 state 范数与读取误差。这个实验只能证明 KDA 的参数空间更
细；它不能证明语言模型一定更好。随后再读论文的 matched-training 消融，才构成“机制
容量 + 经验结果”的完整论证。

## 12.9 第三项实验：operator 性能，而非宣传数字

在同一 GPU 上比较 FLA 的 GDN 与 KDA implementation 时，至少固定：

| 项目 | 必须记录 |
| --- | --- |
| 软件 | repo commit、PyTorch/Triton/CUDA 版本 |
| 硬件 | GPU 型号、可用显存、功耗/时钟是否受限 |
| shape | `B,T,H,Dk,Dv` 与 variable-length 设置 |
| 数值 | input/state dtype、accumulation 行为 |
| 模式 | recurrent、chunk、forward 或 fwd+bwd |
| 计时 | warmup、重复数、CUDA event、同步边界 |
| 正确性 | 对 reference 的最大/平均误差、梯度检查 |

应画 `T` 与延迟/吞吐的曲线，而不是只选一处最快点。若导师说二者“几乎重合”，这项实验
会进一步回答：数学上是 special case；在你的 A100/5090 和常见 shape 上，实际 overhead
又是多少。

## 常见误区

**误区 1：Kimi 是一条 attention 公式。**
Kimi Linear 是模型；KDA 才是本章比较的 token-mixing operator。

**误区 2：KDA 只是 GLA。**
KDA 的通道 gate 与 GLA 有亲缘关系，但它还保留 delta residual/rank-1 erase；GLA 的基本
update 是 gated additive write。

**误区 3：channel-wise 一定很慢。**
一般细粒度 DPLR 可能昂贵；KDA 的研究问题正是约束 transition 并设计专用算法。是否快
仍需在具体 shape/hardware 上测量。

**误区 4：论文中曲线接近，所以 operator 完全相同。**
相似的 TPOT 只说明被测配置中的运行时间接近；scalarization test 才直接验证递推包含关系。

**误区 5：`GDN-H` 是跨论文固定名称。**
它是配置标签。Kimi 报告的 GDN-H 与 GDN 原论文 H1/H2 必须分别记录来源。

## 练习

### 练习 12.1：不用文字，只化公式

把 `alpha_vec=alpha_scalar*1` 代入 KDA，逐行化到 GDN。然后举一个 `Dk=2` 的非等通道
`alpha_vec`，证明不存在单一 scalar `alpha` 能产生相同的 decay matrix。

### 练习 12.2：给导师的一分钟回答

不用“差不多”“更先进”这类词，用四句话回答：

1. GDN operator 是什么；
2. KDA 改了哪一个量的粒度；
3. Kimi Linear 又加了哪些架构层设计；
4. 你准备如何用 reference 和 benchmark 验证。

### 练习 12.3：给论文图加边界

从 Kimi 报告选一张 GDN-H/Kimi/MLA 对照图，抄下模型规模、训练 tokens、context、batch、
指标和硬件。缺失项写“未在该图/附近文字报告”，不要猜。

## 通过条件

你只有同时做到以下几点，才算真正理解“几乎重合”：

- 能写出 GDN/KDA 两条递推，并指出对角 gate 与 scalar gate；
- scalarization test 在 reference 上逐步 state 一致；
- 不会把 Kimi Linear、KDA、GDN、GDN-H 混用；
- 能说出 Kimi Linear 的 3:1 KDA/MLA 混合和 MoE 属于架构层；
- 能把“数学包含关系”“operator 速度”“模型质量”分成三份证据。

## 本章依据

- `yang2025gated`：ICLR 2025 Gated Delta Networks，给出 gated delta rule、GDN block
  与 H1/H2 hybrid；
- `kimi2025linear`：Kimi Linear technical report v2，明确写出 KDA 对 GDN 的 fine-grained
  gate 扩展、constrained DPLR、3:1 KDA/MLA 以及 GDN-H matched baseline；
- `yang2024gated`：GLA 的 channel-wise gate 背景；
- `yang2024delta`：delta/WY/chunkwise 背景。

条目见仓库根目录的 `references/attention.bib`。下一章转向 FLA 源码：不是从 Triton
第一行硬啃，而是把这条递推沿 `layer -> op -> reference -> kernel -> test -> benchmark`
一路追下去。
