# 第 10 章：Linear Attention——把“看全部历史”改写成“维护一个状态”

标准 causal attention 在位置 `t` 的输出，是对所有历史 value 的加权和。Linear
attention 的核心想法不是“让 GPU 更擅长算同一张 `T×T` 表”，而是试图把这件事改写为：

> 历史先被压进一个大小不随序列长度 `T` 增长的 state；当前 token 只读取该 state。

这是后续 GLA、DeltaNet、Gated DeltaNet（GDN）和 Kimi Delta Attention（KDA）的共同
起点。它也解释了为什么这些论文常同时谈 attention、RNN、fast weights 和 state-space
model：它们使用的语言不同，但都在讨论“历史怎样写入一个可递推的状态”。

本章先讲最干净的 kernel-feature 版本；后面几章再讲为什么现代方法会加入 gate 和
delta rule。请先把它当作一座桥，而不要把它误认为所有“线性注意力”都严格采用的公式。

## 学习目标

读完后，你应当能够：

1. 从 causal softmax attention 推导出 kernelized linear attention 的 state；
2. 写出 `S_t`、`z_t` 的形状，并解释它们为什么不随 `T` 增长；
3. 清楚地区分“exact dense attention 的更快实现”和“改变/近似 attention 模型”；
4. 区分 recurrent、parallel、chunkwise 三种计算视角；
5. 解释为什么 `O(T)` 并不自动等于训练或推理更快；
6. 知道阅读本线论文时该先找 state、update、parallelization 还是 benchmark。

这一章的前置是[第 1 章的张量形状](../part1/01-transformer-from-tensors.md)、[第 3
章的 RNN/state/scan](../part1/03-rnn-state-and-scan.md)，以及 dense attention 的
[第 7 章](../part2/07-dense-attention.md)。

## 10.1 先把 dense attention 写成“对历史逐项求和”

为避免多头和 batch 遮住核心，先固定一个 head、一个位置。设

```text
q_t, k_t: [Dk]     # query / key
v_t:      [Dv]     # value
```

标准 causal softmax attention 是

$$
o_t = \sum_{i\le t}
  \frac{\exp(q_t^\top k_i / \sqrt{D_k})}
       {\sum_{j\le t}\exp(q_t^\top k_j / \sqrt{D_k})}
  v_i.
$$

对每一个新的 `q_t`，这条式子都需要和全部历史 key 做匹配。因而它天然有一个随 `t`
增长的历史集合。FlashAttention 可以不把完整分数矩阵落到显存中，但仍然计算这批
`q_t-k_i` 配对；它是 **exact dense attention**。不要把这种“实现更省 IO”与本章的
“改变计算结构”混为一谈。

## 10.2 一个可交换顺序的版本：feature map

如果把相似度写成某个特征映射的内积：

$$
\mathrm{sim}(q,k) = \phi(q)^\top\phi(k),
$$

那么先对历史求和、再和当前 query 相乘就变得可能。令

```text
φ(q_t), φ(k_t): [Dφ]
```

并定义两个累计 state：

$$
S_t = \sum_{i\le t}\phi(k_i)v_i^\top \in \mathbb{R}^{D_\phi\times D_v},
\qquad
z_t = \sum_{i\le t}\phi(k_i) \in \mathbb{R}^{D_\phi}.
$$

则归一化输出可写为

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t + \varepsilon}
\in \mathbb{R}^{D_v}.
$$

这里分子是 `[Dφ] @ [Dφ,Dv] -> [Dv]`；分母是一个标量。于是每一步只需递推：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad z_t=z_{t-1}+\phi(k_t).
$$

这正是“Transformer 可以像 RNN 一样运行”的最小数学原因。历史没有消失，而是被压缩成
矩阵 `S_t` 和向量 `z_t`。

### 一个形状检查

若 `Dφ=64, Dv=128`，每个 head 的 `S_t` 有 `64×128=8192` 个元素。无论序列从
4k 增至 1M，它仍是同样大小；但它绝不是“小标量记忆”。多 head、多层、batch 和 dtype
仍会让这个 state 有真实的显存与读写代价。

## 10.3 “linear”到底线性在哪里？又不线性在哪里？

所谓 linear，通常是指**关于序列长度**的 attention 主体工作量不再是 `T²`。若 `Dφ`、
`Dk`、`Dv` 固定，累计 state 的更新约为 `O(T Dφ Dv)`，而不是对每对位置计算一次的
`O(T²D)`。

但下列说法都太快了：

- “它等价于标准 softmax attention。”通常不对。有限维 `φ` 往往是对 softmax kernel 的
  近似，或者方法干脆采用了不同的更新规则；
- “它只需 `O(T)` 内存。”需说明是在 decode state、训练激活、还是某个特定实现上说的；
- “它总比 FlashAttention 快。”不对。矩阵 state 的常数、低并行度、归一化、gate 和
  backward 都可能改变结果；
- “state 固定，所以记忆无限。”不对。固定容量 state 会发生干扰、遗忘或覆盖，这正是
  后续 delta-rule 研究的动机。

更诚实的表述是：

> Linear attention 用状态化结构换取对序列长度的线性扩展性；它同时改变了模型的记忆
> 机制，并需要新的算法和 kernel 才可能在真实 GPU 上兑现速度。

## 10.4 从公式到三种算法视角

同一组数学式可以有三种实现方式。不要因为论文只展示其中一种，就以为其余两种不存在。

| 视角 | 如何计算 | 最自然的阶段 | 优点 | 主要困难 |
| --- | --- | --- | --- | --- |
| recurrent | 一个 token 更新一次 state | decode / reference | 直观；只维护当前 state | `T` 步依赖，不利于训练时大规模并行 |
| parallel | 展开成大矩阵/scan | 训练或短序列 prefill | 许多位置可同时算 | 需要推导；可能产生大中间量 |
| chunkwise | 一块内并行，块间传 state | 现代训练 kernel | 将 GEMM 与 state dependency 折中 | 边界、数值稳定和 backward 更复杂 |

### Recurrent reference：最值得先自己写的版本

伪代码故意保持慢而清楚：

```python
S = zeros(Dphi, Dv)
z = zeros(Dphi)
for t in range(T):
    kt = phi(k[t])                 # [Dphi]
    qt = phi(q[t])                 # [Dphi]
    S = S + outer(kt, v[t])        # [Dphi, Dv]
    z = z + kt                     # [Dphi]
    out[t] = (qt @ S) / (qt @ z + eps)
```

它的价值不是速度，而是可以把每一行与公式、形状和单元测试一一对应。后续任何 chunkwise
或 Triton 实现都应该先和这种 reference 比较。

### Chunkwise 的直觉

把长度 `T` 切成若干长度 `C` 的 chunk。对每块：

1. 接收上一块末尾的 state；
2. 在块内用矩阵乘或分块 scan 同时处理许多 token；
3. 产生该块的输出和下一块的边界 state。

这样仍保留“前块影响后块”的因果性，却让块内工作更像 GPU 擅长的矩阵运算。`C` 是一个
**性能参数**，不是数学常数：过小可能喂不满 GPU；过大可能让片上 state、中间量或 register
压力上升。

## 10.5 为什么 kernel 是这条线的第二个难题

算法把 `T×T` attention map 换成 `Dφ×Dv` 的 state，并没有保证算子跑得好。阅读实现时，
需要同时看四件事：

1. **状态在哪里？** 整段放不下片上时，哪些 boundary state 必须写回 device DRAM？
2. **块内是什么矩阵乘？** 这决定能否利用 Tensor Core，而不只是循环多少次；
3. **是否 materialize 中间量？** 例如所有时间步的 state、gate 或 backward 缓存；
4. **训练还是 decode？** recurrent decode 很自然，但训练还要处理 backward 与时间维并行。

这也是为什么 GLA、DeltaNet 与 KDA 的论文都会把“能否 chunkwise parallelize”与模型
表达力放在同等重要的位置。只有复杂度公式而没有硬件友好的重写，可能出现理论线性、
实际不如优化 dense kernel 的情况。GPU 的基本推理工具见[第 5 章](../part2/05-gpu-mental-model.md)，
benchmark 的最低规范见[第 6 章](../part2/06-benchmarking-gpu.md)。

## 10.6 现代线性方法不是一条直线

本书接下来采用的阅读顺序如下。它不是历史上的唯一顺序，而是让每一步只新增一个概念：

| 阶段 | 核心问题 | 先看什么 | 先不纠结什么 |
| --- | --- | --- | --- |
| kernel feature | 如何把历史累积成 `S,z`？ | `katharopoulos2020transformers`、`choromanski2021rethinking` | 随机特征的全部理论界 |
| fast weight / state | 为什么 `S` 像可写记忆？ | `schlag2021linear` | 所有 online-learning 推导 |
| forget gate | 固定 state 怎样主动忘记？ | `yang2024gated`（GLA） | 先优化出最快 kernel |
| delta rule | 怎样修改特定 key 对应的记忆？ | `yang2024delta` | 完整 WY 推导细节 |
| SSD/SSM 视角 | 为什么它也可称为 SSM？ | `dao2024transformers` | S4 的全部谱理论 |
| gated delta | 忘记与精确更新怎样结合？ | `yang2025gated` | 大模型训练配方 |
| KDA / Kimi Linear | 标量 gate 怎样变为通道 gate，模型怎样做混合？ | `kimi2025linear` | 一开始复现 48B 模型 |

所有 key 都能在仓库根目录的 `references/attention.bib` 中找到。第一遍阅读只需在
每篇论文上回答下面五个问题：

1. state 是什么形状，是否随 `T` 增长？
2. 新 token 如何写入 state，是否有遗忘或覆盖？
3. 训练时怎样绕开逐 token 的串行依赖？
4. 作者比较的是 forward、forward+backward、prefill 还是 decode？
5. “更快”是在何种 GPU、dtype、batch、长度和 baseline 下得到的？

## 10.7 小实验：先证明“状态化”而不是先追求速度

在真正安装 FLA 或写 Triton 前，做一个只有几十行的小实验即可。目标是证据链，而不是
漂亮曲线。

1. 生成小随机 `q,k,v`，例如 `T=8,Dφ=4,Dv=3`；
2. 写出上面的 recurrent reference；
3. 再用显式下三角循环，按定义直接计算每个 `o_t`；
4. 断言两个输出在 float64 或合理容差内相同；
5. 改成 `T=64,128,256`，只记录 reference 的时间趋势，不把 CPU 小样本结果当 GPU 结论。

建议在实验记录里增加两栏：`计算定义是否相同？` 与 `是否含归一化 z？`。不少看似的
“复现失败”其实是把 unnormalized fast-weight update 和 normalized kernel attention
当成同一个算子了。

## 常见误区

**误区 1：所有线性 attention 都是 Performer。**
Performer 是 feature-map/随机特征路线的代表；GLA、DeltaNet、GDN、KDA 的 state update
更适合放在“现代线性 RNN / fast weight”支线理解。

**误区 2：有 state 就一定是 RNN，因而无法并行训练。**
递推定义确实有依赖，但 associative scan、矩阵重写和 chunkwise 算法可以暴露一部分并行。
问题不是“有没有递推”，而是 transition 能否被高效组合。

**误区 3：固定 state 是 KV cache。**
Dense decode 的 KV cache 通常随 `T` 线性增长；线性方法的 recurrent state 在理想情形下
不随 `T` 增长。两者都保存历史信息，但数据结构与读写模式不同。

**误区 4：论文说 `O(T)` 就不用看 benchmark。**
复杂度忽略 `Dφ×Dv` 的状态大小、gate 的额外计算、kernel launch、低精度与不规则访存。
任何速度结论都要回到测量设置。

## 练习

### 练习 10.1：推导顺序交换

从

$$
\sum_{i\le t}(\phi(q_t)^\top\phi(k_i))v_i
$$

出发，在不跳步的情况下把 `φ(q_t)` 提到求和外，并写出 state 的形状。说明为什么这种
交换对 softmax 的原始指数形式不直接成立。

### 练习 10.2：状态大小与长度

设每层有 `H=8` 个 head、`Dφ=Dv=128`、BF16。计算单层全部 `S` state 的近似字节数；
再比较它与保存 `T=32768` 个 K/V 的 dense cache 在数量级上的区别。明确写出你是否把 batch
算进去了。

### 练习 10.3：先预测，再测量

若把 chunk 长度从 16 改为 128，预测下列哪项可能变好、哪项可能变坏：矩阵乘利用率、
边界 state 次数、片上资源压力、数值误差。之后再用真实实现检验，而不是把预测写成事实。

## 通过条件

进入第 11 章前，你应能在不看正文的情况下：

- 写出 `S_t,z_t` 的递推、每个变量的形状和归一化输出；
- 用一句话说明本章方法与 FlashAttention 的根本区别；
- 解释为什么“固定大小 state”既带来长序列优势，也带来记忆干扰风险；
- 画出 recurrent 与 chunkwise 的数据流；
- 完成一个 state reference 与定义式的数值一致性测试。

下一章将只新增两个问题：**何时忘记？又怎样覆盖一条旧的 key-value 记忆？**
