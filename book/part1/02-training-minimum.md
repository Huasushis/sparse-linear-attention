# 第 2 章：读 attention 论文所需的训练最小闭环

你不需要先修完整的机器学习课程，才有资格研究 attention kernel；但如果不知道模型如何
从 loss 获得梯度，就很容易把“前向算得对”“可以训练”“训练后效果好”混成一件事。
这一章只建立后续研究反复用到的最小闭环：数据如何变成监督信号，loss 衡量什么，
backward 产生什么，optimizer 又改变什么。

本章不会教大规模预训练配方，也不会推导所有优化理论。目标是让你能读懂实验设置、
训练循环与 backward benchmark，并知道自己暂时还没有验证什么。

## 学习目标

读完后，你应当能够：

1. 从一段 token 序列构造 next-token prediction 的输入和标签；
2. 根据 `[B,T,V]` logits 写出交叉熵 loss，并手算一个三分类例子；
3. 用链式法则解释 backward 为什么能为每个参数得到梯度；
4. 区分 parameter、activation、gradient、optimizer state 和 hyperparameter；
5. 读懂最小 PyTorch 训练循环中 `zero_grad/backward/step` 的职责；
6. 解释“只实现 forward”和“可用于训练”之间还缺什么；
7. 看 benchmark 时分清 forward、forward+backward 和完整 training step。

## 2.1 模型学习的任务：猜下一个 token

假设 tokenizer 把一段文本转成 5 个整数：

```text
[11, 42, 7, 9, 3]
```

decoder-only 语言模型通常使用**右移一位**的监督：

```text
输入 x: [11, 42, 7, 9]
标签 y: [42,  7, 9, 3]
```

也就是：看到 `11` 猜 `42`，看到 `[11,42]` 猜 `7`，依此类推。causal mask 保证位置
`t` 不能偷看标签所在的未来位置。

对 batch 中的 token 张量，常见形状是：

```text
tokens:  [B, T+1]
inputs:  tokens[:, :-1] -> [B, T]
targets: tokens[:,  1:] -> [B, T]
```

经过[第 1 章](01-transformer-from-tensors.md)的前向计算，模型输出：

```text
logits: [B, T, V]
```

`V` 是词表大小。每个 `(b,t)` 都对应一个 `V` 分类问题，而不是整条序列只产生一个
答案。

### Teacher forcing 是什么

训练位置 `t` 时，模型看到的前缀通常来自真实数据，而不是它在前一步自己采样的 token，
这叫 teacher forcing。这样所有位置可以并行计算。生成时却必须把模型刚生成的 token
接回输入，逐步 decode。训练和生成的执行形态因此天然不同。

## 2.2 从 logits 到概率

logit 是任意实数，不是概率。对固定位置，softmax 把向量 `z in R^V` 转成分布：

```text
p_k = exp(z_k) / sum_j exp(z_j)
```

它满足 `p_k > 0` 且 `sum_k p_k = 1`。给所有 logit 加同一个常数不会改变概率，所以
稳定实现会先减最大值：

```text
softmax(z) = softmax(z - max(z))
```

这不是近似技巧，而是完全相同的数学结果；它避免 `exp(1000)` 溢出。后面学习
FlashAttention 的 online softmax 时，会再次使用这种“改变计算顺序但保持结果”的思想。

## 2.3 交叉熵在惩罚什么

若正确 token 的 id 是 `y`，单个位置的负对数似然为：

```text
loss = -log p_y
     = -z_y + log(sum_j exp(z_j))
```

模型给正确答案的概率越大，loss 越小。若 `p_y=1`，loss 接近 0；若 `p_y=0.01`，loss
约为 4.605。

### 手算例子

词表只有 3 个 token，logits 为：

```text
z = [ln 2, 0, 0]
```

则指数为 `[2,1,1]`，概率为 `[1/2,1/4,1/4]`。

- 若正确类别是 0，loss 为 `-log(1/2)=log 2`；
- 若正确类别是 1，loss 为 `-log(1/4)=log 4`。

模型不是只在“猜错”时才受惩罚；即使类别 0 的概率最高，只要没有接近 1，loss 仍非零。

### batch 和序列维怎样汇总

每个非 padding 位置先各有一个 loss，训练代码通常取平均：

```text
L = sum_(b,t) valid[b,t] * loss[b,t] / sum_(b,t) valid[b,t]
```

若不同实验使用 `sum`、按 token 平均或按序列平均，梯度尺度可能不同。处理变长序列时，
padding token 必须从 loss 中排除；attention mask 和 loss mask 是两个相关但不同的概念：
前者决定能看谁，后者决定哪些位置贡献训练目标。

## 2.4 参数、激活、梯度分别是什么

把训练中的对象分清，会让显存分析容易很多。

| 对象 | 例子 | 是否由 optimizer 更新 |
| --- | --- | --- |
| parameter | `W_Q`、embedding table | 是 |
| activation | 某一批数据产生的 `Q`、softmax 输出 | 否，下一批会改变 |
| gradient | `W_Q.grad` | 否，它指导参数更新 |
| optimizer state | AdamW 的一阶/二阶动量 | 由 optimizer 维护 |
| hyperparameter | learning rate、batch size | 由研究者配置 |
| buffer | 某些归一化统计或缓存 | 依模型定义，不一定求梯度 |

**参数**跨 batch 保存，是模型学到的内容。**激活**属于一次 forward；为了 backward，
框架可能暂存它们或之后重算。**梯度**的形状与对应参数相同，表示当前 loss 对参数的局部
敏感度。

例如：

```text
W_Q:      [C, H*D]
dL/dW_Q:  [C, H*D]
```

梯度不是“参数应该变成什么”，而是“在当前位置，参数向哪个方向小幅变化会怎样影响
loss”。optimizer 再把梯度与历史状态转成具体更新量。

## 2.5 链式法则：backward 的核心

先看只有一个参数的模型：

```text
y_hat = w * x
L = 1/2 * (y_hat - y)^2
```

loss 对 `w` 的导数通过中间量 `y_hat` 传递：

```text
dL/dw = dL/dy_hat * dy_hat/dw
      = (y_hat - y) * x
```

取 `x=2, y=6, w=1`：`y_hat=2`，`L=8`，梯度为 `(2-6)*2=-8`。梯度为负意味着把
`w` 稍微增大，会让 loss 降低。

Transformer 只是把这条链变得很长并有许多分支。自动微分框架在 forward 时构建或
记录计算关系，然后按相反方向应用每个算子的局部导数。这就是 reverse-mode automatic
differentiation，也就是日常所说的 autograd/backpropagation。

### 一个非常有用的交叉熵梯度

对 softmax + cross entropy，单位置 loss 对 logit 的梯度可简化为：

```text
dL/dz_k = p_k - 1[k = y]
```

仍用概率 `[1/2,1/4,1/4]`，若正确类别为 0，梯度是：

```text
[-1/2, 1/4, 1/4]
```

优化会倾向于提高正确 logit、降低其他 logit。三个分量和为 0，也与“所有 logit 同加
一个常数不改变 softmax”一致。

## 2.6 一次训练 step 的完整顺序

最小流程是：

```text
取一个 batch
  -> forward 得到 logits
  -> logits 与 targets 计算 loss
  -> backward 得到每个参数的 gradient
  -> optimizer 根据 gradient 更新参数
  -> 进入下一 batch
```

PyTorch 代码通常长这样：

```python
model.train()

for tokens in dataloader:                 # tokens: [B, T+1]
    inputs = tokens[:, :-1]                # [B, T]
    targets = tokens[:, 1:]                # [B, T]

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)                 # [B, T, V]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )                                      # scalar
    loss.backward()
    optimizer.step()
```

逐行理解：

- `model.train()` 打开训练模式，例如启用 dropout；它本身不做参数更新；
- `zero_grad()` 清除上一 step 的梯度，因为 PyTorch 默认会累加 `.grad`；
- `model(inputs)` 是 forward；
- reshape 把 `B*T` 个位置当作 `B*T` 个分类样本，词表维保持为最后一维；
- `loss.backward()` 填充参数的 `.grad`，但还没有更新参数；
- `optimizer.step()` 才修改参数。

如果故意做 gradient accumulation，就不是每个 microbatch 都清梯度或 step，而是累积
若干次 backward 后更新一次。论文中的 global batch size 可能由
`microbatch * accumulation steps * data-parallel workers` 共同形成。

## 2.7 SGD 到 AdamW：optimizer 做了什么

最简单的随机梯度下降（SGD）是：

```text
theta_(t+1) = theta_t - learning_rate * grad_t
```

`theta` 代表所有参数。learning rate 太大可能越过低 loss 区域，太小则进展缓慢。

Adam 为每个参数元素维护梯度的一阶、二阶指数移动平均：

```text
m_t = beta1 * m_(t-1) + (1-beta1) * g_t
v_t = beta2 * v_(t-1) + (1-beta2) * g_t^2
```

做偏差修正后，用 `m_hat / (sqrt(v_hat)+eps)` 调整每个元素的步幅。AdamW 再以解耦方式
加入 weight decay。现在无需背下完整实现，只要知道：

1. AdamW 不只保存参数和梯度，还保存通常与参数同形状的 `m`、`v`；
2. optimizer state 会显著占显存；
3. 换 optimizer、学习率或 schedule，训练结果就不再是同一实验；
4. weight decay 不等于“把 loss 简单加一项”在所有 optimizer 中都完全等效。

后面做算子 microbenchmark 时通常不包含 optimizer；做完整训练吞吐时则可能包含。报告
必须写清边界。

## 2.8 为什么一个 forward kernel 还不等于可训练算子

假设你写了新的 attention forward：给定 `Q,K,V`，输出 `O` 正确。用于训练还需要 loss
对输入和参数的梯度，例如：

```text
dO -> dQ, dK, dV
```

若算子有 gating、normalizer 或可学习参数，也需要相应梯度。有三种常见情况：

1. 完全由 PyTorch 原语组成，autograd 自动拼出 backward，正确但未必高效；
2. 自定义 forward + 手写 backward kernel；
3. 利用自动生成/编译的 backward，但仍需验证数值与性能。

因此复现 kernel 至少有两个正确性层次：

- **forward correctness**：输出与 reference 足够接近；
- **backward correctness**：给相同上游梯度，`dQ/dK/dV/...` 与 reference 足够接近。

训练论文若只展示 forward latency，不能据此推出训练加速。反过来，服务系统只做推理，
backward 就不在关键路径中。

## 2.9 为什么训练会占很多显存

训练显存大致来自：

```text
parameters
+ gradients
+ optimizer states
+ saved activations
+ temporary workspaces
```

其中 activations 常随 `B`、`T`、层数增长。普通 attention 若保存 `[B,H,T,T]` 概率供
backward 使用，长序列尤其昂贵。FlashAttention 的一个关键点是 backward 时利用较小的
统计量重算局部块，而不是保存整个大矩阵，体现了**用额外计算换显存/访存**。

activation checkpointing 也使用类似取舍：forward 不保存某些中间激活，backward 时再
算一遍。它不会减少参数或 optimizer state，也不等同于 attention kernel 的分块。

## 2.10 dtype、混合精度和数值容差

训练常用 bf16/fp16 存储和计算大部分张量，同时让某些累加或 optimizer state 保持更高
精度。原因不是“低精度永远一样准”，而是 GPU 对低精度矩阵乘法吞吐更高、显存更省。

这会给复现带来三个直接要求：

- correctness test 不能期待两个不同计算顺序的浮点结果逐 bit 相等；
- 应报告 `dtype`、最大/平均误差以及使用的 `atol/rtol`；
- 数值敏感操作，如 softmax 归一化和长递归累加，需要特别检查稳定性。

fp16 的动态范围比 bf16 小，过去常配合 loss scaling 防止很小的梯度下溢。你暂时不必
实现自动混合精度，但读代码时要能认出 autocast、GradScaler 和高精度 accumulator。

## 2.11 `train()`、`eval()` 与“模型效果”

`model.eval()` 会关闭 dropout 等训练时随机行为，但不会自动关闭梯度记录；推理通常还会
配合 `torch.no_grad()` 或 inference mode。反之，`model.train()` 只是切换模块行为，
不会自动调用 backward。

评估一个新 attention 方法可能包含三层不同证据：

1. **算子正确性**：与数学 reference 对齐；
2. **固定权重替换后的效果**：若属于近似/稀疏化，比较 perplexity 或下游任务退化；
3. **按新架构重新训练后的效果**：比较训练预算、数据和最终质量。

第 2 层失败不一定说明第 3 层必然失败，因为模型可能通过训练适应新结构；第 3 层表现好
也不意味着可以无损替换任意已有模型。读 sparse/linear attention 论文时必须辨别作者
证明的是哪一层。

## 2.12 代码阅读练习：找出训练循环中的错误

阅读下面代码，不要先运行：

```python
model.train()
for tokens in dataloader:
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
```

回答：

1. 哪一行缺失会让梯度跨 step 意外累加？
2. `logits`、reshape 后 logits、`y`、reshape 后 `y` 分别是什么形状？
3. `optimizer.step()` 和 `loss.backward()` 能否交换？为什么？
4. 如果 `tokens` 含 padding，这段 loss 还缺什么处理？
5. 若只想评估 loss，需要哪些模式变化，哪些行应删除？

### 核对要点

缺少 `optimizer.zero_grad()`；reshape 后 logits 为 `[B*T,V]`，target 为 `[B*T]`；必须先
产生梯度再 step；padding 需要 `ignore_index` 或显式有效位置 mask。评估时使用
`model.eval()` 与 `torch.no_grad()`，不调用 backward/step/zero_grad。

## 2.13 手算练习：一次 SGD 更新

仍用：

```text
y_hat = w*x
L = 1/2*(y_hat-y)^2
```

取 `w=1, x=2, y=6, learning_rate=0.1`。

1. 算 forward 的 `y_hat` 和 `L`；
2. 算 `dL/dw`；
3. 做一次 SGD 后的新 `w`；
4. 用新 `w` 再算 loss，检查是否下降；
5. 若 learning rate 改成 1，发生什么？这说明“沿负梯度方向”为什么仍不保证任意步长
   都下降？

答案：初始 `y_hat=2, L=8, gradient=-8`；更新后 `w=1.8`，新预测 3.6，新 loss 2.88。
若学习率为 1，则 `w=9`，预测 18，loss 72，反而增大。

## 2.14 常见误区

**误区 1：backward 更新了参数。**
backward 只计算并累积梯度；optimizer step 才更新参数。

**误区 2：loss 下降证明实现完全正确。**
许多有 bug 的模型也可能在小数据上下降。仍需 reference、梯度检查、shape/causal 测试。

**误区 3：一次 forward 的峰值显存就是训练显存。**
训练还要考虑 saved activations、gradients 和 optimizer states。

**误区 4：推理更快就说明训练更快。**
训练包含 backward，算子的 work partition 和内存需求可能完全不同。

**误区 5：用了相同模型名就是公平对比。**
dtype、batch、长度、硬件、编译 warmup、训练 token 数、数据和评估协议都可能不同。

**误区 6：linear attention 只要 forward 递归能跑，就能高效训练。**
逐 token 递归会限制训练并行。如何把递归改写成 parallel/chunkwise scan，是[第 3 章](03-rnn-state-and-scan.md)
的核心。

## 2.15 这一章怎样接到 attention 研究

读论文或仓库时，今后固定问下面几句：

- 这是 inference-only，还是支持训练？
- benchmark 测的是 forward、forward+backward，还是 optimizer 在内的完整 step？
- 新算子的 backward 是手写、自动微分，还是没有提供？
- 质量结果来自无训练替换、微调，还是从头预训练？训练预算公平吗？
- sparse selector/gate 是否可学习？若可学习，梯度怎样穿过选择；若不可微，作者如何处理？
- recurrent state 在 backward 时要保存多少，能否重算或 scan？

对 Kimi Linear、Gated DeltaNet 等架构，kernel speed 只是证据的一部分。它们还需要在可比
训练配方下证明模型质量。因此之后复现会把任务拆开：先做 toy/reference 正确性，再跑
作者算子 benchmark，最后在算力允许范围内做已有 checkpoint 的评估；不会假装 1 张
GPU、4 小时作业能重现完整大模型预训练。

## 本章小结

语言模型训练把每个前缀变成一个分类样本，用交叉熵衡量正确 next token 的概率。
backward 借助链式法则产生与参数同形状的梯度，optimizer 再结合 learning rate 和历史
状态更新参数。forward 正确、backward 正确、训练稳定、最终质量好是逐层增强的四种
主张，不能互相替代。

研究 attention kernel 时，这套最小闭环让你知道 benchmark 的边界，也能解释为什么
training memory 不等于模型权重大小、为什么 FlashAttention backward 会选择重算、以及
为什么一个 inference-only demo 还不能被称作“完成训练复现”。

## 通过条件

不看正文，完成以下任务，才算通过本章：

- 给定 `[B,T+1]` tokens，写出输入、标签和 logits 的形状；
- 手算三分类 softmax、cross entropy 及 `p-one_hot(y)` 梯度；
- 用两句话区分 backward 与 optimizer step；
- 默写最小训练循环，并解释为什么先清梯度；
- 列出训练显存的四类主要来源；
- 看到“2x training speedup”时，列出至少五项必须核查的实验设置；
- 说明为何 forward reference 对齐仍不足以证明算子可以训练。

## 延伸材料

- PyTorch, [A Gentle Introduction to `torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)：
  用很小的图观察 `.grad_fn` 与 `.grad`。
- Karpathy, [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)：
  适合第一次真正手算计算图，不要求一次看完系列。
- Stanford CS336, [Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)：
  后面需要训练系统背景时按主题查阅，不把整门课作为当前前置门槛。
