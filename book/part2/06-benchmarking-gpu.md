# 第 6 章：GPU Benchmark 不是按一下秒表

性能研究最危险的结果不是“程序报错”，而是得到一个看起来合理、实际上比较错对象的
数字。一个能运行的 benchmark，只说明计时代码没有崩溃；一个可信的 benchmark 还要
回答：算得对不对、比较是否公平、时间是否测全、结论能否复现。

本章建立后续所有 dense、linear、sparse attention 实验共用的测量协议。

## 学习目标

读完后，你应当能够：

1. 把 correctness、operator 性能、模型性能和任务质量分开；
2. 正确测量异步 GPU kernel，解释 warm-up 和重复测量的作用；
3. 设计覆盖 shape、dtype、mode 的最小实验矩阵；
4. 计算 latency、throughput、有效带宽和 FLOP/s，并知道各指标的边界；
5. 记录足够元数据，使同学能在另一张 GPU 上复跑；
6. 识别“只挑有利形状”“漏算预处理”等常见不公平比较。

## 6.1 先写清楚你在证明什么

一次 attention 复现至少有四层问题：

| 层级 | 要证明的事 | 典型输出 |
| --- | --- | --- |
| 数值正确性 | 新实现是否符合所声明的数学定义 | max/mean error、梯度误差 |
| operator 性能 | 固定输入下单次算子多快、多占显存 | latency、GB/s、TFLOP/s、peak memory |
| 模型性能 | 放进模型后真实请求多快 | TTFT、prefill tok/s、decode tok/s |
| 任务质量 | 近似或新架构是否仍完成任务 | loss、perplexity、准确率、长上下文得分 |

FlashAttention 是 exact attention，因此主要对比正确性与系统性能；多数 sparse attention
会改变计算结果，还必须报告质量。一个 sparse kernel 比 dense kernel 快，不等于模型
更好；一个线性模型在任务上更好，也不证明其 kernel 更快。

在打开终端前先写一句可证伪的实验问题，例如：

> 在 A100、BF16、`B=4,H=16,D=64` 下，序列长度从 1K 增到 16K 时，SDPA 的高效后端
> 相对显式 materialize 的 dense reference 如何改变 forward latency 和峰值显存？

这比“测一下 FlashAttention”更好，因为设备、dtype、shape、对象、模式和指标都明确。

## 6.2 正确性先于速度

### 建立独立 reference

reference 的任务是易读和可信，不是快。第一版通常使用 FP32 PyTorch，并把每一步分开：

```python
scores = q.float() @ k.float().transpose(-1, -2)
scores = scores / math.sqrt(q.shape[-1])
scores = scores.masked_fill(~mask, float("-inf"))
prob = torch.softmax(scores, dim=-1)
expected = prob @ v.float()
```

被测实现使用 BF16 时，与 FP32 reference 比较。不要让 reference 和被测实现调用同一个
自定义 helper，否则同一个 bug 可能同时出现在两边。

### 误差不是只看 `allclose=True`

至少记录：

$$
e_{\max}=\max_i |y_i-\hat y_i|,
\qquad
e_{\text{mean}}=\frac{1}{n}\sum_i|y_i-\hat y_i|.
$$

相对误差在 reference 接近 0 时会爆大，因此最好同时报告绝对误差，并说明 `atol/rtol`。
对不同 dtype 使用不同容差，不要为了让测试通过不断放宽到失去意义。

要覆盖容易藏 bug 的小形状：

- `T` 不是 tile size 的整数倍，如 127、257；
- causal 与 non-causal；
- 不相等的 `T_q`、`T_k`；
- head dimension 32、64、128；
- GQA 中 `H_q != H_kv`；
- padding、全 mask 行和极端大/小 logits。

训练 kernel 还要检查 `dQ,dK,dV`。可在很小的 FP64/FP32 输入上使用 autograd reference，
或者做有限差分抽查。有限差分近似为：

$$
\frac{\partial L}{\partial x_i}
\approx \frac{L(x_i+\epsilon)-L(x_i-\epsilon)}{2\epsilon}.
$$

它很慢，所以只用于小尺寸和少量元素。

## 6.3 为什么朴素计时会错

GPU kernel 相对 CPU 是异步提交的。正确计时需要让区间覆盖 GPU 真正完成工作的时间。
可以使用框架提供的 benchmark helper；手写时使用 CUDA events：

```python
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

for _ in range(warmup):
    fn()
torch.cuda.synchronize()

samples_ms = []
for _ in range(repeats):
    starter.record()
    fn()
    ender.record()
    ender.synchronize()
    samples_ms.append(starter.elapsed_time(ender))
```

不要在每个 kernel 内部随意插入全设备同步；这会改变流水与并发。上面的同步是为了划清
一次独立样本的边界。若要测一段端到端流水，应把整个流水包在 events 中，只在末尾同步。

### warm-up 在排除什么

第一次执行可能包含：

- Python import 和 lazy initialization；
- Triton/JIT 编译与 autotune；
- CUDA context 建立；
- allocator 首次申请；
- cache 冷启动和 GPU 时钟状态变化。

warm-up 次数不应成为神秘常数。持续运行到计时稳定，再记录实际次数。若研究的正是
冷启动/JIT 延迟，则应单独定义“首次调用时间”，而不是把它偷偷混进稳态 latency。

### 为什么报告分布

一次运行可能受 OS 调度、共享集群负载、温度和功耗状态影响。建议保留全部样本，至少
报告 median/p50；有抖动时再给 p10/p90。平均值容易被少数异常值拖动。

## 6.4 实验矩阵：不要只挑一个甜点形状

Attention 性能对 shape 极敏感。一个最小 dense baseline 矩阵可从下面选择：

| 轴 | 起步取值 | 为什么 |
| --- | --- | --- |
| `B` | 1, 4 | 小并行量与常规 batch |
| `T` | 128, 1K, 4K, 16K | launch、计算、显存三个区域 |
| `H_q/H_kv` | 16/16, 16/4 | MHA 与 GQA |
| `D` | 64, 128 | 常见 head dimension |
| dtype | FP32 reference, BF16 实验 | 正确性与实际 tensor core 路径 |
| mask | causal, non-causal | causal 可跳过部分 tile |
| mode | forward, fwd+bwd, prefill, decode | 工作负载完全不同 |

不必一开始跑所有笛卡尔积。应先选 8 到 12 个有代表性的点，再根据拐点加密。OOM 不是
“没有结果”，而是显存可扩展性的结果，应记录发生在哪个 shape。

### prefill 与 decode 必须分开

- **prefill**：`T_q` 和 `T_k` 都较大，处理整段 prompt，矩阵乘占比高；
- **decode**：每一步通常 `T_q=1`，读取历史 KV cache，容易受内存带宽和 launch 限制。

把 prefill 的 tok/s 与 decode 的 tok/s 放在同一列比较没有意义。端到端服务还常报告：

- TTFT（time to first token）；
- TPOT（time per output token）；
- inter-token latency；
- 在指定并发和请求长度分布下的吞吐。

## 6.5 明确计时边界

对动态 sparse attention，可能存在：

```text
选择重要 token / 构造索引 -> 稀疏 attention kernel -> scatter/merge 输出
```

只测中间 kernel 会回答“已知索引时算 attention 多快”，不能回答完整方法多快。合理做法是
同时报告：

| 指标 | 包含内容 |
| --- | --- |
| kernel-only | 已准备好输入、mask/index 后的核心 kernel |
| selection/preprocess | 选择、排序、mask/index 构造 |
| end-to-end operator | 从原始 QKV 到最终输出的全部必要工作 |

内存分配、随机输入生成通常放在稳态计时区间外；但如果算法每次请求必须动态分配临时
buffer，就不能假装这部分不存在。计时边界应与所声称的使用场景一致。

## 6.6 从 latency 推导性能指标

### 吞吐

若一次调用处理 `N_token` 个 token，耗时 `t` 秒：

$$
\text{throughput}=\frac{N_{token}}{t}\quad \text{token/s}.
$$

必须说明 token 怎样计数。prefill 常计 `B*T`；decode 可计每步生成的 batch token。不能把
不同 batch 的 token/s 当作单请求延迟。

### 有效带宽

若一个向量加法最低搬运 `bytes_min`，可以定义：

$$
BW_{effective}=\frac{bytes_{min}}{t}.
$$

它是基于模型的有效带宽，不等于 profiler 看到的真实所有内存流量。cache 命中、重复读取
和写分配都会改变实际流量。因此报告时应写清 bytes 的计算方式。

### 有效 FLOP/s

对于 dense attention forward，忽略 softmax 的低阶项，可粗算两次矩阵乘：

$$
F \approx 4BH T_qT_kD.
$$

其中乘加按 2 FLOPs 计。于是：

$$
P_{effective}=F/t.
$$

这适合比较相同数学工作量的 exact kernel。对于 sparse/linear 方法，若各自做的数学工作
不同，单看 TFLOP/s 可能奖励“多做无用计算”的实现，必须同时报告 wall-clock 和质量。

### Speedup 必须带分母

永远写成：

```text
1.8x vs PyTorch SDPA math backend, same B/H/T/D/dtype/causal, forward-only
```

不要只写“快 1.8x”。改变 baseline、backend 或计时范围，数字就变了。

## 6.7 显存测量

区分三种量：

- 参数/输入本身占用；
- 算子额外申请的临时空间；
- 框架 caching allocator 已保留但当前未使用的空间。

PyTorch 可在测试前重置 peak 统计，并在运行后读取 allocated peak。为了比较算子增量，
应先分配并保留相同输入，再重置统计，然后运行被测算子。不同进程或不同方法之间最好
独立启动，避免 allocator 历史状态污染。

训练时还要说明是否包含：

- 保存供 backward 的 activation；
- gradients；
- optimizer state；
- gradient checkpointing/recomputation。

“FlashAttention 使用线性额外内存”不等于整个模型显存随序列长度线性增长；模型其他层、
activation 和 logits 仍占空间。

## 6.8 Profiler 用来解释，不用来装饰

benchmark 告诉你“快多少”，profiler 帮你判断“为什么”。建议顺序：

1. 先得到稳定、可复现的时间；
2. 写下瓶颈假设；
3. 再用 profiler 检查 kernel 数、时间占比、内存流量、tensor core 使用、occupancy 等；
4. 若证据否定假设，修改解释，而不是挑一张漂亮截图。

开始阶段不要一次收集所有指标。先回答一个具体问题，例如“为什么 `T=128` 时 Triton
版本更慢：launch 太多，还是访存模式差？” profiler dump 往往很大，应留在集群工作
目录，只把命令、关键数字和小截图/表格提交 Git。

## 6.9 一张可复跑的实验卡

每张性能表旁至少保存：

```yaml
question: "..."
repo_commit: "..."
author_repo_commit: "..."
gpu: "..."
driver_cuda: "..."
pytorch_triton: "..."
mode: "forward | fwd+bwd | prefill | decode"
shape: {B: 4, T_q: 4096, T_k: 4096, H_q: 16, H_kv: 16, D: 64}
dtype: "bf16"
layout: "contiguous BHTD"
causal: true
baseline_backend: "..."
warmup_repeats: [25, 100]
statistic: "median; p10/p90"
command: "..."
seed: 0
```

还要保存原始小型 CSV，而不只是画出的图。图会隐藏精度和异常点，CSV 允许重新计算。

## 6.10 一套由小到大的执行顺序

```text
1. CPU/FP32 或 GPU/FP32 小 reference
2. 小 shape forward correctness
3. 边界 shape + causal/GQA correctness
4. backward correctness（若实现训练）
5. 固定单 shape，检查 timing 稳定
6. 扫 shape 矩阵并记录 OOM
7. 只对异常或关键拐点做 profiler
8. 汇总 operator 结果
9. 有必要时才进入模型/服务 benchmark
```

这样安排能避免在跑了数小时后才发现 causal mask 写反。

## 常见坑

- 用 `time.time()` 包围异步 CUDA 调用而不同步；
- 把 JIT/autotune 首次开销混入稳态数据，又不说明；
- 方法 A 用 BF16，方法 B 用 FP32；
- 比较不同 `B/T/H/D`，只展示 speedup；
- 新方法漏算 mask/index 构造，baseline 却计入全部工作；
- 只测 `T=4096` 这个最有利点；
- 只看输出前几个元素，没有系统误差和边界测试；
- 运行时 GPU 上还有别人的进程，却不记录；
- 把官方在 H100/A100 上的数字当成本机复现结果；
- 测一次、保留最好成绩，不保存原始样本。

## 练习

### 练习 6.1：找出错误 benchmark

解释下面代码至少三个问题，并写出修正版伪代码：

```python
t0 = time.time()
for _ in range(100):
    y = my_kernel(torch.randn(4096, device="cuda"))
print((time.time() - t0) / 100)
```

### 练习 6.2：设计第一张 attention 表

在总实验矩阵中选 8 个 shape。每个选择写一句理由，必须同时包含一个非整 tile 长度、
一个 GQA、一个 decode 和一个预期 OOM 或接近显存边界的点。

### 练习 6.3：先写实验卡

在真正运行前填写一张实验卡。无法提前填写的字段标记为 `TBD after allocation`，不要猜
集群最终分到的 GPU。

### 练习 6.4：解释矛盾指标

构造一个例子：方法 A 的 TFLOP/s 比 B 高，但 wall-clock 更慢；再构造一个 A 的 token/s
更高但单请求 latency 更差的例子。说明为什么指标并不矛盾。

## 通过条件

进入 dense attention 实验前，你应当提交一页 benchmark 协议，且满足：

- 有明确、可证伪的问题和 baseline；
- correctness 形状与 performance 形状分开；
- 计时使用 event/helper，包含 warm-up、repeats 和统计量；
- 明确 kernel-only 与 end-to-end 边界；
- 表格每一行都能追溯到设备、commit、软件版本和命令；
- 能解释为什么 prefill 与 decode 不能混为一个 speedup。
