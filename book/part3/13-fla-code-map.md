# 第 13 章：沿着 FLA 读代码——从 layer 到 kernel，而不是从第一行 Triton 开始

[Flash Linear Attention（FLA）](https://github.com/fla-org/flash-linear-attention)不是要你在第一天
“读懂一个大型 CUDA 项目”的对象，而是一座可以反复走的实现地图。对当前研究主线，最有用
的路线是：

```text
论文公式
  -> layer 的张量接口
  -> op 的公开入口
  -> naive/reference
  -> recurrent 或 chunk 算法
  -> Triton/TileLang backend
  -> unit test
  -> benchmark / 你的实验记录
```

一旦你能把这条链走通一次，之后读 GLA、DeltaNet、KDA 或新的 sparse operator 都会轻松
很多。本章不要求逐行翻译 kernel；它要求你在任何一层都能回答“输入是什么、输出是什么、
与公式哪个变量对应、证据在哪里”。

## 学习目标

读完后，你应当能够：

1. 找到 FLA 中 KDA、Gated DeltaNet、GLA、DeltaNet 的 layer 和 op 入口；
2. 按正确顺序阅读 `naive -> recurrent -> chunk -> kernel`；
3. 为一个 FLA op 画出输入/输出/state/gradient 的 dataflow card；
4. 通过测试与独立 reference 区分“代码能 import”与“公式正确”；
5. 找到官方 benchmark 框架，但不会未经核查就运行长任务；
6. 知道 TileLang/Ascend backend 目前为什么是旁支，而不是第一阅读目标。

## 13.1 版本纪律：先记录你读的是哪棵树

下面的路径已按 FLA `main` 源码树在 **2026-07-23** 核对。项目更新很快，所以每次真正
运行前先在自己的 clone 中记录：

```powershell
git rev-parse HEAD
git status --short
rg -n "class|def" fla/layers/kda.py
```

教程中的路径是导航，不是对未来版本的承诺。若路径移动，先用 `rg --files fla | rg "kda|gated_delta|delta_rule"`
定位，而不是凭记忆修改 import。把 commit hash 写进实验记录，这会避免“上周能跑、今天
不一样”的幽灵问题。

## 13.2 最小源码地图

下表把当前主线拆成四层。只读第一列也无法理解实现；只读最后一列也无法判断论文公式。

| 目标 | layer（模型调用处） | operator（算子入口） | 先读的 reference / 算法文件 |
| --- | --- | --- | --- |
| GLA | `fla/layers/gla.py` | `fla/ops/gla/__init__.py` | `gla/naive.py`、`fused_recurrent.py`、`chunk.py` |
| DeltaNet | `fla/layers/delta_net.py` | `fla/ops/delta_rule/__init__.py` | `delta_rule/naive.py`、`fused_recurrent.py`、`chunk.py` |
| Gated DeltaNet | `fla/layers/gated_deltanet.py` | `fla/ops/gated_delta_rule/__init__.py` | `gated_delta_rule/naive.py`、`fused_recurrent.py`、`chunk.py` |
| KDA | `fla/layers/kda.py` | `fla/ops/kda/__init__.py` | `kda/naive.py`、`fused_recurrent.py`、`chunk.py` |

对于 KDA，继续向下通常会看到：

```text
fla/ops/kda/
  __init__.py                 # 公开 API / dispatch 入口
  naive.py                    # 清楚优先的实现
  fused_recurrent.py          # 逐步 state 视角的高效版本
  chunk.py                    # chunk 的 Python/封装入口
  chunk_fwd.py, chunk_bwd.py  # 前向和反向主路径
  chunk_intra.py              # 块内工作
  gate.py                     # gate 的辅助计算
  wy_fast.py                  # WY/辅助量相关计算
  backends/                   # 额外后端，例如 flash_kda、TileLang、Ascend 路径
```

Gated Delta Rule 也有相似的 `chunk_fwd.py`、`gate.py`、`wy_fast.py` 等目录结构；
泛化 DPLR/IPLR 的实验路径在 `fla/ops/generalized_delta_rule/`。这些相似性正是本书先讲
统一递推的原因：你不会把每个目录当成一门新语言。

## 13.3 从 layer 开始：先看“模型要什么”

打开 `fla/layers/kda.py` 或 `fla/layers/gated_deltanet.py` 时，不要立即读 kernel import。
先回答：

1. layer 的输入 hidden state 是什么形状，例如 `[B,T,C]`？
2. q/k/v、`alpha`、`beta` 分别由哪条 projection/conv 路径产生？
3. q/k 是否做 L2 normalization？这个选择在数值稳定上服务什么假设？
4. token-mixing op 的输出怎样经过 norm、output gate 和 output projection 回到 `[B,T,C]`？
5. layer 是否支持 `past_key_values` / recurrent state / cache 接口？
6. train 与 inference 是否调用相同 op 变体？

把答案画成下面这种卡片，而不是复制几十行代码：

```text
x [B,T,C]
 ├─ q path: linear -> short conv -> activation -> L2Norm -> q [B,H,T,Dk]
 ├─ k path: linear -> short conv -> activation -> L2Norm -> k [B,H,T,Dk]
 ├─ v path: linear -> short conv -> activation          -> v [B,H,T,Dv]
 ├─ gates: projection(s)                                -> alpha / beta
 └─ op(q,k,v,alpha,beta,initial_state)
       -> o [B,H,T,Dv], final_state [...]
       -> norm / output gate / projection -> y [B,T,C]
```

如果某个 shape 与预期不同，先停下来问“head 维在何时拆开”“GQA/MQA 是否改变 KV head 数”，
不要把 `view` 或 `transpose` 当成无意义样板。

## 13.4 再看 op 入口：它决定了你在测哪一种语义

`fla/ops/*/__init__.py` 通常是理解公开函数参数、默认算法、可选 backend 和返回值的最好
位置。阅读时用一张表记录：

| 项目 | 需要写下的事实 |
| --- | --- |
| q/k/v | layout、dtype、head 数、是否 variable-length |
| gate | `alpha` 是 scalar 还是 `[... ,Dk]`；`beta` 的范围/参数化 |
| causal/state | 是否只支持 causal；initial/final state 如何传递 |
| forward | 返回 output 之外是否返回 final state |
| backward | autograd 是否走专用 kernel；需要保存哪些中间量 |
| algorithm | naive、recurrent、chunk、backend 的选择条件 |

不要把函数名中的 `fused` 自动翻译成“永远最快”。它只说明若干步骤被组合；真正速度仍受
shape、dtype、GPU、版本和输入布局影响。

## 13.5 `naive.py` 是解释器，不是性能基线

先读 `naive.py`，原因不是它一定是最短代码，而是它通常最接近逐时间步的数学定义。你的
阅读顺序应为：

1. 对照第 11/12 章，把 state 的转置统一；
2. 找出“先 decay、后 prediction、后 residual write”的顺序；
3. 找出 output 是更新前 state 还是更新后 state 的读取；这是 causal off-by-one 的常见源头；
4. 用独立的二十行 reference 再写一遍，不能只把 `naive.py` 当裁判；
5. 用小随机张量比较两份实现的 output **和每一步 final state**。

FLA 的 naive 实现与自己的 reference 结果一致，才是“公式层正确”的第一条证据。它不会
证明 chunk kernel、梯度或模型质量正确。

## 13.6 recurrent 与 chunk 文件各该问什么

### `fused_recurrent.py`：为一步一步 state 而读

这种路径最适合建立 decode 心智模型：每来一个 token，读取当前 state、产生 output、写回
新 state。重点看：state 是否显式传入/返回、哪个 dtype 做 accumulation、每步是否需要
写回全局显存。

它**常**是 decode 或短序列 correctness 对照的重要形式，但不要只凭文件名断言它就是
线上 serving 的默认实现；需在当前 API/benchmark 中确认 dispatch。

### `chunk.py` 与 `chunk_fwd.py`：为训练/长 prefill 而读

它们的目标是把一段序列拆成块：

```text
block 0: state_0 -> 并行算 token 0..C-1 -> state_1
block 1: state_1 -> 并行算 token C..2C-1 -> state_2
...
```

读时不要试图一次看懂所有 `tl.program_id`。先标记：

- 哪段是块间 boundary state；
- 哪段是块内 Q/K/V 矩阵乘；
- `alpha/beta` 在哪里被累计或重标度；
- WY/triangular solve 的输入输出是什么；
- forward 保存哪些量供 `chunk_bwd.py` 使用。

理解这五件事后，再查看 tile 尺寸、warp 数、layout、mask 和 pointer arithmetic。此时
Triton 才从“乱码”变成一个可验证的数据流。

## 13.7 测试、benchmark 和模型测试在不同层回答不同问题

当前 FLA 源码树中，值得先定位的测试/工具包括：

```text
tests/ops/test_kda.py
tests/ops/test_gla.py
tests/ops/test_delta.py
tests/ops/test_dplr_delta.py
tests/layers/test_gated_deltanet.py
tests/models/test_modeling_kda.py
tests/models/test_modeling_gated_deltanet.py

benchmarks/ops/registry.py
benchmarks/ops/verify.py
benchmarks/ops/run.py
benchmarks/benchmark_generation.py
benchmarks/benchmark_training_throughput.py
```

| 层 | 它回答的问题 | 不足以回答的问题 |
| --- | --- | --- |
| 你的 independent reference | 我是否写对公式？ | FLA kernel 是否正确/快 |
| `tests/ops` | kernel 与指定 reference/梯度是否一致？ | 模型质量、你的目标 GPU 表现 |
| `tests/layers` | projection/cache/layer 集成是否工作？ | 论文的大规模训练结论 |
| `tests/models` | 模型包装/API 是否可用？ | sparse/linear 算法本身的优越性 |
| `benchmarks/ops` | 指定算子、shape、dtype 的时间/吞吐 | 端到端 LLM serving |
| generation/training benchmark | 给定脚本的端到端指标 | 所有模型、所有 batch 的通用结论 |

因此实验顺序必须是 **reference -> selected test -> small op benchmark -> 需要时才是 layer/model**。
跳过中间层直接跑一个大例子，失败时几乎无法定位问题。

## 13.8 第一次读 FLA 的 90 分钟路线

不必一天读完整个仓库。一次只走一条细路径：**KDA 或 GDN 二选一**。建议按下面的停靠点
推进；每完成一点就留下一个可检查产物。

1. **10 分钟：定位。** 打开相应 `layers/*.py` 与 `ops/*/__init__.py`，写下公开函数和
   q/k/v/gate/state 的形状；
2. **15 分钟：公式。** 阅读 `naive.py`，将每个 update 对应到第 11/12 章的一行公式；
3. **15 分钟：自己重写。** 在教学代码中用小循环写 independent reference，并用 float64
   随机样本比较；
4. **15 分钟：算法。** 只看 `fused_recurrent.py` 与 `chunk.py` 的入口/注释，画出 state
   如何跨 token 或跨 chunk；
5. **15 分钟：测试。** 阅读相关 `tests/ops` 的参数化 case；挑一个最小 case 运行或在
   无 GPU 环境中先读它的容差逻辑；
6. **20 分钟：性能假设。** 看 `benchmarks/ops` 的注册/运行入口，写下你会固定的 shape、
   dtype、阶段和输出指标，但先不启动长跑。

这 90 分钟的完成条件不是“所有 Triton 行都懂”，而是一张 dataflow card、一份 reference
对照和一个待验证的性能假设。

## 13.9 把论文公式映射到源码的工作表

每次阅读一个新 op，在笔记中复制下面空表并填完。它比保存截图更容易长成调研报告：

| 论文对象 | 数学记号/形状 | FLA 文件/变量 | 我如何验证 |
| --- | --- | --- | --- |
| state | `S:[Dk,Dv]` | `initial_state` / state buffer（以当前源码为准） | 单步手算、跨 chunk 对比 |
| gate | scalar `α` 或 vector `α:[Dk]` | gate projection + op 参数 | scalarization test |
| delta | `βk(v-S^Tk)^T` | naive/reference 中的更新 | 展开式等价 |
| output | `S^Tq` | op 返回 tensor | 和显式循环对比 |
| chunk | chunk length `C` | chunk dispatch / tile 参数 | 不同 `C` 的误差和性能 |
| backward | `dQ,dK,dV,dα,dβ` | `chunk_bwd.py` / autograd | gradcheck / 小尺度 autograd |

“以当前源码为准”是有意写上的。不要因为教程使用变量名 `S`，就假设 repository 必然也
叫 `S`；实现可能转置 state、合并 batch/head、或将它拆成多个 buffer。

## 13.10 关于 TileLang、Triton Ascend 与其他 backend

KDA 目录下可见 `backends/flash_kda.py`、`backends/tilelang/` 和
`backends/triton_ascend/` 等分支。这些是很好的“同一算法可有多个实现后端”的证据，但
不应成为当前第一站：

- TileLang 分支适合在已经理解 KDA dataflow 后，比较 layout 和 Tensor Core 表达；
- Ascend 路径面向不同硬件/软件环境，不能拿来替代 A100/5090 上的 CUDA 结论；
- 某个 backend 存在，不表示在你的环境中已编译、已 dispatch 或性能最好；
- 先理解 `naive`、主 recurrent/chunk 路径，才能分辨不同 backend 改变了什么。

你现在的语言优先级仍是：**PyTorch reference -> Triton dataflow -> 阅读 FLA kernel**。
这不是贬低 TileLang/TVM；它是为了让第一次 kernel 实验有最短的“修改—测试—计时”反馈。

## 13.11 在 107 上运行前的安全检查

当后续 Lab 让你进入集群时，先提交一个短 smoke job，再决定安装/测试范围。运行前必须
记录：

```text
Git commit:          __________
GPU / driver / CUDA: __________
Python / PyTorch:    __________
FLA revision:        __________
operator + mode:     __________
shape + dtype:       __________
reference error:     __________
benchmark command:   __________
```

不要在登录节点 `pip install` 一个不明兼容组合后马上跑 4 小时 benchmark；也不要把 checkpoint、
编译 cache 或 profiler dump 提交 Git。集群工作流和 Slurm 模板由后续 Lab 专门处理，本章
只要求你先知道“版本和证据也是实验的一部分”。

## 13.12 ReplaySSM：同一递推，另一种 decode cache policy

[ReplaySSM](https://tridao.me/blog/2026/replayssm/) 是 Tri Dao 与 Ze-Wei Liou 于 2026-06-15
发布的工程技术博客，配套代码是一个 [vLLM research fork](https://github.com/Johnny-Liou/ReplaySSM)。
它不是新的 sparse mask，也不是新的训练目标；它研究的是：**同一个 SSM/linear-attention
递推，在 GPU decode 时究竟应该把什么写回 HBM。** 因此它适合作为本章的 B* 工程补充案例，
而不是 74 篇学术论文中的第 75 篇。

一个容易误引的细节：博客里的 `[PDF]` 链接指向的是
[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) 技术报告，并不是 ReplaySSM 论文。
ReplaySSM 本身在当前资料中应以博客、代码仓库和上游 RFC/PR 作为工程资料引用。

### 13.12.1 先把问题写成访存图

为避免和不同仓库的转置约定混淆，本节采用本书的 state 方向：
`S:[B,H,D_k,D_v]`，`q,k:[B,H,D_k]`，`v:[B,H,D_v]`。简化的 Mamba-2 递推是

$$
S_t=a_tS_{t-1}+\Delta_t k_tv_t^\top,
\qquad y_t=S_t^\top q_t.
$$

标准 decode 每个 token 大致走这条路径：

```text
load S_(t-1) from HBM -> update S_t -> read y_t -> store S_t to HBM
```

矩阵更新很小，反而是完整 state 的读写占主导，所以“理论上 state 是 `O(1)`”并不等于
“GPU decode 免费”。这正好补上第 5、6 章的性能模型与第 11/12 章递推之间的一条实际连接。

### 13.12.2 checkpoint + ring buffer

ReplaySSM 保留一个较少写回的 checkpoint `S_0`，把最近输入放进小 ring buffer；buffer 未满时，
不把新的完整 state 写回 HBM。对 Mamba-2，buffer 记录每步的 `(k,v,Δ)` 及衰减所需量。把
递推展开后，最近窗口内的输出可以写成：

$$
y_t=\bar a_t S_0^\top q_t+
\sum_{j\le t}w_{j,t}\,v_j(k_j^\top q_t).
$$

第二项使用结合律 `((k_jv_j^\top)^\top q_t)=v_j(k_j^\top q_t)`：它直接得到 output，
不必先物化每个 `D_k\times D_v` state。buffer 达到容量时才 flush，把窗口折叠进 checkpoint
并清空/推进 ring cursor。

```text
checkpoint S0 + recent input ring buffer
          |                         |
          | output-only route       | buffer full
          v                         v
   K^T Q -> weighted V       state-and-output flush -> new S0
```

buffer 长度不是越大越好：太短会频繁 flush，太长会增加每步 buffer 读取和重算。它是一个
需要在目标 GPU、batch、dtype 上测出的系统参数，不是论文公式可以单独决定的常数。

### 13.12.3 GDN 的关键陷阱与 speculative decode

GDN 不能简单照搬“缓存原始 `v`”的说法。衰减之后的状态先读取当前 key，再形成修正量

$$
u_t=\beta_t\bigl(v_t-\bar S_t^\top k_t\bigr),
\qquad S_t=\bar S_t+k_tu_t^\top.
$$

`u_t` 已经依赖旧 state；ReplaySSM 因而缓存 `(u,k,g)`（而不是只缓存原始 `v`），这样重放
时不会重新引入逐 token 的 state 依赖。speculative decoding 中，拒绝 draft 只需移动 ring
buffer 指针；GDN 的 draft 输出则用带 causal mask 的 chunkwise triangular solve/矩阵运算，
而不是为每个 draft 保存一个完整 state snapshot。

这不是把 GDN 变成 sparse attention。它改变的是 **cache policy、执行顺序和 kernel dataflow**，
模型的递推语义保持不变（允许浮点舍入差异）。

### 13.12.4 从代码到证据

仓库 README 列出了值得按层阅读的 Triton 文件：

| 目标 | 先读的文件 | 先回答的问题 |
| --- | --- | --- |
| Mamba-2 普通 decode | `selective_state_update_replayssm_output_only.py` | 怎样从 checkpoint/buffer 直接得到 output？ |
| Mamba-2 flush | `selective_state_update_replayssm_state_and_output.py` | 何时物化 state，写回多少？ |
| Mamba-2 speculative | `selective_state_update_replayssm_spec.py` | commit/rollback cursor 怎样保持因果性？ |
| GDN | `fused_recurrent_replayssm.py`、`gdn_replayssm_spec_decode.py` | 为什么要缓存 `u`，哪里解除串行依赖？ |

推荐阅读博客的 2.1、3、5.1、5.3、5.4 和 Appendix A.1/A.2，先跳过完整 vLLM 集成与大模型
图表。代码仓库的 benchmark 需要 H100/B300、CUDA Graph 和 4B--550B 模型；博客中最高约
`1.48x` 标准 decode 端到端、`1.87--1.96x` speculative decode 是作者在特定硬件/版本上的
结果，不是 107 RTX 5090 的预期数字。上游 [RFC #47572](https://github.com/vllm-project/vllm/issues/47572)
仍是 Open，[PR #47576](https://github.com/vllm-project/vllm/pull/47576) 仍是 Draft；因此本案例
的重点是读懂因果链和做小算子验证，不是宣称已经生产化。

### 13.12.5 你的最小复现阶梯

不要从 vLLM 或大模型权重开始。按下面顺序留下证据：

1. **L1 reference：** `B=H=1,T<=32,float64`，逐步 recurrent Mamba-2 与 output-only 重结合
   的 output、每步 state 对齐；
2. **L1 cache：** 实现 buffer length `4/8/16/32` 和 flush，测试 ring cursor 的 commit/rollback；
3. **L2 kernel：** 只在 GPU 可用时写一个小 Triton operator，比较 recurrent 与 replay 的 CUDA
   Event 中位数；
4. **L3 serving（可选）：** 读 vLLM 接入点，再决定是否有合适模型/节点。不要把下载权重或复现
   B300/NVFP4 表格设为入门通过条件。

计时和解释沿用 [Lab 4B：从计时到 profiler 证据](../labs/04b-profiling.md)：CUDA Event
负责稳态延迟，PyTorch Profiler/`nsys` 负责调用与时间线，`ncu`（若权限和架构支持）才适合
查看真实 memory throughput。Profiler 的 kernel 时间不能直接替代主 benchmark，也不能仅凭
PyTorch trace 声称“写了多少 HBM”；后者要靠 kernel 数据流、字节模型或硬件 counter 交叉验证。

## 常见误区

**误区 1：从 kernel 第一行读到最后一行。**
这样很容易被 pointer arithmetic 淹没。先从 layer/entry/naive 建立变量词典。

**误区 2：测试全绿就等于论文复现成功。**
测试通常验证指定算子/形状的数值或梯度；它不证明训练数据、模型容量、长上下文质量或
作者 benchmark 数字。

**误区 3：`naive.py` 很慢，所以可以不看。**
恰恰相反，它是最接近数学定义的解释器。性能 baseline 应是 dense SDPA/FlashAttention 或
同类优化 op，而不是 Python loop。

**误区 4：一个 `pytest` 命令失败就是 kernel 有 bug。**
先检查环境、GPU 架构、dtype、测试选择、编译器、repo revision 和 reference 容差。把完整
traceback、命令、commit 写进实验记录。

**误区 5：看到 `backends` 就要全部安装。**
先只走与你实际 GPU 和当前目标对应的主路径；额外后端会扩大变量空间。

## 练习

### 练习 13.1：画 KDA dataflow card

不查 kernel 内部，把 `fla/layers/kda.py` 与 `fla/ops/kda/__init__.py` 的输入输出画成一张图。
图中必须有 batch/head/sequence 维、`alpha` 的粒度、initial/final state 和 output gate。

### 练习 13.2：双 reference

写自己的 GDN 或 KDA reference，与 FLA 的 naive 路径在 `T=4,Dk=3,Dv=2` 上比较。故意把
`Diag(alpha)` 放到 rank-1 transition 的另一侧一次，观察从第几步开始不一致，并解释原因。

### 练习 13.3：不运行也能审查 benchmark

打开 `benchmarks/ops/registry.py`、`verify.py`、`run.py`，列出一次 benchmark 至少应固定的
五个参数。再写一条你认为最可能导致不公平对比的变量。

## 通过条件

你已经可以进入 sparse 主线，前提是能够：

- 从 KDA/GDN layer 找到对应 op 与 naive 文件；
- 用独立 reference 解释一个 update，而非只说“FLA 已实现”；
- 分辨 recurrent、chunk、forward、backward、prefill、decode；
- 说清选中测试和 benchmark 分别验证什么；
- 在未提交 GPU 作业前，写完整一张实验运行卡。

本章的路径依据 FLA 当前公开源码树以及 Kimi 报告指向的 `fla/ops/kda`。下一部分会把视野
转向 sparse attention：这次历史不被压进一个固定 state，而是决定哪些 query-key 配对根本
不计算。
