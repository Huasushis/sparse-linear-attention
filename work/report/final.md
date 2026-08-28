# Sparse 与 Linear Attention：算法、GPU Kernel 与受控复现

> 报告版本：2026-08-28  
> 研究范围：算法与算子级复现为主，模型训练与 serving 作为论文证据和系统边界讨论  
> 引用格式：正文中的 `[@key]` 对应 [`references/attention.bib`](../../references/attention.bib)

## 摘要

标准 softmax attention 需要为每个 query 与所有可见 key 建立配对，计算量随序列长度
$T$ 以 $O(T^2D)$ 增长。降低长上下文成本主要形成了三条路线：第一，FlashAttention
保持 dense softmax 的数学定义不变，通过 tiling、online softmax 与重计算减少 HBM
读写；第二，linear attention 把历史压入固定形状的 recurrent state，并用 parallel、scan
或 chunkwise 算法恢复训练并行；第三，sparse attention 只计算结构规则或动态选择后的
query-key 子集，并围绕 selector、block layout、在线归一化和 serving cache 设计 kernel。

本报告以 FlashAttention、kernelized linear attention、GLA、DeltaNet、Gated DeltaNet
（GDN）、Kimi Delta Attention（KDA）、Mamba-2/SSD，以及 MInference、Native Sparse
Attention（NSA）、MoBA、SpargeAttention、HiLS-Attention 为重点，说明算法更新式和 GPU
执行路径。复现部分在 107 Slurm 集群上固定代码提交，完成 dense/linear reference、FLA
GDN/KDA 退化关系、FLA chunk/recurrent kernel、NSA block-selected kernel，以及
prefill/decode benchmark。报告严格区分三类证据：本报告实测的 operator 数据、代码阅读
得到的实现事实、论文报告的模型质量或 serving 结果。我们没有重训大模型，因此不把算子
正确性或速度外推为模型质量结论。

## 1. 研究问题与证据边界

本调研回答五个问题：

1. dense、linear、sparse attention 分别改变了什么；
2. recurrent、parallel、chunkwise 三种表示为什么能对应同一 linear operator；
3. chunk、tile、scan、online softmax、WY/DPLR 怎样把算法映射到 GPU；
4. 理论复杂度降低为什么不保证 wall-clock 加速；
5. 当前硬件上能复现哪些结果，哪些结论仍只能引用论文。

### 1.1 三个层级不能混写

| 层级 | 核心问题 | 本报告的主要证据 |
| --- | --- | --- |
| 算法 | 数学定义、状态、mask、复杂度与信息损失 | 公式推导、独立 reference、论文 |
| 算子/kernel | 相同输入契约下是否正确、快、省显存 | pytest、CUDA Event、PyTorch profiler、FLA 源码 |
| 模型/系统 | 训练后质量、TTFT、TPOT、吞吐、cache 与调度 | 论文结果；本报告不声称完成大模型重训 |

“固定 token”“固定 FLOPs”“固定 GPU 时间”回答的是不同问题。固定 token 更接近数据量
相同时的质量；固定 FLOPs 比较理论计算预算；固定 GPU 时间把 kernel 成熟度、利用率和
通信也纳入实际成本。训练方法最终应同时报告 validation loss、time-to-quality、总 GPU
时间和显存，而不是只报告 step time。

### 1.2 本报告的复现等级

- **L0 定义：** 小张量公式和 mask；
- **L1 reference：** PyTorch/FP32 或 FP64 正确性；
- **L2 operator：** 固定 GPU、shape、dtype 的优化 kernel；
- **L3 layer/model：** 投影、卷积、norm、cache 接口；
- **L4 training/serving：** 模型训练质量或端到端请求指标。

本报告完成 L0--L2，并阅读部分 L3 代码；Kimi 48B、NSA 预训练、MInference/HiLS 7B
serving 等 L4 结果仅作为论文证据。

## 2. Dense attention 与 exact kernel 基线

### 2.1 数学定义

对一个 head，设 $Q,K\in\mathbb R^{T\times D}$、$V\in\mathbb R^{T\times D_v}$：

$$
S=\frac{QK^\top}{\sqrt D},\qquad
P=\operatorname{softmax}(S+M),\qquad
O=PV,
$$

其中 causal mask $M_{ij}=0$（$j\le i$），否则为 $-\infty$。逻辑 score/probability
矩阵为 $T\times T$，所以主计算量为 $O(T^2D)$。训练、prefill 和 decode 的形态不同：

- 训练/prefill 同时处理大量 query，适合大块 GEMM；
- 单步 decode 通常 $T_q=1$，历史 $K/V$ 来自 KV cache，常受显存带宽与调度限制；
- KV cache 随上下文长度增长，而历史 query 不需要保存。

### 2.2 FlashAttention：不改 attention，只改数据流

FlashAttention 是 exact dense attention，而非 sparse 方法。朴素路径将 $S$、$P$ 写入
HBM；FlashAttention 将 $Q$ 切为行 tile，将 $K/V$ 切为列 tile，让临时 score 在片上产生、
消费后丢弃，只维护每个 query 行的 running maximum、normalizer 和未归一化输出
[@dao2022flashattention]。

对新 score block $s_b$，online softmax 维护

$$
m=\max s,\quad l=\sum_j e^{s_j-m},\quad
u=\sum_j e^{s_j-m}v_j.
$$

若 $m'=\max(m,\max s_b)$，则

$$
l'=e^{m-m'}l+\sum_{j\in b}e^{s_j-m'},
$$

$$
u'=e^{m-m'}u+\sum_{j\in b}e^{s_j-m'}v_j,qquad o=u'/l'.
$$

旧状态必须乘 $e^{m-m'}$，否则两个 block 使用不同指数基准。causal kernel 还可以跳过
对角线上方的整块，对角块才做逐元素 mask。Backward 不保存完整 $P$，而是在 tile 内重算
score/probability，再累积 $dQ,dK,dV$；这增加部分 FLOPs，却减少 activation 与 HBM 流量。

FlashAttention-2 的重点是减少非 matmul FLOPs、增加 sequence 维并行、改进 warp 间工作
划分，而不是发明另一种 attention [@dao2024flashattention2]。FA-3/4 继续利用 Hopper/
Blackwell 的异步 pipeline 与低精度；这些硬件特化数字不能直接当作 A100/5090 结果
[@shah2024flashattention3; @zadouri2026flashattention4]。

### 2.3 Dense kernel 的 tile 生命周期

```text
HBM: Q_i ---------------------------┐
HBM: K_j,V_j -> SRAM/register tile  |
                    |               |
             Q_i K_j^T              |
                    | mask          |
                    v               |
          online (m,l,u) update ----┘  对所有可见 KV tile 循环
                    |
                    v
                 write O_i
```

关键不是“完全不读 HBM”，而是不 materialize 完整 $T\times T$ 中间量，并在片上复用 tile。
因此 sparse/linear 的最终 baseline 应是成熟 SDPA/FlashAttention，而不是三行显式 PyTorch。

## 3. Linear attention：从配对矩阵到固定状态

### 3.1 Kernel feature 推导

若相似度核可写或近似为

$$
\kappa(q,k)\approx\phi(q)^\top\phi(k),
$$

则 causal attention 可以交换求和次序。定义

$$
S_t=\sum_{i\le t}\phi(k_i)v_i^\top
\in\mathbb R^{D_\phi\times D_v},\qquad
z_t=\sum_{i\le t}\phi(k_i)\in\mathbb R^{D_\phi},
$$

$$
o_t=\frac{\phi(q_t)^\top S_t}
          {\phi(q_t)^\top z_t+\varepsilon}.
$$

状态更新为

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t).
$$

当 $D_\phi,D_v$ 固定时，关于序列长度的工作为 $O(TD_\phi D_v)$，decode state 不随
$T$ 增长。这是“Transformers are RNNs”的基本形式 [@katharopoulos2020transformers]。
Performer 用随机正特征近似 softmax kernel [@choromanski2021rethinking]；cosFormer 等则
直接采用不同的特征或归一化 [@hua2022transformer]。有限 state 可能发生碰撞、遗忘和
表达损失，所以“线性复杂度”不等于“与 dense softmax 完全等价”。

### 3.2 Recurrent、parallel 与 chunkwise

三种形式描述的是同一个 operator 的不同执行计划：

| 形式 | 数据依赖 | 适合阶段 | 主要 kernel 特征 |
| --- | --- | --- | --- |
| recurrent | 每 token 依赖前一 state | decode、正确性 oracle | 固定 state；并行量小、频繁读写 state |
| parallel/scan | 展开全部位置 | 训练、短 prefill | 并行强；可能产生大中间量 |
| chunkwise | chunk 间递推、chunk 内并行 | 训练、长 prefill | 大部分工作转 GEMM；保留少量 boundary state |

对最简单的 additive state，令一个 chunk 的 $Q,K\in\mathbb R^{C\times D_k}$、
$V\in\mathbb R^{C\times D_v}$，则

$$
S_{c+1}=S_c+K_c^\top V_c,
$$

$$
O_c=\underbrace{Q_cS_c}_{\text{inter-chunk}}
 +\underbrace{\bigl((Q_cK_c^\top)\odot M_C\bigr)V_c}_{\text{intra-chunk}}.
$$

串行边界从每 token 一次降为每 chunk 一次；块内使用 $C\times D$ 与 $D\times C$ GEMM。
GLA 论文指出 $C$ 取 Tensor Core 友好的倍数（如 16 的倍数）有利于利用矩阵乘，并通过
tiling 在片上复用张量块，减少 HBM 往返 [@yang2024gated]。FLA 固定提交中 GDN 支持
chunk size 16/32/64，KDA 支持 32/64，默认均为 64。

### 3.3 GLA：主动遗忘

普通累加 state 不会主动忘记。GLA 为 key channel 加入数据依赖 gate：

$$
S_t=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}+k_tv_t^\top,
\qquad \boldsymbol\alpha_t\in(0,1)^{D_k}.
$$

不同通道可以有不同记忆寿命。它仍是 additive write：相同 key 的新旧 value 可能叠加。
GLA 的研究贡献同时包括模型机制与 I/O-aware chunkwise training，二者需要模型消融和
operator benchmark 两套证据 [@yang2024gated]。

### 3.4 Delta rule：沿当前 key 方向擦除再写入

令 state 方向为 $S_t\in\mathbb R^{D_k\times D_v}$，旧预测为
$\hat v_t=S_{t-1}^\top k_t$。DeltaNet 更新为

$$
S_t=S_{t-1}+\beta_tk_t(v_t-\hat v_t)^\top.
$$

它也可由在线最小二乘得到。令

$$
L_t(S)=\frac12\|S^\top k_t-v_t\|_2^2,
$$

则

$$
\nabla_SL_t=k_t(S^\top k_t-v_t)^\top,
$$

一次步长 $\beta_t$ 的梯度下降即为上式。展开得到

$$
S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top.
$$

$I-\beta kk^\top$ 是 rank-1 修正的 generalized Householder transition；实现不应每步
显式创建 $D_k\times D_k$ identity。DeltaNet 使用 compact WY 表示压缩一串 rank-1
transition，使 chunk 内 pseudo-key/pseudo-value 与边界 state 的计算转为 GEMM，并避免为
每个 token materialize 矩阵 state [@yang2024delta]。

### 3.5 GDN：gate 与 delta 的组合

GDN 先以每 head scalar gate 衰减，再执行 delta write：

$$
\bar S_t=\alpha_tS_{t-1},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

$\alpha$ 决定旧状态寿命，$\beta$ 决定当前 key 方向纠正幅度，两者不是同一个 gate。论文
在合成 recall、语言模型和 LongBench 中提供模型证据，并报告 GDN 与 DeltaNet 训练吞吐
接近、因 transition 更强而略慢于 Mamba-2；这些是单 H100、给定模型/shape 的论文结果，
不是本报告 GPU 的预定答案 [@yang2025gated]。

### 3.6 Mamba-2 / SSD：state-space 与 attention 的块对偶

SSD 将一类 scalar-identity state transition 的 SSM 写成结构化半可分（semiseparable）
矩阵，也可以反向把 attention 看作结构化矩阵乘。其工程意义是：同一 operator 可在
recurrent state、卷积/scan、block matrix 三种视角中选择执行计划；限制 transition 结构
换取块算法与 Tensor Core 友好性 [@dao2024transformers]。SSD/Mamba-2 与 GDN/KDA 都是
固定矩阵 state 路线，但更新规则和表达能力不应混为一谈。

### 3.7 KDA 与 Kimi Linear：GDN 的细粒度扩展

KDA 把 GDN 的 scalar decay 换为 key-channel diagonal decay：

$$
\bar S_t=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1},
$$

$$
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

其 transition 为

$$
A_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\boldsymbol\alpha_t)
=\operatorname{Diag}(\boldsymbol\alpha_t)
-\beta_tk_t\bigl(k_t^\top\operatorname{Diag}(\boldsymbol\alpha_t)\bigr).
$$

这是受约束的 diagonal-plus-rank-1（DPLR）结构。当
$\boldsymbol\alpha_t=\alpha_t\mathbf1$ 时，KDA 精确退化为 GDN；本报告用 FLA 的 naive/
fused recurrent kernel 验证该 scalarization test。KDA 绑定低秩两侧的变量结构，论文称其
特化 chunkwise 算法比一般 DPLR 少做第二级 chunk 矩阵计算与若干 GEMM [@kimi2025linear]。

Kimi Linear 并不等于 KDA：报告模型以 3:1 交错 KDA 与全局 MLA，并加入 MoE backbone、
位置处理和完整训练配方。论文所述“最多约 75% KV cache 减少”来自这一层比例；周期性
MLA 层仍维护 KV cache。因而 operator 复现不能冒充 48B 模型质量复现。

### 3.8 Linear kernel 的 tile/chunk 数据流

```text
token/chunk 输入 Q,K,V,gate,beta
        |
        +--> gate prefix / triangular solve / WY auxiliary
        |
boundary state S_c ------------------------------┐
        |                                        |
        +--> inter-chunk: Q_c @ S_c              |
        +--> intra-chunk: tiled QK^T -> mask/WY  |
        |                         -> pseudo-V GEMM|
        +--> tiled state update K^T @ U ----------┘
                               |
                               v
                         boundary S_(c+1)
```

实现层的主要选择包括：是否把 chunk boundary state materialize 到 HBM、forward/backward
是否重算辅助量、state 使用 `[K,V]` 还是 `[V,K]` layout、tile 的 `BK/BV` 如何覆盖 head
dimension、chunk 是否 16/32/64、variable-length 的 chunk index 如何生成。FLA 的
`chunk.py` 是算法/dispatch，`chunk_fwd.py`/`chunk_bwd.py` 负责主路径，`wy_fast.py`、
`gate.py` 与 `chunk_intra.py` 处理辅助量；`fused_recurrent.py` 则面向逐步 state 路径。

### 3.9 本次读取和复现的算子接口

| 算子 | 主要输入布局 | gate/state | 典型用途 | 本次证据 |
| --- | --- | --- | --- | --- |
| PyTorch SDPA | `[B,H,T,D]` | KV cache 由上层管理 | dense prefill/decode | correctness + sweep + profiler |
| FLA `chunk_gated_delta_rule` | `q/k:[B,T,H,K]`, `v:[B,T,HV,V]` | `g,beta:[B,T,HV]`；state `[B,HV,K,V]` | 训练/长 prefill | forward benchmark |
| FLA `fused_recurrent_gated_delta_rule` | 同上 | scalar log-decay；initial/final state | decode/顺序 reference | correctness + benchmark |
| FLA `chunk_kda` | 同上 | `g:[B,T,HV,K]` channel log-decay | KDA 训练/长 prefill | scalarized correctness + benchmark |
| FLA `fused_recurrent_kda` | 同上 | channel decay + fixed matrix state | KDA decode/reference | scalarized correctness + benchmark |
| FLA `parallel_nsa` | `q:[B,TQ,HQ,K]`, `k/v:[B,T,Hkv,*]` | `block_indices:[B,TQ,Hkv,S]` | block-selected sparse | naive 对齐 + kernel/selector benchmark |
| FLA `parallel_moba` | `[B,T,H,D]` + packed `cu_seqlens` | chunk size、top-k block | block router + two-stream attention | 源码审计；依赖不足未计时 |

FLA 还允许 grouped-value attention、variable-length `cu_seqlens`、不同 state layout 和 fused
gate activation。本次固定最小契约，避免把 dispatch/模型外围差异混入核心算法比较。

## 4. Sparse attention：选择哪些 query-key 配对

### 4.1 统一定义

令 $\mathcal A(i)\subseteq\{0,\ldots,i\}$ 为 query $i$ 可见的 key 集合：

$$
o_i=\sum_{j\in\mathcal A(i)}
\frac{e^{q_i^\top k_j/\sqrt D}}
     {\sum_{r\in\mathcal A(i)}e^{q_i^\top k_r/\sqrt D}}v_j.
$$

计算量约为 $O(D\sum_i|\mathcal A(i)|)$。但 density 不是速度：细粒度不规则 gather、每行
不同工作量、selector/index、softmax reduction、launch 与同步可能吞掉省下的 FLOPs。
GPU 通常更喜欢 block sparse，因为一个 program 可以读取连续 K/V block，执行规整 MMA，
再用 online softmax 合并 selected tiles。

### 4.2 方法分类

| 类别 | 代表方法 | 选择规则 | 主要代价 |
| --- | --- | --- | --- |
| 固定结构 | Sparse Transformer、Longformer、BigBird、LongNet | window/global/strided/random/dilated | 规则快，但结构先验可能漏掉内容相关远程依赖 |
| hashing/routing | Reformer、Routing Transformer、Sinkhorn | LSH、聚类或可学习排序 | router、排序、负载均衡 |
| prefill 动态模式 | MInference、FlexPrefill、XAttention | pattern/head + 动态 block index | selector 与不规则 block 分布 |
| 可训练稀疏架构 | NSA、MoBA、SeerAttention、HiLS | compression/router/top-k/window | 训练、backward 与选择质量 |
| training-free 近似 | SparQ、QUEST、Loki、SpargeAttention | proxy、量化、低秩 key、在线过滤 | proxy recall 与 selector 带宽 |
| KV/cache policy | StreamingLLM、H2O、DuoAttention、InfiniGen | sink/heavy hitter/head 分类/eviction | decode 质量、cache page 与调度 |

完整 74 篇分类见附录 A 和 `study/PAPER_MAP.md`。

### 4.3 MInference：按 head 选择预填充稀疏模式

MInference 针对长上下文 **prefill**，离线为 attention head 分配 A-shape、Vertical-Slash
或 Block-Sparse 模式，运行时按输入建立具体 index，再调用相应 Triton/FlashAttention 风格
kernel [@jiang2024minference]：

- A-shape：初始 token + local window，结构较稳定；
- Vertical-Slash：少量动态垂直列与斜线；
- Block-Sparse：用 64×64 等块表达更分散的内容相关区域。

Vertical-Slash 可用少量尾部 query 与 K 的乘积估计重要列/斜线；Block-Sparse 可对 Q/K 做
64-token block mean pooling，再计算 coarse attention 选块。论文目标是 1M-token prefill，
不能把其结果直接外推为单 token decode 加速。

### 4.4 NSA：compression、selection、sliding 三分支

NSA 是 natively trainable、hardware-aligned sparse architecture [@yuan2025native]：

1. **Compression：** 将连续 K/V block 聚合成 coarse token，保留全局概览；
2. **Selection：** 复用 compression attention score 估计 block importance，选择少量原始
   K/V block 做 fine-grained attention；
3. **Sliding window：** 单独保留近期局部 token，避免其他分支被局部模式“走捷径”；
4. **Gated merge：** 三分支输出由学习 gate 聚合，并可使用独立 K/V 投影降低梯度干扰。

FLA 的 NSA selected kernel 使用 `[B,TQ,H,S]` block indices；当前固定提交默认 block size
64，论文常用每 query 16 个 selected blocks。kernel 以 `BK/BV` power-of-two tile 覆盖
head dimension，Q tile 留在片上，遍历 selected K/V blocks 并做 sparse online softmax。
Backward 将 query-to-block 选择倒排成 CSR，让 dK/dV kernel 以被选择 block 为单位 gather
query，避免扫描所有 query。

本报告分别测量固定 block indices（selector 排除）和 compression/top-k + selected 路径，
因此可以直接观察 selector/压缩路径的增量开销；未运行 sliding branch，因为当前 FLA
实现该分支依赖额外 `flash-attn` 包。

### 4.5 MoBA：block router + 两路 FlashAttention

MoBA 把序列切成固定 block，以 block mean key 作为 router representative；每个 query/head
选择 top-k block，其中 local block 始终保留 [@lu2025moba]。当前 FLA 实现的流程是：

```text
block mean K -> query-to-block score -> causal top-k-1 nonlocal blocks
      |                                      |
local causal FlashAttention          gathered varlen FlashAttention
      |                                      |
      +-------- online LSE softmax merge -----+
```

两个 stream 分别返回 output 与 log-sum-exp，再用稳定 LSE 合并，等价于在二者 key 并集上
做一次 softmax。优点是 selected block 内保持连续 FlashAttention；代价是 router、gather、
变长序列打包和负载不均。本环境没有安装 MoBA 所需的额外 `flash-attn` Python 包，因此只做
源码路径复核，不报告本机 MoBA latency。

### 4.6 SpargeAttention：两阶段在线过滤

SpargeAttention 是 training-free sparse/quantized inference kernel [@zhang2025spargeattention]：

1. 将每个 Q/K block 压缩成 representative，按 block 内 token 相似度决定是否相信压缩；
2. 以 coarse score 的 softmax CDF 生成 block mask，跳过对应 $Q_iK_j^\top$ 与 $P_{ij}V_j$；
3. 对已计算 score tile，在 online softmax 中比较 local/global max，若该 warp 的概率贡献
   足够小，再跳过 $\tilde P_{ij}V_j$；
4. 与 SageAttention 的低比特 QK 路径结合。

第二阶段利用已经计算的 online-softmax 状态，新增判断开销小；但“无精度损失”是论文在
特定模型/任务/阈值下的经验结论，不是数学上的 exact dense attention。

### 4.7 HiLS-Attention：分层 chunk mass 与 query packing

HiLS 是 2026 年的新工作，使用层级 chunk summary/质量分配选择远程内容，并将相邻 query
的 selected chunk 并集打包，使一次 K/V load 服务多个 query [@hu2026hils]。这种
one-load-multiple-compute 可以增加 Tensor Core 左矩阵规模与 K/V 复用，但并集会引入额外
无效块；必须同时报告目标 top-k、并集后的实际 block 数、selector、attention kernel 和
端到端 latency。其公开代码依赖 TileLang/VeOmni 与独立 serving fork，且发布较新，本报告
仅完成方法定位，不声称复现 4M/infinite-context 模型结果。

### 4.8 Sparse tile 数据流

```text
Q block r
  |
  +--> selector / fixed rule --> key block ids [j1,j2,...]
                                  |
                                  v
                 gather/load contiguous K_j,V_j tiles
                                  |
                         Q_r K_j^T + mask
                                  |
                     sparse online (m,l,u)
                                  |
                             write O_r
```

真实 kernel 还要解决：每行 block 数不同造成的 load imbalance；index/page-table 的带宽；
selected K/V 是否 coalesced；多个 query 是否共享 block；backward 如何倒排选择；动态
selector 是否 materialize dense proxy。block 更粗会牺牲理想 sparsity，却常能换取更高
Tensor Core 利用率。

## 5. Serving：prefill、decode 与 cache 不是同一问题

| 阶段 | 形态 | 常见瓶颈 | 主要指标 |
| --- | --- | --- | --- |
| prefill | 大量 query 一次处理 prompt | $T^2$ work、HBM IO、selector | TTFT、prefill latency、token/s |
| decode | 每请求每步一个新 query | KV/state 带宽、batch、page/scheduler | TPOT/ITL、aggregate token/s、p95 |

StreamingLLM 保留 attention sink 与 recent window [@xiao2024streamingllm]；H2O 保留
heavy hitters [@zhang2023h2o]；QUEST/SparQ/Loki 用 query-aware proxy 或低秩 key 缩小 KV
读取 [@tang2024quest; @ribar2024sparq; @singhania2024loki]。这些更接近 decode/cache
policy，而 MInference 更接近 prefill。FlashInfer 进一步把 paged/composable KV layout、
JIT attention template、load-balanced scheduling 与 CUDA Graph 约束放进 serving engine
[@ye2025flashinfer]；LServe 则尝试统一 sparse prefill/decode [@yang2025lserve]。

Linear recurrent decode 用固定 $D_k\times D_v$ state 替代随 $T$ 增长的 KV cache，但每步
读写完整矩阵 state 仍可能受 HBM 限制。ReplaySSM 的工程思路是保存较少写回的 checkpoint
与近期输入 ring buffer，直接重结合 output，buffer 满时才 flush state；它不改变模型数学，
而是改变 cache policy 和执行顺序。本报告只引用其工程资料，不把它计入 74 篇论文。

任何 serving speedup 都应说明是否包含 tokenizer、模型其他层、selector、cache page、
scheduler、batching 与通信；TTFT 和 TPOT 不能合成一个模糊的“推理速度”。

## 6. 复现设计

### 6.1 环境与版本

| 项目 | 记录 |
| --- | --- |
| 教程仓库 | `study/sparse-linear-attention`，正式实验提交 `4d086aa` |
| FLA 仓库 | `~/sparse_linear/flash-linear-attention` |
| FLA commit | `d1ce07369d581813553f30a750af3b6b5f9af6a9` |
| 作业 | Slurm job `46628`，节点 `anode02`，1 GPU，4 CPU，最长 2 h；stderr 为空 |
| GPU/软件 | RTX 5090 32607 MiB，driver 580.173.02；Python 3.12.13；PyTorch 2.11.0+cu128；CUDA runtime 12.8 |
| Python 环境 | `~/sparse_linear/.envs/sla-tutorial-py312` |
| 原始产物 | 107：`artifacts/final-46628/`（不提交 Git） |
| 小型摘要 | `work/runs/final-reproduction-46628.md` |

GPU、driver、PyTorch/CUDA/Triton 由计算节点日志填写，不能从登录节点猜测。

### 6.2 实验矩阵

- dtype：BF16；
- $B=1,D=64$；dense/教学 linear 使用 $H=4$；NSA 因 FLA kernel 契约使用
  $H_q/H_{kv}=16/1$；
- $T\in\{512,2048,8192\}$；
- warm-up：dense/teaching linear 10，FLA 5；repeats：20；
- 计时：CUDA Event；每项保存全部样本与 p10/p50/p90；
- JIT/autotune 在 warm-up 中完成；输入分配和固定 block-index 构造不计入 kernel-only；
- NSA 另测 compression/top-k + selected 路径以体现 selector/压缩开销；
- prefill/decode 分开；不同语义的 dense/linear/sparse 不宣称输出等价。

### 6.3 正确性阶梯

1. 本地/107 pytest 覆盖 dense、linear 三种形式、GDN/KDA toy、sparse mask、Triton；
2. FLA probe 比较 naive/fused GDN、naive/fused scalarized KDA 的 output/final state；
3. NSA selected kernel 与 naive selected-block oracle 在小 shape 比较；
4. 性能 sweep 只在 correctness gate 后执行。

## 7. 复现结果

原始 JSON 与全部 samples 位于 107 `artifacts/final-46628/`；可提交摘要见
[`work/runs/final-reproduction-46628.md`](../runs/final-reproduction-46628.md)。

### 7.1 Correctness

- tests + dense/Triton grader：`20 passed in 9.75s`；
- FLA FP32 probe 中，naive KDA/GDN scalarization 的 output/state max abs 为
  `1.49e-8/5.96e-8`；fused GDN 与 scalarized fused KDA 为 `2.98e-8/5.96e-8`；
- final suite 的另一组 FP32 scalarization 得到 output/state max abs 均为 `0.0`；
- NSA selected kernel 与 naive BF16 oracle 的 max abs 为 `0.015625`。

这些结果验证指定 operator/shape，不验证模型训练质量。

### 7.2 Dense 与教学 linear 的扩展趋势

共同配置 `B=1,H=4,D=64,BF16`。表中为 p50 ms；括号为 peak allocated delta MiB。

| mode/operator | T=512 | T=2048 | T=8192 |
| --- | ---: | ---: | ---: |
| explicit dense prefill | 0.2538 (14.38) | 0.2818 (108.13) | 4.8040 (1608.13) |
| torch SDPA prefill | 0.04859 (1.27) | 0.04653 (5.09) | 0.3709 (4.13) |
| torch SDPA decode | 0.03922 | 0.03818 | 0.03832 |
| teaching linear parallel prefill | 0.2465 (41.38) | 1.4415 (141.16) | 6.0358 (540.25) |
| teaching linear state decode | 0.1094 | 0.1095 | 0.1068 |

在 `T=8192`，SDPA 相对显式 dense p50 约快 `12.95x`，显式路径峰值临时分配约为
SDPA 的 `390x`。教学 linear reference 没有自动超过 SDPA，fixed-state decode 在这些小
shape 上也约慢 `2.8x`；这正是“复杂度不等于 kernel 性能”的反例。

教学 linear prefill 会显式构造所有 prefix state，是语义 reference，不是优化 kernel；它的
显存和 latency 不能代表 FLA。linear decode 从预计算 state 读取，state 构建不计入单步
读出时间。dense softmax 与 ELU+1 linear attention 数学定义不同，表格只展示执行形态。

### 7.3 FLA GDN/KDA

共同配置 `B=1,H=4,K=V=64,BF16`；表中为 p50 ms。

| method | T=512 | T=2048 | T=8192 |
| --- | ---: | ---: | ---: |
| chunk GDN | 0.6061 | 0.5941 | 0.5922 |
| chunk KDA（scalar gate） | 0.7407 | 0.7169 | 0.7460 |
| fused recurrent GDN | 0.3621 | 1.1937 | 4.5224 |
| fused recurrent KDA（scalar gate） | 0.4044 | 1.3691 | 5.2234 |

`T=512` 时 recurrent GDN 比 chunk 快约 `1.67x`；`T=2048/8192` 时 chunk 分别快约
`2.01x/7.64x`，显示 chunk parallelism 的 crossover。scalarized KDA chunk 比 GDN chunk
慢约 `21%--26%`，recurrent 慢约 `12%--16%`。单 token decode 为 GDN `0.1142 ms`、
KDA `0.1106 ms`；该小差异按噪声处理。

只有 scalarized KDA 与 GDN 具有相同递推语义；KDA 的真实 channel gate 参数空间更大。chunk
结果描述长 prefill/训练 forward，fused recurrent 描述顺序 state path，不应把二者混成
一个 speedup。

### 7.4 FLA NSA block-sparse

共同配置 `B=1,HQ/Hkv=16/1,D=64,BF16`，block size 64，最多选 4 block。表中为 p50 ms。

| T | selected density | full causal SDPA | fixed selected | compression/top-k + selected |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 68.81% | 0.03725 | 0.3026 | 0.5833 |
| 2048 | 20.74% | 0.1218 | 0.3043 | 0.5827 |
| 8192 | 5.41% | 0.8200 | 0.3773 | 0.6427 |

前两点 sparse 仍更慢；`T=8192` 时 fixed selected 相对 full dense latency 约低 `2.17x`，
计入 compression/top-k 后仅约低 `1.28x`。selector-included 路径比给定 indices 慢
`1.70--1.93x`，证明 selector 不能从动态 sparse 主结果中省略。

`nsa_selected_fixed_blocks` 排除了 selector，回答“给定 block ids 的 kernel 有多快”；
`nsa_compression_topk_plus_selected` 包含 compression、top-k 与 selected path，但未包含
sliding branch。full causal SDPA 是性能上下文，不是相同 sparse 语义 baseline。

### 7.5 Profiler 证据

此前 job `40083` 在 RTX 5090 对 `B=1,H=4,T=512,D=64,BF16` 的 PyTorch SDPA prefill
profile 5 步：每步观察到 split-KV 主 kernel 与 combine kernel 两次 CUDA launch；主/合并
kernel 平均约 6.042/2.368 us。PyTorch profiler 的 Activity Buffer Request 是插桩开销，
不能替代 CUDA Event benchmark，也不能据此单独证明 launch-bound。该节点无 `nsys`，旧
`ncu` 不支持 GB202，因此没有 DRAM throughput/occupancy counter；报告不虚构这些指标。

## 8. 结果解释与局限

### 8.1 可以支持的结论

- Flash/SDPA 相对显式 dense reference 的优势来自融合、tiling 和中间量 IO，而不是改变
  softmax 语义；
- KDA channel gate 广播成 scalar 时，operator 退化为 GDN；
- chunk 与 recurrent kernel 的适用阶段不同，性能交叉依赖 $T$、state/head 维度和 GPU；
- block-sparse kernel 的固定-mask 时间与 selector-included 时间必须分开；
- density、FLOPs、GPU utilization 任一单指标都不足以解释真实延迟。

### 8.2 不能支持的结论

- 未重训模型，不能声称 linear/sparse 的 perplexity、LongBench 或 RL 质量已复现；
- 未运行完整 Kimi 48B、NSA 训练、MInference/HiLS serving，不能把论文速度当成本机结果；
- 不同 attention 语义的 latency 不能直接解释为无损 speedup；
- operator 随机张量结果不等于端到端 TTFT/TPOT；
- 单 GPU、单 batch、三个序列长度不能代表多卡训练、并发 serving 或所有 head dimension。

### 8.3 复现中的实现限制

- 教学 linear parallel reference materialize prefix state，长序列显存复杂度高；
- FLA 固定 commit 比远端 main 落后，但版本固定保证本次证据可追踪；
- MoBA 路径依赖未安装的 `flash-attn`，所以只做源码审计；
- NSA 实测只覆盖 selection 与 compression/top-k 子路径，未覆盖 sliding/window、模型 gate
  训练与 backward；
- 5090 节点缺少兼容 Nsight 工具，硬件 counter 仍是证据缺口。

## 9. 方法全景与后续研究问题

### 9.1 74 篇方法图谱（压缩索引）

| 分组 | 方法 |
| --- | --- |
| Linear 基础/架构 | Transformers are RNNs、Performer、Fast Weight Programmers、cosFormer、RetNet、GLA、DeltaNet、Mamba-2/SSD、Based、GDN、Kimi Linear、MDN |
| Linear kernel/并行/量化 | Lightning Attention、LASP、LASP-2、Tiled Flash Linear Attention、Optimized GPU Kernel、GRU sub-8-bit、SSDi8、State Reduction |
| 经典 sparse | Sparse Transformer、Reformer、Longformer、BigBird、ETC、Sparse Sinkhorn、Routing Transformer、Scatterbrain、LongNet、HyperAttention |
| Sparse kernel/serving | SpAtten、Sanger、SALO、Dynamic Sparse FlashAttention、FlexAttention、FlashInfer、InfiniGen、LServe |
| 长上下文 sparse | StreamingLLM、H2O、MInference、QUEST、SparQ、Loki、MagicPIG、RetrievalAttention、SampleAttention、SparseK、HiP、DuoAttention、FlexPrefill、NSA、HiLS、SeerAttention、MoBA、Star Attention、XAttention、UNIQUE |
| 通用/视觉 sparse | SpargeAttention、AdaSplash、SpargeAttention2、AdaSplash-2、Sparsifiner、FPSAttention、DFSAttn、DSV、FG-Attn、VSA、Sparse VideoGen2、db-SP |
| Exact dense baseline | FlashAttention、FlashAttention-2、FlashAttention-3、FlashAttention-4 |

### 9.2 值得继续验证的三个问题

1. **Linear crossover：** 在 A100/5090 上，固定 $B,H,D_k,D_v$ 后，GDN/KDA 的 chunk
   kernel 从什么 $T$ 开始超过成熟 dense SDPA？forward 与 fwd+bwd 的 crossover 是否不同？
2. **Selector 税：** 对相同 block-selected kernel，compression/top-k、index packing 与
   kernel 分别占多少；改变 selected blocks 时何处出现最优点？
3. **Serving state/cache：** 固定模型和请求分布，linear state 或 sparse KV policy 对 TTFT、
   TPOT、p95、显存和质量的 Pareto 前沿是什么，而不是只看 kernel-only p50？

## 10. 结论

Sparse 与 linear attention 不是一个“把 $T^2$ 改成 $T$”的单一技巧，而是算法、kernel 和
系统共同设计问题。FlashAttention 证明 exact dense attention 可以在不降低 FLOPs
复杂度的情况下，通过减少 HBM IO 获得巨大收益；linear attention 通过固定 state 改变
记忆机制，需要 recurrent/chunkwise/WY/DPLR 把串行定义映射到 Tensor Core；sparse attention
通过 block selector 减少配对，却必须支付选择、索引、gather、负载均衡和 serving 集成成本。

当前最稳妥的工程结论不是“某一类永远更快”，而是：短序列和成熟 dense backend 往往仍
强；长 prefill 给 block-sparse/chunkwise 更多机会；长 decode 的关键是 KV/state 带宽与
cache policy；模型质量必须由匹配训练预算的实验验证。后续性能优化应沿“固定语义和 shape
→ correctness → kernel-only → selector-included → layer/model → serving”的证据阶梯推进。

## 参考文献说明

本文使用 Pandoc 风格 citation key，例如 `[@dao2022flashattention]`。完整 74 篇学术论文
的作者、题目、年份、venue、DOI/URL 见 [`references/attention.bib`](../../references/attention.bib)；
分类和精读等级见 [`study/PAPER_MAP.md`](../../study/PAPER_MAP.md)。核心引用包括：

- `katharopoulos2020transformers`、`choromanski2021rethinking`；
- `yang2024gated`、`yang2024delta`、`dao2024transformers`、`yang2025gated`、
  `kimi2025linear`；
- `dao2022flashattention`、`dao2024flashattention2`、`shah2024flashattention3`、
  `zadouri2026flashattention4`；
- `jiang2024minference`、`yuan2025native`、`lu2025moba`、`zhang2025spargeattention`、
  `hu2026hils`；
- `dong2025flexattention`、`ye2025flashinfer`、`yang2025lserve`；
- `xiao2024streamingllm`、`zhang2023h2o`、`tang2024quest`、`ribar2024sparq`、
  `singhania2024loki`。

## 致谢

感谢中国科学技术大学孙经纬老师的指导，以及 107 本科生算力平台提供的 GPU 资源。
