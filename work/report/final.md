# Sparse 与 Linear Attention 调研及复现报告

> 报告日期：2026-08-30
> 调研重点：核心算法、chunkwise/tiled GPU kernel、训练与推理性能

## 摘要

标准 softmax attention 需要为每个 query 与所有可见 key 建立配对，计算量随序列长度
$T$ 以 $O(T^2D)$ 增长。降低长上下文成本主要形成了三条路线：第一，FlashAttention
保持 dense softmax 的数学定义不变，通过 tiling、online softmax 与重计算减少 HBM
读写；第二，linear attention 把历史压入固定形状的 recurrent state，并用 parallel、scan
或 chunkwise 算法恢复训练并行；第三，sparse attention 只计算结构规则或动态选择后的
query-key 子集，并围绕 selector、block layout、在线归一化和 serving cache 设计 kernel。

本报告从 dense attention 出发，依次讲清 linear attention 如何把历史压缩为固定大小的
状态、sparse attention 如何选择少量 query-key 配对，以及这些算法怎样落实成 GPU 上的
chunk、tile、scan 和 fused kernel。重点方法包括 GLA、DeltaNet、Gated DeltaNet（GDN）、
Kimi Delta Attention（KDA）、Mamba-2/SSD、MInference、Native Sparse Attention（NSA）、
MoBA、SpargeAttention 和 HiLS-Attention。复现使用 FLA 官方实现，在 107 Slurm 集群上完成
前向、反向、显存和长序列参数扫描，并用 MQAR 小模型训练观察 GDN/KDA 的实际学习效果。

## 1. 背景与研究问题

Transformer 的注意力层负责从上下文中取回当前 token 所需的信息。每个位置先生成 query、
key 和 value：query 表示当前位置希望查找的内容，key 表示各位置可供匹配的特征，value
表示匹配后真正被聚合的信息。query 与 key 的内积给出位置之间的相关性，softmax 将相关性
变为权重，最后对 value 加权求和。与按时间步传递隐藏状态的传统 RNN 相比，这种两两交互
既便于训练时并行，也能直接建立远距离联系。

代价来自 query-key 的两两配对。对于长度为 $T$、head dimension 为 $D$ 的序列，标准
attention 需要计算 $T\times T$ 个分数，主要计算量为 $O(T^2D)$。上下文从 4K 增长到
64K 时，token 数只增加 16 倍，分数矩阵却增加 256 倍。这个代价在不同阶段表现不同：

- 训练和 prefill 一次处理整段序列，主要压力是二次增长的矩阵乘、激活和临时存储；
- autoregressive decode 每次只有一个新 query，单步计算量随已有上下文线性增长，同时还要
  从显存读取不断扩大的 KV cache；
- 长上下文 serving 还会受到请求调度、分页 KV cache 和并发 batch 的影响。

因此，长上下文优化不能只看公式中的 FLOPs，还要同时处理算法复杂度、显存访问和 GPU
执行效率。现有方法主要沿三条路线发展：

| 路线 | 核心做法 | 保存的历史 | 代表方法 |
| --- | --- | --- | --- |
| Dense kernel 优化 | 保留完整 softmax attention，改变数据在 HBM 与 SRAM 间的流动方式 | 完整 K/V | FlashAttention |
| Linear attention | 把历史 key-value 累积到固定形状的矩阵状态 | recurrent state | GLA、DeltaNet、GDN、KDA、Mamba-2/SSD |
| Sparse attention | 为每个 query 计算局部窗口或动态选出的少量 KV block | 稀疏 KV 子集或完整 KV cache | MInference、NSA、MoBA、Sparge、HiLS |

这三条路线分别改变数据流、历史表示和实际配对数量。它们并不互斥：sparse kernel 仍可使用
FlashAttention 的 online softmax；Kimi Linear 则把三层 KDA 与一层全局 MLA 交错，在固定
状态与完整注意力之间折中。

本报告围绕四个问题展开：

1. 每种方法如何表示和更新历史信息；
2. recurrent、parallel、chunkwise 三种计算形式如何互相转换；
3. chunk、tile、WY 表示和 online softmax 如何映射到 GPU；
4. 算法复杂度、kernel 时间、训练效果和 serving 指标之间是什么关系。

评价一个方法需要把模型效果和系统性能分开测量。模型效果包括 validation loss、下游任务
准确率和长上下文能力；训练性能包括每秒 token 数、达到相同 loss 所需时间和峰值显存；
推理性能则包括 prefill latency、首 token 延迟（TTFT）、单 token 延迟（TPOT）、吞吐和
KV cache 占用。只有明确训练 token、FLOPs、硬件时间和模型规模中哪一项保持相同，两个
结果才具有可解释的比较基础。

## 2. Dense attention 与性能基线

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

### 2.2 FlashAttention：用分块数据流计算完整 attention

FlashAttention 计算完整的 dense softmax attention。朴素路径将 $S$、$P$ 写入 HBM；
FlashAttention 将 $Q$ 切为行 tile，将 $K/V$ 切为列 tile，让临时 score 在片上产生、
消费后丢弃，只维护每个 query 行的 running maximum、normalizer 和未归一化输出
[71]。

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
u'=e^{m-m'}u+\sum_{j\in b}e^{s_j-m'}v_j,\qquad o=u'/l'.
$$

旧状态必须乘 $e^{m-m'}$，否则两个 block 使用不同指数基准。causal kernel 还可以跳过
对角线上方的整块，对角块才做逐元素 mask。Backward 不保存完整 $P$，而是在 tile 内重算
score/probability，再累积 $dQ,dK,dV$；这增加部分 FLOPs，却减少 activation 与 HBM 流量。

FlashAttention-2 进一步减少非 matmul FLOPs、增加 sequence 维并行并改进 warp 间工作
划分 [72]。FA-3/4 利用 Hopper/Blackwell 的异步 pipeline 与低精度
继续提高新架构上的吞吐 [73, 74]。

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

这种数据流省去了完整 $T\times T$ 中间量的 HBM 写回，并在片上反复使用 Q/K/V tile。
因此本报告使用成熟 SDPA/FlashAttention 作为 dense 性能基线，显式 PyTorch 承担公式验证。

## 3. Linear attention：算法分类与原理

### 3.1 从 softmax kernel 到 recurrent state

若相似度核可写或近似为

$$
\kappa(q,k)\approx\phi(q)^\top\phi(k),
$$

则 causal attention 可以交换求和次序。定义

$$
S_t=\sum_{i\le t}v_i\phi(k_i)^\top
\in\mathbb R^{D_v\times D_\phi},\qquad
z_t=\sum_{i\le t}\phi(k_i)\in\mathbb R^{D_\phi},
$$

$$
o_t=\frac{S_t\phi(q_t)}
          {\phi(q_t)^\top z_t+\varepsilon}.
$$

状态更新为

$$
S_t=S_{t-1}+v_t\phi(k_t)^\top,\qquad
z_t=z_{t-1}+\phi(k_t).
$$

softmax 的未归一化 kernel 是 $\exp(q^\top k)$。它确实存在内积特征展开：

$$
e^{q^\top k}
=\sum_{r=0}^{\infty}\frac{(q^\top k)^r}{r!}
=\left\langle
\bigoplus_{r=0}^{\infty}\frac{q^{\otimes r}}{\sqrt{r!}},
\bigoplus_{r=0}^{\infty}\frac{k^{\otimes r}}{\sqrt{r!}}
\right\rangle.
$$

这个精确特征是无限维的。Performer 从高斯分布采样 $\omega$，使用正随机特征

$$
\phi_\omega(x)=\exp\left(\omega^\top x-\frac{\|x\|_2^2}{2}\right),
$$

它满足

$$
\mathbb E_\omega[\phi_\omega(q)\phi_\omega(k)]=e^{q^\top k}.
$$

采样 $D_\phi$ 个 $\omega$ 就得到有限维近似。另一类方法直接选择
$\phi(x)=\operatorname{ELU}(x)+1$、ReLU、SiLU 等正特征，再配合新的归一化和训练方式。
从矩阵角度看，$\Phi(Q)\Phi(K)^\top$ 的秩最多为 $D_\phi$，因此特征维度也决定了这张
近似 attention map 能表达多少独立的匹配模式。

当 $D_\phi,D_v$ 固定时，关于序列长度的工作为 $O(TD_\phi D_v)$，decode state 不随
$T$ 增长。这是“Transformers are RNNs”的基本形式 [1]。
Performer 用随机正特征近似 softmax kernel [2]；cosFormer 等
直接采用新的特征或归一化 [4]。有限维 state 将全部历史压缩到固定
容量，其表达能力取决于特征维度、更新规则和遗忘机制。

### 3.2 Recurrent、parallel 与 chunkwise 三种计算形式

三种形式描述的是同一个 operator 的不同执行计划：

| 形式 | 数据依赖 | 适合阶段 | 主要 kernel 特征 |
| --- | --- | --- | --- |
| recurrent | 每 token 依赖前一 state | decode、正确性 oracle | 固定 state；并行量小、频繁读写 state |
| parallel/scan | 展开全部位置 | 训练、短 prefill | 并行强；可能产生大中间量 |
| chunkwise | chunk 间递推、chunk 内并行 | 训练、长 prefill | 大部分工作转 GEMM；保留少量 boundary state |

对最简单的 additive state，令一个 chunk 的 $Q,K\in\mathbb R^{C\times D_k}$、
$V\in\mathbb R^{C\times D_v}$，则

$$
S_{c+1}=S_c+V_c^\top K_c,
$$

$$
O_c=\underbrace{Q_cS_c^\top}_{\text{inter-chunk}}
 +\underbrace{\bigl((Q_cK_c^\top)\odot M_C\bigr)V_c}_{\text{intra-chunk}}.
$$

串行边界从每 token 一次降为每 chunk 一次；块内使用 $C\times D$ 与 $D\times C$ GEMM。
GLA 论文指出 $C$ 取 Tensor Core 友好的倍数（如 16 的倍数）有利于利用矩阵乘，并通过
tiling 在片上复用张量块，减少 HBM 往返 [6]。在实验所用的 FLA 版本中，GDN 支持
chunk size 16/32/64，KDA 支持 32/64，默认均为 64。

### 3.3 GLA：主动遗忘

GLA 为普通累加 state 加入数据依赖的 key-channel gate：

$$
S_t=S_{t-1}\operatorname{Diag}(\boldsymbol\alpha_t)+v_tk_t^\top,
\qquad \boldsymbol\alpha_t\in(0,1)^{D_k}.
$$

不同通道可以有不同记忆寿命。它仍是 additive write：相同 key 的新旧 value 可能叠加。
GLA 的研究贡献同时包括模型机制与 I/O-aware chunkwise training，二者需要模型消融和
operator benchmark 两套证据 [6]。

### 3.4 Delta rule：沿当前 key 方向擦除再写入

令 state 方向为 $S_t\in\mathbb R^{D_v\times D_k}$，旧预测为
$\hat v_t=S_{t-1}k_t$。DeltaNet 更新为

$$
S_t=S_{t-1}+\beta_t(v_t-\hat v_t)k_t^\top.
$$

它也可由在线最小二乘得到。令

$$
L_t(S)=\frac12\|S k_t-v_t\|_2^2,
$$

则

$$
\nabla_SL_t=(S k_t-v_t)k_t^\top,
$$

一次步长 $\beta_t$ 的梯度下降即为上式。展开得到

$$
S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top.
$$

$I-\beta kk^\top$ 是 rank-1 修正的 generalized Householder transition；kernel 直接使用
$k$ 与 $\beta$ 完成这次修正。DeltaNet 使用 compact WY 表示压缩一串 rank-1
transition，使 chunk 内 pseudo-key/pseudo-value 与边界 state 的计算转为 GEMM，并避免为
每个 token materialize 矩阵 state [7]。

### 3.5 GDN：gate 与 delta 的组合

GDN 先以每 head scalar gate 衰减，再执行 delta write：

$$
\bar S_t=S_{t-1}\alpha_t,
$$

$$
S_t=\bar S_t+\beta_t(v_t-\bar S_t k_t)k_t^\top.
$$

$\alpha$ 决定旧状态寿命，$\beta$ 决定当前 key 方向的纠正幅度。论文在合成 recall、语言
模型和 LongBench 中验证了这种组合，并在单张 H100 上报告了与 DeltaNet 接近的训练吞吐
[10]。

### 3.6 Mamba-2 / SSD：state-space 与 attention 的块对偶

SSD 将一类 scalar-identity state transition 的 SSM 写成结构化半可分（semiseparable）
矩阵，也可以反向把 attention 看作结构化矩阵乘。其工程意义是：同一 operator 可在
recurrent state、卷积/scan、block matrix 三种视角中选择执行计划；结构化 transition
带来了块算法与 Tensor Core 友好性 [8]。SSD/Mamba-2、GDN 和 KDA
都维护固定矩阵 state，并分别使用 scalar transition、gated delta 和 channel-wise delta 更新。

### 3.7 KDA 与 Kimi Linear：GDN 的细粒度扩展

KDA 把 GDN 的 scalar decay 换为 key-channel diagonal decay：

$$
\bar S_t=S_{t-1}\operatorname{Diag}(\boldsymbol\alpha_t),
$$

$$
S_t=\bar S_t+\beta_t(v_t-\bar S_t k_t)k_t^\top.
$$

写成 $S_t=S_{t-1}A_t+\beta_tv_tk_t^\top$ 后，其 transition 为

$$
A_t=\operatorname{Diag}(\boldsymbol\alpha_t)(I-\beta_tk_tk_t^\top)
=\operatorname{Diag}(\boldsymbol\alpha_t)
-\beta_t\bigl(\operatorname{Diag}(\boldsymbol\alpha_t)k_t\bigr)k_t^\top.
$$

这是受约束的 diagonal-plus-rank-1（DPLR）结构。当
$\boldsymbol\alpha_t=\alpha_t\mathbf1$ 时，KDA 精确退化为 GDN。KDA 绑定低秩两侧的
变量结构，论文称其
特化 chunkwise 算法比一般 DPLR 少做第二级 chunk 矩阵计算与若干 GEMM [11]。

Kimi Linear 在 KDA 之上构建完整模型：以 3:1 交错 KDA 与全局 MLA，并加入 MoE
backbone、位置处理和训练配方。三层 KDA 使用固定 state，一层 MLA 维护 KV cache，因此
这组层比例最多减少约 75% 的 attention KV cache [11]。

## 4. Linear attention：kernel 与实现

### 4.1 普通线性注意力的 chunkwise 矩阵化

核心思想是把长度为 $C$（通常为 64）的 chunk 内逐 token 更新改写为矩阵乘法。以下统一
采用 $S\in\mathbb R^{D_v\times D_k}$ 的 state 方向：

$$
S_t=S_{t-1}+v_tk_t^\top,\qquad o_t=q_tS_t^\top.
$$

将一个 chunk 的输入按行堆成
$Q,K\in\mathbb R^{C\times D_k}$、$V\in\mathbb R^{C\times D_v}$。

#### 1. Chunk 边界状态更新

chunk 中的 $C$ 次外积可以合并成一次 GEMM：

$$
S_{\text{next}}=S_{\text{in}}+V^\top K.
$$

维度变化为

$$
[D_v,C]\times[C,D_k]\longrightarrow[D_v,D_k].
$$

#### 2. Chunk 内所有输出并行计算

每个输出由两部分组成：上一 chunk 留下的历史状态，以及当前 chunk 中位于该 token
之前的局部信息。

$$
O=
\underbrace{QS_{\text{in}}^\top}_{\text{历史状态（inter-chunk）}}
+
\underbrace{\left((QK^\top)\odot M_C\right)V}_{\text{当前块（intra-chunk）}}.
$$

其中 $M_C$ 是 $C\times C$ 下三角 causal mask。实际计算可拆为：

1. `Q @ S_in.T`：$[C,D_k]\times[D_k,D_v]\to[C,D_v]$；
2. `Q @ K.T`：$[C,D_k]\times[D_k,C]\to[C,C]$；
3. 对 $C\times C$ score 施加 causal mask；
4. `score @ V`：$[C,C]\times[C,D_v]\to[C,D_v]$；
5. `V.T @ K`：生成下一个 chunk 的边界 state。

chunk 之间仍按顺序传递 $S_{\text{in}}$，串行步数从 $T$ 次减少为 $T/C$ 次；chunk 内
的主要工作都变成 Tensor Core 擅长的 GEMM。$C=1$ 时回到 recurrent，$C=T$ 时接近完全
parallel，常用的 $C=64$ 在并行度、片上空间和 $C^2$ 局部矩阵之间取得平衡
[4, 6]。

### 4.2 Delta Rule 为什么需要 WY 表示

Delta Rule 会先读出当前 key 在旧 state 中对应的 value，再用误差修正它：

$$
S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top
=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top.
$$

普通线性注意力的 $v_t$ 在进入 state 以前彼此独立；这里的修正量

$$
u_t=\beta_t(v_t-S_{t-1}k_t)
$$

依赖 $S_{t-1}$，所以直接计算 $u_1,u_2,\ldots,u_C$ 仍然是串行的。DeltaNet 将连续的
$I-\beta_tk_tk_t^\top$ 看作 generalized Householder transformation，并用 compact WY
表示把一串 rank-1 transition 压缩为两个瘦矩阵 [7]。

### 4.3 DeltaNet Chunk-wise 计算公式

对一个大小为 $C$ 的 chunk，定义 $\beta\in\mathbb R^C$，并构造下三角矩阵

$$
T=
\left(I+\operatorname{tril}
\left(\operatorname{Diag}(\beta)KK^\top,-1\right)\right)^{-1}
\operatorname{Diag}(\beta).
$$

这里的“逆”通过 lower-triangular solve 完成。随后得到两组伪向量：

$$
W=TK\in\mathbb R^{C\times D_k},\qquad
U=TV\in\mathbb R^{C\times D_v}.
$$

$W$ 描述 chunk 内连续擦除操作的组合，$U$ 描述对应的写入。把 chunk 入口 state 加入后，
得到修正后的 pseudo value：

$$
G=U-WS_{\text{in}}^\top
\in\mathbb R^{C\times D_v}.
$$

于是状态更新和输出分别变为

$$
S_{\text{out}}=S_{\text{in}}+G^\top K,
$$

$$
O=
\underbrace{QS_{\text{in}}^\top}_{\text{历史状态}}
+
\underbrace{\left((QK^\top)\odot M_C\right)G}_{\text{当前 chunk 的 delta 修正}}.
$$

由此可以得到五个核心 GEMM：

1. `W @ S_in.T` 生成 $G$ 中的历史修正项；
2. `G.T @ K` 更新 $S_{\text{out}}$；
3. `Q @ S_in.T` 读取 chunk 以前的历史；
4. `Q @ K.T` 计算 chunk 内 key-query 配对；
5. `(Q @ K.T * M_C) @ G` 聚合当前 chunk 的修正 value。

WY 表示精确保持 Delta Rule 的数学结果，同时重新安排执行顺序：token 级递推被整理为
一次小型三角求解和数次大矩阵乘，使训练能够沿序列维度并行。

### 4.4 GDN 与 KDA 如何扩展这条路径

GDN 在 Delta Rule 前加入每个 head 的 scalar decay。一个 chunk 内先对 log gate 做局部
prefix sum，由前缀差得到任意两个位置间的累计衰减；然后将衰减吸收到 $Q/K/W/U$ 的缩放
中，继续使用 WY、inter-chunk state 和 intra-chunk GEMM。FLA 的 GDN forward 顺序为：

1. `chunk_local_cumsum` 计算 chunk 内 gate 累积；
2. `chunk_gated_delta_rule_fwd_intra` 计算 $W/U$ 和三角辅助矩阵；
3. `chunk_gated_delta_rule_fwd_h` 传播 chunk 边界 state；
4. `chunk_fwd_o` 合并历史输出与当前 chunk 输出。

KDA 将 scalar decay 扩展为每个 key channel 一个 decay，因此 transition 变为
diagonal-plus-rank-1。通道 gate 能更细地控制 state 的行，但也会增加 chunk 内辅助矩阵。
KDA 把 rank-1 两侧向量绑定到同一个 $k_t$，专用算法比一般 DPLR 少两次二级 chunk 矩阵
计算和三次额外矩阵乘；这也是 Kimi Linear Figure 2 比较 KDA 与 DPLR kernel 的原因
[11]。

### 4.5 Chunk、tile 与 GPU kernel

chunk 和 tile 位于两个层次：

- **chunk size $C$** 是算法分组，决定多少 token 共用一个边界 state，以及局部矩阵
  $C\times C$ 的大小；
- **tile size $B_M,B_N,B_K$** 是单个 GPU program 处理的数据块，决定 SRAM/register
  占用、Tensor Core 形状和并行 program 数量。

一个 $C=64$ 的 chunk 可以由多个 tile 完成。典型数据流是：

```text
HBM: Q, K, V, gate, beta
          |
          v
  gate prefix + triangular/WY preparation
          |
          +---------------- boundary state S_c ----------------+
          |                                                     |
          v                                                     v
  intra-chunk QK^T tiles                              inter-chunk Q @ S_c
          |                                                     |
       mask / decay                                             |
          |                                                     |
          +------------------- pseudo-V GEMM -------------------+
                                |
                                v
                         output O and S_(c+1)
```

kernel 会让 $Q/K/V$ tile 在 SRAM 和寄存器中复用，state 累加通常使用 FP32。forward 保存
少量三角辅助量；backward 根据这些辅助量重算 $W/U$ 和 chunk state，以增加一些计算换取
更少的 HBM 写入。head dimension 大于一个 tile 时，kernel 还要沿 $D_k/D_v$ 分块并在
不同 program 之间合并部分结果。

FLA 中 `chunk.py` 负责接口和调度，`chunk_fwd.py`/`chunk_bwd.py` 负责主路径，
`wy_fast.py`、`gate.py` 和 `chunk_intra.py` 生成辅助量；`fused_recurrent.py` 将一个或少量
token 的递推融合为 decode kernel。该 FLA 版本的 GDN 支持 chunk size 16/32/64，KDA 支持
32/64，默认使用 64。

### 4.6 实验所用算子

| 算子 | 输入布局 | 核心状态或索引 | 用途 |
| --- | --- | --- | --- |
| `chunk_delta_rule` | `[B,T,H,D]` | state `[B,H,D,D]` | DeltaNet chunkwise 训练 |
| `fused_recurrent_delta_rule` | `[B,T,H,D]` | 同一矩阵 state | DeltaNet recurrent 对照 |
| `chunk_gated_delta_rule` | `q/k:[B,T,H,K]`, `v:[B,T,HV,V]` | scalar `g, beta` | GDN chunkwise 训练 |
| `chunk_kda` | 同上 | channel gate `[B,T,HV,K]` | KDA chunkwise 训练 |
| `chunk_dplr_delta_rule` | `q/k/v/a/b/gk` | 一般 diagonal-plus-rank-1 transition | Kimi kernel 对照 |
| `parallel_nsa` | `q:[B,T,Hq,Dk]`, `k/v:[B,T,Hkv,*]` | selected block indices | NSA block-sparse 前反向 |

## 5. Sparse attention：算法分类与原理

### 5.1 统一定义

令 $\mathcal A(i)\subseteq\{0,\ldots,i\}$ 为 query $i$ 可见的 key 集合：

$$
o_i=\sum_{j\in\mathcal A(i)}
\frac{e^{q_i^\top k_j/\sqrt D}}
     {\sum_{r\in\mathcal A(i)}e^{q_i^\top k_r/\sqrt D}}v_j.
$$

计算量约为 $O(D\sum_i|\mathcal A(i)|)$。实际时间还包含细粒度 gather、selector/index、
softmax reduction、kernel launch 与同步。block sparse 将相邻 token 一起选择，一个 program
可以读取连续 K/V block，执行规整 MMA，再用 online softmax 合并 selected tiles，因此更
容易把理论稀疏率转化为 GPU 加速。

### 5.2 方法分类

| 类别 | 代表方法 | 选择规则 | 主要代价 |
| --- | --- | --- | --- |
| 固定结构 | Sparse Transformer、Longformer、BigBird、LongNet | window/global/strided/random/dilated | 规则快，但结构先验可能漏掉内容相关远程依赖 |
| hashing/routing | Reformer、Routing Transformer、Sinkhorn | LSH、聚类或可学习排序 | router、排序、负载均衡 |
| prefill 动态模式 | MInference、FlexPrefill、XAttention | pattern/head + 动态 block index | selector 与不规则 block 分布 |
| 可训练稀疏架构 | NSA、MoBA、SeerAttention、HiLS | compression/router/top-k/window | 训练、backward 与选择质量 |
| training-free 近似 | SparQ、QUEST、Loki、SpargeAttention | proxy、量化、低秩 key、在线过滤 | proxy recall 与 selector 带宽 |
| KV/cache policy | StreamingLLM、H2O、DuoAttention、InfiniGen | sink/heavy hitter/head 分类/eviction | decode 质量、cache page 与调度 |

全部 74 篇文献按研究方向列于第 9.4 节，并在文末给出完整出处。

### 5.3 MInference：按 head 选择预填充稀疏模式

MInference 针对长上下文 **prefill**，离线为 attention head 分配 A-shape、Vertical-Slash
或 Block-Sparse 模式，运行时按输入建立具体 index，再调用相应 Triton/FlashAttention 风格
kernel [41]：

- A-shape：初始 token + local window，结构较稳定；
- Vertical-Slash：少量动态垂直列与斜线；
- Block-Sparse：用 64×64 等块表达更分散的内容相关区域。

Vertical-Slash 可用少量尾部 query 与 K 的乘积估计重要列/斜线；Block-Sparse 可对 Q/K 做
64-token block mean pooling，再计算 coarse attention 选块。它的主要应用阶段是长 prompt
的 prefill；decode 由 KV-cache 类方法负责。

### 5.4 NSA：compression、selection、sliding 三分支

NSA 是 natively trainable、hardware-aligned sparse architecture [52]：

1. **Compression：** 将连续 K/V block 聚合成 coarse token，保留全局概览；
2. **Selection：** 复用 compression attention score 估计 block importance，选择少量原始
   K/V block 做 fine-grained attention；
3. **Sliding window：** 单独保留近期局部 token，避免其他分支被局部模式“走捷径”；
4. **Gated merge：** 三分支输出由学习 gate 聚合，并可使用独立 K/V 投影降低梯度干扰。

FLA 的 NSA selected kernel 使用 `[B,TQ,H,S]` block indices；实验所用版本的 block size
64，论文常用每 query 16 个 selected blocks。kernel 以 `BK/BV` power-of-two tile 覆盖
head dimension，Q tile 留在片上，遍历 selected K/V blocks 并做 sparse online softmax。
Backward 将 query-to-block 选择倒排成 CSR，让 dK/dV kernel 以被选择 block 为单位 gather
query，避免扫描所有 query。

### 5.5 MoBA：block router + 两路 FlashAttention

MoBA 把序列切成固定 block，以 block mean key 作为 router representative；每个 query/head
选择 top-k block，其中 local block 始终保留 [54]。FLA 实现的流程是：

```text
block mean K -> query-to-block score -> causal top-k-1 nonlocal blocks
      |                                      |
local causal FlashAttention          gathered varlen FlashAttention
      |                                      |
      +-------- online LSE softmax merge -----+
```

两个 stream 分别返回 output 与 log-sum-exp，再用稳定 LSE 合并，等价于在二者 key 并集上
做一次 softmax。selected block 内保持连续 FlashAttention；router、gather、变长序列打包和
负载均衡则构成其额外系统工作。

### 5.6 SpargeAttention：两阶段在线过滤

SpargeAttention 是 training-free sparse/quantized inference kernel [59]：

1. 将每个 Q/K block 压缩成 representative，按 block 内 token 相似度决定是否相信压缩；
2. 以 coarse score 的 softmax CDF 生成 block mask，跳过对应 $Q_iK_j^\top$ 与 $P_{ij}V_j$；
3. 对已计算 score tile，在 online softmax 中比较 local/global max，若该 warp 的概率贡献
   足够小，再跳过 $\tilde P_{ij}V_j$；
4. 与 SageAttention 的低比特 QK 路径结合。

第二阶段复用已经计算的 online-softmax 状态，因此新增判断开销较小。最终输出是由阈值
控制的 dense-attention 近似，论文通过模型评测选择兼顾精度和速度的阈值。

### 5.7 HiLS-Attention：分层 chunk mass 与 query packing

HiLS 是 2026 年的新工作，使用层级 chunk summary/质量分配选择远程内容，并将相邻 query
的 selected chunk 并集打包，使一次 K/V load 服务多个 query [58]。这种
one-load-multiple-compute 可以增加 Tensor Core 左矩阵规模与 K/V 复用；query 并集同时会
扩大实际加载的 block 数。因此它的实际性能由目标 top-k、合并后的 block 数、selector
开销、attention kernel 和端到端 latency 共同决定。

### 5.8 Sparse tile 数据流

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

## 6. Sparse attention：kernel、serving 与部署

### 6.1 Block-sparse kernel 如何执行

以 NSA selected attention 为例，输入除了 $Q/K/V$ 以外，还包含
`block_indices[B,T,Hkv,S]`。GPU grid 通常把 query token、KV head 和 value tile 映射到
不同 program。每个 program 将一组 GQA query heads 保留在片上，然后依次完成：

1. 从 index 中取得一个连续 KV block 的起点；
2. 合并读取该 block 的 $K/V$，让同一 KV load 服务一组 query heads；
3. 用 Tensor Core 计算 $QK^\top$ tile；
4. 更新 running max、normalizer 和 output accumulator；
5. 遍历完 $S$ 个 block 后归一化并写回输出。

backward 的 $dQ$ 可以继续按 query 遍历 selected blocks。$dK/dV$ 需要知道哪些 query 选择
了当前 block，因此 FLA 先把 query-to-block 关系整理为 CSR，再让每个 program 汇总指向同一
KV block 的 query。这个倒排步骤换来了连续的 dK/dV 写入和更规则的负载。

### 6.2 Prefill 与 decode

| 阶段 | 张量形态 | 主要工作 | 主要指标 |
| --- | --- | --- | --- |
| prefill | 大量 query 同时处理 prompt | attention GEMM、selector、HBM IO | TTFT、prefill latency、token/s |
| decode | 每个请求新增一个 query | 读取 KV/state、batching、page/scheduler | TPOT/ITL、aggregate token/s、p95 |

prefill 有足够多的 query，可以用大 tile 和 block-sparse GEMM；decode 的 $T_q=1$，算术强度
较低，时间更接近实际读取的 KV 字节数。因此同一个稀疏模式需要分别报告 prefill 与 decode。

StreamingLLM 保留 attention sink 与 recent window [39]；H2O 保留
heavy hitters [40]；QUEST/SparQ/Loki 用 query-aware proxy 或低秩 key 缩小 KV
读取 [42–44]。MInference 主要优化 prefill，LServe 进一步统一 sparse prefill/decode [38]。

### 6.3 KV cache、page table 与调度

serving engine 会把不同请求的 KV cache 切成 page，并在每一步把活跃请求组成 batch。
sparse attention 的 block index 最终需要映射到物理 page；连续逻辑 block 可能落在不同物理
地址。FlashInfer 将 paged/composable KV layout、JIT attention template、load-balanced
scheduling 与 CUDA Graph 约束放进统一接口 [36]。因此端到端吞吐还取决于
batch 大小、请求长度分布、page 命中、通信和其他模型层。

Linear recurrent decode 用固定 $D_k\times D_v$ state 代替随 $T$ 增长的 KV cache。
ReplaySSM 进一步保存一个较少写回的 state checkpoint 和近期输入 ring buffer，先直接重组
output，buffer 满时再把更新 flush 到 state。它优化的是 state 写回频率和 speculative decode
执行顺序，适合 linear attention serving [75]。

### 6.4 性能测量与瓶颈定位

GPU kernel 由 CPU 异步提交，CPU 端函数返回并不代表 GPU 已经完成计算。因而本实验采用
CUDA Event 与 `triton.testing.do_bench` 在 GPU 时间线上计时，并在正式测量前完成 Triton
编译、autotune 和缓存预热。每个配置记录第 20、50、80 百分位（p20/p50/p80）；正文使用
p50，即中位数，表示稳态延迟。
峰值显存由 `torch.cuda.max_memory_allocated` 记录，输入张量在测量前已经常驻显存，因此
结果表示算子输出、激活、临时量和梯度带来的额外分配。

延迟只能说明“慢了多少”，profiler 用于解释“时间花在哪里”。PyTorch Profiler 记录一次
前向和反向中的 kernel 名称、调用次数与 self device time，由此可以判断时间集中在状态传播、
矩阵乘、索引准备还是梯度聚合。Nsight Systems 适合检查 CPU launch gap、memcpy、通信与
GPU 执行是否重叠；Nsight Compute 则进一步提供 DRAM 吞吐、Tensor Core 指令、occupancy
和 warp stall 等硬件计数器，可用于区分访存受限、计算受限和并行度不足。本报告的绝对延迟
来自未插桩 benchmark，PyTorch Profiler 数据只用于分析各 kernel 的时间构成。

## 7. 复现方法与环境

### 7.1 环境与版本

实验在 Slurm 集群的单张 A100 上完成。算子实现来自 Flash Linear Attention（FLA），固定
版本可以保证算法接口、Triton kernel 和 autotune 配置保持一致。

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB（可用显存 81151.75 MiB） |
| FLA commit | `d1ce07369d581813553f30a750af3b6b5f9af6a9` |
| 软件 | Python 3.12.13；PyTorch 2.11.0+cu128；CUDA 12.8；Triton 3.6.0 |
| 算子精度 | Q/K/V 与主要输出为 BF16；gate 累计和数值敏感的辅助量为 FP32 |

### 7.2 计时、显存与输入规则

每个 shape 先完成编译和 autotune，再以 25 ms warmup、100 ms repeat 测量 p20、p50、
p80。64K NSA 另外使用 1000 ms warmup 和 6500 ms repeat 复测。`forward+backward` 每次
重新执行 forward，并对所有浮点输入计算梯度。显存结果是在输入已经创建以后记录的
PyTorch allocator 额外峰值。

本文统一定义

$$
\text{speedup}=\frac{\text{基线方法的 p50 延迟}}
                       {\text{被比较方法的 p50 延迟}}.
$$

因此 speedup 大于 1 表示被比较方法更快。三组性能实验的基线分别为：

| 实验 | 被比较方法 | 基线方法 |
| --- | --- | --- |
| DeltaNet | `chunk_delta_rule` | 数学等价的 `fused_recurrent_delta_rule` |
| Kimi Linear | 专用 `chunk_kda` | 一般 DPLR transition 的 `chunk_dplr_delta_rule` |
| NSA | block-sparse selected 与 compression+selection 路径 | 相同 B/T/head/dtype 的 PyTorch Flash SDPA dense GQA |

四组主要配置为：

| 复现对象 | 配置 | 对应论文实验 |
| --- | --- | --- |
| DeltaNet | model dim 2048；$B\times T=16384$；head dim 64/128/256；$T=512\ldots16K$ | DeltaNet Figure 1 |
| KDA / DPLR | `B=1,H=16,D=128`；$T=2K\ldots64K$；KDA chunk 64、DPLR chunk 16 | Kimi Linear Figure 2 |
| NSA | `B=1,Hq/Hkv=64/4,Dk=Dv=128`；block size 64；16 blocks；$T=8K\ldots64K$ | NSA Figure 6 的 GQA/head/block 配置 |
| MQAR | 2 layers、2 heads、head dim 128；$T=512$；128 pairs、64 queries；8000 steps | Kimi Linear Section 5.1 |

NSA 论文的效率实验使用 $D_k=192,D_v=128$。FLA 的 A100 kernel 将单个 key tile 上限设为
128，并要求 key dimension 只占一个 tile；PyTorch Flash SDPA 还要求 Q/K/V 的最后一维
相同。为使 sparse 与 dense 基线能够在同一后端运行，实验将 $D_k$ 和 $D_v$ 统一为 128，
保留论文中的 GQA 比例、block size、selected block 数和序列长度。

### 7.3 MQAR 训练设置

MQAR（Multi-Query Associative Recall）要求模型从序列前部记住多组 key-value，并在后部
看到 key 时输出对应 value，用于测量状态模型的关联记忆能力。GDN 和 KDA 使用相同数据、
随机种子、AdamW、learning rate `5e-4`、batch size 16 和 BF16 autocast。每条长度 512 的
序列包含 128 组 key-value、64 个查询和填充噪声，loss 只计算查询后的答案 token。GDN
含 2,385,672 个参数，KDA 含 2,516,740 个参数；KDA 多出的约 5.5% 参数来自 channel-wise
gate 投影。

## 8. 正确性、性能与效果结果

性能比较的前提是矩阵化、分块和 fused kernel 没有改变算子的数学结果。本实验将优化
kernel 与逐 token 或朴素并行参考实现输入相同的随机张量，比较输出、最终 state 以及所有
可训练输入的梯度。最大绝对误差反映单个元素的最坏偏差；归一化误差定义为误差的 RMSE
除以参考张量的 RMS，用于消除不同张量尺度的影响。

### 8.1 数值正确性：优化 kernel 与参考实现的一致性

| 算子 | 覆盖范围 | 输出最大绝对误差 | 输出/梯度最大归一化误差 |
| --- | --- | ---: | ---: |
| GDN chunk | FP32；chunk 16/32/64；output、state 与全部梯度 | $9\times10^{-6}$ | $2.4\times10^{-5}$ |
| KDA chunk | FP16/BF16；chunk 32/64；GVA、L2 norm、gate、output、state 与全部梯度 | $1.465\times10^{-3}$ | $6.677\times10^{-3}$ |
| DPLR chunk | FP16；$D=60/64/100/128$；safe gate、recompute、output、state 与全部梯度 | $6.282\times10^{-2}$ | $9.32\times10^{-4}$ |
| NSA selected | BF16；$D=60/64/100/128$；GQA；output 与 $dQ/dK/dV$ | $1.953\times10^{-3}$ | $7.82\times10^{-4}$ |

GDN 的 FP32 结果几乎与参考递推重合。KDA 的测试覆盖 FP16/BF16，NSA 使用 BF16，绝对
误差会受低精度量化步长影响，但相对于参考张量尺度的误差均低于 0.7%。DPLR 的部分 FP16
梯度最大绝对差较大，
其最大归一化误差仍低于 $10^{-3}$。这些结果说明后续性能差异来自执行方式，而不是算子
定义发生了变化。

### 8.2 DeltaNet：chunkwise 相对 recurrent 的加速

DeltaNet 论文 Figure 1 固定模型维度 2048，并让 $B\times T=16384$。这样每个点的 token
总数一致：序列越长，batch 越小。被比较对象是 `chunk_delta_rule`，基线是计算同一 Delta
Rule 的 `fused_recurrent_delta_rule`；表中 speedup 等于 recurrent 延迟除以 chunkwise
延迟。下面给出 head dim 128 的 p50，head 数为 $2048/128=16$。

| T / B | recurrent fwd (ms) | chunk fwd (ms) | recurrent/chunk | recurrent fwd+bwd (ms) | chunk fwd+bwd (ms) | recurrent/chunk |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 32 | 4.924 | 1.259 | 3.91× | 31.200 | 3.469 | 8.99× |
| 1K / 16 | 5.303 | 1.273 | 4.17× | 33.771 | 3.488 | 9.68× |
| 2K / 8 | 6.200 | 1.304 | 4.75× | 40.568 | 3.529 | 11.50× |
| 4K / 4 | 7.786 | 1.385 | 5.62× | 46.678 | 3.759 | 12.42× |
| 8K / 2 | 14.189 | 1.337 | 10.62× | 84.280 | 3.762 | 22.40× |
| 16K / 1 | 26.629 | 1.617 | 16.47× | 173.534 | 4.319 | 40.18× |

![DeltaNet recurrent 与 chunkwise 速度比较](figures/deltanet-speedup.png)

head dim 64/128/256 都呈现相同趋势。固定 token 数后，chunkwise 的大 GEMM 工作量接近
恒定；recurrent 路径随着 $T$ 增大而失去 batch 并行度，串行循环也更长。因此 head dim
128 的 chunkwise forward 相对 recurrent 从快 3.91× 增长到快 16.47×，前向加反向从
快 8.99× 增长到快 40.18×。head dim 256 的 16K forward+backward 相对 recurrent 快
47.28×。这复现了论文中
“序列越长、head dimension 越大，chunkwise 优势越明显”的主要曲线形状
[7]。

### 8.3 Kimi Linear：复现 KDA 与 DPLR kernel

Kimi Linear Figure 2 使用 `B=1,H=16,D=128` 扫描 2K–64K。被比较对象是利用 KDA 约束
结构的专用 chunk kernel，基线是表达一般 diagonal-plus-rank-1 transition 的 DPLR chunk
kernel。表中 `DPLR/KDA` 等于 DPLR 延迟除以 KDA 延迟，时间均为 A100 上的 p50。

| T | DPLR fwd | KDA fwd | DPLR/KDA | DPLR fwd+bwd | KDA fwd+bwd | DPLR/KDA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0.914 | 0.711 | 1.29× | 2.517 | 3.305 | 0.76× |
| 4K | 1.784 | 0.850 | 2.10× | 4.911 | 3.511 | 1.40× |
| 8K | 3.540 | 1.633 | 2.17× | 9.759 | 5.174 | 1.89× |
| 16K | 7.039 | 3.189 | 2.21× | 19.375 | 10.275 | 1.89× |
| 32K | 14.074 | 6.381 | 2.21× | 39.048 | 20.382 | 1.92× |
| 64K | 28.452 | 12.761 | 2.23× | 78.649 | 40.838 | 1.93× |

![KDA 与一般 DPLR 的 kernel 时间](figures/kimi-dplr-kda.png)

从 4K 开始，KDA forward 稳定约为 DPLR 的一半；64K 时，KDA 的 forward 和
forward+backward 相对 DPLR 分别快 2.23× 与 1.93×。2K forward+backward 的 KDA 固定
准备开销仍占较大比例，因而慢于 DPLR；4K 后专用 chunk 公式节省的辅助矩阵和 GEMM
开始占主导。64K 时 KDA 的 forward 额外峰值为 3080 MiB，DPLR 为 5376 MiB；
forward+backward 分别为 7712 与
11648 MiB。实验同时复现了论文所述算子效率约翻倍的长序列趋势及其稳定区间
[11]。

### 8.4 MQAR：channel-wise gate 带来的学习效果

| 模型 | 参数量 | 训练吞吐 | 最终验证 loss | step 3500 准确率 | step 4000 | step 5000 | step 8000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN | 2.386M | 201.6K token/s | 5.5511 | 0.43% | 0.45% | 0.35% | 0.52% |
| KDA | 2.517M | 145.1K token/s | 0.00665 | 55.41% | 97.22% | 99.29% | 99.88% |

![GDN 与 KDA 的 MQAR 学习曲线](figures/mqar-gdn-kda.png)

随机猜测 256 个 value 的准确率约为 $1/256=0.39\%$。在这组 512-token、128 对记忆、
64 个查询的设置中，GDN 到 8000 step 仍在随机水平附近；KDA 在约 3200 step 后开始快速
学习，3500 step 达到 55.4%，4000 step 达到 97.2%，最后达到 99.88%。KDA 的 per-channel
decay 可以为同一个 head 内的不同 state 行设置不同记忆寿命，正适合同时维护大量独立
key-value 映射。代价是本实验中参数量多 5.5%，训练吞吐低约 28%。这组结果复现了 Kimi
论文合成任务中 KDA 比 GDN 收敛更快的方向 [11]。

### 8.5 NSA：64K block-sparse 前向与反向

共同配置为 `B=1,Hq/Hkv=64/4,Dk=Dv=128,BF16`，每个 query 最多选择 16 个大小为
64 的 block。dense 基线是相同 shape 的 PyTorch Flash SDPA GQA；`selected` 使用预先
给定的稀疏索引；`compression+selection` 还执行 compression、top-k 和 selected attention。表中为
p50 ms，所有 NSA speedup 均用 dense SDPA 延迟除以对应 sparse 路径延迟。

| T | dense fwd | selected fwd | compression+selection fwd | dense fwd+bwd | selected fwd+bwd | compression+selection fwd+bwd |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 6.566 | 4.316 | 6.751 | 26.219 | 23.291 | 51.517 |
| 16K | 21.334 | 8.888 | 15.418 | 83.081 | 46.262 | 107.059 |
| 32K | 84.316 | 18.217 | 38.055 | 323.557 | 92.911 | 227.753 |
| 64K | 343.004 | 37.001 | 105.870 | 1291.925 | 186.869 | 502.871 |

![NSA 长序列前向与前反向时间](figures/nsa-long-sequence.png)

上表的 64K 点来自 100 ms repeat 窗口，而 dense backward 单次已经超过该窗口。为降低
短测量窗口带来的波动，又在另一张 A100 上以 1000 ms warmup、6500 ms repeat 重测 64K：

| 64K 长窗口复测 | dense | selected | compression+selection | dense/selected | dense/compression+selection |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 326.374 | 23.417 | 59.743 | 13.94× | 5.46× |
| forward+backward | 1131.922 | 139.381 | 332.676 | 8.12× | 3.40× |

两次 A100 测量中，64K selected forward 相对 dense 快 9.27×--13.94×，包含 compression
和 selector 的路径相对 dense 快 3.24×--5.46×；forward+backward 分别快
6.91×--8.12× 和
2.57×--3.40×。同型号节点
仍会受时钟、共置负载和 Triton autotune 选择影响，因此曲线用于说明扩展趋势，精确回归
应固定节点并锁定功耗/时钟。两次实验都显示 8K selector 路径接近或慢于 dense，32K 后
compression+selection 前反向开始加速；动态稀疏的主要优化空间已从 selected attention 转移到 compression、
top-k 和 index 生成。

64K forward+backward 的算子额外峰值为：dense 8352 MiB、selected 3104 MiB、
compression+selection 路径 7234 MiB。selected kernel 的 activation/gradient 峰值降低
2.69×；compression
分支和 selector 的中间量会使用其中一部分节省。NSA 原论文同时加入 512-token sliding
window 和独立分支 gate，这组曲线测量的是 FLA 已公开的 compression/selection 路径。

### 8.6 Profiler：时间花在了哪些 kernel

PyTorch Profiler 对三个代表 shape 各记录一次 forward+backward。Profiler 插桩会增加
绝对时间，下面使用 self device time 识别主 kernel：

| operator / shape | 主要 kernel | self device time |
| --- | --- | ---: |
| Delta chunk `B=4,T=4096,H=16,D=128` | state forward `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（2 次） | 764.9 us |
| 同上 | `chunk_bwd_kernel_dqkwg` | 565.3 us |
| 同上 | state backward `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 481.1 us |
| 同上 | `prepare_wy_repr_bwd_kernel` | 403.9 us |
| KDA `B=1,T=16384,H=16,D=128` | `chunk_kda_bwd_kernel_wy_dqkg_fused` | 3089.1 us |
| 同上 | `chunk_kda_bwd_kernel_intra` | 2787.1 us |
| 同上 | `chunk_kda_fwd_kernel_inter_solve_fused` | 1248.4 us |
| 同上 | state forward kernel（2 次） | 1222.1 us |
| NSA `B=1,T=16384,Hq/Hkv=64/4,D=128` | `parallel_nsa_bwd_kernel_dkv` | 22314.6 us |
| 同上 | `parallel_nsa_bwd_kernel_dq` | 5984.6 us |
| 同上 | `parallel_nsa_fwd_kernel` | 5456.5 us |
| 同上 | `prepare_block_csr_kernel`（2 次） | 194.6 us |

DeltaNet 的时间分散在 state propagation、$dQ/dK/dW/dg$、state backward、WY backward
和 triangular merge；KDA 的两个大型 backward kernel 占据主导，说明继续优化时应优先
研究 WY/dQKG 融合和 intra-chunk backward，而非只缩短 gate prefix sum。NSA 中 dK/dV
backward 是 16K selected operator 最重的单个 kernel，约为 dQ 的 3.73 倍、forward 的
4.09 倍；CSR 准备只占约 0.195 ms。因此下一步优化重点是被多个 query 选择的 KV block
如何聚合梯度，以及 dK/dV program 的负载均衡。

## 9. 结论与研究展望

### 9.1 主要结论

Dense、linear 和 sparse attention 的加速分别来自三种不同来源。FlashAttention 保留完整
softmax，通过 tile 与 online softmax 降低 HBM 流量；linear attention 把历史压缩为固定
state，再用 chunkwise/WY/DPLR 把递推改写为 GEMM；sparse attention 保留显式 KV，通过
block selector 减少实际配对。

实验得到四个直接结论：

1. **Chunkwise 是 Delta Rule 训练的关键执行形式。** 固定总 token 数为 16384、head dim
   为 128 时，chunkwise 前向加反向相对逐 token recurrent 的加速从 512-token 序列上的
   8.99× 增长到 16K 序列上的 40.18×。
2. **KDA 的约束 DPLR 结构提高了专用 kernel 的效率。** 在 8K–64K 上，KDA forward
   相对一般 DPLR kernel 快约 2.2×，forward+backward 快约 1.9×，同时减少长序列下的
   额外显存。
3. **Channel-wise decay 提高了关联记忆能力。** 在相同 MQAR 数据与训练设置下，KDA
   最终验证准确率达到 99.88%，GDN 为 0.52%；KDA 参数量多 5.5%，训练吞吐低约 28%。
4. **NSA 的长序列收益来自减少实际访问的 KV block。** 64K 时，给定索引的 selected
   kernel 相对 dense SDPA 的 forward 快 9.27×--13.94×，forward+backward 快
   6.91×--8.12×。计入 compression 和 top-k 后，compression+selection 路径仍分别快
   3.24×--5.46× 和
   2.57×--3.40×。

这些结果也说明 theoretical FLOPs、稀疏率和 wall-clock 各自回答不同问题。GPU 上的最终
速度由矩阵形状、tile、片上空间、索引、前反向算法和 batch 并行度共同决定。

### 9.2 结果的适用范围

本报告的实验分为三个层次。第一层比较优化 kernel 与参考实现，确认分块和融合以后，前向
输出、最终 state 和反向梯度仍保持一致。第二层在单张 A100 上测量算子延迟、额外显存和
kernel 时间构成，结果适用于本文列出的 shape、精度和 FLA 版本。第三层训练两层 GDN/KDA
模型完成 MQAR，用来观察两种更新规则在关联记忆任务上的学习差异。

DeltaNet、Kimi Linear 和 NSA 论文中的十亿级模型训练结果分别来自原论文 [7, 11, 52]。
这些论文结果说明方法可以扩展到真实语言模型；本文的实测数据则补充说明相应算子在单张
A100 上如何运行，以及速度差异来自哪些 kernel。NSA 实验采用 $D_k=D_v=128$，因为当前
A100 kernel 只支持单个不超过 128 的 key tile；其 GQA 比例、block size、selected block
数和 8K–64K 长度扫描与论文效率设置保持一致。

### 9.3 后续研究问题

1. **KDA backward：** `chunk_kda_bwd_kernel_wy_dqkg_fused` 与 intra-chunk backward
   占据主要时间，能否通过更好的 tile、recompute 策略或融合减少 HBM 往返？
2. **选择器开销：** NSA compression、top-k、index packing 各占多少时间？selected block
   数从 4/8/16/32 变化时，质量、selector 与 attention kernel 的最优点在哪里？
3. **真实 serving：** 在固定 checkpoint 和请求长度分布下，KDA state、NSA sparse KV 和
   dense paged KV 对 TTFT、TPOT、p95、显存与质量形成怎样的 Pareto 前沿？
4. **跨 GPU 泛化：** A100、H100、RTX 5090 的 SRAM、Tensor Core 和编译器不同，chunk
   size、head dim 与 crossover 是否随架构移动？
5. **模型效果：** 将 MQAR 扩展到 palindrome、stack、不同序列长度和多个学习率，并加入
   matched-parameter GDN，可以进一步定位 channel-wise decay 的收益来源。

### 9.4 文献范围

| 分组 | 文献编号 | 代表内容 |
| --- | --- | --- |
| Linear attention 基础与架构 | [1]–[12] | 核特征、fast weight、GLA、DeltaNet、SSD、GDN、KDA |
| Linear kernel、并行与量化 | [13]–[20] | Lightning Attention、LASP、tiled kernel、低比特和 state reduction |
| 经典 sparse attention | [21]–[30] | 固定稀疏、LSH、routing、低秩与稀疏结合 |
| Sparse kernel 与 serving | [31]–[38] | 稀疏硬件、动态 kernel、FlexAttention、FlashInfer、LServe |
| 长上下文 sparse attention | [39]–[58] | KV 筛选、动态 prefill、NSA、MoBA、HiLS 等 |
| 视觉与视频 sparse attention | [59]–[70] | 在线过滤、可训练稀疏、视频生成和序列并行 |
| Exact dense baseline | [71]–[74] | FlashAttention 1–4 |
| Linear attention serving 工程资料 | [75] | ReplaySSM decode 数据流 |

## 参考文献

[1] Katharopoulos A., Vyas A., Pappas N., et al. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *Proceedings of the 37th International Conference on Machine Learning*, 2020. [链接](https://proceedings.mlr.press/v119/katharopoulos20a.html)

[2] Choromanski K., Likhosherstov V., Dohan D., et al. Rethinking Attention with Performers. *International Conference on Learning Representations*, 2021. [链接](https://openreview.net/forum?id=Ua6zuk0WRH)

[3] Schlag I., Irie K., Schmidhuber J. Linear Transformers Are Secretly Fast Weight Programmers. *Proceedings of the 38th International Conference on Machine Learning*, 2021. [链接](https://proceedings.mlr.press/v139/schlag21a.html)

[4] Hua W., Dai Z., Liu H., et al. Transformer Quality in Linear Time. *Proceedings of the 39th International Conference on Machine Learning*, 2022. [链接](https://proceedings.mlr.press/v162/hua22a.html)

[5] Sun Y., Dong L., Huang S., et al. Retentive Network: A Successor to Transformer for Large Language Models. *arXiv:2307.08621*, 2023. [链接](https://arxiv.org/abs/2307.08621)

[6] Yang S., Wang B., Shen Y., et al. Gated Linear Attention Transformers with Hardware-Efficient Training. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/yang24ab.html)

[7] Yang S., Wang B., Zhang Y., et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *Advances in Neural Information Processing Systems*, 2024. [链接](https://arxiv.org/abs/2406.06484)

[8] Dao T., Gu A. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/dao24a.html)

[9] Arora S., Eyuboglu S., Zhang M., et al. Simple Linear Attention Language Models Balance the Recall--Throughput Tradeoff. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/arora24a.html)

[10] Yang S., Kautz J., Hatamizadeh A. Gated Delta Networks: Improving Mamba2 with Delta Rule. *The Thirteenth International Conference on Learning Representations*, 2025. [链接](https://openreview.net/forum?id=r8H7xhYPwz)

[11] Zhang Y., Lin Z., Yao X., et al. Kimi Linear: An Expressive, Efficient Attention Architecture. *arXiv:2510.26692*, 2025. [链接](https://arxiv.org/abs/2510.26692)

[12] Huang Y., Liu X., Huang H., et al. MDN: Parallelizing Stepwise Momentum for Delta Linear Attention. *arXiv:2605.05838*, 2026. [链接](https://arxiv.org/abs/2605.05838)

[13] Qin Z., Sun W., Li D., et al. Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/qin24c.html)

[14] Sun W., Qin Z., Li D., et al. LASP: Linear Attention Sequence Parallelism. *Transactions on Machine Learning Research*, 2025. [链接](https://openreview.net/forum?id=gG8sQUUtN7)

[15] Sun W., Lan D., Zhong Y., et al. LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid. *arXiv:2502.07563*, 2025. [链接](https://arxiv.org/abs/2502.07563)

[16] Beck M., Poppel K., Lippe P., et al. Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels. *Advances in Neural Information Processing Systems*, 2025. [链接](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cb81234ab47027e991728ed7dd76735-Abstract-Conference.html)

[17] Gerami A., Duraiswami R. Transformer Based Linear Attention with Optimized GPU Kernel Implementation. *arXiv:2510.21956*, 2025. [链接](https://arxiv.org/abs/2510.21956)

[18] Miccini R., Cerioli A., Laroche C., et al. Towards a Tailored Mixed-Precision Sub-8-Bit Quantization Scheme for Gated Recurrent Units Using Genetic Algorithms. *tinyML Research Symposium*, 2024. [链接](https://arxiv.org/abs/2402.12263)

[19] Kim H., Ko B., Kang M., et al. SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality. *The Fourteenth International Conference on Learning Representations*, 2026. [链接](https://openreview.net/forum?id=pjMDZJd4rT)

[20] Nazari P., Rusch T. K. The Key to State Reduction in Linear Attention: A Rank-Based Perspective. *arXiv:2602.04852*, 2026. [链接](https://arxiv.org/abs/2602.04852)

[21] Child R., Gray S., Radford A., et al. Generating Long Sequences with Sparse Transformers. *arXiv:1904.10509*, 2019. [链接](https://arxiv.org/abs/1904.10509)

[22] Kitaev N., Kaiser L., Levskaya A. Reformer: The Efficient Transformer. *International Conference on Learning Representations*, 2020. [链接](https://arxiv.org/abs/2001.04451)

[23] Beltagy I., Peters M. E., Cohan A. Longformer: The Long-Document Transformer. *arXiv:2004.05150*, 2020. [链接](https://arxiv.org/abs/2004.05150)

[24] Zaheer M., Guruganesh G., Dubey K. A., et al. Big Bird: Transformers for Longer Sequences. *Advances in Neural Information Processing Systems*, 2020. [链接](https://arxiv.org/abs/2007.14062)

[25] Ainslie J., Ontanon S., Alberti C., et al. ETC: Encoding Long and Structured Inputs in Transformers. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing*, 2020. [链接](https://aclanthology.org/2020.emnlp-main.19/)

[26] Tay Y., Bahri D., Yang L., et al. Sparse Sinkhorn Attention. *Proceedings of the 37th International Conference on Machine Learning*, 2020. [链接](https://proceedings.mlr.press/v119/tay20a.html)

[27] Roy A., Saffar M., Vaswani A., et al. Efficient Content-Based Sparse Attention with Routing Transformers. *Transactions of the Association for Computational Linguistics*, 2021. [链接](https://aclanthology.org/2021.tacl-1.4/)

[28] Chen B., Dao T., Winsor E., et al. Scatterbrain: Unifying Sparse and Low-Rank Attention Approximation. *Advances in Neural Information Processing Systems*, 2021. [链接](https://proceedings.neurips.cc/paper/2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html)

[29] Ding J., Ma S., Dong L., et al. LongNet: Scaling Transformers to 1,000,000,000 Tokens. *arXiv:2307.02486*, 2023. [链接](https://arxiv.org/abs/2307.02486)

[30] Han I., Jayaram R., Karbasi A., et al. HyperAttention: Long-Context Attention in Near-Linear Time. *The Twelfth International Conference on Learning Representations*, 2024. [链接](https://arxiv.org/abs/2310.05869)

[31] Wang H., Zhang Z., Han S. SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning. *IEEE International Symposium on High-Performance Computer Architecture*, 2021. [链接](https://ieeexplore.ieee.org/document/9407232/)

[32] Lu L., Jin Y., Bi H., et al. Sanger: A Co-Design Framework for Enabling Sparse Attention Using Reconfigurable Architecture. *IEEE/ACM International Symposium on Microarchitecture*, 2021. [链接](https://doi.org/10.1145/3466752.3480125)

[33] Shen G., Zhao J., Chen Q., et al. SALO: An Efficient Spatial Accelerator Enabling Hybrid Sparse Attention Mechanisms for Long Sequences. *ACM/IEEE Design Automation Conference*, 2022. [链接](https://doi.org/10.1145/3489517.3530504)

[34] Pagliardini M., Paliotta D., Jaggi M., et al. Fast Attention over Long Sequences with Dynamic Sparse Flash Attention. *Advances in Neural Information Processing Systems*, 2023. [链接](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bc222e8153a49c1b30a1b8ba96b35117-Abstract-Conference.html)

[35] Dong J., Feng B., Guessous D., et al. FlexAttention: A Programming Model for Generating Fused Attention Variants. *Proceedings of Machine Learning and Systems*, 2025. [链接](https://proceedings.mlsys.org/paper_files/paper/2025/hash/61a9278dfef5f871b5e472389f8d6fa1-Abstract-Conference.html)

[36] Ye Z., Chen L., Lai R., et al. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. *Proceedings of Machine Learning and Systems*, 2025. [链接](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)

[37] Lee W., Lee J., Seo J., et al. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. *18th USENIX Symposium on Operating Systems Design and Implementation*, 2024. [链接](https://www.usenix.org/conference/osdi24/presentation/lee)

[38] Yang S., Guo J., Tang H., et al. LServe: Efficient Long-Sequence LLM Serving with Unified Sparse Attention. *Proceedings of Machine Learning and Systems*, 2025. [链接](https://arxiv.org/abs/2502.14866)

[39] Xiao G., Tian Y., Chen B., et al. Efficient Streaming Language Models with Attention Sinks. *The Twelfth International Conference on Learning Representations*, 2024. [链接](https://arxiv.org/abs/2309.17453)

[40] Zhang Z., Sheng Y., Zhou T., et al. H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. *Advances in Neural Information Processing Systems*, 2023. [链接](https://arxiv.org/abs/2306.14048)

[41] Jiang H., Li Y., Zhang C., et al. MInference 1.0: Accelerating Pre-Filling for Long-Context LLMs via Dynamic Sparse Attention. *Advances in Neural Information Processing Systems*, 2024. [链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5dfbe6f5671e82c76841ba687a8a9ecb-Abstract-Conference.html)

[42] Tang J., Zhao Y., Zhu K., et al. QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/tang24l.html)

[43] Ribar L., Chelombiev I., Hudlass-Galley L., et al. SparQ Attention: Bandwidth-Efficient LLM Inference. *Proceedings of the 41st International Conference on Machine Learning*, 2024. [链接](https://proceedings.mlr.press/v235/ribar24a.html)

[44] Singhania P., Singh S., He S., et al. Loki: Low-Rank Keys for Efficient Sparse Attention. *Advances in Neural Information Processing Systems*, 2024. [链接](https://arxiv.org/abs/2406.02542)

[45] Chen Z., Sadhukhan R., Ye Z., et al. MagicPIG: LSH Sampling for Efficient LLM Generation. *The Thirteenth International Conference on Learning Representations*, 2025. [链接](https://openreview.net/forum?id=ALzTQUgW8a)

[46] Liu D., Chen M., Lu B., et al. RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval. *arXiv:2409.10516*, 2024. [链接](https://arxiv.org/abs/2409.10516)

[47] Zhu Q., Duan J., Chen C., et al. SampleAttention: Near-Lossless Acceleration of Long-Context LLM Inference with Adaptive Structured Sparse Attention. *arXiv:2406.15486*, 2024. [链接](https://arxiv.org/abs/2406.15486)

[48] Lou C., Jia Z., Zheng Z., et al. Sparser Is Faster and Less Is More: Efficient Sparse Attention for Long-Range Transformers. *arXiv:2406.16747*, 2024. [链接](https://arxiv.org/abs/2406.16747)

[49] Lee H., Park G., Lee Y., et al. A Training-Free Sub-Quadratic Cost Transformer Model Serving Framework with Hierarchically Pruned Attention. *The Thirteenth International Conference on Learning Representations*, 2025. [链接](https://openreview.net/forum?id=PTcMzQgKmn)

[50] Xiao G., Tang J., Zuo J., et al. DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads. *The Thirteenth International Conference on Learning Representations*, 2025. [链接](https://openreview.net/forum?id=cFu7ze7xUm)

[51] Lai X., Lu J., Luo Y., et al. FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference. *The Thirteenth International Conference on Learning Representations*, 2025. [链接](https://openreview.net/forum?id=r5GJDVJHmr)

[52] Yuan J., Gao H., Dai D., et al. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. *arXiv:2502.11089*, 2025. [链接](https://arxiv.org/abs/2502.11089)

[53] Gao Y., Zeng Z., Du D., et al. SeerAttention: Self-Distilled Attention Gating for Efficient Long-Context Prefilling. *Advances in Neural Information Processing Systems*, 2025. [链接](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50e9dbc4ab68d94f15261ddc26c8ca2b-Abstract-Conference.html)

[54] Lu E., Jiang Z., Liu J., et al. MoBA: Mixture of Block Attention for Long-Context LLMs. *arXiv:2502.13189*, 2025. [链接](https://arxiv.org/abs/2502.13189)

[55] Acharya S., Jia F., Ginsburg B. Star Attention: Efficient LLM Inference over Long Sequences. *Proceedings of the 42nd International Conference on Machine Learning*, 2025. [链接](https://proceedings.mlr.press/v267/acharya25a.html)

[56] Xu R., Xiao G., Huang H., et al. XAttention: Block Sparse Attention with Antidiagonal Scoring. *Proceedings of the 42nd International Conference on Machine Learning*, 2025. [链接](https://arxiv.org/abs/2503.16428)

[57] Deng K., Ling S., Fan R., et al. UNIQUE: Universal Top-K Sparse Attention for Training-Free Inference and Sparsity-Aware Training. *arXiv:2605.27740*, 2026. [链接](https://arxiv.org/abs/2605.27740)

[58] Hu X., Wei X., Gu H., et al. Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling. *arXiv:2607.02980*, 2026. [链接](https://arxiv.org/abs/2607.02980)

[59] Zhang J., Xiang C., Huang H., et al. SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference. *Proceedings of the 42nd International Conference on Machine Learning*, 2025. [链接](https://proceedings.mlr.press/v267/zhang25ch.html)

[60] Goncalves N., Treviso M., Martins A. F. T. AdaSplash: Adaptive Sparse Flash Attention. *Proceedings of the 42nd International Conference on Machine Learning*, 2025. [链接](https://proceedings.mlr.press/v267/goncalves25a.html)

[61] Zhang J., Jiang K., Xiang C., et al. SpargeAttention2: Trainable Sparse Attention via Hybrid Top-K+Top-P Masking and Distillation Fine-Tuning. *arXiv:2602.13515*, 2026. [链接](https://arxiv.org/abs/2602.13515)

[62] Goncalves N., Pitorro H., Niculae V., et al. AdaSplash-2: Faster Differentiable Sparse Attention. *arXiv:2604.15180*, 2026. [链接](https://arxiv.org/abs/2604.15180)

[63] Wei C., Duke B., Jiang R., et al. Sparsifiner: Learning Sparse Instance-Dependent Attention for Efficient Vision Transformers. *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023. [链接](https://doi.org/10.1109/CVPR52729.2023.02172)

[64] Liu A., Zhang Z., Li Z., et al. FPSAttention: Training-Aware FP8 and Sparsity Co-Design for Fast Video Diffusion. *arXiv:2506.04648*, 2025. [链接](https://arxiv.org/abs/2506.04648)

[65] Hu J., Gao Z., He Y., et al. DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation. *arXiv:2605.23445*, 2026. [链接](https://arxiv.org/abs/2605.23445)

[66] Tan X., Chen Y., Jiang Y., et al. DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training. *Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems*, 2026. [链接](https://arxiv.org/abs/2502.07590)

[67] Durvasula S., Sreedhar K., Moustafa Z., et al. FG-Attn: Leveraging Fine-Grained Sparse Attention in Video Diffusion Models. *arXiv:2509.16518*, 2025. [链接](https://arxiv.org/abs/2509.16518)

[68] Zhang P., Chen Y., Huang H., et al. VSA: Faster Video Diffusion with Trainable Sparse Attention. *arXiv:2505.13389*, 2025. [链接](https://arxiv.org/abs/2505.13389)

[69] Yang S., Xi H., Zhao Y., et al. Sparse VideoGen2: Accelerating Video Generation with Sparse Attention via Semantic-Aware Permutation. *arXiv:2505.18875*, 2025. [链接](https://arxiv.org/abs/2505.18875)

[70] Chen S., Hong K., Zhao T., et al. db-SP: Accelerating Sparse Attention for Visual Generative Models with Dual-Balanced Sequence Parallelism. *arXiv:2511.23113*, 2025. [链接](https://arxiv.org/abs/2511.23113)

[71] Dao T., Fu D. Y., Ermon S., et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *Advances in Neural Information Processing Systems*, 2022. [链接](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

[72] Dao T. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *The Twelfth International Conference on Learning Representations*, 2024. [链接](https://arxiv.org/abs/2307.08691)

[73] Shah J., Bikshandi G., Zhang Y., et al. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision. *Advances in Neural Information Processing Systems*, 2024. [链接](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)

[74] Zadouri T., Hoehnerbach M., Shah J., et al. FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling. *Proceedings of Machine Learning and Systems*, 2026. [链接](https://arxiv.org/abs/2603.05451)

[75] Dao T. ReplaySSM: Cache SSM Inputs, Not State. *Tri Dao's Blog*, 2026. [链接](https://tridao.me/blog/2026/replayssm/)
