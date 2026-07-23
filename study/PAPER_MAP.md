# 74 篇论文阅读图：分层，而非平均用力

## 如何使用本表

- **A0（当前必读）**：读公式/算法、实现和实验；写完整论文笔记。A0 不等于完整重训。
- **A1（选题后升级精读）**：先按 B 级速度建立位置；确认与复现目标直接相关后再精读。
- **B（重点略读）**：摘要、引言、核心图/算法、实验设定、结论；写半页比较笔记。
- **C（脉络浏览）**：摘要与一张代表图；记录它属于哪个篮子、解决什么问题。
- 标签：`ALG` 算法，`KER` kernel，`SYS` serving/系统，`TRAIN` 训练/架构，`DIST` 分布式，`QUANT` 量化，`VISION` 视觉/视频，`BASE` dense 基线。

阅读顺序由[总路线](https://github.com/Huasushis/sparse-linear-attention/blob/main/study/ROADMAP.md)决定，不由表中编号决定。共有 **8 篇 A0、7 篇 A1、26 篇 B、33 篇 C**；这是为了避免初期同时啃 15 篇精读论文。

## 1. Linear attention：基础与现代架构（12 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `katharopoulos2020transformers` | Transformers are RNNs | A0 | ALG | kernel feature、recurrent state、causal 计算 |
| `choromanski2021rethinking` | Performer | B | ALG | 随机特征近似与误差/复杂度权衡 |
| `schlag2021linear` | Fast Weight Programmers | B | ALG | fast-weight 视角与 state 更新 |
| `hua2022transformer` | cosFormer | C | ALG | 线性注意力的另一种特征构造 |
| `sun2023retentive` | RetNet | B | ALG, TRAIN | parallel/recurrent/chunkwise 三种表示 |
| `yang2024gated` | GLA | A0 | ALG, KER | gating、chunkwise algorithm、硬件实验 |
| `yang2024delta` | DeltaNet | A0 | ALG, KER | delta rule 与 sequence-length parallelism |
| `dao2024transformers` | Mamba-2 / SSD | A1 | ALG, TRAIN | 理解 GDN 后回补 SSD 对偶、分块 scan 与 attention 的关系 |
| `arora2024simple` | Based | B | ALG, TRAIN | recall–throughput 取舍 |
| `yang2025gated` | Gated DeltaNet | A0 | ALG, KER | GLA + delta rule 的组合与 ablation |
| `kimi2025linear` | Kimi Linear | A0 | ALG, KER, SYS | 真实 LLM 架构、KDA、实验范围 |
| `huang2026mdn` | MDN | C | ALG | momentum / delta 的后续扩展 |

## 2. Linear attention：kernel、并行、量化（8 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `qin2024various` | Lightning Attention | B | ALG, KER | 常速与不同长度、分块实现思路 |
| `sun2025lasp` | LASP | C | DIST, KER | 线性注意力为何需要序列并行 |
| `sun2025lasp2` | LASP-2 | C | DIST, KER | hybrid 模型的并行边界 |
| `beck2025tiled` | Tiled Flash Linear Attention | A1 | KER | 选择 linear kernel 方向后精读 tile/layout 与实现证据 |
| `gerami2025transformer` | Optimized GPU Kernel | C | KER | 作为补充实现案例 |
| `miccini2024towards` | GRU sub-8-bit | C | QUANT | 知道量化问题存在即可 |
| `kim2026ssdi8` | SSDi8 | C | QUANT | state-space 量化接口 |
| `nazari2026key` | State Reduction | C | ALG, QUANT | rank/state 压缩的后续方向 |

## 3. Sparse attention：经典算法脉络（10 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `child2019sparse` | Sparse Transformer | B | ALG | strided/local 模式的起点 |
| `kitaev2020reformer` | Reformer | B | ALG | LSH 路由与 reversible layer 的边界 |
| `beltagy2020longformer` | Longformer | B | ALG, TRAIN | sliding window + global token；做 toy mask |
| `zaheer2020bigbird` | BigBird | B | ALG, TRAIN | random/global/window 与理论主张 |
| `ainslie2020etc` | ETC | C | ALG | global-local 的结构化输入 |
| `tay2020sparse` | Sparse Sinkhorn | C | ALG | 可学习排序/稀疏 |
| `roy2021routing` | Routing Transformer | C | ALG | content routing 开销 |
| `chen2021scatterbrain` | Scatterbrain | B | ALG | sparse + low-rank 的组合位置 |
| `ding2023longnet` | LongNet | C | ALG, TRAIN | dilated attention 的直观与长度实验 |
| `han2024hyperattention` | HyperAttention | C | ALG | 近线性理论方法的结论边界 |

## 4. Sparse attention：kernel 与 serving（8 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `wang2021spatten` | SpAtten | C | KER, SYS | pruning / hardware co-design 历史 |
| `lu2021sanger` | Sanger | C | KER | reconfigurable hardware 的思路 |
| `shen2022salo` | SALO | C | KER | accelerator 路线，非当前复现目标 |
| `pagliardini2023dynamic` | Dynamic Sparse Flash Attention | B | ALG, KER | dynamic mask 如何接入 Flash 计算 |
| `dong2025flexattention` | FlexAttention | A1 | KER, SYS | 选择可编程 sparse kernel 方向后精读接口与 lowering |
| `ye2025flashinfer` | FlashInfer | A1 | KER, SYS | 选择 serving 方向后精读 attention engine 与 benchmark 语境 |
| `lee2024infinigen` | InfiniGen | C | SYS | dynamic KV cache management |
| `yang2025lserve` | LServe | B | SYS, KER | unified sparse attention 的 serving 视角 |

## 5. Sparse attention：长上下文 LLM（20 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `xiao2024streamingllm` | StreamingLLM | B | SYS | attention sink、streaming decode |
| `zhang2023h2o` | H2O | B | SYS, ALG | heavy-hitter KV eviction |
| `jiang2024minference` | MInference | A0 | ALG, KER, SYS | prefill 动态稀疏、选择与 kernel 的关系 |
| `tang2024quest` | QUEST | B | ALG, SYS | query-aware selection 的 proxy/开销 |
| `ribar2024sparq` | SparQ | B | ALG, SYS | bandwidth 而非 FLOPs 为中心的设计 |
| `singhania2024loki` | Loki | B | ALG | low-rank key 作为 sparse selector |
| `chen2025magicpig` | MagicPIG | C | ALG | LSH sampling 的现代案例 |
| `liu2024retrievalattention` | RetrievalAttention | C | SYS | vector retrieval 路线 |
| `zhu2024sampleattention` | SampleAttention | C | ALG | adaptive structured sparse |
| `lou2024sparsek` | SparseK | B | ALG | 稀疏率、选择规则与实际速度的关系 |
| `lee2025hip` | HiP | C | SYS, ALG | hierarchical pruning |
| `xiao2025duoattention` | DuoAttention | B | SYS | retrieval / streaming head 分类 |
| `lai2025flexprefill` | FlexPrefill | B | ALG, SYS | context-aware prefill 选择 |
| `yuan2025native` | Native Sparse Attention | A1 | ALG, KER, TRAIN | 与 MoBA/Sparge 三选一升级精读；hardware-aligned block sparse、可训练性 |
| `hu2026hils` | HiLS-Attention | B | ALG, KER, SYS, TRAIN | 分层 chunk mass、端到端 selector 与相邻 query packing；先做七问卡，独立跑通算子后再考虑升 A1 |
| `gao2025seerattention` | SeerAttention | B | ALG, TRAIN | self-distilled attention gating |
| `lu2025moba` | MoBA | A1 | ALG, KER, TRAIN | 与 NSA/Sparge 三选一升级精读；FLA 有实现接口 |
| `acharya2025starattention` | Star Attention | B | SYS | 分块/分布式推理思路 |
| `xu2025xattention` | XAttention | B | ALG, KER | antidiagonal block scoring |
| `deng2026unique` | UNIQUE | C | ALG, TRAIN | Top-K 与训练感知稀疏的后续工作 |

## 6. Sparse attention：通用 kernel 与视觉支线（12 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `zhang2025spargeattention` | SpargeAttention | A1 | ALG, KER, SYS | 与 NSA/MoBA 三选一升级精读；training-free pipeline |
| `goncalves2025adasplash` | AdaSplash | B | ALG, KER | adaptive sparse Flash attention |
| `zhang2026spargeattention2` | SpargeAttention2 | C | ALG, TRAIN | training-aware 后续版本 |
| `goncalves2026adasplash2` | AdaSplash-2 | C | ALG, KER | 可微 sparse kernel 后续版本 |
| `wei2023sparsifiner` | Sparsifiner | C | VISION, ALG | ViT 的 instance-dependent sparse |
| `liu2025fpsattention` | FPSAttention | C | VISION, QUANT | video diffusion + FP8 |
| `hu2026dfsattn` | DFSAttn | C | VISION, ALG | dynamic fine-grained sparse |
| `tan2026dsv` | DSV | C | VISION, KER | video training dynamic sparsity |
| `durvasula2025fgattn` | FG-Attn | C | VISION, ALG | fine-grained sparse video |
| `zhang2025vsa` | VSA | C | VISION, TRAIN | 可训练 sparse video attention |
| `yang2025sparsevideogen2` | Sparse VideoGen2 | C | VISION, ALG | permutation / semantic-aware sparse |
| `chen2025dbsp` | db-SP | C | VISION, DIST | video sparse sequence parallelism |

## 7. Exact dense-attention 基线（4 篇）

| Key | 简称 | 层级 | 标签 | 本阶段要看什么 |
| --- | --- | --- | --- | --- |
| `dao2022flashattention` | FlashAttention | A0 | BASE, KER | IO-aware exact attention、online softmax |
| `dao2024flashattention2` | FlashAttention-2 | A0 | BASE, KER | work partition、parallelism、benchmark fairness |
| `shah2024flashattention3` | FlashAttention-3 | B | BASE, KER | asynchrony、low precision；注意硬件前提 |
| `zadouri2026flashattention4` | FlashAttention-4 | C | BASE, KER | Blackwell 特化；读思想，不把运行作为门槛 |

## 核心复现池（不是全部 A0/A1）

按门槛从低到高选择：

1. dense reference + PyTorch SDPA/FlashAttention benchmark；
2. linear attention 的 parallel / recurrent / chunkwise toy operator；
3. FLA 中 GLA 或 Gated DeltaNet 的已实现算子；
4. Longformer/BigBird 风格 structured mask + FlexAttention；
5. MInference、Native Sparse Attention、MoBA、SpargeAttention 中根据代码、GPU 和模型权重可得性选 **一个** 做小范围 serving / operator 复现。

Kimi Linear 是重点案例，但初期以读架构、查看 FLA 实现、做算子级验证为目标；完整训练或最大模型 benchmark 不作为本科入门阶段的完成条件。

HiLS-Attention 是 2026-07 的近期观察项：论文与训练/算子/serving 代码都值得跟踪，但发布尚新、完整环境较重，且仓库当前未声明许可证。它暂不进入核心复现池；只有在固定 commit 的独立环境中跑通 selector、forward/backward correctness 和小型 benchmark 后，才从 B 升为 A1。
