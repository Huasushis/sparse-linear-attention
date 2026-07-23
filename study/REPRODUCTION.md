# 什么叫“复现”：把一个主张变成可检验的证据

## 复现不是从零抄论文

复现的最小单位是一个明确主张，例如：“这个算子和 reference 输出一致”、“在这些形状下该 kernel 的中位延迟较低”、“这种稀疏选择在该任务上保持了质量”。先运行作者实现并固定 commit，再写最小 reference，最后才做改动或重写；这比一开始重造完整项目可靠得多。

| 级别 | 交付物 | 适合的例子 | 现在是否需要 |
| --- | --- | --- | --- |
| L0 概念复核 | 手推、小数组、图 | mask 可见性、online softmax | 必须 |
| L1 reference operator | 简洁 PyTorch 实现 + 单元测试 | dense / linear / block mask attention | 必须 |
| L2 作者算子复跑 | 固定 commit 的测试 + microbenchmark | FLA、FlexAttention、FlashInfer | 主线 |
| L3 小范围模型/服务复现 | 小模型或已有 checkpoint 的质量与速度 | MInference/NSA/MoBA 候选 | 选 1 个 |
| L4 全量论文复现 | 原始训练、数据、规模与全套表格 | 大模型预训练 | 当前不做 |

## 推荐的复现组合

| 关卡 | 目标 | 层级 | 关键对比 | 完成边界 |
| --- | --- | --- | --- | --- |
| R0 | Dense causal attention | L1 | 自己的 FP32 reference | 小 shape 输出/梯度正确 |
| R1 | Dense backend | L2 | PyTorch SDPA、可用 FlashAttention | 同形状的延迟/显存/误差表 |
| R2 | Linear forms | L1/L2 | parallel、recurrent、chunkwise | 说明何处等价、误差来自何处 |
| R3 | FLA 的 GLA 或 Gated DeltaNet | L2 | FLA 算子 vs dense baseline | test 通过、读懂调用路径、测 forward |
| R4 | Structured sparse mask | L1/L2 | Longformer/BigBird mask + dense | mask 正确、选择开销也被计时 |
| R5 | 一个现代 sparse 系统 | L2/L3 | MInference / NSA / MoBA / SpargeAttention 四选一 | operator 或小模型结论可复跑 |

不要同时启动 R3 和 R5。每个目标先写一张[复现卡](templates/reproduction-card.md)，明确“我这次不做什么”。

## 每个实验的固定流程

1. **声明主张。** 只测一个因果问题，例如“block size 改变会如何影响 kernel latency”。
2. **冻结版本。** 记录 Git commit、作者 repo commit、CUDA/PyTorch、GPU 型号、driver、dtype。
3. **正确性先行。** exact 方法对 reference 比较误差；近似/稀疏方法报告输出误差或质量代理，不能假装完全相等。
4. **建立公平 baseline。** 相同 `B,T,H,D,dtype,causal`，相同 prefill/decode 阶段；不要拿不同 batch 或不同精度的数字相除。
5. **warm-up 和同步后计时。** 用中位数与分位数，而不是一次 `time.time()`。
6. **解释结果。** 问“瓶颈是 HBM、算力、选择/索引、launch、KV cache，还是没有落到高效 kernel？”
7. **保存最小证据。** 命令、配置、原始小表、图、解释都登记在实验记录；大日志留在集群，不进 Git。

## 正确性与效果不要混在一起

- **exact dense / FlashAttention：** 输出和梯度应在合理浮点容差内匹配 reference。
- **linear model：** 常常改变模型计算，不应要求逐 token 等于 softmax attention；检查公式、数值稳定性、训练/预训练 checkpoint 的质量或论文设定。
- **sparse / approximate method：** 同时看选择规则、稀疏率、输出偏差、下游任务质量和端到端速度。稀疏率高不是胜利，若选择阶段或不规则访存很慢，总时间仍可更差。

## 何时可以说“复现成功”

只有同时满足以下三项时，才在报告里写“在本设置下复现”：

1. scope 写得清楚（例如“单张 A100，BF16，prefill，`T=8k`，算子级”）；
2. 对照和计时方法足够公平；
3. 主张被数据支持，或者不支持时也如实记录差异与可能原因。

“作者仓库安装成功”“输出了一段文本”“屏幕上出现一个 speedup”都只是中间状态。
