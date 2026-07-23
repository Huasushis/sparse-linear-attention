# 研究路线与交付物

书的章节负责教会你概念和工具；研究路线负责避免你在 74 篇论文与多个仓库之间迷路。
[可勾选任务表](task-board.md)由仓库中的 `study/TASKS.md` 单一来源嵌入本站；这里给出与书的
对应关系。

| 阶段 | 读书关卡 | 需要交出的最小产物 |
| --- | --- | --- |
| P0 | 使用本书、第 1 章 | 一页 Transformer 形状图和第一张论文定位笔记 |
| P1 | 第 2--6 章 | 可复跑环境记录、训练/推理阶段说明 |
| P2 | 第 7--8 章、Lab 1/4 | dense reference + SDPA/Flash baseline 表 |
| P3/P4 | 第 9--13 章、Lab 2/3/6 | linear operator 对照、FLA 调用路径图 |
| P5/P6 | 第 14--16 章、Lab 7 | structured sparse reference + 一个现代方法复现卡 |
| P7 | 全部 | 受配置和结论边界约束的调研/复现报告 |

## 深读池

当前主线分为 8 篇 A0 必读与 7 篇 A1 条件精读。A1 不是立即开始的作业：

1. FlashAttention、FlashAttention-2；
2. Transformers are RNNs；
3. GLA、DeltaNet、Gated DeltaNet；
4. Kimi Linear；
5. MInference；
6. 进入具体方向后，再从 Mamba-2/SSD、Tiled FLA、FlexAttention、FlashInfer、NSA、MoBA、
   SpargeAttention 中选择直接相关的 A1；NSA/MoBA/Sparge 三者只选一个。

每次只同时保留一个“读论文”目标和一个“小代码/Lab”目标。若一个术语卡住，回到本书前置章节，不要立刻新增另一篇论文。
