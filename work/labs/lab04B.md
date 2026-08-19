workload: torch_sdpa / prefill / B=1,H=4,T=512,D=64 / BF16

已有 benchmark p50: 0.06905599683523178

我预计一次调用的 kernel 数: 16

我预计主要受限于: launch

能推翻我的观测: 不知道

## 失败记录

- job `40056`：环境报告成功后，profiler 子进程报
  `Intel oneMKL FATAL ERROR: Cannot load .../torch/lib/libtorch_cpu.so`。同一作业中前一次
  import 正常，说明更像 NFS 上共享环境的瞬时读取失败，而不是 PyTorch/CUDA 版本永久不兼容。
- job `40063`：profiler 已输出算子表和正确的 output shape，但 Kineto 无法打开
  `/tmp/sla-profile-pb25000226-40063/torch-profiler-40063.json`，随后复制 trace 失败；因此本次
  聚合表可用于初步阅读，但作业不算完整成功。
- 下一次只改动 profiler 运行基础设施：导出前重新创建 trace 父目录、验证 trace 非空，并对
  NFS/临时目录瞬时失败进行一次有边界的重试；workload、shape、dtype 和 profile steps 不变。

## 成功重跑证据

- commit：`ef743da`
- job：`40083`
- GPU：NVIDIA GeForce RTX 5090，32607 MiB；driver 580.173.02
- 环境：Python 3.12.13，PyTorch 2.11.0+cu128，PyTorch CUDA runtime 12.8
- workload：`torch_sdpa / prefill / B=1,H=4,T=512,D=64 / BF16`；warm-up 10 次，profile 5 次
- 产物：`artifacts/torch-profiler-40083.json`（90133 bytes）与聚合摘要均成功生成；job stderr 为空
- 实测 kernel：5 个 step 共 10 次 `cudaLaunchKernel`，即每个 step 2 个 CUDA kernel；原先预测的
  “一次调用 16 个 kernel”被推翻。
- 两个 CUDA kernel 分别为 split-KV 主 kernel 与 combine kernel：5 次合计 self CUDA time
  分别为 30.208 us 和 11.840 us，平均每次为 6.042 us 和 2.368 us。
- 聚合表中没有出现 H2D/D2H memcpy；出现了 allocation 相关的 `aten::empty*` 项。
- 节点没有 `nsys`；本次没有运行 `ncu`。这些工具缺失/未运行不能用来证明硬件带宽或 occupancy。
- profiler 插桩时间不能替代已有 benchmark p50；主性能数字仍使用未插桩 benchmark。

## 留给我的解释

- 这条证据是否支持“主要受 launch 限制”：
- 原假设保留、修改还是放弃，理由：
- 下一次只改变的变量：
