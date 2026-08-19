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
