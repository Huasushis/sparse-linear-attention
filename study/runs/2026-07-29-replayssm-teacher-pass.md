# ReplaySSM 教师先行记录

这份记录只说明教程作者先走过了 ReplaySSM 的最小阅读与 toy reference 路线，不是作者
论文表格的独立复现。

## 资料定位

- 博客：[ReplaySSM: Cache SSM Inputs, Not State](https://tridao.me/blog/2026/replayssm/)
- 代码：[Johnny-Liou/ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM)
- 上游状态：[RFC #47572](https://github.com/vllm-project/vllm/issues/47572)、[PR #47576](https://github.com/vllm-project/vllm/pull/47576)
- 本教程提交：`e8db9ad6cf18c3a28a05e4804b5effebf349e525`

## 已走通的最小路径

1. 读博客的问题、checkpoint/ring-buffer、output-only、speculative 和 kernel design 部分；
2. 在本地用 FP64 小张量对齐 recurrent 与 output-only/flush reference；
3. 在 107 工作树 `~/sparse_linear/sla-tutorial-run` 的同一提交上运行：

   ```bash
   ~/sparse_linear/.envs/sla-tutorial-py312/bin/python -m pytest tutorial_code/tests -q
   ```

   结果：`11 passed, 4 skipped`。这是直接 SSH 会话中的 CPU/reference 回归；跳过项需要
   Slurm GPU 分配，不能据此推断 Triton kernel 性能。

## 尚未声称的结果

- 没有下载 4B/120B 权重，也没有运行完整 vLLM fork；
- 没有把博客在 H100/B300、BF16/NVFP4、CUDA Graph 下的 `1.48x`/`1.87--1.96x` 写成 107 结果；
- 没有在登录节点运行 CUDA。后续若做 L2 Triton microbenchmark，应通过 Slurm 固定 GPU、
  buffer length、warm-up、CUDA Event 和 profiler 证据。
