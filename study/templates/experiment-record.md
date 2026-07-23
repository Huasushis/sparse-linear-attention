# 实验记录：`YYYY-MM-DD` — 简短实验名

## 主张与范围

- 想验证的主张：
- 范围：operator / model；forward / fwd+bwd / prefill / decode
- 明确不回答的问题：

## 可复跑信息

- 本仓库 commit：
- 外部仓库/模型 commit：
- GPU / driver：
- CUDA / PyTorch / Triton / Python：
- job id、命令、配置文件、随机种子：
- `B,T,H,H_kv,D,dtype,causal`：

## 正确性

- reference：
- 容差与误差（max/mean abs、relative、gradient）：
- 近似方法的质量代理（如适用）：

## 性能

- warmup / repeats / 同步 / 统计量：
- 是否计入 mask/selector/KV cache/数据搬运：

| method | p50 latency | p10/p90 | peak memory | 备注 |
| --- | ---: | ---: | ---: | --- |
| baseline |  |  |  |  |
| method |  |  |  |  |

## 观察与解释

- 数据直接说明什么：
- 可能的瓶颈：
- 反例或异常：
- 下一步只改变的一个变量：

## 产物位置

- 小型汇总表/图（提交到 Git）：
- 完整日志、profiling、权重（远程路径，不提交）：
