# 论文尺度复现记录（2026-08-28）

## 1. 固定环境

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB（81151.75 MiB） |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.11.0+cu128 / 12.8 |
| Triton | 3.6.0 |
| FLA | `d1ce07369d581813553f30a750af3b6b5f9af6a9` |
| 环境 | `~/sparse_linear/.envs/sla-tutorial-py312` |

核心作业：

- `46740`：FLA 官方测试、DeltaNet Figure 1 路线、Kimi Figure 2 路线、NSA 原始维度探测；
- `46741`：两层 GDN/KDA MQAR 训练；
- `46839`：A100/FLA 支持维度的 NSA 8K--64K rerun；
- `46856`：NSA 64K，1000 ms warmup / 6500 ms repeat 跨节点确认；
- `46745`、`46842`：DeltaNet、KDA、NSA profiler；
- `46848`：KDA channel-wise chunk 前反向官方测试。

原始文件保留在 107：

```text
artifacts/paper-46740/{delta,kimi,nsa}.json
artifacts/paper-46740/test-{gdn,kda,dplr,nsa}.txt
artifacts/sla-kda-correct-46848.out
artifacts/mqar-46741/mqar.json
artifacts/nsa-46839/nsa.json
artifacts/nsa-confirm-46856.json
artifacts/profile-46745/{delta,kda}*.json
artifacts/profile-46842/{delta,kda,nsa}*.json
```

`46740` 提交时 HEAD 为 `97f750d`。作业运行期间主 worktree 后续快进到了 `562fbbf`；
`paper_operator_reproduction.py` 在这两个提交之间内容相同，Delta/Kimi 结果对应同一份
benchmark driver。NSA rerun 固定在 `59110bb`。

## 2. 方法

- dtype：BF16；gate/prefix 需要的辅助量为 FP32；
- 计时：`triton.testing.do_bench`，p20/p50/p80，warmup 25 ms，repeat 100 ms；
- forward+backward：每次重新 forward，再对全部浮点输入调用 `torch.autograd.grad`；
- 显存：输入已经常驻后，记录 `torch.cuda.max_memory_allocated` 的额外峰值；
- 输入生成、Triton 编译和 autotune 位于计时区间外。

## 3. 正确性

| 测试 | 结果 |
| --- | ---: |
| GDN chunk 16/32/64，output/state/全部梯度 | 3 passed |
| KDA channel-wise chunk，FP16/BF16、GVA、chunk 32/64、output/state/全部梯度 | 16 passed |
| DPLR chunk 多 shape，output/state/全部梯度 | 11 passed |
| NSA selected 多 shape，output/dQ/dK/dV | 6 passed |

NSA 代表性 `B=3,T=1024,Hkv=2,Hq=32,D=128` 的最大绝对差：output `0.001953`，
dK/dV `0.007812`。

## 4. DeltaNet fixed-token sweep

固定 model dim 2048 与 `B*T=16384`，head dim 为 64/128/256。下表是 head dim 128。

| T / B | recurrent fwd | chunk fwd | speedup | recurrent fwd+bwd | chunk fwd+bwd | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 32 | 4.924 | 1.259 | 3.91× | 31.200 | 3.469 | 8.99× |
| 1K / 16 | 5.303 | 1.273 | 4.17× | 33.771 | 3.488 | 9.68× |
| 2K / 8 | 6.200 | 1.304 | 4.75× | 40.568 | 3.529 | 11.50× |
| 4K / 4 | 7.786 | 1.385 | 5.62× | 46.678 | 3.759 | 12.42× |
| 8K / 2 | 14.189 | 1.337 | 10.62× | 84.280 | 3.762 | 22.40× |
| 16K / 1 | 26.629 | 1.617 | 16.47× | 173.534 | 4.319 | 40.18× |

## 5. KDA versus DPLR

`B=1,H=16,D=128`，KDA chunk 64，DPLR chunk 16。

| T | DPLR fwd | KDA fwd | speedup | DPLR fwd+bwd | KDA fwd+bwd | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0.914 | 0.711 | 1.29× | 2.517 | 3.305 | 0.76× |
| 4K | 1.784 | 0.850 | 2.10× | 4.911 | 3.511 | 1.40× |
| 8K | 3.540 | 1.633 | 2.17× | 9.759 | 5.174 | 1.89× |
| 16K | 7.039 | 3.189 | 2.21× | 19.375 | 10.275 | 1.89× |
| 32K | 14.074 | 6.381 | 2.21× | 39.048 | 20.382 | 1.92× |
| 64K | 28.452 | 12.761 | 2.23× | 78.649 | 40.838 | 1.93× |

64K 额外峰值 MiB：DPLR/KDA forward `5376/3080`，forward+backward `11648/7712`。

## 6. MQAR

- 2 layers，2 heads，head dim 128；
- T=512，128 key-value pairs，64 queries；
- batch 16，AdamW，lr `5e-4`，8000 steps；
- GDN 2.386M 参数，KDA 2.517M 参数。

| 模型 | 最终 validation loss | 最终 accuracy | throughput |
| --- | ---: | ---: | ---: |
| GDN | 5.5511 | 0.52% | 201.6K token/s |
| KDA | 0.00665 | 99.88% | 145.1K token/s |

KDA 在 step 3500/4000/5000 的验证准确率为 55.41%/97.22%/99.29%。

## 7. NSA

原论文效率设置为 `Hkv=4,Hq=64,Dk=192,Dv=128`。FLA 的 A100 path 设置 `BK<=128`
并要求 `NK==1`，因此原始 Dk=192 探测触发保护；PyTorch Flash SDPA 还要求 Q/K/V 的
最后一维相同。rerun 使用 `Dk=Dv=128`，保留 GQA、block size 64、16 blocks 与长度。

| T | dense fwd | selected fwd | + selector | dense fwd+bwd | selected fwd+bwd | + selector |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 6.566 | 4.316 | 6.751 | 26.219 | 23.291 | 51.517 |
| 16K | 21.334 | 8.888 | 15.418 | 83.081 | 46.262 | 107.059 |
| 32K | 84.316 | 18.217 | 38.055 | 323.557 | 92.911 | 227.753 |
| 64K | 343.004 | 37.001 | 105.870 | 1291.925 | 186.869 | 502.871 |

64K 额外峰值 MiB：dense `8352`，selected `3104`，selector-included `7234`。

64K 长 repeat 确认（job `46856`，A100 `anode02`）：

| mode | dense | selected | + selector | selected speedup | full speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| fwd | 326.374 | 23.417 | 59.743 | 13.94× | 5.46× |
| fwd+bwd | 1131.922 | 139.381 | 332.676 | 8.12× | 3.40× |

## 8. Profiler 摘要

- Delta 4K：state forward 2 次合计 764.9 us；`dqkwg` 565.3 us；state backward
  481.1 us；WY backward 403.9 us。
- KDA 16K：WY/dQKG fused backward 3089.1 us；intra backward 2787.1 us；inter solve
  1248.4 us；state forward 1222.1 us。
- NSA 16K：dK/dV 22314.6 us；dQ 5984.6 us；forward 5456.5 us；CSR 准备 194.6 us。
