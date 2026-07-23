# Lab 7：从 mask 语义走到真正的 sparse operator

## 这次实验要分开的三层

同一个“滑动窗口 + global token”规则可以有三种完全不同的实现：

| 层 | 本 Lab 的对象 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| mask 语义 | `[T,T]` boolean mask + dense attention | 哪些 key 可见、输出定义是否正确 | 没有证明跳过了计算 |
| sparse reference | 逐 query 只收集选中的 K/V | 不需要构造 dense score matrix | Python loop 不能代表 GPU 性能 |
| compiled sparse kernel | FlexAttention 或后续 Triton kernel | 特定布局/shape 下是否真的更快 | 不能自动证明模型质量不变 |

本 Lab 先完成前两层。第三层要在 GPU 环境确认后进行，并继续使用同一个 mask 作为正确性
oracle。

## 1. 看见 mask

运行：

```bash
python -m tutorial_code.scripts.show_sparse_mask --length 16 --window 4 --global-token 0 --global-token 8
```

输出中的 `#` 表示需要计算的 query-key pair，`.` 表示跳过。逐行检查：

- 对角线上方必须全是 `.`，因为未来 token 不可见；
- 普通 query 能看本地窗口和已经出现的 global key；
- global query 能看它之前的全部历史，但仍不能看未来。

把这张文本图和 density 一起放进第一份 sparse 实验记录。它比只写“window=4”更容易发现
off-by-one 与 global-token 方向错误。

## 2. 验证两种实现的语义

```bash
python -m pytest tutorial_code/tests/test_sparse_attention.py -q
```

测试比较：

```text
masked dense oracle
    scores: [T,T] 全部计算，mask 后 softmax

sparse Python oracle
    每个 query 只 gather 被选中的 K/V，再计算该行 softmax
```

二者输出匹配，说明 selected-key 集合与归一化范围一致。这个测试不关心哪个更快。

## 3. 完成 TODO，并使用独立判题

编辑 `tutorial_code/exercises/03_sparse_attention_todo.py`，实现四条契约：causal、local window、
global query、global key。然后运行：

```bash
python -m pytest tutorial_code/graders/test_exercise_sparse.py -q
```

判题测试与 reference 测试故意分开。默认测试全绿只表示课程提供的 oracle 没坏；只有 grader
全绿才表示你的 TODO 已完成。

## 4. 从“正确”升级到“GPU 上跳过”

在 PyTorch 2.5+ 的 GPU 环境中，下一步可用 FlexAttention 的 `create_block_mask` 表达同一
规则。第一次实验固定 `B,H,D,dtype,T,window`，比较：

1. PyTorch SDPA 的 dense causal baseline；
2. materialized boolean mask 的 dense 实现；
3. FlexAttention block mask 实现。

记录 compile/第一次调用与 steady-state 两组时间。若只比较第一次调用，编译成本会遮住
kernel 时间；若 boolean mask 版本变慢，也不能据此说“sparse attention 更慢”，因为它仍
可能计算了 dense `QK^T`。

## 交付物与通过条件

- 一张 `T=16` 的 mask 文本图和 density；
- grader 全绿；
- masked-dense 与 sparse-loop 的最大误差；
- 一段不超过五句的解释，区分 mask、selector 和 sparse kernel；
- GPU 可用后，再补同 shape 的 steady-state 计时表。

满足前四项即可继续读第 15 章。最后一项属于 kernel 阶段，不要求在 CPU 本地环境完成。
