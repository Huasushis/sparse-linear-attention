# Lab 1：写出并验证 dense causal attention

## 目标

从 `QK^T → mask → softmax → PV` 写出一个正确的 oracle，理解它和 PyTorch 高效实现的分工。

## 先运行 reference

```bash
python -m pytest tutorial_code/tests/test_dense_attention.py -q
```

打开 `tutorial_code/reference/dense_attention.py`，跟着 `scaled_dot_product_attention` 写下每一个张量的形状。再看 `online_attention_single_query`：它不是完整 FlashAttention kernel，但精确演示了 online softmax 为什么可以分块保持正确。

## 练习

在 `tutorial_code/exercises/01_dense_attention_todo.py` 填完：

1. causal mask；
2. masked softmax；
3. score 与 `V` 的矩阵乘。

先用 `T=3, D=2` 手算一个位置，再运行自己的实现和 reference 比较。

```bash
python -m pytest tutorial_code/graders/test_exercise_dense.py -q
```

这个 grader 会直接导入你的 TODO 文件；默认 reference 测试全绿不代表练习已经完成。

## 通过条件

- 你能说明 decode 时 `T_q=1,T_k>1` 的 causal mask 为什么不能简单取左上角三角矩阵；
- 你的输出与 reference 在 `float64` 小样本上匹配；
- 你知道它很慢是正常的：它的职责是正确性，不是速度。
