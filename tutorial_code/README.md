# 教学代码

这不是 FLA 的替代实现。`reference/` 选择可读性和可测性，故意保留 dense score matrix 或 Python loop；它们是理解论文、验证 kernel 的 oracle。

```bash
python -m pytest tutorial_code/tests -q
python -m tutorial_code.scripts.gdn_kda_probe
python -m tutorial_code.scripts.show_sparse_mask --length 16 --window 4 --global-token 0
python -m tutorial_code.benchmarks.benchmark_attention --device auto --seq-len 256
```

默认的 `tests/` 只检查课程 oracle。完成 `exercises/` 的 TODO 后，使用独立 grader：

```bash
python -m pytest tutorial_code/graders/test_exercise_dense.py -q
python -m pytest tutorial_code/graders/test_exercise_linear.py -q
python -m pytest tutorial_code/graders/test_exercise_sparse.py -q
```

TODO 尚未填写时，grader 失败是预期行为。不要为了“更快”改掉 reference；性能版本应单独
实现并明确测量范围。
