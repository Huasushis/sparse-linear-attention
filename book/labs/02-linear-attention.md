# Lab 2：把 causal linear attention 看成三种同一计算

## 目标

对同一个特征映射线性注意力，比较：

1. parallel prefix-sum；
2. token-by-token recurrent state；
3. chunkwise state carry。

## 运行

```bash
python -m pytest tutorial_code/tests/test_linear_attention.py -q
```

`tutorial_code/reference/linear_attention.py` 中三个函数都应给出相同结果（浮点误差范围内）。先读 recurrent 版本：它最直观地显示 state；然后才读 parallel 版本，理解为什么 `sum(k_i v_i^T)` 可以 prefix-sum。

## 练习

在 `tutorial_code/exercises/02_linear_attention_todo.py` 中实现一次 state 更新，并回答：若把 sequence 切成两个 chunk，第二块需要从第一块带走什么？

```bash
python -m pytest tutorial_code/graders/test_exercise_linear.py -q
```

默认测试验证课程提供的三种 reference；这个 grader 才会直接检查你的 TODO。

## 通过条件

你能写出 state 和 normalizer 的形状，并解释“线性复杂度”并没有自动保证 kernel 更快。
