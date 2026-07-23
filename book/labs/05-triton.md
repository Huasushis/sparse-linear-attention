# Lab 5：Triton 起步（在有匹配 CUDA 环境后）

## 目标

先在本仓库完成一个有边界测试的 vector add，再通过 Triton 的 softmax、matmul 教程建立
block/program/layout 的感觉，最后接触 attention kernel。

!!! note "这不是当前的阻塞条件"
    你的本地机器没有 CUDA PyTorch，107 上的 GPU 又可能不同。先完成 Lab 1--4；在 GPU job
    内确认版本并完成 `cluster/setup-env.sbatch`。CPU 环境中本 Lab 显示 skip 是预期结果。

## 1. 先跑课程 reference

```bash
python -m pytest tutorial_code/tests/test_triton_vector_add.py -q
```

打开 `tutorial_code/kernels/vector_add_triton.py`，把代码逐行标成四类：program id、offset、
masked load/store、Python wrapper contract。测试故意包含 `N=257` 与 `N=4099`，它们不能被
block size 整除，因而会检查尾块 mask。

## 2. 填一行 TODO，再跑独立 grader

编辑 `tutorial_code/exercises/04_triton_vector_add_todo.py`。你只需补 store；grid、offset、
load 和 wrapper 已经提供：

```bash
python -m pytest tutorial_code/graders/test_exercise_triton.py -q
```

不要通过把 output 改成 `x + y` 绕过 kernel；grader 的意义是让你第一次亲手把一个 Triton
program 写完整，而不是考 Python 加法。

## 3. 再读官方递进教程

依次阅读 [Triton 官方 vector add、fused softmax、matmul](https://triton-lang.org/main/getting-started/tutorials/index.html)。
不要求复制全部代码；每篇只画一个 program 负责的输出 tile，并回答下面三个问题。

## 读代码时的三个问题

1. 一个 program 负责输出的哪一块？
2. 哪些元素连续读取，哪些需要 reduction？
3. 这个 tile 的工作是算力受限还是带宽受限？

完成官方 softmax 后，回到第 8 章：你会更容易理解为什么 FlashAttention 要在片上维护 `m`、`l`、accumulator。

## 通过条件

- reference 与 grader 都通过；
- 能解释 `program_id * BLOCK_SIZE + arange`；
- 能说明 load 与 store 为什么都需要 tail mask；
- 能指出 vector add 主要受显存带宽限制，而 attention 还包含 reduction、矩阵乘与片上状态。
