# 第 9 章：第一个 Triton Kernel，从“能跑”走到“能解释”

你已经写过最简单的 CUDA，因此知道 thread id、block 和 device pointer。本章不从 CUDA
语法重新开始，而是学习 Triton 的核心视角：**一个 program instance 负责一个 tile，
你描述 tile 上的数据流，编译器把它映射到 GPU 线程与指令。**

第一个 kernel 选择向量加法。它不炫技，却能完整走通研究中最重要的闭环：数学契约、
边界、JIT、正确性、benchmark、性能模型和实验记录。之后把“向量块”换成“Q tile”，
思考方式仍然成立。

## 学习目标

读完并完成练习后，你应当能够：

1. 解释 `program_id`、`arange`、mask 和 `tl.constexpr`；
2. 写出支持任意 N 的一维 Triton kernel；
3. 使用 PyTorch reference 验证输出，而不是只打印几个元素；
4. 排除 JIT 与分配后正确 benchmark kernel；
5. 从最低字节流量估算有效带宽，并解释 block size 扫描；
6. 画出二维 attention tile 将怎样映射到 Triton program grid。

## 9.1 Triton 解决什么，不解决什么

Triton 是面向 GPU tile 的语言与编译器。对当前阶段，它比直接从手写 CUDA attention
kernel 开始更适合，因为你可以先显式控制：

- program grid 有多少 tile；
- 每个 tile 的逻辑 offsets；
- 哪些值 load/store；
- reduction 和矩阵乘发生在哪些维；
- compile-time block size 与其他配置。

编译器处理许多线程级映射、向量化和指令选择。但它不会替你解决：

- 数学定义写错；
- layout/stride 与访问模式不匹配；
- tile 太大导致 register/shared memory 压力；
- 工作量太小喂不满 GPU；
- 不公平或异步的 benchmark；
- 算法本身不适合规则 tile。

所以 Triton 是更高层的控制，不是“写 Python 就自动快”。

## 9.2 只在分配到 GPU 的作业里建立环境

不要根据登录节点猜 CUDA/GPU。进入 Slurm 分配的计算节点后记录：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())"
python -c "import triton; print(triton.__version__)"
```

环境原则：

1. 为本项目创建独立 conda 环境；
2. 先选择与节点 driver/CUDA 兼容的 PyTorch，再确认其携带或兼容的 Triton；
3. 把实际版本和安装命令写入实验记录，不把整个环境目录提交 Git；
4. JIT cache、wheel cache 和 profiler dump 放远程用户目录；
5. 不在登录节点编译、benchmark 或据其硬件信息下结论。

不要把某篇教程的精确版本号永远写死。GPU 软件栈更新快，真正需要固定的是你完成实验
时的环境文件、commit 和命令。

## 9.3 数学契约先写在 kernel 前面

目标：给定长度 N 的同 dtype、同 device、连续一维张量 x/y，计算：

$$
\forall i\in[0,N),\qquad out_i=x_i+y_i.
$$

契约还应说明：

- `out` 与输入形状相同；
- 第一版只支持 contiguous tensor；
- N 可以不是 block size 的整数倍；
- dtype 从输入继承；
- 不在第一版支持广播或原地 alias。

写清边界比急着支持所有功能更好。之后每扩展一种 stride、dtype 或 alias 情况，都增加
对应测试。

## 9.4 完整的第一个 kernel

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def add(x, y, block_size=256):
    assert x.is_cuda and y.is_cuda
    assert x.is_contiguous() and y.is_contiguous()
    assert x.shape == y.shape and x.dtype == y.dtype

    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    add_kernel[grid](
        x, y, out, n,
        BLOCK_SIZE=block_size,
    )
    return out
```

先逐行理解，不要背模板。

### `@triton.jit`

函数第一次遇到一组相关编译配置时由 Triton JIT 编译。Python 语法只是书写表面；函数中
能使用的是 Triton 支持的编译期/张量操作，不能随意调用任意 Python 或 PyTorch 函数。

### `tl.program_id(axis=0)`

一次 launch 创建许多 program instance。`pid` 表示当前 program 在第 0 个 grid 维的位置。
它类似 CUDA block id，而不是 thread id。这里每个 program 负责 `BLOCK_SIZE` 个元素。

### `tl.arange(0, BLOCK_SIZE)`

它产生一个向量化的逻辑 offset 集合，而不是 Python 循环：

```text
pid = 2, BLOCK_SIZE = 256
offsets = [512, 513, ..., 767]
```

编译器再把这些元素的工作映射到线程/warp。

### `BLOCK_SIZE: tl.constexpr`

`BLOCK_SIZE` 在编译期已知，因此可以决定 `arange` 长度、展开和资源配置。不同 block size
可能生成不同 kernel 变体，也会触发新的 JIT 编译。

### mask

若 `N=1000,BLOCK_SIZE=256`，grid 有 4 个 program，最后一个逻辑处理 768..1023。只有
768..999 有效：

```python
mask = offsets < n_elements
```

load 和 store 都使用 mask，避免越界访问。只在 load 上加 mask、store 忘记加，仍会越界。

## 9.5 第一个 grid 怎样算

$$
n_{program}=\left\lceil\frac{N}{BLOCK\_SIZE}\right\rceil.
$$

Triton 接受 tuple grid，也接受依赖 meta 参数的函数：

```python
grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
```

后者在 autotune 有多个 `BLOCK_SIZE` 候选时方便。第一版使用简单 tuple 即可。关键不在
语法，而在每个 grid 轴必须对应清楚的逻辑工作。

## 9.6 正确性：先攻击边界

```python
def check_add(n, dtype=torch.float32):
    torch.manual_seed(0)
    x = torch.randn(n, device="cuda", dtype=dtype)
    y = torch.randn(n, device="cuda", dtype=dtype)
    actual = add(x, y)
    expected = x + y
    torch.testing.assert_close(actual, expected)
```

第一轮至少测试：

```text
N = 1, 31, 32, 255, 256, 257, 1000, 4096, 1_000_003
dtype = float32, float16 或 bfloat16（以当前 GPU 支持为准）
BLOCK_SIZE = 128, 256, 512
```

这些长度覆盖空闲 lane、warp/tile 边界、整除与非整除。若只测 `N=4096`，即使完全删除
mask 也可能“正确”。

### 一个可控的失败实验

不要通过删除 `tl.store` 的 mask 来学习边界：越界写的结果不确定，也可能让后续报错远离
真正来源。改为把 `mask = offsets < n_elements` 故意写成 `offsets <= n_elements`，然后让
grader 在 `N=257` 的带 guard 区域输出上检查第 258 个位置没有被改写。这样仍能观察
off-by-one，却不会把任意地址交给 kernel 写。实验后立即恢复正确条件。

### 异步错误定位

kernel launch 后错误可能在后续同步才出现。调试时可以在测试边界同步，让报错更接近来源；
定位完成后不要把到处同步留在性能路径中。

## 9.7 Benchmark：分配不能偷偷进来

上面的 `add()` 每次都会 `torch.empty_like`。若目标是测纯 kernel，应预分配输出：

```python
n = 16 * 1024 * 1024
x = torch.randn(n, device="cuda")
y = torch.randn_like(x)
out = torch.empty_like(x)
block_size = 256
grid = (triton.cdiv(n, block_size),)

# 先调用一次，完成 JIT；再确认正确性
add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
torch.testing.assert_close(out, x + y)

ms = triton.testing.do_bench(
    lambda: add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
)
```

不同 Triton 版本的 benchmark helper 参数可能变化；运行前查看当前版本文档或 `help()`，
并在实验记录中保留版本。若手写 CUDA events，遵循第 6 章协议。

PyTorch baseline 也应避免把分配混入纯 operator 对比：

```python
torch.add(x, y, out=out_torch)
```

另外保存一个 end-to-end 表，测 `add(x,y)` 包含输出分配的真实 wrapper 成本。kernel-only 与
end-to-end 都有意义，只是回答不同问题。

## 9.8 这个 kernel 应该有多快

对 FP32 向量加：

- 读 x：4N bytes；
- 读 y：4N bytes；
- 写 out：4N bytes；
- 加法：N FLOPs。

最低流量约 $12N$ bytes，算术强度约 $1/12$ FLOP/byte，是典型 bandwidth-bound kernel。
若计时为 `ms`：

$$
BW_{effective}=\frac{12N}{ms\times10^{-3}}\ \text{bytes/s}.
$$

这里按最低逻辑流量估算；cache 和实际内存事务会影响物理流量。报告时称“有效带宽”，
不要冒充硬件精确 DRAM 带宽。

### 小 N 为什么完全不同

当 N 很小时，数据搬运量很少，固定 launch/JIT/dispatch 成本占主导。曲线常有三个区域：

```text
很小 N：launch-bound
中等 N：并行度逐步增加
大 N：接近 bandwidth-bound 稳态
```

因此只报一个大 N 会漏掉调用实际小张量时的表现；只报小 N 又无法评价访存效率。

## 9.9 扫 block size，但不要把搜索当理解

测试 `BLOCK_SIZE=64,128,256,512,1024`（具体合法性以版本/目标为准），每个配置第一次编译
后再计时。先预测：

- 太小：program 太多，管理开销增加，每 program 工作少；
- 适中：并行量和连续访存平衡；
- 太大：可能增加 register 压力，且对简单算子未必有额外收益。

向量加的最佳 block 不一定能迁移到 softmax 或 attention，因为后两者有 reduction、矩阵
乘和更大的活跃状态。autotune 能找较快配置，但研究报告仍要解释候选空间和 shape 依赖。

## 9.10 第一次 fusion：把中间张量留在 program 内

将数学契约改为：

$$
out_i=\max(0,a x_i+b).
$$

在同一个 program 中依次执行 load、乘加、maximum、store，中间 `a*x+b` 不写回 HBM。
对比 eager PyTorch 的多个逐元素操作时，要明确框架是否自动融合；若使用 `torch.compile`，
它可能同样生成 fused kernel，baseline 就不再是“三次独立往返”。

这个练习比向量加更接近 attention 的思想：收益常来自避免中间 materialization，而不是
让某次加法本身更快。

## 9.11 从一维 block 走向二维 tile

一个简化 attention forward grid 可以想成：

```text
axis 0: 第几个 Q tile
axis 1: 第几个 batch-head
```

伪代码：

```python
pid_q = tl.program_id(0)
pid_bh = tl.program_id(1)

q_rows = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
d_cols = tl.arange(0, BLOCK_D)

# load Q tile [BLOCK_M, BLOCK_D]
# loop over K/V tiles [BLOCK_N, BLOCK_D]
# tl.dot(Q_tile, transposed K_tile)
# row-wise max/sum and online state update
# store O tile
```

这时必须同时处理：

- 二维 pointer arithmetic 与 stride；
- Q/K/V 的不同 tile 形状；
- `tl.dot` 适合的 dtype/layout；
- 行 reduction；
- causal 和边界 mask；
- `(m,l,u)` 状态的 register/片上压力；
- grid 是否有足够 program。

不要从向量加直接跳到完整 FlashAttention。合理阶梯是：

```text
vector add
-> fused pointwise
-> row reduction / stable softmax
-> tiled matmul
-> online softmax over two blocks
-> 教学 attention forward
-> 阅读成熟实现
```

每一步都保留独立 reference 和 benchmark。这样出错时知道是哪一种新机制造成的。

## 9.12 怎样阅读 FLA 中的 Triton

面对一个长 kernel，不要逐行翻译语法。先做一张 dataflow 卡：

```text
operator:
inputs/outputs + shapes:
program grid axes:
tile sizes:
state carried across chunks:
HBM reads/writes:
on-chip temporaries:
forward/backward variants:
boundary/causal handling:
autotune keys:
```

再沿着一条调用链读：layer/API -> operator dispatch -> 一个 kernel variant -> correctness test
-> benchmark。看到 `tl.load` 就问加载的逻辑 tile；看到 `tl.dot` 就标注两个维度；看到
`tl.sum/tl.max` 就标注 reduction axis。目标不是记住每个装饰器参数，而是重建数据流。

## 9.13 Triton 与 CUDA 的对应关系

| CUDA 直觉 | Triton 直觉 |
| --- | --- |
| thread/block id | program id + tile offsets |
| 每线程一个标量 | 一个 program 操作一组张量值 |
| 手工 shared-memory tile | 用 block pointer/tile 描述数据，编译器参与放置 |
| `__syncthreads()` | 许多同步由编译模型隐藏，但跨 program 仍不能随意共享 |
| 模板常量 | `tl.constexpr` meta-parameter |
| grid/block dimensions | program grid + `num_warps/num_stages` 等配置 |

这是概念映射，不是逐项等价。Triton program 不能被简单理解成“一块 CUDA shared
memory”；实际资源分配应看编译结果和 profiler。

完成本章后进入 [Lab 5：Triton 起步](../labs/05-triton.md)。课程仓库提供可运行 reference、
只留一处 TODO 的 kernel 和 GPU grader；不需要从空文件开始。

## 常见坑

- 在 CPU tensor 上 launch kernel；
- 假设输入 contiguous，却未 assert 或传 stride；
- 只给 load 加 mask，store 越界；
- 用错误 `axis` 的 program id；
- 把 runtime N 标成 `tl.constexpr`，为每个长度重复编译；
- 把 BLOCK_SIZE 当成 CUDA thread 数逐字对应；
- benchmark 包含 JIT、输入生成或输出分配，却称 kernel latency；
- 每轮创建新 tensor，allocator 噪声掩盖 kernel；
- 只测试整 block N；
- 看到 Triton 比 PyTorch 慢就认定失败，却没检查 PyTorch 是否调用了高度优化/融合 backend；
- 直接优化一行汇编，而没有先确认 bottleneck；
- autotune 使用测试集形状并只报告最有利点，却不记录候选配置和搜索成本。

## 练习

### 练习 9.1：补全骨架

把完整代码中的 `offsets`、`mask`、两次 load 和 store 暂时替换为 `TODO`，在不看本章的
情况下补回。用 `N=1,257,1_000_003` 验证。

### 练习 9.2：Block size 曲线

对小、中、大三个 N 扫至少四个 block size。运行前写预测，结果表包含 median latency 和
有效带宽。解释最佳配置为什么随 N 变化或不变化。

### 练习 9.3：Fusion

实现 `relu(a*x+b)` fused kernel，与数学上相同的 PyTorch reference 比正确性。分别比较：

1. eager 多算子；
2. Triton fused；
3. 若环境允许，`torch.compile` 后的版本。

使用 profiler 或 trace 确认实际 kernel 数，再解释结果。

### 练习 9.4：支持 stride

扩展契约以支持一维 strided view。显式传 `stride_x/stride_y/stride_out`，构造非连续输入
测试。比较连续与跨步访问的有效性能，并用 coalescing 解释。

### 练习 9.5：两块 online softmax

先不写 attention。每个 program 负责一行，把该行拆成两个 block，维护 `(m,l)` 计算 stable
softmax。与 `torch.softmax` 比较，加入“第二块有极大值”的测试。

### 练习 9.6：画 attention grid

对 `B=2,H=4,T=1024,D=64,BLOCK_M=64,BLOCK_N=64`，计算二维 grid 的两个轴大小。再讨论
causal 时第 0、7、15 个 query program 分别循环多少个 KV tile；不要写完整 kernel，只画
职责与状态。

## 通过条件

完成本章时，你应提交一份小型 lab 记录，且能现场完成：

- 从空白写出支持非整除 N 的 vector add kernel；
- 用至少 9 个边界长度和两种 dtype 验证；
- 分开报告 kernel-only 与 wrapper end-to-end 时间；
- 用 $12N/t$ 计算 FP32 向量加的有效带宽，并说明其假设；
- 给 block size 曲线提出并检验一个解释；
- 画出二维 attention program grid 和 Q tile 的 `(m,l,u)` 生命周期；
- 明确下一步只新增一种复杂性：row reduction，而不是直接复制完整 FlashAttention。

## 延伸阅读

- [Triton 官方 Vector Addition 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton 官方 Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton 官方 Matrix Multiplication 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton 官方 Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
