# 第 5 章：先建立 GPU 性能直觉

如果把 GPU 只理解成“有很多线程的 CPU”，后面读 FlashAttention 或 FLA kernel 时会
不断迷路。真正有用的问题不是“这段代码有多少行”，而是：**数据从哪里来，经过哪一
级存储，哪些执行单元同时工作，最后被什么资源卡住**。

这一章不会背某张显卡的完整参数表，也不会教 Blackwell 专属指令。目标是建立一个能从
A100 迁移到 5090、也能从 CUDA 迁移到 Triton 的性能模型。

## 学习目标

读完后，你应当能够：

1. 区分 host、device、kernel、grid、block、warp 和 thread；
2. 解释 HBM、cache、shared memory/SRAM、register 的不同作用；
3. 用算术强度判断一个算子更可能受算力还是带宽限制；
4. 解释 coalescing、tiling、reuse、occupancy 各自在解决什么问题；
5. 把 CUDA 的线程视角翻译为 Triton 的 tile/program 视角；
6. 知道 MLC GPU 教程中哪些内容现在应当学习，哪些 Blackwell 细节可以暂缓。

## 5.1 CPU 发起工作，GPU 异步执行

考虑一行 PyTorch：

```python
y = torch.relu(x @ w)
```

Python 在 CPU 上运行。矩阵 `x`、`w` 若位于 CUDA device，框架会向某条 CUDA stream
提交 GPU 工作，然后 CPU 通常立刻继续向下走。GPU 不一定已经算完。因此下面的计时
通常是错的：

```python
t0 = time.perf_counter()
y = torch.relu(x @ w)
elapsed = time.perf_counter() - t0  # 很可能主要测到 launch 时间
```

这条事实同时解释两个常见现象：

- GPU 出错可能在之后某个同步点才暴露，而不是在真正出错的那一行立即暴露；
- benchmark 必须使用 CUDA event，或在计时区间边界显式同步。

先记住一个最小数据流：

```text
CPU/Python
   |  launch kernel，参数中含 device pointer
   v
CUDA stream 中排队的 kernel
   |  从显存读取输入，在芯片上计算
   v
device memory 中的输出
```

kernel launch 本身有固定开销。一个只做几百次加法的小 kernel，即使计算“零成本”，
也可能主要耗在 launch 上。这是后面理解 kernel fusion 的第一把钥匙。

## 5.2 CUDA 的工作层级

在最基础的 CUDA 心智模型中：

```text
grid
└── thread block（许多个，调度到 SM）
    └── warp（通常每 32 个线程组成一组执行）
        └── thread
```

- **grid** 是一次 kernel launch 创建的全部工作；
- **thread block** 是能共享 shared memory、进行 block 内同步的一组线程；
- **warp** 是硬件发射指令的重要粒度；同一 warp 的线程执行同一条指令，但处理不同数据；
- **SM**（streaming multiprocessor）是承载多个 block/warp 的硬件单元。

不要把“一个 block 对应一个 SM”当成定律。一个 SM 可以同时驻留多个 block，一个
block 在其生命周期内则只在一个 SM 上执行。实际驻留数量受线程数、register 用量、
shared memory 用量和硬件上限共同约束。

### 分支为什么可能昂贵

假设一个 warp 中一半线程走 `if`，另一半走 `else`。硬件通常需要分别执行两条路径，
每次屏蔽不属于该路径的线程。这叫 divergence。关键不是“GPU 不能写 if”，而是同一
warp 内线程是否分歧。若整个 warp 都走相同分支，代价小得多。

注意：现代编译器和硬件会做谓词化等优化，不能只凭源码里出现 `if` 就断言性能差。
应当把它视为需要 profiler 验证的假设。

## 5.3 存储层级：容量越大，通常离计算越远

为了学习，可以使用下面这个简化模型：

| 层级 | 谁管理 | 特点 | 常见用途 |
| --- | --- | --- | --- |
| device DRAM（A100 常为 HBM；许多消费卡为 GDDR） | 程序分配，硬件搬运 | 容量大、片外、访问代价高 | 输入、输出、大张量 |
| L2/L1 cache | 主要由硬件 | 自动缓存，命中与访问模式有关 | 重复或邻近访问 |
| shared memory / SRAM | kernel 显式组织 | 片上、容量小、block 内共享 | tile、线程间交换 |
| register | 编译器分配 | 每线程私有、最快、最稀缺 | 标量、局部累加器 |

FlashAttention 论文把 A100 等数据中心 GPU 的片外显存写作 **HBM**。本书在讲该论文时
沿用 HBM；讨论你实际可能拿到的 5090 等设备时，更准确的泛称是 **device DRAM/global
memory**。二者在本章的性能角色相同：都远大于片上存储、但访问代价也高。

这里的名称是教学近似。不同架构的 cache/shared memory 组织不同，Triton 编译器也可能
改变值实际落在哪一级。读论文时，“SRAM”常泛指程序可利用的片上快速存储，并不一定
严格等同于 CUDA 源码中的 `__shared__` 数组。

### 带宽和延迟不是一回事

- **延迟**：一次依赖性读取从发出到数据可用需要多久；
- **带宽**：稳定流水后单位时间最多搬多少字节。

GPU 依靠同时保留许多可运行 warp，在某些 warp 等待内存时调度其他 warp，以隐藏延迟。
但隐藏延迟不会凭空增加 HBM 总带宽。一旦大量 warp 都持续搬数据，系统仍会撞到带宽
上限。

### 连续访问为什么重要

假设 32 个线程各读取一个 `float32`：

```text
好：thread 0..31 -> x[i + 0..31]      连续地址
差：thread 0..31 -> x[i + 0..31 * s]  大步长、分散地址
```

连续、对齐的访问更容易合并为较少的内存事务，称为 coalesced access。矩阵的逻辑形状
相同，不代表物理访问相同；stride 和 layout 是 kernel 性能的一部分。

## 5.4 Roofline：先问算力，还是带宽

定义算术强度（arithmetic intensity）：

$$
I = \frac{\text{执行的浮点运算数 FLOPs}}
         {\text{从目标存储层级搬运的字节数 Bytes}}.
$$

对某一层存储而言，粗略可达到的性能受下面两者中较小者限制：

$$
P_{\text{attainable}}
\leq \min(P_{\text{peak}},\; BW \times I).
$$

其中 $P_{\text{peak}}$ 是峰值计算吞吐，$BW$ 是该层级的带宽。这不是精确预测器，但很适合
在写 kernel 前排除错误方向。

### 例 1：向量加法通常受带宽限制

`c[i] = a[i] + b[i]` 对每个元素大致：

- 读 `a` 4 bytes；
- 读 `b` 4 bytes；
- 写 `c` 4 bytes；
- 做 1 次浮点加法。

忽略 cache 后，$I \approx 1/12$ FLOP/byte。这个数很低。继续优化加法指令通常没有意义，
重点应是合并访存、避免多余中间张量、把多个逐元素操作融合在同一 kernel 中。

### 例 2：分块矩阵乘能重复利用数据

对 `C[M,N] = A[M,K] @ B[K,N]`，若每个输出元素都从 HBM 重新读取一整行 `A` 和一整
列 `B`，数据搬运会非常浪费。分块后，把 `A`、`B` 的 tile 搬入片上存储，同一 tile 被
许多乘加重复使用，算术强度显著上升。这就是 tiling 的核心：

> tiling 不是为了把循环写得更复杂，而是用小容量的快速存储换取数据复用。

Attention 中的 `QK^T` 和 `PV` 也是矩阵乘。FlashAttention 的关键不是减少这两部分的
渐近 FLOPs，而是避免把巨大的分数矩阵和概率矩阵反复写回、读回 HBM。

## 5.5 四个经常混淆的优化词

### Tiling：把大问题切成能放进片上存储的小块

tile 尺寸太小，数据复用不足、launch/循环开销占比变大；太大，则可能耗尽 shared
memory/register，降低同时驻留的工作数量。tile size 是需要按 shape 和硬件搜索的参数，
不是越大越好。

### Fusion：不让中间量离开芯片

例如 `bias -> activation -> dropout` 若由三个 kernel 完成，中间张量可能两次写回 HBM，
还要付三次 launch 开销。融合后可在一个 kernel 中读取输入、连续计算、只写最终输出。

融合也不是无限好。过度融合会扩大活跃值集合，导致 register spilling，或使单个 kernel
的并行划分不适合其中某一步。

### Occupancy：有多少工作能同时驻留

occupancy 高有助于隐藏延迟，但它不是最终目标。一个复用良好、每个线程使用更多
register 的 GEMM kernel，occupancy 可能较低却更快。正确问法是：当前是否因为没有足够
可运行 warp 而暴露了延迟？而不是盲目追求 100%。

### Layout：谁连续、谁负责哪一块

layout 同时影响：

- HBM 访问是否连续；
- shared memory 是否出现 bank conflict；
- tensor core 喜欢的矩阵片段怎样分给 warp；
- reduction 是否需要跨 warp 通信。

后面阅读 FLA 时，先画输入输出和 tile 的 layout，再看每一行 Triton，通常比从函数第一
行顺读更有效。

## 5.6 从 CUDA 线程转向 Triton program

CUDA 常让你思考“线程 `i` 处理元素 `i`”。Triton 更常让一个 **program instance** 处理
一整个向量或二维 tile：

```text
CUDA:   thread id -> scalar element
Triton: program id -> block/tile -> compiler映射到线程与指令
```

例如长度为 `N` 的向量可分为大小 `BLOCK_SIZE` 的块：

```text
program 0 -> offsets [0, BLOCK_SIZE)
program 1 -> offsets [BLOCK_SIZE, 2*BLOCK_SIZE)
...
```

Triton 并没有消除 GPU 的 warp、register 或访存规则。它把线程级安排和许多指令选择交给
编译器，让学习者先把注意力放在 tile、数据流和边界 mask 上。第 9 章会写第一个 kernel。

## 5.7 怎样阅读 MLC 的现代 GPU 教程

导师给出的 [Modern GPU Programming for MLSys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/)
很有价值，但课程的最新可运行实现会使用 Blackwell 上的新机制。你的第一遍应按下表取舍：

| 现在精读：可迁移概念 | 现在只认识名字：硬件专属细节 |
| --- | --- |
| GPU 异步执行与存储层级 | Blackwell `tcgen05.mma` 的具体编码 |
| Roofline、带宽、算术强度 | TMEM 的精确使用协议 |
| coalescing、tiling、layout、fusion | TMA descriptor 的每个字段 |
| GEMM 分块和数据复用 | `mbarrier` 与特定异步 pipeline 的指令细节 |
| online softmax、causal mask、GQA | cluster launch control 与 FA-4 专属调度 |
| profiler 证据与性能假设 | 为某个 SM 版本手调的 stage 数和资源上限 |

划分依据不是“旧显卡永远用不到新思想”。恰恰相反，异步搬运、流水化、生产者-消费者
分工都是可迁移思想；只是你暂时不需要在没有相应硬件时复现其指令级实现。当前路线是：

```text
PyTorch reference -> Triton -> 阅读 CUDA/FLA -> 有明确硬件需求时再学 TileLang/TIRx
```

Triton 更适合作为第一门 GPU DSL，不代表 TVM/TIR 或 TileLang 不重要。它只是在目前目标
下提供更短的“写出 kernel -> 检查正确性 -> benchmark -> profiler”反馈回路。

## 5.8 一套性能推理模板

面对任何新 kernel，先在纸上回答：

1. 输入输出形状和 dtype 是什么？
2. 大约执行多少 FLOPs？
3. 至少读写多少 bytes？是否产生大中间量？
4. 哪些数据能在 tile 内复用？复用发生在 register 还是 shared memory？
5. 并行工作有多少？小 batch/少 head 时能否喂满 GPU？
6. 是否有 reduction、同步、分支或原子操作？
7. 由此预测 compute-bound、bandwidth-bound、latency-bound，还是 launch-bound？
8. 用什么测量能推翻自己的预测？

最后一问最重要。性能模型是待检验的解释，不是凭术语下结论。

## 常见坑

- **把显存容量当作带宽。** 80 GB 表示能放多少数据，不表示每秒能搬多少数据。
- **把峰值 TFLOP/s 当作所有算子的速度。** 峰值往往对应特定 dtype 和矩阵指令。
- **认为 FLOPs 少就一定快。** 不规则 gather、索引构造和小 kernel launch 可能吃掉收益。
- **认为 shared memory 一定更快。** 如果数据只用一次，额外搬入搬出反而可能更慢。
- **只看 occupancy 数字。** 最终应看时间和瓶颈证据。
- **把论文中的 A100 数字套到 5090。** 架构、软件栈、功耗和 shape 都不同。
- **一开始就追 Blackwell 指令。** 先掌握跨架构原理，之后学习专属机制会更快。

## 练习

### 练习 5.1：手算算术强度

对 `y = a * x + b`，假设 `a`、`b` 是标量且已在 register，`x`、`y` 为 FP32 长向量。
估算每个元素的 FLOPs、最低 HBM bytes 和算术强度。说明它更可能受什么限制。

### 练习 5.2：画数据旅程

选一个你写过的 CUDA 向量加法，画出 `host -> HBM -> register -> HBM`。在图上标出 grid、
block、warp、thread 各自负责的范围，并说明连续访问发生在哪里。

### 练习 5.3：预测再验证

从以下改动中选两个，先写下速度预测和理由，之后在 GPU 上实验：

1. 把三个逐元素 PyTorch 操作合为一个表达式；
2. 对矩阵做转置后再逐行读取；
3. 把很短的向量 batch 扩大 100 倍；
4. 把 FP32 改为 BF16。

预测必须写在结果之前，避免看到数字后编故事。

### 练习 5.4：MLC 取舍卡

浏览课程目录，任选一个 Blackwell 章节，写两列：其中三个可迁移思想，以及三个暂缓的
硬件接口。不要尝试背指令名称。

## 通过条件

只有当你能在不看本章的情况下完成下面四件事，才进入第 6 章：

- 用两分钟讲清“异步 launch 为什么会让朴素计时出错”；
- 给向量加法和分块 GEMM 各写一份 FLOPs/bytes 粗算；
- 画出 GPU 存储层级，并为每一级给出一个 attention 中的例子；
- 面对一个“新算法 FLOPs 降低 50%”的主张，主动追问访存、shape、dtype 和实际时间。
