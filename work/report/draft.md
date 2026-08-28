# Sparse 与 Linear Attention 调研及复现报告（草稿）

> 从 P1 起逐节填写。没有证据的结论先写成问题，不要补成肯定句。
>
> 本文件保留学习过程中的原始记号与推导，未统一 state 转置约定。已校订、带引用和实测数据的
> 提交版本见 [`final.md`](final.md)；正式引用公式和结论时以该文件为准。

## 1. 背景与研究问题

## 2. Dense attention 与性能基线

## 3. Linear attention：算法分类与原理

原始的 attention 是 $\mathrm{O}(n^2)$ 的，

$$
y_i
=
\frac{
\sum_{j=1}^{n}
\exp(q_i^\top k_j)\,v_j
}{
\sum_{j=1}^{n}
\exp(q_i^\top k_j)
}
$$

暂时忽略掉 $1/\sqrt{d_k}$。

其中未归一化的相似度核是 $\kappa(q,k) = \exp(q^\top k)$，每个 query 都要和所有 key 比较因此会要 $\mathrm O(n^2)$

假设存在一个特征映射 $$\phi: \mathbb{R}^d \rightarrow \mathbb{R}^m$$ 满足 $$\kappa(q,k) \approx \phi(q)^\top \phi(k)$$ 代回得到：

$$\sum_j \phi(q_i)^\top \phi(k_j)\,v_j$$

则令

$$
S = \sum_j \phi(k_j)\,v_j^\top \in \mathbb{R}^{m\times d_v} \\
z = \sum_j \phi(k_j) \in \mathbb{R}^{m}
$$

则

$$y_i \approx \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}$$

复杂度从 $\mathrm O(n^2 d)$ 降低到 $\mathrm O(n m d)$。

在 Causal Attention 中，可以变为 recurrent state，$S_i = S_{i-1} + \phi(k_i)v_i^\top$，$z_i = z_{i-1} + \phi(k_i)$，则 $y_i = \frac{\phi(q_i)^\top S_i}{\phi(q_i)^\top z_i}$。

缺点是用矩阵秩来看，$n$ 个 query 和 $n$ 个 key 的相似度矩阵是 $n\times n$ 的，rank 最多为 $n$，而 $\phi(q_i)^\top \phi(k_j)$ 的 rank 最多为 $m$，因此如果 $m<n$，则会损失表达能力。

softmax 确实可以映射映射，但是无限维：
$$
\exp(q^\top k) = \sum_{i=0}^{\infty} \frac{(q^\top k)^i}{i!}\\
(q^\top k)^i = \langle q^{\otimes i}, k^{\otimes i} \rangle \\
\phi(q) = \bigoplus_{i=0}^{\infty} \frac{q^{\otimes i}}{\sqrt{i!}} \\
$$

可以取有限项的近似近似，但是显式张量维度会随着 $i$ 的增加而指数增长，因此需要用随机特征映射（Random Feature Map）来近似。

令 $\omega \sim \mathcal{N}(0, I)$，定义正特征 $\phi_{\omega}(x) = \exp \left( \omega^\top x -\frac{||x||^2}{2} \right)$，那么：
$$
\mathbb{E}_{\omega}[\phi_{\omega}(q)\phi_{\omega}(k)] = \exp \left( q^\top k \right)。
$$

于是随机取 $m$ 个 $\omega_i$，定义 $\phi(q) = \frac{1}{\sqrt{m}}[\phi_{\omega_1}(q), \ldots, \phi_{\omega_m}(q)]^\top$，则有：
$$
\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)。
$$

\[
\boxed{\;\mathrm{Var}\big(\hat{K}_m(q,k)\big)
=\frac{1}{m}\,e^{2q^\top k}\big(e^{\|q+k\|^2}-1\big)\; }.
\]

还有一些方法重新定义了 kernel，或者用不同的特征映射 $\phi$，比如 $ELU(x)+1$。

Delta Rule:

$$S_t = S_{t-1} - \beta_t(S_{t-1}k_t - v_t)k_t^\top$$

实际上就是梯度下降：
$$L_t(S) = \frac12 ||S k_t - v_t||^2$$

对 $S$ 求梯度，得到 $\nabla L_t(S) = (S k_t - v_t)k_t^\top$，然后用学习率 $\beta_t$ 更新。

## 4. Linear attention：kernel 与实现

### Chunk-wise 线性注意力/状态更新矩阵化总结

核心思想：将长度为 $C$（如 $C=64$）的 Chunk 内逐 Token 串行计算，转换为 GPU 高效的矩阵乘法（GEMM）。

---

#### 1. 状态更新矩阵化 ($S_{\text{next}}$)

* **优化目标**：替代 $C$ 次串行外积与状态更新。
* **计算公式**：
  $$S_{\text{next}} = S_{\text{in}} + V^\top K$$
* **维度变化 (GEMM)**：
  $$[d, C] \times [C, d] \to [d, d]$$

---

#### 2. Chunk 输出并行计算 ($O$)

Chunk 内所有 Token 的输出 $O$ 由**历史状态投影**与**Chunk 内因果注意力**两部分叠加而成：

$$O = \underbrace{Q S_{\text{in}}^\top}_{\text{历史状态（Inter-chunk）}} + \underbrace{\left( (Q K^\top) \odot M_C \right) V}_{\text{局部因果注意力（Intra-chunk）}}$$

##### 计算拆解：
1. **历史状态投影**：$C$ 个 Query 一次性作用于历史状态 $S_{\text{in}}$
   * **GEMM 维度**：$[C, d] \times [d, d] \to [C, d]$
2. **局部因果注意力**：
   * **注意力得分矩阵**：$Q K^\top$（维度：$[C, d] \times [d, C] \to [C, C]$）
   * **施加因果掩码**：$A_{\text{chunk}} = Q K^\top \odot M_C$
   * **加权聚合 V**：$A_{\text{chunk}} V$（维度：$[C, C] \times [C, d] \to [C, d]$）

### DeltaNet Chunk-wise 矩阵化与并行化总结

核心难点：Delta Rule 中的更新项 $v_t - S_{t-1}k_t$ 强依赖前一时刻状态 $S_{t-1}$，无法直接并行。
解决思路：利用 **WY 表示法（Generalized Householder）** 重参数化 Chunk 内递推，构造 $C \times C$ 的下三角矩阵 $T$ 解耦时序依赖，全面转化为 GEMM 矩阵计算。

---

#### 1. 解耦递推：构造伪键值矩阵 ($T, W, U$)

针对大小为 $C$（如 $C=64$）的 Chunk，计算下三角求解矩阵 $T$ 及伪键值 (Pseudo Keys/Values)：

$$T = \left( I + \text{tril}(\text{diag}(\beta) K K^\top, -1) \right)^{-1} \text{diag}(\beta) \quad \in \mathbb{R}^{C \times C}$$

$$W = T K, \quad U = T V \quad (K, V, U, W \in \mathbb{R}^{C \times d})$$

---

#### 2. Delta Chunk 核心计算公式

引入结合历史状态 $S_{\text{in}}$ 的修正 Value 矩阵 $G$：

$$G = U - W S_{\text{in}}^\top \quad (\text{shape: } [C, d] - [C, d][d, d] \to [C, d])$$

##### ① 状态更新方程 ($S_{\text{out}}$)
$$S_{\text{out}} = S_{\text{in}} + G^\top K \quad (\text{shape: } [d, d] + [d, C][C, d] \to [d, d])$$

##### ② Chunk 输出方程 ($O$)
$$O = \underbrace{Q S_{\text{in}}^\top}_{\text{历史状态投影}} + \underbrace{\left( (Q K^\top) \odot M_C \right) G}_{\text{Chunk 内因果加权}}$$

---

#### 3. GEMM 算子分解汇总

整套 ChunkWise 计算全部转化为 GPU 高效的 GEMM 乘法：

1. `W @ S_in.T` $\to$ 生成修正项矩阵 $G$
2. `G.T @ K` $\to$ 状态更新 $S_{\text{out}}$
3. `Q @ S_in.T` $\to$ 计算历史状态输出
4. `Q @ K.T` $\to$ 计算 Chunk 内 Attention 分数
5. `(Q @ K.T * M_C) @ G` $\to$ 因果 Mask 后加权 $G$ 得到局部输出

## 5. Sparse attention：算法分类与原理

## 6. Sparse attention：kernel、serving 与部署

## 7. 复现方法与环境

## 8. 正确性、性能与效果结果

## 9. 局限、结论边界与后续问题
