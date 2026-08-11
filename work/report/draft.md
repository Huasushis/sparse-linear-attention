# Sparse 与 Linear Attention 调研及复现报告（草稿）

> 从 P1 起逐节填写。没有证据的结论先写成问题，不要补成肯定句。

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

## 5. Sparse attention：算法分类与原理

## 6. Sparse attention：kernel、serving 与部署

## 7. 复现方法与环境

## 8. 正确性、性能与效果结果

## 9. 局限、结论边界与后续问题
