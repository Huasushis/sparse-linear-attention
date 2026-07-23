# 资料与引用：按用途读，而不是收集链接

## 动画和第一直觉

- [3Blue1Brown: But what is a GPT?](https://www.3blue1brown.com/lessons/gpt)：先建立 Transformer 的整体直觉。
- [3Blue1Brown: Attention in transformers](https://www.3blue1brown.com/lessons/attention)：与第 1、7 章并读，专注 Q/K/V 和权重。

## 官方文档与实现

- [PyTorch Autograd tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：第 2 章练习后的查阅对象。
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：第 7、Lab 1 的框架基线。
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)：第 9、Lab 5 的后续练习。
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)：第 13、Lab 6 的实现样本库。

## GPU 课程：现在读什么，稍后读什么

- [Modern GPU Programming for MLSys（中文版）](https://mlc.ai/modern-gpu-programming-for-mlsys/zh/)：优先读 GPU 执行模型、性能、数据布局；Blackwell 的 TMA/TMEM/tcgen05/mbarrier 可先理解概念，不要求在 A100/5090 上运行。

## 主线论文

- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)；Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691)。
- Katharopoulos et al., [Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html)。
- Yang et al., [Gated Linear Attention](https://arxiv.org/abs/2312.06635)、[DeltaNet](https://arxiv.org/abs/2406.06484)、[Gated Delta Networks](https://arxiv.org/abs/2412.06464)。
- Dao & Gu, [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)。
- Kimi Team, [Kimi Linear](https://arxiv.org/abs/2510.26692)。

完整的 73 篇原始 BibTeX 在仓库根目录的 `references/attention.bib`。按优先级标注的
[阅读图](paper-map.md)从 `study/PAPER_MAP.md` 单一来源嵌入本站，因此网站与研究文件不会
维护两份相互漂移的副本。
