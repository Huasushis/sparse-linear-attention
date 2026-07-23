# Lab 3：GDN 与 KDA 的“几乎重合”做成可检验命题

## 目标

把导师的判断从一句话改写成一个可运行的命题：

> 在教学版的同头数、同 q/k/v、相同 `β` 下，若 KDA 的每个 key-channel gate 都等于 GDN 的 per-head scalar gate，则两者的 recurrent state update 相同。

## 运行探针

```bash
python -m tutorial_code.scripts.gdn_kda_probe
python -m pytest tutorial_code/tests/test_gated_delta.py -q
```

这是**算子骨架的退化关系**，不是说完整 Kimi Linear 与 GDN 的所有架构、训练数据、MLA 层或 kernel 一样。第 12 章会逐层区分它们。

## 改一个变量

将 KDA 的 `g[..., key_dim]` 改成每维不同，然后再运行。观察它不再等于 scalar GDN；这正是 KDA 的一个表达能力来源。

## 通过条件

你能区分以下三句话：

1. GDN 和 KDA 的 delta-update skeleton 高度相似；
2. KDA 对 gate 粒度、head/值头关系和 chunk kernel 有扩展；
3. Kimi Linear 是含 KDA 与 MLA 的整体 hybrid architecture，不能被缩写成一个 GDN layer。
