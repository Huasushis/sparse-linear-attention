# 小型运行摘要

这里提交可以被 Git 审查的小型 Markdown、CSV 或 JSON 摘要。建议文件名为 `YYYY-MM-DD-topic.md`。

每条记录至少包含 commit、Slurm job id、GPU、软件版本、输入 shape、dtype、命令、计时方法和结果。完整日志、PyTorch trace、Nsight 文件与模型权重放 `artifacts/` 或远程用户目录，不放这里。
