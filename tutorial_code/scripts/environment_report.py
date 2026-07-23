"""Print a concise environment report suitable for an experiment record."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys

import torch


def nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        return subprocess.check_output([executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "available but query failed"


print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "torch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "nvidia_smi": nvidia_smi(),
}, indent=2, ensure_ascii=False))
