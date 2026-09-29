#!/usr/bin/env python3
"""Report the compute environment (works locally AND on RunPod).

Prints: Python version, PyTorch version, CUDA availability, GPU name, GPU memory
(VRAM), plus platform and disk space. Designed to give REAL values wherever it
runs — on the RunPod A100 it will show the GPU; locally it honestly reports no
CUDA. It trains nothing, downloads nothing, and imports no model code.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def _disk(path: str):
    try:
        total, used, free = shutil.disk_usage(path)
        gb = 1024 ** 3
        return {"path": path, "total_gb": round(total / gb, 1),
                "used_gb": round(used / gb, 1), "free_gb": round(free / gb, 1)}
    except OSError:
        return {"path": path, "error": "not available"}


def collect() -> dict:
    info: dict = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    # PyTorch / CUDA / GPU — optional; report honestly if torch is absent.
    try:
        import torch  # noqa: WPS433
        info["torch_version"] = torch.__version__
        cuda = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if cuda:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 1)
        else:
            info["gpu_name"] = None
            info["gpu_total_memory_gb"] = None
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False
        info["gpu_name"] = None
        info["gpu_total_memory_gb"] = None
        info["torch_note"] = "PyTorch not installed in this environment."

    # Disk space (cwd, root, and RunPod /workspace if present).
    disks = [_disk(str(Path.cwd()))]
    for p in ("/", "/workspace"):
        if Path(p).exists():
            disks.append(_disk(p))
    info["disk"] = disks
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Report the compute environment")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    info = collect()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print("=" * 60)
    print("Environment check — chexlocalize-medical-imaging")
    print("=" * 60)
    print(f"Python version   : {info['python_version']}")
    print(f"Python exe       : {info['python_executable']}")
    print(f"Platform         : {info['platform']}")
    print(f"PyTorch version  : {info.get('torch_version')}")
    print(f"CUDA available   : {info.get('cuda_available')}")
    print(f"Torch CUDA       : {info.get('torch_cuda_version')}")
    print(f"GPU name         : {info.get('gpu_name')}")
    print(f"GPU memory (VRAM): "
          f"{str(info.get('gpu_total_memory_gb')) + ' GB' if info.get('gpu_total_memory_gb') else 'n/a'}")
    print("Disk:")
    for d in info["disk"]:
        if "error" in d:
            print(f"  {d['path']}: {d['error']}")
        else:
            print(f"  {d['path']}: {d['free_gb']} GB free / {d['total_gb']} GB total")
    if info.get("torch_note"):
        print(f"\nNote: {info['torch_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
