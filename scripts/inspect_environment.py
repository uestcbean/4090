"""Read-only Linux/WSL inventory with optional tiny PyTorch CUDA smoke check.

No dependency installation, model downloads or performance benchmark is performed.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {"status": "ok" if p.returncode == 0 else "error",
                "returncode": p.returncode, "stdout": p.stdout.strip(),
                "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def inspect(disk_path, smoke):
    result = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(), "kernel": platform.release(),
        "python": platform.python_version(), "cpu_logical_count": os.cpu_count(),
        "is_wsl": "microsoft" in platform.release().lower(),
        "git": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--porcelain"]),
    }
    mem = Path("/proc/meminfo")
    if mem.exists():
        values = dict(line.split(":", 1) for line in mem.read_text().splitlines())
        result["memory_bytes"] = {key: int(values[key].split()[0]) * 1024
                                  for key in ("MemTotal", "MemAvailable", "SwapTotal")}
    else:
        result["memory_bytes"] = {"status": "unavailable", "reason": "requires /proc/meminfo"}
    usage = shutil.disk_usage(disk_path)
    result["disk"] = {"path": str(disk_path), "total_bytes": usage.total,
                      "free_bytes": usage.free,
                      "note": "WSL filesystem capacity is not proof of host backing-volume free space"}
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    gpu = command([smi, "--query-gpu=name,memory.total,memory.free,driver_version",
                   "--format=csv,nounits"])
    if gpu["status"] == "ok":
        gpu["rows"] = list(csv.DictReader(io.StringIO(gpu["stdout"]), skipinitialspace=True))
    result["nvidia_smi"] = gpu
    if not smoke:
        result["torch"] = {"status": "not_tested", "reason": "pass --cuda-smoke"}
        return result
    try:
        import torch
        info = {"version": torch.__version__, "built_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available()}
        result["torch"] = info
        if not info["cuda_available"]:
            info["status"] = "failed"
            return result
        info["device"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        # Small exact-integer products stay exactly representable in float32.
        a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        expected = a @ a.T
        gpu_a = a.to("cuda")
        actual = gpu_a @ gpu_a.T
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        info["status"] = "passed"
        info["smoke"] = "4x4 float32 CUDA matmul equals CPU reference; not a benchmark"
        info["result"] = actual.cpu().tolist()
    except Exception as exc:
        result.setdefault("torch", {}).update({"status": "failed",
                                                "error_type": type(exc).__name__,
                                                "reason": str(exc)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-path", type=Path, default=Path.home())
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.disk_path, args.cuda_smoke)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.cuda_smoke and result["torch"]["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
