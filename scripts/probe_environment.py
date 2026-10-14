"""Record measured execution resources without installing dependencies or reading secrets."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request


def command(argv, timeout=20):
    started = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"command": argv, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "seconds": time.perf_counter() - started}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc), "seconds": time.perf_counter() - started}


def probe(label, network):
    disk = shutil.disk_usage(Path.cwd())
    result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "execution_location_label": label, "platform": platform.platform(),
              "python": platform.python_version(), "cpu_count": os.cpu_count(),
              "disk_bytes": {"total": disk.total, "free": disk.free},
              "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
              "cuda_compiler": command(["nvcc", "--version"])}
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["memory_bytes"] = {"total": mem.total, "available": mem.available}
    except ImportError:
        if Path("/proc/meminfo").exists():
            result["memory_linux_meminfo"] = Path("/proc/meminfo").read_text()
    # A separate process contains import failures and releases imported libraries.
    torch_code = ("import torch,json; r={'version':torch.__version__, 'cuda_build':torch.version.cuda,"
                  "'cuda_available':torch.cuda.is_available()}; "
                  "r.update({'device':torch.cuda.get_device_name(0),"
                  "'tensor_gpu_result':(torch.ones(8,device='cuda')*2).sum().item()} "
                  "if torch.cuda.is_available() else {}); print(json.dumps(r))")
    result["torch"] = command([os.sys.executable, "-c", torch_code], timeout=90)
    result["network"] = []
    if network:
        for url in ["https://raw.githubusercontent.com/NVIDIA/physicsnemo/main/pyproject.toml",
                    "https://pypi.org/pypi/nvidia-physicsnemo/json",
                    "https://huggingface.co/api/datasets/AIRBORNEPANDA/BumperBeamCrashExample"]:
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(url, timeout=15) as response:
                    response.read(512)
                    row = {"url": url, "status": response.status}
            except Exception as exc:
                row = {"url": url, "error": str(exc)}
            row["seconds"] = time.perf_counter() - started
            result["network"].append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Explicit location, e.g. local-windows or codex-cloud")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    data = probe(args.label, args.network)
    assert data["cpu_count"] is None or data["cpu_count"] > 0
    assert 0 <= data["disk_bytes"]["free"] <= data["disk_bytes"]["total"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
