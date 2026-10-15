"""Record an actual CUDA operation; this is an environment test, not FEM validation."""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import psutil

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=True)


def smi():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "torch_path": torch.__file__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "nvidia_smi_before": smi(),
    "disk_free_bytes": shutil.disk_usage(output).free,
    "pid": os.getpid(),
    "host_memory_before_bytes": dict(psutil.virtual_memory()._asdict()),
}
try:
    assert report["cuda_available"], "PyTorch cannot use CUDA"
    torch.manual_seed(17)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    y = x.square().mean()
    y.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert x.is_cuda and x.grad.is_cuda
    assert torch.allclose(x.grad, 2 * x.detach() / x.numel(), atol=1e-8)
    report.update({
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "tensor_shape": list(x.shape),
        "tensor_device": str(x.device),
        "loss": y.item(),
        "gradient_assertion_passed": True,
        "cuda_forward_backward_wall_seconds_including_first_operation": elapsed,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvidia_smi_tensors_alive": smi(),
        "status": "passed",
        "process_rss_after_bytes": psutil.Process().memory_info().rss,
    })
except Exception as exc:
    report.update({"status": "failed", "error": repr(exc)})
    raise
finally:
    (output / "gpu-probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
