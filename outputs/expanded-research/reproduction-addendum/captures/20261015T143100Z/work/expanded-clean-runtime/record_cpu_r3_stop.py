"""Preserve the observed one-build result and finalization stop; never retry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

assert sys.flags.no_site
root = Path(__file__).resolve().parents[2]
own = root / "work/expanded-clean-runtime"
out = root / "outputs/expanded-research/reproduction-clean-runtime"
build = json.loads((out / "antlr-build-execution.json").read_text())
failed = json.loads((out / "finalize-antlr-20261015T140744621390Z.json").read_text())
assert build["status"] == "completed" and build["exit_code"] == 0
assert failed["status"] == "failed_child" and failed["child_reaped"] and failed["descendants_reaped"]
wheels = list((own / "antlr-built-wheelhouse").glob("*.whl"))
assert len(wheels) == 1
wheel = wheels[0]
result = {"status": "one_ANTLR_build_completed_finalizer_path_spelling_check_failed_install_not_started",
          "observed_utc": datetime.now(timezone.utc).isoformat(), "build_attempts_started": 1,
          "new_wheel_observed": {"filename": wheel.name, "bytes": wheel.stat().st_size,
                                 "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                                 "full_source_and_lock_finalization_passed": False},
          "build_receipt": "antlr-build-execution.json", "build_pid": build["pid"],
          "build_finished_utc": build["finished_utc"], "build_descendants_reaped": build["descendants_reaped"],
          "finalizer_failure_receipt": "finalize-antlr-20261015T140744621390Z.json",
          "finalizer_failure_log": failed["log"], "finalizer_pid": failed["pid"],
          "finalizer_exit_code": failed["exit_code"], "finalizer_finished_utc": failed["finished_utc"],
          "finalizer_child_and_descendants_reaped": True, "host_available_bytes_at_stop": failed["available_bytes_at_end"],
          "cause_from_read_only_source_comparison": "Approval source_root uses forward separators; actual standard Windows argv uses backslashes. Strict source_root-in-command spelling check rejects the same path. Original receipt/sentinel argv equality passed.",
          "runtime_lock_r2_created": (out / "requirements-lock-r2.txt").exists(),
          "build_venv_created": (own / "build-venv").exists(), "runtime_venv_created": (own / "venv").exists(),
          "original_R2_source_sha256": hashlib.sha256((own / "run_guarded_r2.py").read_bytes()).hexdigest(),
          "CPU_R3_source_sha256": hashlib.sha256((own / "run_guarded_cpu_r3.py").read_bytes()).hexdigest(),
          "framework_or_GPU_or_arrays_used": False, "automatic_retry_planned": False,
          "additional_ANTLR_build_forbidden": True, "active_owned_children": 0,
          "next_action": "Actual stages held after failure; preserve all outputs and await a bounded finalizer correction allocation."}
path = out / "CPU-progress-r3-finalizer-stop.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"status": result["status"], "new_wheel_observed": result["new_wheel_observed"], "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
