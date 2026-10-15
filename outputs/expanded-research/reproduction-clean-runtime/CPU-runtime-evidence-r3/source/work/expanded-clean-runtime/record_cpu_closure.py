"""Record completed CPU stages and a no-child memory observation; no ML imports."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

assert sys.flags.no_site
root = Path(__file__).resolve().parents[2]
own = root / "work/expanded-clean-runtime"
out = root / "outputs/expanded-research/reproduction-clean-runtime"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    return json.loads((out / name).read_text(encoding="utf-8"))


stages = []
for stage in ("antlr-source", "build-tools-download", "build-create", "build-bootstrap", "build-tools-install", "build-wheel", "finalize-antlr", "create", "bootstrap", "resolve-local", "install", "check", "audit"):
    paths = sorted(out.glob(stage + "-20*.json"))
    path = paths[-1]
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "completed" and record["exit_code"] == 0 and record["child_reaped"]
    assert record["descendants_reaped"] and not record["remaining_descendant_pids_at_end"]
    assert record["supervisor_source_sha256"] == "ff9737545332ad418f5b1ab654668c10d55c85d348ee3e1ad7fd9d22e5505541"
    stages.append({key: record[key] for key in ("stage", "attempt", "pid", "child_started_utc", "finished_utc", "wall_seconds", "peak_sampled_family_RSS_bytes", "available_bytes_at_end", "disk_free_bytes_at_end", "child_reaped", "descendants_reaped")})
    stages[-1].update(receipt=path.name, receipt_sha256=digest(path))
assert len(list(out.glob("build-wheel-20*.json"))) == 1
audit = load("environment-static-audit.json")
assert audit["status"] == "passed_static_clean_environment_audit_only"
assert len(audit["packages"]) == 90 and len(audit["core_source_files_checked_against_wheel_and_RECORD"]) == 40
site = own / "venv/Lib/site-packages"
installed = sorted(({"name": dist.metadata["Name"], "version": dist.version} for dist in importlib.metadata.distributions(path=[str(site)])), key=lambda row: row["name"].lower())
lock = out / "requirements-lock-r2.txt"
antlr_line = next(line for line in lock.read_text().splitlines() if line.startswith("antlr4-python3-runtime=="))
assert "815dd98f0ae9191076788662e065b7bf86a7e8c53dc241f4c1ac588197bae439" in antlr_line
guard_path = own / "run_guarded_r2.py"
assert digest(guard_path) == "45d6bfb697b95f8df6721d2d42146c53095dc81e062e885f9fa91d17b5fdda1a"
specification = importlib.util.spec_from_file_location("stdlib_memory_observer_only", guard_path)
guard = importlib.util.module_from_spec(specification)
specification.loader.exec_module(guard)
observations = []
for index in range(2):
    observations.append({"utc": datetime.now(timezone.utc).isoformat(), "available_bytes": guard.available_memory()})
    if index == 0:
        time.sleep(10)
assert not any(name in sys.modules for name in ("numpy", "torch", "physicsnemo"))
result = {"status": "passed_independent_installation_and_static_audit_only", "closed_utc": datetime.now(timezone.utc).isoformat(),
          "runtime_environment": "${WORKSPACE}/work/expanded-clean-runtime/venv", "system_or_user_site_inherited": False,
          "locked_runtime_packages": 90, "installed_metadata_including_bootstrap_tools": installed,
          "pip_check_conflicts": 0, "core40_wheel_and_RECORD_byte_matches": 40,
          "requirements_lock": "requirements-lock-r2.txt", "requirements_lock_sha256": digest(lock),
          "new_ANTLR_wheel_sha256": "815dd98f0ae9191076788662e065b7bf86a7e8c53dc241f4c1ac588197bae439",
          "ANTLR_build_attempts": 1, "full_repeated_build_determinism_tested": False,
          "source_and_build_receipts": ["antlr-source-audit-r2.json", "antlr-build-execution.json", "antlr-build-verification.json"],
          "stages": stages, "active_owned_children_after_all_stages": 0,
          "peak_sampled_CPU_stage_family_RSS_bytes": max(row["peak_sampled_family_RSS_bytes"] for row in stages),
          "memory_sampling_limit": "One-second process-family samples are lower bounds on instantaneous peak RSS.",
          "post_CPU_no_child_memory_observations": observations,
          "future_functional_1p5GiB_operating_admission_observations_pass": all(row["available_bytes"] >= 1610612736 for row in observations),
          "functional_runtime_source_sha256": digest(own / "newfunctional_runtime.py"),
          "functional_protocol_sha256": digest(out / "newfunctional-protocol-v1.json"),
          "actual_functional_checks_completed": 0, "framework_imported": False, "GPU_probed_or_used": False,
          "data_arrays_or_modelweights_read": False, "whole_fit_reproduction_claimed": False,
          "original_R2_guard_unchanged": True, "previous_failed_receipts_preserved": True,
          "current_GPU_execution_authorized_by_this_record": False,
          "next_action": "Report closure and await explicit root GPU allocation before six prepared functional checks."}
path = out / "CPU-clean-runtime-closure-r3.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps({"status": result["status"], "receipt_sha256": digest(path), "observations": observations,
                  "installed_metadata_count": len(installed), "locked_runtime_packages": 90,
                  "pip_check_conflicts": 0, "core40_matches": 40, "functional_runtime_source_sha256": result["functional_runtime_source_sha256"],
                  "functional_protocol_sha256": result["functional_protocol_sha256"]}))
