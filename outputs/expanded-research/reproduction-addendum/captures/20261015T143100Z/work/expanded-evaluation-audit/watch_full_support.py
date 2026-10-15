"""Lightweight >=60s completed-result watcher; delegates only CPU saved-array audits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import audit_full_support as audit
import source_preservation

WATCHER_SOURCES = source_preservation.sources(__file__, audit.__file__, source_preservation.__file__)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    plan = audit.read(audit.PLAN)
    assert len(plan["runs"]) == len({(row["domain"], row["architecture"]) for row in plan["runs"]}) == 12
    suite = audit.PLAN.with_name("budgeted-suite-execution.json")
    metadata_checked, terminal, last_memory_state, mean_failures = set(), set(), None, set()
    mean_path = audit.OUT.with_name("full-support-mean-baselines.json")
    while True:
        report = audit.read(audit.OUT) if audit.OUT.exists() else {"models": {}}
        baseline_pending = {key for key, result in report["models"].items()
            if result.get("supplementary_audits") and result["supplementary_audits"][-1]["status"] == "deferred_memory"}
        for row in plan["runs"]:
            key = row["domain"]+"/"+row["architecture"]
            existing = report["models"].get(key, {})
            if existing.get("status") in ("passed", "integrity_passed_insufficient_budget", "failed"):
                terminal.add(key)
            if existing.get("status", "").startswith("metadata_passed"):
                metadata_checked.add(key)
        mean_report = audit.read(mean_path) if mean_path.exists() else {"domains": {}}
        means_done = all(mean_report["domains"].get(domain, {}).get("status") == "computed_and_self_checked" for domain in ("phone", "bumper"))
        if len(terminal) == len(plan["runs"]) and not baseline_pending and means_done:
            latest_path = audit.OUT.with_name("full-support-latest-audit.json")
            if latest_path.exists():
                latest = audit.read(latest_path)
                if latest.get("auditor", {}).get("script_sha256") != audit.EXECUTION_SHA:
                    emit({"watcher": "latest_audit_source_mismatch_preserved_no_overwrite",
                        "recorded": latest.get("auditor", {}).get("script_sha256"), "expected": audit.EXECUTION_SHA})
                    return
                if latest["status"] == "complete" or any(item["status"] == "failed" for item in latest["models"].values()):
                    emit({"watcher": "all_completed_results_audited", "models": len(terminal), "latest_audit_status": latest["status"]})
                    return
            try:
                ready = audit.memory()["host_available_bytes"] >= audit.FREE_MIN+audit.RSS_CAP
            except audit.DeferredMemory:
                ready = False
            if ready:
                result = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("audit_full_support.py")), "--latest-audit"],
                    cwd=audit.ROOT, env=dict(os.environ), capture_output=True, text=True)
                if result.stdout.strip(): print(result.stdout.strip(), flush=True)
                if result.returncode:
                    emit({"watcher": "latest_audit_executable_failed", "stderr": result.stderr[-2000:]})
                    return
            time.sleep(60)
            continue
        for row in plan["runs"]:
            key = row["domain"]+"/"+row["architecture"]
            if key in terminal and key not in baseline_pending:
                continue
            path = Path(row["out"])/"test_report.json"
            if not path.is_file():
                continue
            try:
                response_status = audit.read(path).get("status")
                completed = response_status == "completed"
            except (OSError, ValueError):
                continue
            if not completed and not str(response_status).startswith("failed"):
                continue
            try:
                memory = audit.memory()
                available = memory["host_available_bytes"] >= audit.FREE_MIN+audit.RSS_CAP
                if not available:
                    memory["startup_headroom_required_bytes"] = audit.FREE_MIN+audit.RSS_CAP
            except audit.DeferredMemory as error:
                memory, available = {"deferral": str(error)}, False
            if not available and (key in metadata_checked or key in baseline_pending):
                if last_memory_state is not False:
                    emit({"watcher": "saved_array_audit_waiting_for_memory", **memory})
                last_memory_state = False
                continue
            if available and last_memory_state is False:
                emit({"watcher": "memory_gate_recovered", **memory})
            last_memory_state = available
            command = [sys.executable, "-u", str(Path(__file__).with_name("audit_full_support.py")),
                       "--domain", row["domain"], "--architecture", row["architecture"]]
            if key in baseline_pending:
                command.append("--baseline-only")
            result = subprocess.run(command, cwd=audit.ROOT, env=dict(os.environ), capture_output=True, text=True)
            if result.stdout.strip():
                print(result.stdout.strip(), flush=True)
            if result.returncode:
                emit({"watcher": "audit_executable_failed", "model": key, "exit_code": result.returncode,
                      "stderr": result.stderr[-2000:]})
                return
            refreshed = audit.read(audit.OUT)["models"][key]
            metadata_checked.add(key)
            if refreshed["status"] in ("passed", "integrity_passed_insufficient_budget", "failed"):
                terminal.add(key)
        report = audit.read(audit.OUT)
        queued_model = False
        for row in plan["runs"]:
            key = row["domain"]+"/"+row["architecture"]
            path = Path(row["out"])/"test_report.json"
            if key not in terminal and path.is_file():
                queued_model = True
        if not queued_model and not baseline_pending:
            for domain in ("phone", "bumper"):
                if domain in mean_failures or mean_report["domains"].get(domain, {}).get("status") == "computed_and_self_checked":
                    continue
                ready = any(row["domain"] == domain and report["models"].get(domain+"/"+row["architecture"], {}).get("status") in ("passed", "integrity_passed_insufficient_budget") for row in plan["runs"])
                if not ready:
                    continue
                try:
                    headroom = audit.memory()["host_available_bytes"] >= audit.FREE_MIN+audit.RSS_CAP
                except audit.DeferredMemory:
                    headroom = False
                if not headroom:
                    continue
                result = subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("compute_mean_baselines.py")), "--domain", domain],
                    cwd=audit.ROOT, env=dict(os.environ), capture_output=True, text=True)
                if result.stdout.strip():
                    print(result.stdout.strip(), flush=True)
                if result.returncode:
                    mean_failures.add(domain)
                    emit({"watcher": "mean_baseline_computation_failed", "domain": domain,
                        "exit_code": result.returncode, "stderr": result.stderr[-2000:]})
        if suite.is_file():
            try:
                suite_status = audit.read(suite).get("status")
            except (OSError, ValueError):
                suite_status = None
            if suite_status in ("completed", "completed_with_failures"):
                not_ready = [row["domain"]+"/"+row["architecture"] for row in plan["runs"]
                    if not (Path(row["out"])/"test_report.json").is_file()]
                if not_ready:
                    emit({"watcher": "suite_ended_with_unauditable_runs", "models": not_ready})
                    return
        time.sleep(60)


if __name__ == "__main__":
    main()
