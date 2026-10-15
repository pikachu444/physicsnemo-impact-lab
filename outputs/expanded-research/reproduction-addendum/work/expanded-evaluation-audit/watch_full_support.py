"""Lightweight >=60s completed-result watcher; delegates only CPU saved-array audits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import audit_full_support as audit


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main():
    plan = audit.read(audit.PLAN)
    assert len(plan["runs"]) == len({(row["domain"], row["architecture"]) for row in plan["runs"]}) == 12
    suite = audit.PLAN.with_name("budgeted-suite-execution.json")
    metadata_checked, terminal, last_memory_state = set(), set(), None
    while True:
        report = audit.read(audit.OUT) if audit.OUT.exists() else {"models": {}}
        for row in plan["runs"]:
            key = row["domain"]+"/"+row["architecture"]
            existing = report["models"].get(key, {})
            if existing.get("status") in ("passed", "integrity_passed_insufficient_budget", "failed"):
                terminal.add(key)
            if existing.get("status", "").startswith("metadata_passed"):
                metadata_checked.add(key)
        if len(terminal) == len(plan["runs"]):
            emit({"watcher": "all_completed_results_audited", "models": len(terminal)})
            return
        for row in plan["runs"]:
            key = row["domain"]+"/"+row["architecture"]
            if key in terminal:
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
                available = True
            except audit.DeferredMemory as error:
                memory, available = {"deferral": str(error)}, False
            if not available and key in metadata_checked:
                if last_memory_state is not False:
                    emit({"watcher": "saved_array_audit_waiting_for_memory", **memory})
                last_memory_state = False
                continue
            if available and last_memory_state is False:
                emit({"watcher": "memory_gate_recovered", **memory})
            last_memory_state = available
            command = [sys.executable, "-u", str(Path(__file__).with_name("audit_full_support.py")),
                       "--domain", row["domain"], "--architecture", row["architecture"]]
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
