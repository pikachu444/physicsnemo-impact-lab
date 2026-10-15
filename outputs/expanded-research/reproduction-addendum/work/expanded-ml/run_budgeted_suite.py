"""Sequential process isolation for the prospectively fixed GPU comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-slot-authorized", action="store_true")
    args = parser.parse_args()
    if not args.gpu_slot_authorized or not __debug__:
        raise RuntimeError("An explicit GPU slot and normal Python are required")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for path, expected in plan["source_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Source changed after preflight audit: {path}")
    root = Path(plan["working_directory"])
    os.chdir(root)
    logs = args.plan.resolve().with_name("budgeted-suite-logs")
    logs.mkdir(exist_ok=True)
    report_path = args.plan.resolve().with_name("budgeted-suite-execution.json")
    if report_path.exists():
        raise RuntimeError("Suite evidence already exists; choose an explicitly reviewed recovery plan")
    report = {"status": "running", "started_utc": stamp(), "pid": os.getpid(),
              "plan_sha256": digest(args.plan), "commands": [], "results": []}
    save(report_path, report)
    env = dict(os.environ, PYTHONUTF8="1", PHYSICSNEMO_FORCE_TE="False",
               TORCHDYNAMO_DISABLE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    python = str(Path(sys.executable).resolve())
    for row in plan["runs"]:
        output = Path(row["out"])
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Refusing nonempty training output: {output}")
        train = [python, "-X", "utf8", "work/expanded-ml/train_budgeted.py",
                 "--cache", row["cache"], "--expected-manifest-sha256", row["manifest_sha256"],
                 "--normalization", row["normalization"], "--out", row["out"],
                 "--architecture", row["architecture"], "--batch-size", str(row["batch_size"]),
                 "--protocol", plan["protocol"], "--protocol-sha256", plan["protocol_sha256"],
                 "--gpu-slot-authorized"]
        if row.get("batch_adapter"):
            train += ["--batch-adapter", row["batch_adapter"]]
        for approval in row.get("batch_approvals", []):
            train += ["--batch-approval", approval]
        if row.get("data_clarification"):
            train += ["--data-clarification", row["data_clarification"]]
        for pid in plan.get("concurrent_pids", []):
            train += ["--concurrent-pid", str(pid)]
        result = {"domain": row["domain"], "architecture": row["architecture"],
                  "batch_size": row["batch_size"], "out": row["out"], "stages": []}
        for mode in ("train", "validation", "test"):
            command = train if mode == "train" else [python, "-X", "utf8",
                "work/expanded-ml/evaluate_budgeted.py", "--run", row["out"],
                "--cache", row["cache"], "--expected-manifest-sha256", row["manifest_sha256"],
                "--mode", mode, "--gpu-slot-authorized"]
            log = logs/f"{row['domain']}-{row['architecture']}-{mode}.log"
            event = {"domain": row["domain"], "architecture": row["architecture"], "stage": mode,
                     "command": command, "working_directory": str(root), "started_utc": stamp(),
                     "log": str(log), "status": "running"}
            report["commands"].append(event)
            save(report_path, report)
            print(json.dumps(event), flush=True)
            started = time.perf_counter()
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            event.update({"finished_utc": stamp(), "wall_seconds": time.perf_counter()-started,
                          "exit_code": completed.returncode,
                          "status": "passed" if completed.returncode == 0 else "failed"})
            result["stages"].append(dict(event))
            if output.exists():
                shutil.copyfile(log, output/f"{mode}.log")
                save(output/f"{mode}-command.json", event)
            save(report_path, report)
            print(json.dumps(event), flush=True)
            if completed.returncode:
                break
        result["status"] = "passed" if len(result["stages"]) == 3 and all(x["exit_code"] == 0 for x in result["stages"]) else "failed"
        report["results"].append(result)
        save(report_path, report)
    report.update({"status": "completed" if all(x["status"] == "passed" for x in report["results"]) else "completed_with_failures",
                   "finished_utc": stamp()})
    save(report_path, report)
    print(json.dumps({"status": report["status"], "results": report["results"]}), flush=True)


if __name__ == "__main__":
    main()
