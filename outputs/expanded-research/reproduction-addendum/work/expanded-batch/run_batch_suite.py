"""Run ascending, isolated native batch probes; stop larger B after any failure."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_batch_probe import ARCHITECTURES, DOMAINS, MAIN, OUTPUT_ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", choices=DOMAINS, nargs="+", default=list(DOMAINS))
    parser.add_argument("--architectures", choices=ARCHITECTURES, nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--batches", type=int, choices=(1, 2, 5), nargs="+", default=[1, 2, 5])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=MAIN/"venv/Scripts/python.exe")
    parser.add_argument("--timing-repeats", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--gpu-slot-authorized", action="store_true")
    args = parser.parse_args()
    if not args.gpu_slot_authorized:
        raise SystemExit("GPU slot flag missing: no ML library import or child process was performed.")
    assert args.batches == sorted(set(args.batches)), "Use ascending unique B1, B2, B5"
    assert args.timing_repeats >= 2 and args.timeout_seconds > 0
    assert len(args.domains) == len(set(args.domains)) and len(args.architectures) == len(set(args.architectures))
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT_ROOT.resolve())
    out.mkdir(parents=True, exist_ok=False)
    suite = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
             "batch_order": args.batches, "results": [], "command": sys.argv,
             "preceding_B1_note": "If B1 is omitted, scheduler supplied preceding main B1 evidence; each B2/B5 still compares original B1 calls at identical initial weights.",
             "stop_rule": "Any OOM/failure/timeout/validation failure stops larger batches for that domain/model. Every attempted batch has a fresh subprocess."}
    driver = Path(__file__).with_name("run_batch_probe.py")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for domain in args.domains:
        for architecture in args.architectures:
            stopped = None
            for batch_size in args.batches:
                row = {"domain": domain, "architecture": architecture, "batch_size": batch_size}
                if stopped:
                    row.update(status="not_attempted", reason=f"Stopped after B{stopped['batch_size']} {stopped['status']}; no larger-batch feasibility inferred")
                    suite["results"].append(row)
                    write_json(out/"suite.json", suite)
                    continue
                child_out = out/domain/architecture/f"B{batch_size}"
                child_out.parent.mkdir(parents=True, exist_ok=True)
                command = [str(args.python), str(driver), "--domain", domain, "--architecture", architecture,
                           "--batch-size", str(batch_size), "--out", str(child_out),
                           "--timing-repeats", str(args.timing_repeats), "--gpu-slot-authorized"]
                raw_log = child_out.with_suffix(".raw.log")
                tick = time.perf_counter()
                print(json.dumps({**row, "status": "starting", "raw_log": str(raw_log)}), flush=True)
                try:
                    with raw_log.open("wb") as stream:
                        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   env=env, timeout=args.timeout_seconds, check=False)
                    row["exit_code"] = completed.returncode
                    if (child_out/"result.json").exists():
                        payload = json.loads((child_out/"result.json").read_text(encoding="utf-8"))
                        row.update(status=payload["status"], result=str(child_out/"result.json"),
                                   timing=payload.get("timing"), failed_stage=payload.get("failed_stage"))
                    else:
                        row.update(status="failed", reason="Worker exited without result.json; raw log preserved")
                    if completed.returncode != 0 and row["status"] == "passed":
                        row.update(status="failed", reason="Nonzero worker exit despite passed payload")
                except subprocess.TimeoutExpired:
                    row.update(status="timeout", reason=f"Isolated child killed after {args.timeout_seconds}s; raw log and partial events retained")
                except Exception as error:
                    row.update(status="failed", reason=f"{type(error).__name__}: {error}")
                row.update(seconds=time.perf_counter()-tick, raw_log=str(raw_log))
                suite["results"].append(row)
                if row["status"] != "passed":
                    stopped = row
                write_json(out/"suite.json", suite)
                print(json.dumps({k: v for k, v in row.items() if k != "timing"}), flush=True)
    attempted = [r for r in suite["results"] if r["status"] != "not_attempted"]
    suite["status"] = "passed" if all(row["status"] == "passed" for row in attempted) else "completed_with_failures"
    suite["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out/"suite.json", suite)
    return 0 if suite["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
