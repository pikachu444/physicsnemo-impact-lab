"""Bounded recovery runner; preserves the original probe and suite evidence."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/expanded-research/benchmarks/batching/allocated-slot-02"
DRIVER = Path(__file__).with_name("run_batch_probe.py")
PYTHON = ROOT / "work/expanded-ml/venv/Scripts/python.exe"
MIB = 1024 ** 2


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                ("total_commit", ctypes.c_ulonglong), ("available_commit", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                ("extended_virtual", ctypes.c_ulonglong)]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def host_memory():
    row = MemoryStatus()
    row.length = ctypes.sizeof(row)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(row))
    return {"available_physical_bytes": row.available_physical,
            "available_commit_bytes": row.available_commit, "total_commit_bytes": row.total_commit}


def main():
    import psutil

    assert not OUT.exists(), "Recovery evidence must never overwrite an existing output directory"
    OUT.mkdir(parents=True)
    original_suite = OUT.parent / "allocated-slot-01/suite.json"
    before = hashlib.sha256(original_suite.read_bytes()).hexdigest()
    suite = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Only phone FIGConvUNet B2, then B5 if B2 passes; no training run or held-out evaluation",
             "original_suite_sha256": before, "results": [], "timeout_seconds": 180,
             "guard": {"physical_available_min_bytes": 128*MIB, "commit_available_min_bytes": 1024*MIB,
                       "commit_definition": "Windows GlobalMemoryStatusEx ullAvailPageFile; available system commit, not process address space",
                       "sample_interval_seconds": 0.5},
             "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT/"suite.json", suite)
    stopped = None
    for batch_size in (2, 5):
        row = {"domain": "phone", "architecture": "figconvunet", "batch_size": batch_size}
        if stopped:
            row.update(status="not_attempted", reason=f"Stopped after B{stopped['batch_size']} {stopped['status']}")
            suite["results"].append(row)
            write(OUT/"suite.json", suite)
            continue
        child_out = OUT/"phone/figconvunet"/f"B{batch_size}"
        child_out.parent.mkdir(parents=True, exist_ok=True)
        raw_log = child_out.with_suffix(".raw.log")
        command = [str(PYTHON), str(DRIVER), "--domain", "phone", "--architecture", "figconvunet",
                   "--batch-size", str(batch_size), "--out", str(child_out),
                   "--timing-repeats", "2", "--gpu-slot-authorized"]
        row.update(command=command, raw_log=str(raw_log))
        tick = time.perf_counter()
        print(json.dumps({**row, "status": "starting"}), flush=True)
        guard_path = OUT/f"phone-figconvunet-B{batch_size}-resources.jsonl"
        samples = []
        state = None
        with raw_log.open("wb") as stream, guard_path.open("w", encoding="utf-8") as monitor:
            initial = host_memory()
            if initial["available_physical_bytes"] < 128*MIB or initial["available_commit_bytes"] < 1024*MIB:
                row.update(status="not_attempted", reason="Resource guard prevented launch", initial_memory=initial)
                state = "not_attempted"
            else:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
                row["pid"] = child.pid
                while True:
                    snapshot = {"utc": datetime.now(timezone.utc).isoformat(),
                                "elapsed_seconds": time.perf_counter()-tick, **host_memory()}
                    try:
                        info = psutil.Process(child.pid).memory_info()
                        snapshot.update(rss_bytes=info.rss, windows_process_peak_wset_bytes=getattr(info, "peak_wset", None))
                    except psutil.NoSuchProcess:
                        pass
                    monitor.write(json.dumps(snapshot)+"\n")
                    monitor.flush()
                    samples.append(snapshot)
                    code = child.poll()
                    if code is not None:
                        row["exit_code"] = code
                        break
                    if snapshot["available_physical_bytes"] < 128*MIB or snapshot["available_commit_bytes"] < 1024*MIB:
                        state = "resource_guard_stop"
                    elif snapshot["elapsed_seconds"] >= 180:
                        state = "timeout"
                    if state:
                        # This runner owns only this freshly launched isolated probe.
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
                        row.update(exit_code=child.returncode, status=state, stop_sample=snapshot,
                                   reason="Owned probe terminated by explicit OS resource guard" if state == "resource_guard_stop" else "Owned probe terminated at 180 second limit")
                        break
                    time.sleep(0.5)
        if not state:
            result_path = child_out/"result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                row.update(status=result["status"], result=str(result_path), timing=result.get("timing"),
                           failed_stage=result.get("failed_stage"))
                if row["exit_code"] != 0 and row["status"] == "passed":
                    row.update(status="failed", reason="Nonzero exit despite passed payload")
            else:
                row.update(status="failed", reason="Child exited without result.json")
        row.update(seconds=time.perf_counter()-tick, resource_log=str(guard_path), samples=len(samples))
        if samples:
            row["resource_summary"] = {"minimum_available_physical_bytes": min(x["available_physical_bytes"] for x in samples),
                                       "minimum_available_commit_bytes": min(x["available_commit_bytes"] for x in samples),
                                       "peak_sampled_rss_bytes": max(x.get("rss_bytes", 0) for x in samples)}
        if row["status"] != "passed":
            stopped = row
        suite["results"].append(row)
        write(OUT/"suite.json", suite)
        print(json.dumps({k: v for k, v in row.items() if k != "timing"}), flush=True)
    assert hashlib.sha256(original_suite.read_bytes()).hexdigest() == before, "Original suite was changed"
    suite.update(status="passed" if all(r["status"] == "passed" for r in suite["results"]) else "completed_with_stops",
                 finished_utc=datetime.now(timezone.utc).isoformat())
    write(OUT/"suite.json", suite)


if __name__ == "__main__":
    main()
