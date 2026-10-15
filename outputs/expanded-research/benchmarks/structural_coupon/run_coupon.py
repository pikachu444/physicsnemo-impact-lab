"""Run only the explicitly selected prepared case, bounded by CPU/RAM/time."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil


def run(case, solver, timeout, memory_cap):
    assert (case / "parameters.json").exists() and (solver / "exec/engine_win64.exe").exists()
    assert not list(case.glob("*.rst")), "Use a fresh case folder; do not overwrite completed solver evidence"
    env = os.environ.copy()
    env.update(OPENCOURANT_PATH=str(solver), RAD_CFG_PATH=str(solver / "hm_cfg_files"),
               RAD_H3D_PATH=str(solver / "extlib/h3d/lib/win64"), KMP_STACKSIZE="400m", OMP_NUM_THREADS="1")
    env["PATH"] = os.pathsep.join(str(solver / p) for p in ["extlib/hm_reader/win64", "extlib/h3d/lib/win64", "extlib/intelOneAPI_runtime/win64"]) + os.pathsep + env["PATH"]
    results = []
    for stage, exe, arguments in [
        ("starter", "starter_win64.exe", ["-i", "tensile_LAW2_0000.rad", "-np", "1", "-nt", "1"]),
        ("engine", "engine_win64.exe", ["-i", "tensile_LAW2_0001.rad", "-nt", "1"]),
    ]:
        started = datetime.now(timezone.utc).isoformat()
        clock = time.perf_counter()
        peak = 0
        reason = None
        with (case / (stage + ".stdout.log")).open("wb") as stdout, (case / (stage + ".stderr.log")).open("wb") as stderr:
            proc = subprocess.Popen([str(solver / "exec" / exe), *arguments], cwd=case, env=env,
                                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            monitored = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    peak = max(peak, monitored.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > memory_cap:
                    reason = "memory_cap"
                if time.perf_counter() - clock > timeout:
                    reason = "timeout"
                if reason:
                    proc.terminate()
                    proc.wait(timeout=10)
                    break
                time.sleep(0.05)
        result = {"stage": stage, "started_utc": started, "threads": 1, "pid": proc.pid,
                  "exit_code": proc.returncode, "wall_seconds": time.perf_counter() - clock,
                  "sampled_peak_rss_bytes": peak, "memory_cap_bytes": memory_cap,
                  "sample_interval_seconds": 0.05, "terminated_reason": reason, "executable": exe,
                  "arguments": arguments}
        results.append(result)
        (case / "execution-metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        if proc.returncode != 0 or reason:
            raise RuntimeError(f"{stage} failed; inspect preserved logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--memory-cap-bytes", type=int, default=400_000_000)
    args = parser.parse_args()
    run(args.case.resolve(), args.solver.resolve(), args.timeout, args.memory_cap_bytes)
