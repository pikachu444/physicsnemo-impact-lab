"""Tiny owned-child containment test; no package, network, or GPU operation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_guarded_r2 as guard


def main():
    assert os.name == "nt" and sys.flags.no_site
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), "Keep prior selfcheck receipts"
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    before = guard.available_memory()
    assert before >= 512 * 1024**2, "Tiny check requires the existing host floor"
    own = Path(__file__).resolve().parent
    temporary = own / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", TEMP=str(temporary), TMP=str(temporary))
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-S','-B','-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print(p.pid,flush=True)"
    with guard.ProcessJob() as job:
        process = subprocess.Popen([sys.executable, "-S", "-B", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=own, env=environment, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 4)
        try:
            job.assign_and_resume(process)
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0 and not error, error
            descendant = int(output.strip())
            members = job.pids()
            assert descendant in members and process.pid not in members, "A surviving grandchild must remain in the job after its parent exits"
            family_rss = sum(guard.rss(member) or 0 for member in members)
            assert family_rss < 350 * 1024**2
            job.terminate()
            deadline = time.perf_counter() + 5
            while job.pids() and time.perf_counter() < deadline:
                time.sleep(0.05)
            assert not job.pids(), "Owned child family did not terminate"
        finally:
            job.terminate()
            process.wait(timeout=10)
    class RefusingKernel:
        def AssignProcessToJobObject(self, *_):
            return False
    class OwnedChild:
        _handle = 1
        terminated = False
        waited = False
        def terminate(self):
            self.terminated = True
        def wait(self):
            self.waited = True
    rejected = object.__new__(guard.ProcessJob)
    rejected.kernel, rejected.handle = RefusingKernel(), 1
    child = OwnedChild()
    try:
        rejected.assign_and_resume(child)
    except OSError:
        pass
    else:
        raise AssertionError("Assignment failure was accepted")
    assert child.terminated and child.waited
    assert "taskkill" not in Path(guard.__file__).read_text(encoding="utf-8")
    assert not any(name in sys.modules for name in ("numpy", "torch", "physicsnemo"))
    result = {"status": "passed_tiny_owned_child_job_check", "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - start, "root_child_pid": process.pid, "grandchild_pid": descendant, "root_exit_code": process.returncode, "surviving_grandchild_observed_after_parent_exit": True, "family_sample_RSS_bytes": family_rss, "job_empty_after_termination": True, "assignment_failure_owned_handle_termination_checked_with_mock": True, "host_available_bytes_before": before, "host_available_bytes_after": guard.available_memory(), "guard_source_sha256": hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest(), "framework_imported": False, "network_used": False, "build_or_install_performed": False}
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
