"""Run one metadata/download stage under Windows host/process resource guards."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [(name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_page", "available_page", "total_virtual", "available_virtual", "available_extended")]


class ProcessCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private_usage")]


class BasicJobLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD), ("scheduling_class", wintypes.DWORD)]


class ExtendedJobLimits(ctypes.Structure):
    _fields_ = [("basic", BasicJobLimits), ("io_counters", ctypes.c_ulonglong * 6), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class JobProcessList(ctypes.Structure):
    _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD), ("pids", ctypes.c_size_t * 256)]


class ThreadEntry(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("tid", wintypes.DWORD), ("owner_pid", wintypes.DWORD), ("base_priority", ctypes.c_long), ("delta_priority", ctypes.c_long), ("flags", wintypes.DWORD)]


class ProcessJob:
    """The child is assigned while suspended, so every descendant stays contained."""
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, arguments, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
            ("Thread32First", [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)], wintypes.BOOL),
            ("Thread32Next", [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)], wintypes.BOOL),
            ("OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
        ):
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedJobLimits()
        limits.basic.flags = 0x2000  # Kill contained processes if this guard exits unexpectedly.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_and_resume(self, process):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            process.terminate()
            process.wait()
            raise ctypes.WinError(ctypes.get_last_error())
        snapshot = self.kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            valid = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while valid and entry.owner_pid != process.pid:
                valid = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            if not valid:
                raise RuntimeError("Suspended child main thread was not found")
            thread = self.kernel.OpenThread(2, False, entry.tid)
            if not thread:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if self.kernel.ResumeThread(thread) != 1:
                    raise RuntimeError("Child did not have exactly one startup suspension")
            finally:
                self.kernel.CloseHandle(thread)
        finally:
            self.kernel.CloseHandle(snapshot)

    def pids(self):
        value = JobProcessList()
        if not self.kernel.QueryInformationJobObject(self.handle, 3, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if value.assigned > len(value.pids) or value.count > len(value.pids):
            raise RuntimeError("Contained process count exceeds bounded inspection buffer")
        return sorted(value.pids[:value.count])

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def available_memory():
    value = MemoryStatus()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("Cannot read host memory")
    return value.available_physical


def rss(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
    if not handle:
        return None
    try:
        value = ProcessCounters()
        value.cb = ctypes.sizeof(value)
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessCounters), wintypes.DWORD]
        function.restype = wintypes.BOOL
        if not function(handle, ctypes.byref(value), value.cb):
            raise OSError("Cannot measure child memory")
        return value.working_set
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=("metadata", "download", "create", "bootstrap", "resolve-local", "install", "check", "audit", "antlr-source", "build-tools-download", "build-create", "build-bootstrap", "build-tools-install", "build-wheel", "finalize-antlr", "wheel-startup-audit"), required=True)
    args = parser.parse_args()
    assert os.name == "nt" and sys.flags.no_site, "Use Windows Python -S"
    root = args.workspace.resolve()
    own = root / "work/expanded-clean-runtime"
    out = root / "outputs/expanded-research/reproduction-clean-runtime"
    for path in (root / "work", own):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("Linked runtime destination refused")
    own.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    scripts = {"metadata": "resolve_wheels.py", "download": "download_wheels.py", "create": "create_environment.py", "audit": "audit_environment.py", "antlr-source": "prepare_antlr_source.py", "build-tools-download": "resolve_build_tools.py", "build-create": "create_environment.py", "finalize-antlr": "finalize_antlr_build.py", "wheel-startup-audit": "audit_wheel_startup.py"}
    script = Path(__file__).resolve().parent / scripts[args.stage] if args.stage in scripts else None
    if script is not None and not script.is_file():
        raise FileNotFoundError("Stage source is not ready: " + script.name)
    if args.stage == "antlr-source":
        completed_download = json.loads((out / "wheel-download-manifest.json").read_text(encoding="utf-8"))
        if completed_download.get("status") not in ("complete_exact_wheels_verified", "complete_exact_wheels_verified_with_antlr_deferred"):
            raise ValueError("ANTLR source exception begins only after exact Torch/all-wheel acquisition completes")
    now = datetime.now(timezone.utc)
    attempt = args.stage + "-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    record = {"stage": args.stage, "attempt": attempt, "started_utc": now.isoformat(),
              "CPU_threads": 1, "priority": "BelowNormal", "RSS_cap_bytes": 350 * 1024**2,
              "preflight_min_available_bytes": int(1.0 * 1024**3), "running_min_available_bytes": 512 * 1024**2,
              "preflight_min_disk_bytes": 15 * 1024**3, "running_min_disk_bytes": 5 * 1024**3,
              "framework_import_authorized": False, "GPU_used": False}
    own_data = Path(__file__).read_bytes()
    own_sha = hashlib.sha256(own_data).hexdigest()
    own_snapshot = own / "executed-sources" / (attempt + "-guard-" + own_sha[:12] + ".py")
    own_snapshot.parent.mkdir(exist_ok=True)
    with own_snapshot.open("xb") as stream:
        stream.write(own_data)
    record.update({"supervisor_source_sha256": own_sha, "supervisor_source_snapshot": own_snapshot.relative_to(root).as_posix()})

    def save():
        (out / (attempt + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    free = available_memory()
    disk = shutil.disk_usage(root).free
    record.update({"preflight_available_bytes": free, "preflight_disk_free_bytes": disk})
    if free < record["preflight_min_available_bytes"] or disk < record["preflight_min_disk_bytes"]:
        record.update({"status": "deferred_resource_preflight", "child_started": False})
        save()
        print(json.dumps(record, indent=2), flush=True)
        return 3
    time.sleep(10)
    second = available_memory()
    record["stable_preflight_available_observations_bytes"] = [free, second]
    if second < record["preflight_min_available_bytes"]:
        record.update({"status": "deferred_stable_memory_preflight", "child_started": False})
        save()
        print(json.dumps(record, indent=2), flush=True)
        return 3
    build_stage = args.stage in ("build-bootstrap", "build-tools-install", "build-wheel")
    environment_root = own / ("build-venv" if build_stage else "venv")
    interpreter = environment_root / "Scripts/python.exe"
    needs_environment = args.stage in ("bootstrap", "resolve-local", "install", "check", "audit", "build-bootstrap", "build-tools-install", "build-wheel")
    if needs_environment and not interpreter.is_file():
        raise FileNotFoundError("Clean interpreter does not exist; create/bootstrap stages are required")
    if needs_environment:
        if environment_root.is_symlink() or getattr(environment_root, "is_junction", lambda: False)():
            raise ValueError("Linked environment refused")
        if "include-system-site-packages = false" not in (environment_root / "pyvenv.cfg").read_text():
            raise ValueError("Environment inherits system packages")
    if script is not None:
        data = script.read_bytes()
        source_sha = hashlib.sha256(data).hexdigest()
        snapshot = own / "executed-sources" / (attempt + "-" + source_sha[:12] + "-" + script.name)
        snapshot.parent.mkdir(exist_ok=True)
        with snapshot.open("xb") as stream:
            stream.write(data)
        record.update({"source_origin": "work/expanded-clean-runtime/" + script.name, "executed_source_sha256": source_sha,
                       "executed_source_snapshot": snapshot.relative_to(root).as_posix()})
        command = [str(interpreter) if args.stage == "audit" else sys.executable, "-I" if args.stage == "audit" else "-S", "-B", "-X", "utf8", str(snapshot), "--workspace", str(root)]
        flags = {"metadata": "--resolve-metadata", "download": "--download-approved", "antlr-source": "--prepare-source-approved"}
        if args.stage in flags:
            command.append(flags[args.stage])
        if args.stage == "build-tools-download":
            command.append("--download-approved")
        elif args.stage == "build-create":
            command.append("--build-venv")
        elif args.stage == "finalize-antlr":
            command += ["--finalize-approved", "--build-receipt", str(out / "antlr-build-execution.json")]
    else:
        command = [str(interpreter), "-I", "-B", "-m"]
        if args.stage in ("bootstrap", "build-bootstrap"):
            command += ["ensurepip", "--default-pip"]
        elif args.stage == "check":
            command += ["pip", "check"]
        elif args.stage == "build-wheel":
            source_audit_path = out / "antlr-source-audit-r2.json"
            source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
            if source_audit["status"] != "prepared_not_built" or source_audit["source_audit"]["findings"]:
                raise ValueError("Pure ANTLR source audit did not pass")
            source_root = Path(source_audit["source_root"].replace("${WORKSPACE}", str(root)))
            if not source_root.resolve().is_relative_to((own / "antlr-source/extracted-r2").resolve()):
                raise ValueError("Build source escaped approved extraction")
            for member in source_audit["source_audit"]["members"]:
                if member["type"] == "file":
                    path = source_root.parent / Path(member["path"])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
                        raise ValueError("Source changed after static audit: " + member["path"])
            destination = own / "antlr-built-wheelhouse"
            if destination.exists():
                raise FileExistsError("Prior ANTLR build preserved; only one build is authorized")
            destination.mkdir()
            command += ["pip", "--disable-pip-version-check", "wheel", "--no-deps", "--no-build-isolation", "--no-index", "--no-cache-dir", "--wheel-dir", str(destination), str(source_root)]
            record.update({"source_root": source_audit["source_root"], "source_sha256": "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
                           "source_audit_sha256": hashlib.sha256(source_audit_path.read_bytes()).hexdigest(),
                           "build_tool_pins": {"setuptools": "80.9.0", "wheel": "0.45.1"}, "no_index": True, "no_deps": True, "no_build_isolation": True})
        else:
            lock = out / ("build-tools-lock.txt" if args.stage == "build-tools-install" else "requirements-lock-r2.txt")
            wheelhouse = own / ("wheelhouse" if args.stage == "build-tools-install" else "wheelhouse-r2")
            if not lock.is_file() or not wheelhouse.is_dir():
                raise FileNotFoundError("Required verified lock/wheelhouse revision is not ready")
            command += ["pip", "--disable-pip-version-check", "install", "--no-index", "--no-cache-dir", "--only-binary=:all:", "--require-hashes", "--no-user", "--find-links", str(wheelhouse), "-r", str(lock)]
            if args.stage == "resolve-local":
                command += ["--dry-run", "--ignore-installed", "--report", str(own / "local-resolver-report.json")]
            else:
                command += ["--no-deps", "--prefix", str(environment_root)]
        record["stdlib_or_clean_pip_stage"] = args.stage
    temporary = own / "tmp"
    if temporary.is_symlink() or getattr(temporary, "is_junction", lambda: False)():
        raise ValueError("Linked temporary destination refused")
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", TEMP=str(temporary), TMP=str(temporary))
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    for key in list(environment):
        if key.startswith("PIP_"):
            environment.pop(key)
    environment.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1")
    if args.stage == "build-wheel":
        environment.update(SOURCE_DATE_EPOCH="1704067200", PYTHONHASHSEED="0", TZ="UTC", PIP_NO_INDEX="1")
        record["environment"] = {name: environment[name] for name in ("SOURCE_DATE_EPOCH", "PYTHONHASHSEED", "TZ")}
    record["command"] = [value.replace(str(root), "${WORKSPACE}").replace(sys.base_prefix, "${PYTHON_HOME}") for value in command]
    if args.stage == "build-wheel":
        with (own / "antlr-one-build-started.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
    log = own / (attempt + ".log")
    start = time.perf_counter()
    with log.open("x", encoding="utf-8") as stream, ProcessJob() as job:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 4)
        record.update({"status": "running", "child_started": True, "pid": process.pid, "child_started_utc": datetime.now(timezone.utc).isoformat(), "log": "work/expanded-clean-runtime/" + log.name, "peak_sampled_RSS_bytes": 0, "peak_sampled_family_RSS_bytes": 0, "observed_descendant_pids": []})
        try:
            job.assign_and_resume(process)
            record["contained_before_resume"] = True
        except Exception as error:
            job.terminate()
            process.wait(timeout=30)
            record.update({"status": "failed_child_containment", "contained_before_resume": False, "containment_error": str(error)})
        save()
        print(json.dumps({"status": "running", "stage": args.stage, "pid": process.pid, "started_utc": record["started_utc"]}), flush=True)
        while process.poll() is None:
            memory = rss(process.pid)
            members = job.pids()
            family_memory = sum(rss(member) or 0 for member in members)
            record["observed_descendant_pids"] = sorted(set(record["observed_descendant_pids"]) | (set(members) - {process.pid}))
            record["peak_sampled_family_RSS_bytes"] = max(record["peak_sampled_family_RSS_bytes"], family_memory)
            free = available_memory()
            disk = shutil.disk_usage(root).free
            if memory is not None:
                record["peak_sampled_RSS_bytes"] = max(record["peak_sampled_RSS_bytes"], memory)
            reason = None
            if memory is None and process.poll() is None: reason = "child_memory_measurement_unavailable"
            elif family_memory > record["RSS_cap_bytes"]: reason = "child_family_RSS_cap"
            elif free < record["running_min_available_bytes"]: reason = "host_available_memory"
            elif disk < record["running_min_disk_bytes"]: reason = "disk_free"
            elif time.perf_counter() - start > 7200: reason = "two_hour_stage_wall_cap"
            if reason:
                job.terminate()
                process.wait(timeout=30)
                record.update({"status": "stopped_resource_guard", "guard_reason": reason, "available_bytes_at_stop": free, "disk_free_bytes_at_stop": disk})
                break
            time.sleep(1)
        process.wait()
        remaining = job.pids()
        record["remaining_descendant_pids_before_cleanup"] = remaining
        if remaining:
            job.terminate()
            deadline = time.perf_counter() + 10
            while job.pids() and time.perf_counter() < deadline:
                time.sleep(0.1)
        remaining = job.pids()
        record.update({"remaining_descendant_pids_at_end": remaining, "descendants_reaped": not remaining})
        if remaining:
            record["status"] = "failed_descendant_cleanup"
    if record["status"] == "running":
        record["status"] = "completed" if process.returncode == 0 else "failed_child"
    record.update({"exit_code": process.returncode, "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": time.perf_counter() - start, "child_reaped": True})
    record.update({"available_bytes_at_end": available_memory(), "disk_free_bytes_at_end": shutil.disk_usage(root).free})
    save()
    if args.stage == "build-wheel":
        with (out / "antlr-build-execution.json").open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    print(json.dumps(record, indent=2), flush=True)
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
