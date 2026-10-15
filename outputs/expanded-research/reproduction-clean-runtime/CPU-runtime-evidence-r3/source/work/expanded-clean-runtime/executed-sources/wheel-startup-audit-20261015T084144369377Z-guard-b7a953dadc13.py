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


class ProcessEntry(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD), ("parent_pid", wintypes.DWORD), ("priority", ctypes.c_long), ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]


def family_pids(pid):
    kernel = ctypes.windll.kernel32
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError("Cannot inspect child process family")
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        first, following = kernel.Process32FirstW, kernel.Process32NextW
        for function in (first, following):
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
            function.restype = wintypes.BOOL
        parents = {}
        valid = first(handle, ctypes.byref(entry))
        while valid:
            parents[entry.pid] = entry.parent_pid
            valid = following(handle, ctypes.byref(entry))
        found = {pid}
        while True:
            expanded = found | {child for child, parent in parents.items() if parent in found}
            if expanded == found:
                return sorted(found)
            found = expanded
    finally:
        kernel.CloseHandle(handle)


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
        if completed_download.get("status") != "complete_exact_wheels_verified":
            raise ValueError("ANTLR source exception begins only after exact Torch/all-wheel acquisition completes")
    now = datetime.now(timezone.utc)
    attempt = args.stage + "-" + now.strftime("%Y%m%dT%H%M%S%fZ")
    record = {"stage": args.stage, "attempt": attempt, "started_utc": now.isoformat(),
              "CPU_threads": 1, "priority": "BelowNormal", "RSS_cap_bytes": 350 * 1024**2,
              "preflight_min_available_bytes": int(1.5 * 1024**3), "running_min_available_bytes": 512 * 1024**2,
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
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="-1")
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
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        record.update({"status": "running", "child_started": True, "pid": process.pid, "child_started_utc": datetime.now(timezone.utc).isoformat(), "log": "work/expanded-clean-runtime/" + log.name, "peak_sampled_RSS_bytes": 0, "peak_sampled_family_RSS_bytes": 0, "observed_descendant_pids": []})
        save()
        print(json.dumps({"status": "running", "stage": args.stage, "pid": process.pid, "started_utc": record["started_utc"]}), flush=True)
        while process.poll() is None:
            memory = rss(process.pid)
            members = family_pids(process.pid)
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
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
                record.update({"status": "stopped_resource_guard", "guard_reason": reason, "available_bytes_at_stop": free, "disk_free_bytes_at_stop": disk})
                break
            time.sleep(1)
        process.wait()
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
