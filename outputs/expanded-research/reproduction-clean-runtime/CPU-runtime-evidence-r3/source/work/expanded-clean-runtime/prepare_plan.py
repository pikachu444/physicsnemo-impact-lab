"""Preserve a small reviewable clean-runtime plan; does not install anything."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from inspect_runtime import ROOT, OUT, public, sha


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    assert sys.flags.no_site
    inventory = json.loads((OUT / "installed-metadata-inventory.json").read_text(encoding="utf-8"))
    roots = ROOT / "work/expanded-ml/wheels"
    sources = json.loads((ROOT / "work/expanded-ml/install-sources.json").read_text(encoding="utf-8"))
    known = json.loads((ROOT / "work/expanded-ml/install-wheels.json").read_text(encoding="utf-8"))
    wheels = []
    for original in known:
        path = roots / original["filename"]
        row = {**original, **sources[original["filename"]], "local_path": public(path), "present_now": path.exists(),
               "hash_status": "historically_recorded_not_currently_present"}
        if path.exists():
            assert path.stat().st_size < 20 * 1024**2
            assert path.stat().st_size == original["bytes"] and sha(path.read_bytes()) == original["sha256"]
            row["hash_status"] = "current_local_bytes_verified"
            with zipfile.ZipFile(path) as archive:
                wheel_path = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
                row["wheel_tags"] = [line[5:].strip() for line in archive.read(wheel_path).decode().splitlines() if line.startswith("Tag: ")]
        wheels.append(row)
    cached = Path.home() / "AppData/Local/pip/Cache/http-v2/9/5/3/9/d/9539db41fd237aca378f99a587f85c4608c282b947f112399447f575.body"
    assert cached.stat().st_size == 2942045
    assert sha(cached.read_bytes()) == "c447771384d92f5c31547e293f65b70e0947f8829981f34c7e8300daf1b8d2eb"
    wheels.append({"filename": "nvidia_physicsnemo-2.2.2-py3-none-any.whl", "bytes": cached.stat().st_size, "sha256": sha(cached.read_bytes()),
                   "url": "https://files.pythonhosted.org/packages/5d/22/1750a9651df15d81440f4a5ad7a7610c3f8cdbe7fd69245a9879b72da8b3/nvidia_physicsnemo-2.2.2-py3-none-any.whl",
                   "url_evidence": "outputs/experiment/pip-physicsnemo.log:26,36", "local_path": public(cached), "present_now": True,
                   "hash_status": "current_local_bytes_verified_no_fresh_remote_metadata_check", "wheel_tags": ["py3-none-any"]})
    for path in sorted((Path.home() / "AppData/Local/pip/Cache/wheels").rglob("antlr4_python3_runtime-4.9.3-*.whl")):
        origin = json.loads((path.parent / "origin.json").read_text())
        wheels.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": sha(path.read_bytes()), "local_path": public(path), "present_now": True,
                       "wheel_tags": ["py3-none-any"], "origin": origin, "provenance_limit": "Previously cached pure-Python wheel built from the recorded source archive; not a publisher-distributed wheel. No source build occurs in this task."})
    save("cached-wheel-inventory.json", {"status": "local_metadata_and_current_small_wheel_hashes_only", "wheels": wheels,
         "not_claimed": ["All 90 dependency wheels locally cached", "Every dependency has a publisher binary wheel", "Fresh verification of remote published hashes", "Torch 2.10 wheel absent everywhere on disk"]})
    package = ROOT / "work/gpu-experiment/venv/Lib/site-packages/physicsnemo"
    paths = set()
    for directory in ("models/transolver", "models/flare", "models/geotransolver", "models/meshgraphnet", "models/figconvnet", "core", "distributed"):
        paths.update((package / directory).glob("*.py"))
    paths.add(package / "__init__.py")
    if (package / "compat/__init__.py").exists():
        paths.add(package / "compat/__init__.py")
    checked = []
    with zipfile.ZipFile(cached) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            internal = "physicsnemo/" + path.relative_to(package).as_posix()
            assert data == archive.read(internal), internal
            imports = set()
            for node in ast.walk(ast.parse(data)):
                if isinstance(node, ast.Import): imports.update(item.name.split(".")[0] for item in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module: imports.add(node.module.split(".")[0])
            checked.append({"path": internal, "installed_origin": public(path), "bytes": len(data), "sha256": sha(data),
                            "cached_wheel_bytes_equal": True, "static_import_roots_overapproximation": sorted(imports)})
    save("core-source-contract.json", {"distribution": "nvidia-physicsnemo", "version": "2.2.2", "cached_wheel_sha256": wheels[5]["sha256"], "files": checked,
         "not_the_same_as_official_example_source_pin": "11685edb3ec1fb6f25665e96cf7a8c4d11a23f16 / development example source must not replace installed 2.2.2 core",
         "static_import_limit": "AST lists are conservative and include guarded optional imports; no framework imports performed"})
    pins = [name + "==" + row["version"] for name, row in inventory["selected"].items()]
    (OUT / "constraints-installed.txt").write_text("# Candidate exact installed-version constraints, NOT an installation/hash lock.\n" + "\n".join(pins) + "\n", encoding="utf-8")
    plan = {"status": "PLAN_PREPARED_NOT_INSTALLED_OR_RUNTIME_VALIDATED", "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "target": {"python": "3.13.5 CPython win_amd64", "new_environment": "work/expanded-clean-runtime/venv", "system_site_packages": False,
                       "user_site": False, "inherited_pth_paths": False, "core": "PhysicsNeMo 2.2.2 exact source hashes", "torch": "2.10.0+cu126", "scatter": "2.1.2+pt210cu126"},
            "dependency_scope": {"packages": len(pins), "reason": "Runtime roots plus complete declared Windows dependencies, including required tensordict[zarr]; unrelated applications excluded for clean pip check",
                                 "metadata_missing": inventory["missing_declared_dependencies"], "marker_unresolved": inventory["unresolved_requirement_markers"],
                                 "installed_record_bytes_estimate": inventory["selected_installed_record_bytes"], "not_yet_a_hash_lock": True},
            "resource_proposal": {"CPU_threads": 1, "priority": "BelowNormal", "non_ML_child_RSS_cap_bytes": 350 * 1024**2,
                                  "minimum_available_host_bytes_before_non_ML_stage": 1024**3, "minimum_free_disk_bytes": 15 * 1024**3,
                                  "expected_incremental_disk_budget_bytes": 12 * 1024**3, "current_disk_free_bytes": inventory["disk_free_bytes"],
                                  "Torch_missing_wheel_expected_download_bytes": 2589881709, "stage_policy": "Fail closed on resource guard; preserve interrupted staging; no repeated automatic retries during active model training"},
            "steps": [
                {"phase": 0, "status": "done_metadata_only", "action": "Inspect exact current source hashes, explicit metadata paths, local wheel availability, disk and inheritance"},
                {"phase": 1, "status": "requires_root_plan_approval", "action": "Resolve official metadata and binary-wheel availability against exact constraints; recover matching cached wheels first. Record package/version, URL, tags, size, publisher SHA where available, observed bytes SHA and cache provenance. Do not build from source."},
                {"phase": 2, "status": "requires_root_plan_approval_and_memory_gate", "action": "Create a separate venv without --system-site-packages. Clear PYTHONPATH/PYTHONHOME/PIP_USER; disable user-site. Use the existing CPython base executable. No old .pth files or package directories are copied."},
                {"phase": 3, "status": "requires_root_plan_approval_and_memory_gate", "action": "Retrieve only unresolved binary wheels, including the exact official Torch wheel if unavailable locally. Stream downloads and hashes, preserve exact URL/SHA/size. Generate a complete --require-hashes lock; install from an offline wheelhouse with --no-index --only-binary=:all:. Never update/uninstall global packages."},
                {"phase": 4, "status": "not_run", "action": "Use isolated environment pip check; require zero missing/conflicting dependencies. Verify pyvenv.cfg false, no external .pth path, user site disabled and no old environment/global/user package paths. Compare all recorded core source hashes without importing frameworks."},
                {"phase": 5, "status": "requires_separate_explicit_framework_GPU_slot", "action": "Import frameworks only in the allocated slot. Log actual imported module files/versions under the clean venv; standard library and application source are separately identified. Run tiny train-only CUDA forward/backward/update checks for needed model families. Do not call this full training reproduction."},
                {"phase": 6, "status": "requires_separate_explicit_framework_GPU_slot", "action": "Reload one already-frozen validation checkpoint and compare its existing validation anchor with recorded tolerance/hashes. No held-out test, retuning or repeat full experiments. Report exactly which architecture/anchor was checked."}
            ],
            "known_limits": ["Exact Torch wheel is absent in the bounded caches checked; about 2.59GB retrieval may be necessary after approval", "Only the listed cached wheel URLs/hashes are known locally; remaining official metadata/compatible wheel availability is pending approval", "ANTLR4 4.9.3 cached wheels were historically built from source. If publisher-only wheels are required, keep this as an explicit unresolved dependency rather than build one", "Existing plate driver prepends work/expanded-plate/deps. Clean runtime verification must use a separate clean launcher and reject fallback package imports; existing driver/source stays untouched", "A clean environment passing representative functional and frozen validation checks does not prove reproduction of every training run"],
            "actions_performed": {"venv_created": False, "pip_run": False, "network_or_download": False, "framework_import": False, "GPU_used": False, "arrays_or_weights_read": False}}
    save("plan.json", plan)
    print(json.dumps({"status": plan["status"], "candidate_packages": len(pins), "source_files_hash_matched": len(checked), "cached_wheel_candidates": len(wheels), "installation_started": False}, indent=2))


if __name__ == "__main__":
    main()
