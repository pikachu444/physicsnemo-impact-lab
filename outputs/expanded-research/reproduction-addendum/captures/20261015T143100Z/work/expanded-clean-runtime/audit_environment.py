"""Metadata/source/path audit under the clean interpreter; no framework imports."""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import site
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace.resolve()
    target = (root / "work/expanded-clean-runtime/venv").resolve()
    assert Path(sys.prefix).resolve() == target and sys.flags.isolated
    assert not site.ENABLE_USER_SITE
    assert "include-system-site-packages = false" in (target / "pyvenv.cfg").read_text()
    violations = [value for value in sys.path if "site-packages" in value.lower() and not Path(value).resolve().is_relative_to(target)]
    assert not violations, "External site-packages on sys.path"
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    inventory = json.loads((output / "installed-metadata-inventory.json").read_text(encoding="utf-8"))
    source_contract = json.loads((output / "core-source-contract.json").read_text(encoding="utf-8"))
    packages = []
    for name, recorded in inventory["selected"].items():
        dist = metadata.distribution(name)
        assert dist.version == recorded["version"], name
        assert Path(dist.locate_file("")).resolve().is_relative_to(target), name
        packages.append({"name": name, "version": dist.version, "origin": "${CLEAN_ENV}/Lib/site-packages"})
    modules = {}
    for name in ("torch", "torchvision", "torch_geometric", "torch_scatter", "physicsnemo", "numpy", "scipy", "matplotlib", "warp"):
        specification = importlib.util.find_spec(name)
        assert specification is not None and specification.origin
        path = Path(specification.origin).resolve()
        assert path.is_relative_to(target), name
        modules[name] = "${CLEAN_ENV}/" + path.relative_to(target).as_posix()
        assert name not in sys.modules, "Framework module imported during static audit"
    pth_files = []
    for path in (target / "Lib/site-packages").glob("*.pth"):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "import ", "import\t")):
                resolved = (path.parent / line).resolve()
                assert resolved.is_relative_to(target), "External .pth path"
        assert "zz_reuse_physicsnemo" not in path.name
        pth_files.append(path.name)
    dist = metadata.distribution("nvidia-physicsnemo")
    record_file = next(path for path in dist.files if str(path).endswith(".dist-info/RECORD"))
    with Path(dist.locate_file(record_file)).open(newline="", encoding="utf-8") as stream:
        record = {row[0]: row[1] for row in csv.reader(stream)}
    sources = []
    for expected in source_contract["files"]:
        path = Path(dist.locate_file(expected["path"]))
        value = hashlib.sha256(path.read_bytes()).digest()
        assert value.hex() == expected["sha256"], expected["path"]
        assert record[expected["path"]] == "sha256=" + base64.urlsafe_b64encode(value).decode().rstrip("="), expected["path"]
        sources.append({"path": expected["path"], "sha256": value.hex(), "wheel_and_RECORD_equal": True})
    result = {"status": "passed_static_clean_environment_audit_only", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "site_packages_inheritance": False, "user_site_enabled": site.ENABLE_USER_SITE, "isolated_interpreter": bool(sys.flags.isolated),
              "packages": packages, "top_level_module_locations_without_import": modules, "pth_files": pth_files,
              "core_source_files_checked_against_wheel_and_RECORD": sources, "framework_imported": False, "GPU_used": False,
              "runtime_functional_and_validation_reload_checked": False}
    (output / "environment-static-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "packages": len(packages), "source_files": len(sources), "framework_imported": False}))


if __name__ == "__main__":
    main()
