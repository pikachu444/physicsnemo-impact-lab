"""Audit installation using metadata only; never import ML frameworks."""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
OUT = WORKSPACE / "outputs/expanded-research/benchmarks/bumper/environment"
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED = {
    "torch": "2.10.0+cu126",
    "torchvision": "0.25.0+cu126",
    "torch-scatter": "2.1.2+pt210cu126",
    "torch-geometric": "2.8.1",
    "nvidia-physicsnemo": "2.2.2",
}
TARGETS = [*EXPECTED, "xxhash"]

packages = []
dependency_findings = []
for name in TARGETS:
    dist = metadata.distribution(name)
    location = Path(dist.locate_file("")).resolve()
    is_own_site = location == (ROOT / "venv/Lib/site-packages").resolve()
    if name in EXPECTED:
        assert dist.version == EXPECTED[name], (name, dist.version)
    if name != "nvidia-physicsnemo":
        assert is_own_site, (name, location)
    else:
        assert location == (WORKSPACE / "work/gpu-experiment/venv/Lib/site-packages").resolve()
    active_requirements = []
    for text in dist.requires or []:
        req = Requirement(text)
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            installed = None
        satisfied = installed is not None and (not req.specifier or installed in req.specifier)
        item = {"requirement": text, "installed_version": installed, "satisfied": satisfied}
        active_requirements.append(item)
        if not satisfied:
            dependency_findings.append({"package": name, **item})
    packages.append({
        "name": dist.metadata["Name"], "version": dist.version,
        "site_packages": str(location), "own_site": is_own_site,
        "requires_python": dist.metadata.get("Requires-Python"),
        "active_dependencies": active_requirements,
    })

wheel_records = json.loads((ROOT / "install-wheels.json").read_text(encoding="utf-8-sig"))
sources = json.loads((ROOT / "install-sources.json").read_text(encoding="utf-8-sig"))
for wheel in wheel_records:
    filename = wheel["filename"]
    source = sources[filename]
    if source.get("published_sha256"):
        assert source["published_sha256"] == wheel["sha256"], filename
    if source.get("published_bytes"):
        assert source["published_bytes"] == wheel["bytes"], filename
    wheel.update(source)
    with zipfile.ZipFile(ROOT / "wheels" / filename) as archive:
        wheel_text = archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/WHEEL"))).decode()
        wheel["tags"] = [line.removeprefix("Tag: ") for line in wheel_text.splitlines() if line.startswith("Tag: ")]

inventory = sorted({
    (canonicalize_name(d.metadata["Name"]), d.version, str(Path(d.locate_file("")).resolve()))
    for d in metadata.distributions() if d.metadata.get("Name")
})
module_locations = {
    name: importlib.util.find_spec(name).origin
    for name in ["torch", "torchvision", "torch_scatter", "torch_geometric", "physicsnemo", "xxhash"]
}
for name in ["torch", "torchvision", "torch_scatter", "torch_geometric", "xxhash"]:
    assert Path(module_locations[name]).resolve().is_relative_to((ROOT / "venv/Lib/site-packages").resolve()), (name, module_locations[name])
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "system_site_packages": True,
    "sys_path": sys.path,
    "module_locations_without_import": module_locations,
    "shared_torch_metadata": [
        {"metadata_directory": str(path), "version": metadata.Distribution.at(path).version}
        for path in sorted((WORKSPACE / "work/gpu-experiment/venv/Lib/site-packages").glob("torch-*.dist-info"))
    ],
    "packages": packages,
    "wheel_files": wheel_records,
    "pypi_release_metadata": json.loads((ROOT / "install-release-metadata.json").read_text(encoding="utf-8-sig")),
    "pyg_version_selection": "Requested 2.9.0 is absent from the live official PyPI releases response. Selected the latest released 2.8.1 wheel instead; no source checkout or unpublished package was installed.",
    "commands": json.loads((ROOT / "install-commands.json").read_text(encoding="utf-8-sig")),
    "prior_attempt_command_records": {
        path.name: json.loads(path.read_text(encoding="utf-8-sig"))
        for path in sorted(ROOT.glob("install-commands.failed-*.json"))
    },
    "dependency_findings": dependency_findings,
    "framework_import_performed": False,
    "gpu_execution_performed": False,
    "shared_environment_modified": False,
    "validation_limit": "Metadata, wheel hashes, package paths and dependency checks only. Model imports, CUDA/extension loading and model execution belong to the exclusive execution worker.",
}
for path in [ROOT / "install-manifest.json", OUT / "installation-manifest.json"]:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
(OUT / "package-inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
pins = [f"{p['name']}=={p['version']}" for p in packages]
(OUT / "selected-package-versions.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
print(json.dumps({
    "python": sys.executable,
    "selected_versions": pins,
    "wheel_count": len(wheel_records),
    "dependency_findings": dependency_findings,
    "manifest": str(OUT / "installation-manifest.json"),
    "framework_import_performed": False,
    "gpu_execution_performed": False,
}, indent=2, ensure_ascii=False))
