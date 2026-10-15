"""Read installed metadata without site initialization, framework imports or pip."""
from __future__ import annotations
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import site
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/expanded-research/reproduction-clean-runtime"
SITES = (ROOT / "work/expanded-ml/venv/Lib/site-packages", ROOT / "work/gpu-experiment/venv/Lib/site-packages", Path(site.getusersitepackages()), Path(sys.base_prefix) / "Lib/site-packages")
ROOTS = ("torch", "torchvision", "torch-geometric", "torch-scatter", "nvidia-physicsnemo", "numpy", "scipy", "matplotlib", "psutil", "hydra-core", "omegaconf", "warp-lang", "h5py")
ENVIRONMENT = {"python_version": "3.13", "python_full_version": "3.13.5", "os_name": "nt", "sys_platform": "win32", "platform_system": "Windows", "platform_machine": "AMD64", "platform_python_implementation": "CPython", "implementation_name": "cpython", "implementation_version": "3.13.5", "extra": ""}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def public(path):
    value = str(path)
    value = re.sub(re.escape(str(ROOT)), "${WORKSPACE}", value, flags=re.I)
    value = re.sub(re.escape(str(Path(sys.base_prefix))), "${PYTHON_HOME}", value, flags=re.I)
    value = re.sub(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+", "${USER_HOME}", value, flags=re.I)
    return value.replace("\\", "/")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def marker_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return ENVIRONMENT[node.id]
    if isinstance(node, ast.BoolOp):
        values = [bool(marker_value(value)) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = marker_value(node.left)
        for operator, other in zip(node.ops, node.comparators):
            right = marker_value(other)
            if isinstance(operator, (ast.In, ast.NotIn)):
                accepted = left in right
                if isinstance(operator, ast.NotIn):
                    accepted = not accepted
            else:
                a, b = left, right
                if re.fullmatch(r"\d+(?:\.\d+)*", str(a)) and re.fullmatch(r"\d+(?:\.\d+)*", str(b)):
                    a, b = tuple(map(int, a.split("."))), tuple(map(int, b.split(".")))
                if isinstance(operator, ast.Eq): accepted = a == b
                elif isinstance(operator, ast.NotEq): accepted = a != b
                elif isinstance(operator, ast.Lt): accepted = a < b
                elif isinstance(operator, ast.LtE): accepted = a <= b
                elif isinstance(operator, ast.Gt): accepted = a > b
                elif isinstance(operator, ast.GtE): accepted = a >= b
                else: raise ValueError("Unsupported marker comparison")
            if not accepted:
                return False
            left = right
        return True
    raise ValueError("Unsupported metadata marker syntax")


def enabled(requirement, extra=""):
    ENVIRONMENT["extra"] = extra
    if ";" not in requirement:
        return True
    return bool(marker_value(ast.parse(requirement.split(";", 1)[1].strip(), mode="eval").body))


def inspect():
    available = {}
    for site in SITES:
        for directory in sorted(site.glob("*.dist-info")):
            name = canonical(directory.name[:-10].rsplit("-", 1)[0])
            available.setdefault(name, []).append(directory)
    pending, selected, missing, excluded, unknown = [(name, ()) for name in ROOTS], {}, set(), [], []
    active_extras = {}
    while pending:
        requested_name, extras = pending.pop(0)
        name = canonical(requested_name)
        extras = {"", *extras}
        if (name in selected and extras <= active_extras[name]) or name in missing:
            continue
        if name not in available:
            missing.add(name)
            continue
        active_extras.setdefault(name, set()).update(extras)
        info = available[name][0]
        if name not in selected:
            dist = metadata.PathDistribution(info)
            meta = dist.metadata
            requirements = meta.get_all("Requires-Dist", [])
            selected[name] = {"name": meta["Name"], "version": meta["Version"], "metadata_path": public(info / "METADATA"),
                          "metadata_sha256": sha((info / "METADATA").read_bytes()), "requires_python": meta.get("Requires-Python"),
                          "requires_dist": requirements, "alternate_installed_metadata": [public(item) for item in available[name][1:]]}
            wheel = info / "WHEEL"
            selected[name]["installed_wheel_tags"] = [line[5:].strip() for line in wheel.read_text(encoding="utf-8").splitlines() if line.startswith("Tag: ")] if wheel.exists() else []
            record_path = info / "RECORD"
            if record_path.exists():
                with record_path.open(newline="", encoding="utf-8") as stream:
                    selected[name]["installed_record_bytes"] = sum(int(row[2]) for row in csv.reader(stream) if len(row) > 2 and row[2].isdigit())
        requirements = selected[name]["requires_dist"]
        selected[name]["requested_extras"] = sorted(active_extras[name] - {""})
        direct = info / "direct_url.json"
        if direct.exists():
            value = json.loads(direct.read_text(encoding="utf-8"))
            if "url" in value and str(value["url"]).startswith("file:"):
                value["url"] = re.sub(r"[A-Za-z]:[/\\]Users[/\\][^/\\]+", "${USER_HOME}", value["url"], flags=re.I)
            selected[name]["direct_url"] = value
        for requirement in requirements:
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[([^\]]+)\])?", requirement)
            if not match:
                unknown.append({"parent": name, "requirement": requirement, "reason": "Unparsed requirement name"})
                continue
            try:
                use = any(enabled(requirement, extra) for extra in active_extras[name])
            except (KeyError, ValueError, SyntaxError, TypeError) as error:
                unknown.append({"parent": name, "requirement": requirement, "reason": str(error)})
                continue
            if use:
                pending.append((match.group(1), tuple(part.strip() for part in (match.group(2) or "").split(",") if part.strip())))
            else:
                excluded.append({"parent": name, "requirement": requirement})
    configs = []
    for path in (ROOT / "work/expanded-ml/venv/pyvenv.cfg", ROOT / "work/gpu-experiment/venv/pyvenv.cfg"):
        configs.append({"path": public(path), "include_system_site_packages": "include-system-site-packages = true" in path.read_text(), "sha256": sha(path.read_bytes())})
    pth = []
    for site in SITES[:2]:
        for path in site.glob("*.pth"):
            data = path.read_bytes()
            pth.append({"path": public(path), "sha256": sha(data), "content": public(data.decode("utf-8"))})
    usage = shutil.disk_usage(ROOT)
    return {"status": "metadata_inventory_only_no_install_or_framework_import", "observed_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "site_initialization_disabled": bool(sys.flags.no_site), "metadata_search_precedence": [public(path) for path in SITES],
            "root_packages": list(ROOTS), "selected_declared_dependency_count": len(selected), "selected": dict(sorted(selected.items())),
            "missing_declared_dependencies": sorted(missing), "excluded_by_default_windows_no_extra_markers": excluded,
            "unresolved_requirement_markers": unknown, "venv_inheritance": configs, "pth_files": pth,
            "disk_free_bytes": usage.free, "disk_total_bytes": usage.total,
            "selected_installed_record_bytes": sum(record.get("installed_record_bytes", 0) for record in selected.values()),
            "limitations": ["Metadata requirements include optional imports used by non-selected PhysicsNeMo features; exact import closure is separately reviewed", "No framework import, GPU use, pip resolution, install, package copying or download performed", "Wheel hashes are researched separately; this is not yet an install lock", "This minimal marker evaluator covers observed default Windows requirements; all unsupported markers are explicit"]}


if __name__ == "__main__":
    assert sys.flags.no_site, "Run with python -S to avoid inherited .pth/site initialization"
    result = inspect()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "installed-metadata-inventory.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "selected_declared_dependency_count", "missing_declared_dependencies", "unresolved_requirement_markers", "disk_free_bytes")}, indent=2))
