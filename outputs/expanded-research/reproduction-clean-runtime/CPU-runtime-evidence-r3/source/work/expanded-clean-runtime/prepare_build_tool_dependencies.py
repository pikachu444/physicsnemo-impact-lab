"""Record the fixed wheel build-tool dependency before offline build-tool install."""
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

assert sys.flags.no_site
root = Path(__file__).resolve().parents[2]
own = root / "work/expanded-clean-runtime"
out = root / "outputs/expanded-research/reproduction-clean-runtime"
tools = json.loads((out / "build-tools-resolution.json").read_text())
runtime = json.loads((out / "wheel-resolution.json").read_text())
rows = tools["packages"]
assert {row["name"]: row["version"] for row in rows} == {"setuptools": "80.9.0", "wheel": "0.45.1"}
observations, needed = [], set()
for row in rows:
    path = own / "wheelhouse" / row["filename"]
    assert path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        data = archive.read(name)
    requires = BytesParser().parsebytes(data).get_all("Requires-Dist", [])
    active = []
    for value in requires:
        if ";" in value and "extra" in value.split(";", 1)[1]:
            continue
        assert re.fullmatch(r"packaging\s*>=\s*24(?:\.0)?", value), "Unexpected default build-tool dependency: " + value
        active.append(value)
        needed.add("packaging")
    observations.append({"name": row["name"], "version": row["version"], "wheel_sha256": row["sha256"],
                         "METADATA_sha256": hashlib.sha256(data).hexdigest(), "Requires_Dist": requires,
                         "default_dependencies_on_Python313": active})
additional = []
for name in sorted(needed):
    row = next(row for row in runtime["packages"] if row["name"] == name)
    assert row["source_kind"] == "publisher_pypi_wheel" and row["url"].startswith("https://files.pythonhosted.org/")
    assert tuple(int(part) for part in row["version"].split(".")) >= (24, 0)
    path = own / "wheelhouse" / row["filename"]
    assert path.stat().st_size == row["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    additional.append({key: row[key] for key in ("name", "version", "filename", "url", "sha256", "bytes")})
protocol = {"status": "exact_build_tool_dependencies_recorded_before_install", "pure_source_exception": "antlr4-python3-runtime4.9.3 only",
            "pinned_build_tools": {"setuptools": "80.9.0", "wheel": "0.45.1"}, "dependency_metadata": observations,
            "additional_official_wheels_from_unchanged_runtime90_lock": additional,
            "sdist_or_native_build_added": False, "runtime90_lock_changed": False, "installation_performed_by_this_script": False}
path = out / "build-tool-dependency-protocol-r3.json"
with path.open("x", encoding="utf-8") as stream:
    json.dump(protocol, stream, indent=2)
    stream.write("\n")
for name in ("build-tools-lock.txt", "build-tools-resolution.json"):
    before = out / name
    preserved = out / name.replace(".txt", "-r1.txt").replace(".json", "-r1.json")
    with preserved.open("xb") as stream:
        stream.write(before.read_bytes())
lock = "\n".join(row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"] for row in [*rows, *additional]) + "\n"
(out / "build-tools-lock.txt").write_text(lock, encoding="utf-8")
with (out / "build-tools-lock-r3.txt").open("x", encoding="utf-8") as stream:
    stream.write(lock)
print(json.dumps({"status": protocol["status"], "additional": additional, "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
