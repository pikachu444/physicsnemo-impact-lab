"""Verify one approved ANTLR build and publish an absent runtime wheelhouse/lock revision."""
from __future__ import annotations

import sys

assert sys.flags.no_site, "Run with python -S; site and .pth initialization must be disabled"

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import uuid
import zipfile

NAME = "antlr4-python3-runtime"
VERSION = "4.9.3"
SOURCE_SHA256 = "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"
CLI_SHA256 = "02d4202c724d47965d015e9d62ad2f243359b840516db9629ce964d4e1150461"
CHUNK_BYTES = 1024**2
MAX_JSON_BYTES = 5 * 1024**2
MAX_WHEEL_BYTES = 8 * 1024**2
MAX_MEMBER_BYTES = 2 * 1024**2
MAX_EXPANDED_BYTES = 16 * 1024**2
EXPECTED_ENVIRONMENT = {"SOURCE_DATE_EPOCH": "1704067200", "PYTHONHASHSEED": "0", "TZ": "UTC"}
APPROVED_BUILD_TOOL_PINS = {"setuptools": "80.9.0", "wheel": "0.45.1"}
NATIVE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".pyx", ".pxd", ".cu", ".cuh", ".rs", ".s", ".asm", ".f", ".f90", ".so", ".pyd", ".dll", ".dylib", ".lib", ".a", ".o", ".obj", ".exe"}
FORBIDDEN_SUFFIXES = NATIVE_SUFFIXES | {".pyc", ".pyo"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def public(value, root):
    value = str(value).replace("\\", "/")
    for path, token in ((root, "${WORKSPACE}"), (Path(sys.base_prefix), "${PYTHON_HOME}"), (Path.home(), "${USER_HOME}")):
        value = re.sub(re.escape(str(path).replace("\\", "/")), lambda match: token, value, flags=re.I)
    return value


def expand(value, root):
    if not isinstance(value, str):
        raise ValueError("Expected a tokenized public path")
    for token, base in (("${WORKSPACE}", root), ("${USER_HOME}", Path.home()), ("${PYTHON_HOME}", Path(sys.base_prefix))):
        if value == token or value.startswith(token + "/"):
            relative = value[len(token):].lstrip("/")
            if "\\" in relative or any(part in {".", ".."} or ":" in part for part in relative.split("/")):
                raise ValueError("Unsafe public path suffix")
            return base / relative
    raise ValueError("Unsupported public path token")


def plain_path(path, anchor, required=True, directory=False):
    path, anchor = path.absolute(), anchor.resolve()
    if not path.is_relative_to(anchor) or not path.resolve().is_relative_to(anchor):
        raise ValueError("Path escapes its approved root")
    for candidate in (path, *path.parents):
        if candidate == anchor.parent:
            break
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise ValueError("Symlink/junction input or destination refused")
        if candidate.exists() and getattr(candidate.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Reparse-point input or destination refused")
        if candidate == anchor:
            break
    if required and not (path.is_dir() if directory else path.is_file()):
        raise ValueError("Required plain " + ("directory" if directory else "file") + " is absent: " + path.name)
    return path


def absent(path, root):
    plain_path(path.parent, root, directory=True)
    if os.path.lexists(path):
        raise ValueError("Destination already exists; no overwrite permitted: " + path.name)


def load_json(path, root):
    plain_path(path, root)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON input exceeds bounded size")
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def fingerprint(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def digest_file(path, expected_size=None):
    before = fingerprint(path)
    if expected_size is not None and before[2] != expected_size:
        raise ValueError("Exact file size mismatch: " + path.name)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(data)
    if fingerprint(path) != before:
        raise ValueError("Input changed while hashing: " + path.name)
    return digest.hexdigest(), before


def save_new(path, data):
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def safe_relative(name, allow_empty=False):
    if allow_empty and name == "":
        return PurePosixPath()
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        raise ValueError("Unsafe relative archive/source path")
    parts = name.removesuffix("/").split("/")
    reserved = {"con", "prn", "aux", "nul", *("com" + str(i) for i in range(1, 10)), *("lpt" + str(i) for i in range(1, 10))}
    if any(not part or part in {".", ".."} or part.endswith((".", " ")) or re.search(r'[\x00-\x1f<>:"|?*]', part) or part.split(".")[0].casefold() in reserved for part in parts):
        raise ValueError("Unsafe Windows archive/source path")
    return PurePosixPath(*parts)


def parse_lock(text):
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+) --hash=sha256:([0-9a-f]{64})", line.strip())
        if not match:
            raise ValueError("Unexpected lock syntax")
        name, version, digest = match.groups()
        name = canonical(name)
        if name in result:
            raise ValueError("Duplicate lock package")
        result[name] = (version, digest)
    return result


def require_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Missing exact SHA256")


def check_build_receipt(receipt, sentinel, tools, source_root, audit_byte_sha):
    if receipt.get("stage") != "build-wheel" or receipt.get("status") != "completed" or receipt.get("exit_code") != 0 or receipt.get("child_started") is not True or receipt.get("child_reaped") is not True:
        raise ValueError("A completed/reaped successful actual build receipt is required")
    if not isinstance(receipt.get("pid"), int) or receipt["pid"] <= 0 or not receipt.get("started_utc") or not receipt.get("finished_utc"):
        raise ValueError("Build receipt lacks actual process/time evidence")
    if sentinel.get("stage") != "build-wheel" or not isinstance(sentinel.get("attempt"), str) or not sentinel["attempt"].startswith("build-wheel-") or not sentinel.get("started_utc"):
        raise ValueError("Single-build prelaunch sentinel is missing or malformed")
    fields = ("stage", "attempt", "command", "source_sha256", "source_audit_sha256", "source_root", "build_tool_pins", "environment", "no_index", "no_deps", "no_build_isolation")
    if any(key not in receipt or key not in sentinel or receipt[key] != sentinel[key] for key in fields):
        raise ValueError("Actual build receipt does not match the single-build sentinel")
    if receipt["source_sha256"] != SOURCE_SHA256 or receipt["source_audit_sha256"] != audit_byte_sha or receipt["source_root"] != source_root:
        raise ValueError("Build receipt is not bound to the approved source/audit bytes")
    if tools != APPROVED_BUILD_TOOL_PINS or receipt["build_tool_pins"] != tools or receipt["environment"] != EXPECTED_ENVIRONMENT:
        raise ValueError("Build-tool pins or deterministic environment differ from approved records")
    command = receipt["command"]
    if not isinstance(command, list) or not command or any(not isinstance(part, str) for part in command) or any(receipt[key] is not True or flag not in command for key, flag in (("no_index", "--no-index"), ("no_deps", "--no-deps"), ("no_build_isolation", "--no-build-isolation"))):
        raise ValueError("Offline/no-dependency/no-isolation build command evidence is incomplete")
    if "wheel" not in command or source_root not in command:
        raise ValueError("Build command does not identify the wheel operation and approved source root")


def approved_python_files(audit):
    facts = audit["source_audit"]
    fields = facts["packaging_ast"]["setup_fields"]
    packages, package_dir = fields["packages"], fields.get("package_dir", {})
    if not isinstance(packages, list) or not packages or not isinstance(package_dir, dict):
        raise ValueError("Approved audit lacks explicit Python packages/package_dir")
    top = safe_relative(facts["archive_top_directory"])
    members = {}
    for row in facts["members"]:
        name = safe_relative(row["path"]).as_posix()
        if row["type"] == "file":
            if name in members:
                raise ValueError("Duplicate approved source member")
            require_digest(row.get("sha256"))
            members[name] = row
    result = {}
    for package in packages:
        if not isinstance(package, str) or not re.fullmatch(r"antlr4(?:\.[A-Za-z_][A-Za-z0-9_]*)*", package):
            raise ValueError("Unexpected approved Python package namespace")
        prefixes = [key for key in package_dir if key == "" or package == key or package.startswith(key + ".")]
        prefix = max(prefixes, key=len) if prefixes else ""
        base = safe_relative(package_dir.get(prefix, ""), allow_empty=True)
        remainder = package[len(prefix):].lstrip(".")
        folder = top / base / PurePosixPath(*filter(None, remainder.split(".")))
        for name, record in members.items():
            member = PurePosixPath(name)
            if member.parent == folder and member.suffix == ".py":
                wheel_name = package.replace(".", "/") + "/" + member.name
                if wheel_name in result:
                    raise ValueError("Overlapping declared source package mappings")
                result[wheel_name] = {"relative_source": member.relative_to(top).as_posix(), "record": record}
    if "antlr4/__init__.py" not in result:
        raise ValueError("Approved source has no antlr4 package initializer")
    return result


def normalized_requirements(values):
    return sorted(re.sub(r"\s+", "", value).replace('"', "'") for value in values)


def normalized_cli_copy(data):
    normalized = data.replace(b"\r\n", b"\n")
    if b"\r" in normalized or not normalized.startswith(b"#!python\n"):
        raise ValueError("Copied CLI has an unexpected wheel shebang or line-ending transformation")
    return normalized


def check_wheel(wheel, source_root, expected_python, old_site, installed, root):
    if not re.fullmatch(r"antlr4_python3_runtime-4\.9\.3(?:-\d[A-Za-z0-9_]*)?-py3-none-any\.whl", wheel.name) or wheel.stat().st_size > MAX_WHEEL_BYTES:
        raise ValueError("New artifact is not the single exact bounded pure ANTLR wheel")
    fresh_sha, wheel_fingerprint = digest_file(wheel)
    files, case_names, total = {}, set(), 0
    dist_info = "antlr4_python3_runtime-4.9.3.dist-info"
    cli_path = "antlr4_python3_runtime-4.9.3.data/scripts/pygrun"
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        if len(entries) > 1000:
            raise ValueError("New wheel has too many members")
        for entry in entries:
            path = safe_relative(entry.filename)
            name, key = path.as_posix(), path.as_posix().casefold()
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if key in case_names or mode not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.external_attr & 0x400 or entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("New wheel contains duplicate/case collisions, links, encryption or unsupported member types")
            case_names.add(key)
            cli_member = name == cli_path or (entry.is_dir() and cli_path.startswith(name + "/"))
            if (path.parts[0] not in {"antlr4", dist_info} and not cli_member) or (len(path.parts) < 2 and not entry.is_dir()):
                raise ValueError("Unexpected wheel root/script/data layout")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.casefold() in {"sitecustomize.py", "usercustomize.py", "entry_points.txt"} or path.name.lower().endswith((".pth", ".bat", ".cmd", ".ps1", ".sh")):
                raise ValueError("Native or executable hook member in purported pure wheel")
            if entry.is_dir():
                if entry.file_size:
                    raise ValueError("Wheel directory declares file data")
                continue
            if mode == stat.S_IFDIR or not 0 <= entry.file_size <= MAX_MEMBER_BYTES:
                raise ValueError("Wheel member type/expanded size is invalid")
            total += entry.file_size
            if total > MAX_EXPANDED_BYTES:
                raise ValueError("Wheel exceeds expanded-size budget")
            data = archive.read(entry)
            if len(data) != entry.file_size:
                raise ValueError("Wheel member expanded size mismatch")
            files[name] = data
    for name in files:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents):
            raise ValueError("Wheel uses a file as another member's parent directory")
    metadata_path, wheel_path, record_path = (dist_info + "/" + name for name in ("METADATA", "WHEEL", "RECORD"))
    if any(name not in files for name in (metadata_path, wheel_path, record_path)):
        raise ValueError("New wheel lacks required unique distribution metadata")
    metadata, wheel_metadata = BytesParser().parsebytes(files[metadata_path]), BytesParser().parsebytes(files[wheel_path])
    if any(len(metadata.get_all(key, [])) != 1 for key in ("Name", "Version")) or len(wheel_metadata.get_all("Root-Is-Purelib", [])) != 1:
        raise ValueError("Required wheel distribution metadata fields are missing or duplicated")
    if canonical(metadata.get("Name", "")) != NAME or metadata.get("Version") != VERSION or wheel_metadata.get("Root-Is-Purelib", "").lower() != "true" or wheel_metadata.get_all("Tag", []) != ["py3-none-any"]:
        raise ValueError("New wheel Name/Version/purelib/tags differ from approved exact package")
    if normalized_requirements(metadata.get_all("Requires-Dist", [])) != normalized_requirements(installed.get("requires_dist", [])):
        raise ValueError("New wheel changes the recorded ANTLR dependency metadata")
    if metadata.get("Requires-Python") != installed.get("requires_python"):
        raise ValueError("New wheel changes the recorded ANTLR Python compatibility metadata")
    if cli_path not in files:
        raise ValueError("New wheel lacks the single approved literal copied CLI")
    cli_source = plain_path(source_root / "bin/pygrun", root)
    if cli_source.stat().st_size != 6111:
        raise ValueError("Approved original CLI source size changed")
    cli_bytes = cli_source.read_bytes()
    if hashlib.sha256(cli_bytes).hexdigest() != CLI_SHA256 or normalized_cli_copy(files[cli_path]) != normalized_cli_copy(cli_bytes):
        raise ValueError("Packaged CLI differs from the exact approved literal script copy")
    cli_check = {"wheel_path": cli_path, "wheel_bytes": len(files[cli_path]), "wheel_sha256": hashlib.sha256(files[cli_path]).hexdigest(),
                 "source_sha256": CLI_SHA256, "source_bytes": 6111, "normalized_copy_equal": True,
                 "permitted_transformation": "Only CRLF/LF line-ending normalization with the fixed #!python wheel shebang", "script_executed": False}
    records = {}
    for row in csv.reader(io.StringIO(files[record_path].decode("utf-8"))):
        if len(row) != 3 or row[0] in records:
            raise ValueError("Malformed or duplicate wheel RECORD row")
        safe_relative(row[0])
        records[row[0]] = row[1:]
    if set(records) != set(files) or records[record_path] != ["", ""]:
        raise ValueError("Wheel RECORD does not cover exactly the packaged files")
    for name, data in files.items():
        if name == record_path:
            continue
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")
        if records[name] != ["sha256=" + encoded, str(len(data))]:
            raise ValueError("Wheel RECORD hash/size does not verify: " + name)
    python_files = {name for name in files if PurePosixPath(name).suffix.casefold() == ".py"}
    if python_files != set(expected_python):
        raise ValueError("New wheel Python file set differs from approved complete source package set")
    comparisons = []
    for name in sorted(python_files):
        expected = expected_python[name]
        source = source_root.joinpath(*safe_relative(expected["relative_source"]).parts)
        plain_path(source, root)
        approved = expected["record"]
        if source.stat().st_size != approved["bytes"] or source.stat().st_size > MAX_MEMBER_BYTES:
            raise ValueError("Approved extracted Python source size changed")
        source_bytes = source.read_bytes()
        actual_sha = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha != approved["sha256"] or source_bytes != files[name]:
            raise ValueError("New wheel code differs from approved sdist/extracted source: " + name)
        old = old_site.joinpath(*safe_relative(name).parts)
        old_present = os.path.lexists(old)
        old_sha = None
        if old_present:
            plain_path(old, root)
            if old.stat().st_size > MAX_MEMBER_BYTES:
                raise ValueError("Old installed Python source exceeds audit budget")
            old_bytes = old.read_bytes()
            old_sha = hashlib.sha256(old_bytes).hexdigest()
            if old_bytes != files[name]:
                raise ValueError("New wheel code differs from available old installed ANTLR source: " + name)
        comparisons.append({"wheel_path": name, "bytes": len(files[name]), "sha256": actual_sha,
                            "approved_source": public(source, root), "approved_archive_member": approved["path"],
                            "approved_sdist_and_extracted_source_equal": True,
                            "old_installed_source": public(old, root), "old_installed_available": old_present,
                            "old_installed_sha256": old_sha, "old_installed_equal": True if old_present else None})
    if fingerprint(wheel) != wheel_fingerprint:
        raise ValueError("Built wheel changed during verification")
    return {"filename": wheel.name, "bytes": wheel.stat().st_size, "sha256": fresh_sha, "wheel_tags": ["py3-none-any"],
            "name": NAME, "version": VERSION, "purelib": True, "record_verified_files": len(files),
            "requires_dist": metadata.get_all("Requires-Dist", []), "requires_python": metadata.get("Requires-Python"),
            "wheel_generator": wheel_metadata.get("Generator"), "python_files": comparisons,
            "approved_literal_cli_copy": cli_check,
            "python_files_compared_to_approved_source": len(comparisons),
            "python_files_compared_to_old_installed_source": sum(row["old_installed_available"] for row in comparisons),
            "old_installed_files_unavailable": sum(not row["old_installed_available"] for row in comparisons)}, wheel_fingerprint


def finalize(root, receipt_path):
    own = root / "work/expanded-clean-runtime"
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    r1, r2 = own / "wheelhouse", own / "wheelhouse-r2"
    lock_r2, resolution_r2, verification_path = (output / name for name in ("requirements-lock-r2.txt", "wheel-resolution-r2.json", "antlr-build-verification.json"))
    for path in (r2, lock_r2, resolution_r2, verification_path):
        absent(path, root)
    inventory, _ = load_json(output / "installed-metadata-inventory.json", root)
    resolution, resolution_sha = load_json(output / "wheel-resolution.json", root)
    manifest, manifest_sha = load_json(output / "wheel-download-manifest.json", root)
    audit, audit_byte_sha = load_json(output / "antlr-source-audit-r2.json", root)
    receipt, receipt_sha = load_json(receipt_path, root)
    sentinel, sentinel_sha = load_json(own / "antlr-one-build-started.json", root)
    tools, tools_sha = load_json(output / "build-tools-resolution.json", root)
    cached, _ = load_json(output / "cached-wheel-inventory.json", root)
    if resolution.get("status") != "resolved" or resolution.get("errors") or manifest.get("status") not in {"complete_exact_wheels_verified", "complete_exact_wheels_verified_with_antlr_deferred"}:
        raise ValueError("Completed exact resolution and download verification are required")
    selected, packages = inventory["selected"], resolution["packages"]
    if len(selected) != 90 or len(packages) != 90 or len({canonical(row["name"]) for row in packages}) != 90:
        raise ValueError("Runtime closure must contain exactly the same 90 unique installed versions")
    previous = {}
    for row in packages:
        name = canonical(row["name"])
        if name not in selected or row["version"] != selected[name]["version"]:
            raise ValueError("Runtime dependency/version closure differs from recorded installed versions")
        if safe_relative(row["filename"]).name != row["filename"] or not row["filename"].endswith(".whl"):
            raise ValueError("Unsafe original wheel filename")
        require_digest(row.get("sha256"))
        previous[name] = row
    if len({row["filename"].casefold() for row in packages}) != 90:
        raise ValueError("Original runtime filenames conflict")
    lock_path = output / "requirements-lock.txt"
    lock_bytes = None
    expected_lock = {name: (row["version"], row["sha256"]) for name, row in previous.items()}
    if os.path.lexists(lock_path):
        plain_path(lock_path, root)
        if lock_path.stat().st_size > 64 * 1024:
            raise ValueError("Original lock exceeds expected bound")
        lock_bytes = lock_path.read_bytes()
        if parse_lock(lock_bytes.decode("utf-8")) != expected_lock:
            raise ValueError("Original lock does not match original exact resolution")
    elif manifest.get("status") != "complete_exact_wheels_verified_with_antlr_deferred":
        raise ValueError("Original complete-wheel run must preserve its original runtime lock")
    facts = audit.get("source_audit", {})
    if audit.get("status") != "prepared_not_built" or not facts.get("packaging_shape_passed") or facts.get("findings") or audit.get("source", {}).get("published_sha256") != SOURCE_SHA256 or audit.get("source_audit_sha256") != hashlib.sha256(json_bytes(facts)).hexdigest():
        raise ValueError("Passing exact source audit/fingerprint is required")
    source_root = expand(audit["source_root"], root)
    plain_path(source_root, root, directory=True)
    source_archive = expand(audit["acquisition"]["path"], root)
    plain_path(source_archive, root)
    source_sha, _ = digest_file(source_archive, audit["source"]["published_bytes"])
    if source_sha != SOURCE_SHA256:
        raise ValueError("Approved preserved sdist bytes changed")
    if tools.get("status") != "exact_build_tools_wheels_verified" or {row["name"] for row in tools["packages"]} != {"setuptools", "wheel"} or len(tools["packages"]) != 2:
        raise ValueError("Exact setuptools/wheel resolution receipt is required")
    tool_pins = {row["name"]: row["version"] for row in tools["packages"]}
    check_build_receipt(receipt, sentinel, tool_pins, audit["source_root"], audit_byte_sha)
    built = plain_path(own / "antlr-built-wheelhouse", root, directory=True)
    built_files = list(built.iterdir())
    if len(built_files) != 1:
        raise ValueError("The isolated build directory must contain exactly one new wheel and no other artifact")
    wheel = plain_path(built_files[0], root)
    old_metadata = expand(selected[NAME]["metadata_path"], root)
    old_site = old_metadata.parent.parent
    # A fresh checkout may have none of the historical installed package tree.
    plain_path(old_site, root, required=False, directory=True)
    old_metadata_present = os.path.lexists(old_metadata)
    if old_metadata_present:
        plain_path(old_metadata, root)
        old_metadata_sha, _ = digest_file(old_metadata)
        if old_metadata_sha != selected[NAME]["metadata_sha256"]:
            raise ValueError("Old installed ANTLR metadata changed from the recorded inventory")
    else:
        old_metadata_sha = None
    expected_python = approved_python_files(audit)
    wheel_check, new_wheel_fingerprint = check_wheel(wheel, source_root, expected_python, old_site, selected[NAME], root)
    verified = {row["filename"]: row for row in manifest["files"]}
    allowed_filenames = {row["filename"] for row in packages}
    required_filenames = {row["filename"] for row in packages if canonical(row["name"]) != NAME}
    if len(manifest["files"]) != len(verified) or not required_filenames <= set(verified) or not set(verified) <= allowed_filenames:
        raise ValueError("Completed download manifest must cover exactly the 89 unchanged runtime wheels plus optional historical ANTLR")
    plain_path(r1, root, directory=True)
    unchanged, original_antlr = [], None
    print(json.dumps({"phase": "finalize_antlr_verification_started", "pid": os.getpid(), "unchanged_wheels": 89, "build": False, "install": False}), flush=True)
    for name, row in sorted(previous.items()):
        item = {"name": name, "version": row["version"], "filename": row["filename"], "bytes": row["bytes"], "sha256": row["sha256"]}
        path = r1 / row["filename"]
        if name == NAME and not os.path.lexists(path):
            original_antlr = {**item, "local_available": False, "hash_evidence": "Historical original resolution only; not reused for the new build"}
            continue
        entry = verified.get(row["filename"], {})
        if entry.get("status") not in {"verified", "already_verified"} or entry.get("sha256") != row["sha256"] or entry.get("bytes") != row["bytes"]:
            raise ValueError("Original wheel lacks a matching completed verification record")
        plain_path(path, root)
        actual, stable = digest_file(path, row["bytes"])
        if actual != row["sha256"]:
            raise ValueError("Original verified wheel bytes changed: " + row["filename"])
        if name == NAME:
            original_antlr = {**item, "local_available": True, "hash_evidence": "Current streaming verification of original historical wheel"}
        else:
            unchanged.append((path, stable, item))
        print(json.dumps({"phase": "original_wheel_sha_verified", "name": name, "pid": os.getpid()}), flush=True)
    if len(unchanged) != 89 or original_antlr is None:
        raise ValueError("Expected exactly 89 unchanged wheels and one replaced ANTLR wheel")
    r2.mkdir(exist_ok=False)
    staging = []
    for path, stable, item in unchanged:
        if fingerprint(path) != stable:
            raise ValueError("Original wheel changed before hardlink staging")
        destination = r2 / item["filename"]
        os.link(path, destination)
        if not os.path.samefile(path, destination) or fingerprint(destination) != stable:
            raise ValueError("Hardlink staging did not preserve the verified file identity")
        staging.append({**item, "method": "hardlink_verified_unchanged", "path": public(destination, root)})
    if fingerprint(wheel) != new_wheel_fingerprint:
        raise ValueError("New built wheel changed before copying")
    new_path = r2 / wheel.name
    digest, copied = hashlib.sha256(), 0
    with wheel.open("rb") as source, new_path.open("xb") as target:
        for data in iter(lambda: source.read(CHUNK_BYTES), b""):
            copied += len(data)
            if copied > wheel_check["bytes"]:
                raise ValueError("New built wheel grew during copying")
            target.write(data)
            digest.update(data)
        target.flush()
        os.fsync(target.fileno())
    if copied != wheel_check["bytes"] or digest.hexdigest() != wheel_check["sha256"]:
        raise ValueError("Copied new ANTLR wheel differs from fresh verified build")
    staging.append({"name": NAME, "version": VERSION, "filename": wheel.name, "bytes": copied, "sha256": digest.hexdigest(), "method": "copied_new_verified_build", "path": public(new_path, root)})
    historical = [{key: row.get(key) for key in ("filename", "bytes", "sha256", "local_path", "origin", "provenance_limit")} for row in cached["wheels"] if row.get("filename") == "antlr4_python3_runtime-4.9.3-py3-none-any.whl"]
    new_row = {"name": NAME, "version": VERSION, "filename": wheel.name, "url": None, "sha256": wheel_check["sha256"], "bytes": wheel_check["bytes"],
               "published_sha256": None, "wheel_tags": ["py3-none-any"], "compatible_tags": ["py3-none-any"], "local_path": public(new_path, root),
               "present_now": True, "source_kind": "new_isolated_pure_build_from_approved_exact_sdist", "hash_source": "Fresh streaming SHA256 of the new actual built wheel; not reused from historical cache",
               "source_archive_sha256": SOURCE_SHA256, "source_archive_url": audit["source"]["url"], "requires_dist": wheel_check["requires_dist"],
               "requires_python": wheel_check["requires_python"], "installed_metadata_sha256": selected[NAME]["metadata_sha256"],
               "build_verification": public(verification_path, root), "build_execution_receipt": public(receipt_path, root),
               "provenance_limit": "Locally built from the exact publisher sdist; the wheel SHA256 is independently computed local build evidence, not a publisher-distributed wheel hash"}
    revised = json.loads(json.dumps(resolution))
    revised["packages"] = [new_row if canonical(row["name"]) == NAME else row for row in packages]
    revised.update(revision="r2", parent_resolution_sha256=resolution_sha, wheelhouse=public(r2, root),
                   only_runtime_artifact_replaced=NAME, runtime_package_versions_unchanged=True,
                   unchanged_wheel_count=89, new_antlr_wheel_sha256=wheel_check["sha256"])
    receipt_keys = ("stage", "attempt", "status", "exit_code", "child_started", "child_reaped", "pid", "started_utc", "finished_utc", "wall_seconds", "peak_sampled_family_RSS_bytes", "stable_preflight_available_observations_bytes", "source_sha256", "source_audit_sha256", "source_root", "build_tool_pins", "command", "environment", "no_index", "no_deps", "no_build_isolation")
    provenance = {key: receipt.get(key, None) for key in receipt_keys}
    verification = {"status": "verified_new_antlr_build_and_runtime_revision", "runtime_package_count": 90, "unchanged_wheels": 89,
                    "source_sha256": source_sha, "source_audit_file_sha256": audit_byte_sha, "source_audit_facts_sha256": audit["source_audit_sha256"],
                    "single_build_start_sentinel_matched": True, "build_start_sentinel_sha256": sentinel_sha,
                    "build_receipt_path": public(receipt_path, root), "build_receipt_sha256": receipt_sha, "build_provenance": provenance,
                    "build_tools_resolution_sha256": tools_sha, "build_tool_versions_independently_import_checked_by_finalizer": False,
                    "original_resolution_sha256": resolution_sha, "original_lock_sha256": hashlib.sha256(lock_bytes).hexdigest() if lock_bytes is not None else None,
                    "original_lock_available": lock_bytes is not None, "original_download_manifest_sha256": manifest_sha,
                    "original_selected_antlr_wheel": original_antlr, "historical_cached_wheels_not_used_for_new_hash": historical,
                    "fresh_new_wheel_sha256": wheel_check["sha256"], "new_wheel_hash_computed_from_actual_bytes": True,
                    "new_hash_equals_a_historical_cached_hash": any(row["sha256"] == wheel_check["sha256"] for row in historical),
                    "new_wheel": wheel_check, "old_installed_metadata_present": old_metadata_present, "old_installed_metadata_sha256": old_metadata_sha,
                    "r2_wheelhouse": public(r2, root), "staged_files": sorted(staging, key=lambda row: row["name"]),
                    "original_wheelhouse_and_lock_preserved": True, "runtime_package_versions_unchanged": True,
                    "framework_imported": False, "network_performed": False, "build_performed_by_finalizer": False, "installation_performed": False,
                    "limitations": ["Build command, fixed environment and tool pins are parent-receipt evidence; this finalizer does not import build tools or repeat a build",
                                    "Unavailable old installed Python files are explicitly counted; all new packaged Python files must match approved source bytes",
                                    "Hardlinked unchanged wheels share their file identity with the preserved original wheelhouse"]}
    pins = [row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"] for row in sorted(revised["packages"], key=lambda row: canonical(row["name"]))]
    lock_text = "# Runtime revision r2: same 90 exact versions; ANTLR uses the newly verified local pure build.\n" + "\n".join(pins) + "\n"
    if parse_lock(lock_text) != {canonical(row["name"]): (row["version"], row["sha256"]) for row in revised["packages"]}:
        raise ValueError("Internal r2 lock mismatch")
    save_new(verification_path, json_bytes(verification))
    save_new(resolution_r2, json_bytes(revised))
    save_new(lock_r2, lock_text.encode("utf-8"))
    print(json.dumps({"status": verification["status"], "pid": os.getpid(), "runtime_packages": 90, "hardlinked_unchanged": 89, "fresh_antlr_sha256": wheel_check["sha256"], "build": False, "install": False}), flush=True)
    return 0


def selfcheck():
    assert safe_relative("antlr4/tree/Trees.py").name == "Trees.py"
    for bad in ("../escape", "/absolute", "C:/escape", "antlr4/../escape", "antlr4\\escape", "antlr4/CON", "antlr4/a//b", "antlr4/trailing."):
        try:
            safe_relative(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe wheel/source path accepted")
    assert parse_lock("# comment\nDemo_pkg==1.2+local --hash=sha256:" + "a" * 64 + "\n") == {"demo-pkg": ("1.2+local", "a" * 64)}
    try:
        parse_lock("demo==2 --index-url=https://example.invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Unapproved lock option accepted")
    assert normalized_requirements(['typing; python_version < "3.5"']) == normalized_requirements(["typing;python_version<'3.5'"])
    assert normalized_cli_copy(b"#!python\r\nprint('test')\r\n") == normalized_cli_copy(b"#!python\nprint('test')\n")
    source_root = "${WORKSPACE}/approved-source"
    sentinel = {"stage": "build-wheel", "attempt": "build-wheel-test", "started_utc": "recorded", "command": ["python", "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation", source_root],
                "source_sha256": SOURCE_SHA256, "source_audit_sha256": "a" * 64, "source_root": source_root,
                "build_tool_pins": APPROVED_BUILD_TOOL_PINS, "environment": EXPECTED_ENVIRONMENT, "no_index": True, "no_deps": True, "no_build_isolation": True}
    receipt = {**sentinel, "status": "completed", "exit_code": 0, "child_started": True, "child_reaped": True, "pid": 1, "finished_utc": "recorded"}
    check_build_receipt(receipt, sentinel, APPROVED_BUILD_TOOL_PINS, source_root, "a" * 64)
    try:
        check_build_receipt(receipt, sentinel, {"setuptools": "unapproved", "wheel": "unapproved"}, source_root, "a" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Unapproved build-tool pins accepted")
    audit = {"source_audit": {"archive_top_directory": "antlr4-python3-runtime-4.9.3", "packaging_ast": {"setup_fields": {"packages": ["antlr4"], "package_dir": {"": "src"}}},
                              "members": [{"path": "antlr4-python3-runtime-4.9.3/src/antlr4/__init__.py", "type": "file", "bytes": 1, "sha256": "a" * 64}]}}
    assert approved_python_files(audit)["antlr4/__init__.py"]["relative_source"] == "src/antlr4/__init__.py"
    print(json.dumps({"status": "selfcheck_passed", "network": False, "writes": False, "build": False, "install": False, "site_initialization_disabled": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--finalize-approved", action="store_true")
    mode.add_argument("--selfcheck", action="store_true")
    arguments = parser.parse_args()
    if arguments.selfcheck:
        selfcheck()
        return 0
    if arguments.build_receipt is None:
        parser.error("--finalize-approved requires the actual --build-receipt JSON")
    root = arguments.workspace.resolve()
    receipt_path = arguments.build_receipt if arguments.build_receipt.is_absolute() else root / arguments.build_receipt
    try:
        return finalize(root, receipt_path)
    except Exception as error:
        print(json.dumps({"status": "antlr_finalization_failed", "pid": os.getpid(), "error_type": type(error).__name__, "error": public(error, root), "build": False, "install": False}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
