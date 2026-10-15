"""Resolve exact installed wheels from bounded official metadata; never download wheels."""
from __future__ import annotations

import sys

assert sys.flags.no_site, "Run with python -S; site initialization must be disabled"

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

MAX_METADATA_BYTES = 5 * 1024**2
CHUNK_BYTES = 1024**2
TARGET_PYTHON = (3, 13, 5)
KNOWN_PRECEDENCE = {"torch", "torchvision", "torch-geometric", "torch-scatter", "xxhash", "nvidia-physicsnemo"}
NON_PYPI_BUILDS = {"torch", "torchvision", "torch-scatter"}
OFFICIAL_FILE_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org", "data.pyg.org"}
ANTLR_OLD_WHEEL_SHA256 = "9a700628218e270a26588ffc54233dc3d8799e82ae4194edf7a953e338812d7c"
ANTLR_SOURCE_SHA256 = "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"
ANTLR_SOURCE_URL = "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz"


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def now():
    return datetime.now(timezone.utc).isoformat()


def public(value, root):
    value = str(value).replace("\\", "/")
    for path, token in ((root, "${WORKSPACE}"), (Path(sys.base_prefix), "${PYTHON_HOME}"), (Path.home(), "${USER_HOME}")):
        value = re.sub(re.escape(str(path).replace("\\", "/")), lambda match: token, value, flags=re.I)
    return value


def expand(value, root):
    for token, path in (("${WORKSPACE}", root), ("${USER_HOME}", Path.home()), ("${PYTHON_HOME}", Path(sys.base_prefix))):
        if value == token or value.startswith(token + "/"):
            return path / value[len(token):].lstrip("/")
    raise ValueError("Cached local_path must start with a supported public path token")


def save_new(path, value):
    # Each successful package cache is immutable; later invocations reuse it.
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Publishing a complete hard link is atomic and refuses an existing name.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def target_tags():
    tags = [("cp313", "cp313", "win_amd64"), ("cp313", "abi3", "win_amd64"), ("cp313", "none", "win_amd64")]
    tags.extend((f"cp3{minor}", "abi3", "win_amd64") for minor in range(12, 1, -1))
    generic = ["py313", "py3", *(f"py3{minor}" for minor in range(12, -1, -1))]
    tags.extend((interpreter, "none", "win_amd64") for interpreter in generic)
    tags.append(("cp313", "none", "any"))
    tags.extend((interpreter, "none", "any") for interpreter in generic)
    return ["-".join(tag) for tag in tags]


TARGET_TAGS = target_tags()
TAG_ORDER = {tag: rank for rank, tag in enumerate(TARGET_TAGS)}


def wheel_identity(filename):
    if not isinstance(filename, str) or "/" in filename or "\\" in filename or not filename.endswith(".whl"):
        raise ValueError("Invalid wheel filename")
    parts = filename[:-4].split("-")
    if len(parts) not in (5, 6) or not re.fullmatch(r"[A-Za-z0-9_.]+", parts[0]):
        raise ValueError("Invalid wheel filename fields")
    if len(parts) == 6 and not re.fullmatch(r"\d[A-Za-z0-9_]*", parts[2]):
        raise ValueError("Invalid wheel build tag")
    if not all(re.fullmatch(r"[A-Za-z0-9_.]+", part) for part in parts[-3:]):
        raise ValueError("Invalid wheel compatibility tags")
    tags = ["-".join(tag) for tag in itertools.product(*(part.split(".") for part in parts[-3:]))]
    compatible = sorted(set(tags) & TAG_ORDER.keys(), key=TAG_ORDER.__getitem__)
    return canonical(parts[0]), parts[1], sorted(set(tags)), compatible


def exact_version(left, right):
    # Observed inventory uses normalized release/post/local versions, with no aliases.
    return left.replace("_", ".").lower() == right.replace("_", ".").lower()


def python_compatible(specification):
    if not specification:
        return True
    for clause in specification.strip().strip("()").split(","):
        match = re.fullmatch(r"\s*(===|==|!=|>=|<=|~=|>|<)\s*(\d+(?:\.\d+)*(?:\.\*)?)\s*", clause)
        if not match:
            raise ValueError("Unsupported Requires-Python constraint: " + specification)
        operator, text = match.groups()
        wildcard = text.endswith(".*")
        wanted = tuple(map(int, text.removesuffix(".*").split(".")))
        length = max(len(wanted), len(TARGET_PYTHON))
        left = TARGET_PYTHON + (0,) * (length - len(TARGET_PYTHON))
        right = wanted + (0,) * (length - len(wanted))
        if wildcard:
            if operator not in ("==", "!="):
                raise ValueError("Wildcard only supported with == or !=")
            accepted = TARGET_PYTHON[:len(wanted)] == wanted
            accepted = not accepted if operator == "!=" else accepted
        elif operator == "~=":
            if len(wanted) < 2:
                raise ValueError("Compatible release requires at least two components")
            prefix = wanted[:-1]
            upper = prefix[:-1] + (prefix[-1] + 1,)
            accepted = left >= right and TARGET_PYTHON < upper
        else:
            accepted = {"==": left == right, "===": left == right, "!=": left != right,
                        ">=": left >= right, "<=": left <= right, ">": left > right, "<": left < right}[operator]
        if not accepted:
            return False
    return True


def require_file_fields(row):
    if not isinstance(row.get("bytes"), int) or isinstance(row["bytes"], bool) or row["bytes"] <= 0:
        raise ValueError("Missing positive wheel byte size")
    if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256") or ""):
        raise ValueError("Missing wheel SHA256")
    if row.get("url"):
        parsed = urllib.parse.urlsplit(row["url"])
        if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_FILE_HOSTS or parsed.username or parsed.password:
            raise ValueError("Wheel URL is not an allowed official HTTPS host")


def pip_http_path(url):
    digest = hashlib.sha224(url.encode("utf-8")).hexdigest()
    return Path.home() / "AppData/Local/pip/Cache/http-v2" / Path(*digest[:5]) / (digest + ".body")


def cache_availability(row, known, root):
    paths = []
    for record in known:
        if record["filename"] == row["filename"] and record.get("sha256") == row["sha256"] and record.get("local_path"):
            paths.append(expand(record["local_path"], root))
    if row.get("url"):
        paths.append(pip_http_path(row["url"]))
    checked = []
    for path in dict.fromkeys(paths):
        item = {"local_path": public(path, root), "present_now": path.is_file()}
        if item["present_now"]:
            item["observed_bytes"] = path.stat().st_size
            if item["observed_bytes"] != row["bytes"]:
                item["validation"] = "size_mismatch_not_used"
            else:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(CHUNK_BYTES):
                        digest.update(chunk)
                item["observed_sha256"] = digest.hexdigest()
                item["validation"] = "verified" if digest.hexdigest() == row["sha256"] else "hash_mismatch_not_used"
        else:
            item["validation"] = "absent_not_opened"
        checked.append(item)
        if item["validation"] == "verified":
            return item["local_path"], checked
    return None, checked


class OfficialMetadataRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.hostname != "pypi.org":
            raise ValueError("Refusing metadata redirect outside official PyPI HTTPS")
        return super().redirect_request(request, response, code, message, headers, newurl)


def official_metadata(name, version, directory):
    url = "https://pypi.org/pypi/" + urllib.parse.quote(name, safe="") + "/" + urllib.parse.quote(version, safe="") + "/json"
    filename = name + "__" + urllib.parse.quote(version, safe=".+!") + ".json"
    path = directory / filename
    if path.exists():
        if path.stat().st_size > MAX_METADATA_BYTES + 16384:
            raise ValueError("Cached metadata exceeds response cap")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("request_url") != url or envelope.get("http_status") not in (200, 404):
            raise ValueError("Cached official metadata identity/status mismatch")
        return envelope, path
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "ExactCleanRuntimeMetadata/1.0", "Accept-Encoding": "identity"})
    opener = urllib.request.build_opener(OfficialMetadataRedirects())
    try:
        with opener.open(request, timeout=30) as response:
            if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > MAX_METADATA_BYTES:
                raise ValueError("Official PyPI response exceeds 5 MiB cap")
            data = response.read(MAX_METADATA_BYTES + 1)
            if len(data) > MAX_METADATA_BYTES:
                raise ValueError("Official PyPI response exceeds 5 MiB cap")
            envelope = {"request_url": url, "fetched_utc": now(), "http_status": response.status,
                        "response_bytes": len(data), "response_sha256": hashlib.sha256(data).hexdigest(),
                        "document": json.loads(data.decode("utf-8"))}
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        envelope = {"request_url": url, "fetched_utc": now(), "http_status": 404, "document": None,
                    "reason": "Exact version endpoint returned HTTP 404; no alternate version queried"}
        error.close()
    save_new(path, envelope)
    return envelope, path


def pypi_candidates(name, version, envelope):
    if envelope["http_status"] == 404:
        return [], []
    document = envelope["document"]
    if canonical(document["info"]["name"]) != name or not exact_version(document["info"]["version"], version):
        raise ValueError("Official PyPI exact-version response identity mismatch")
    candidates = []
    for file in document["urls"]:
        if file.get("packagetype") != "bdist_wheel":
            continue
        parsed_name, parsed_version, tags, compatible = wheel_identity(file["filename"])
        if parsed_name != name or not exact_version(parsed_version, version):
            raise ValueError("Official release contains a wheel with mismatched name/version")
        if not compatible or not python_compatible(file.get("requires_python") or document["info"].get("requires_python")):
            continue
        row = {"name": name, "version": version, "filename": file["filename"], "url": file["url"],
               "sha256": file.get("digests", {}).get("sha256"), "published_sha256": file.get("digests", {}).get("sha256"),
               "bytes": file["size"], "published_bytes": file["size"], "wheel_tags": tags, "compatible_tags": compatible,
               "requires_python": file.get("requires_python"), "source_kind": "publisher_pypi_wheel",
               "yanked": bool(file.get("yanked")), "yanked_reason": file.get("yanked_reason"),
               "hash_source": "Exact-version official PyPI JSON file digests"}
        require_file_fields(row)
        candidates.append(row)
    return sorted(candidates, key=lambda row: (row["yanked"], TAG_ORDER[row["compatible_tags"][0]], row["filename"])), document["urls"]


def known_candidate(record, name, version, publisher_candidates):
    _, _, tags, compatible = wheel_identity(record["filename"])
    row = {key: value for key, value in record.items() if key not in {"present_now", "hash_status", "wheel_tags"}}
    row.update(name=name, version=version, wheel_tags=tags, compatible_tags=compatible,
               source_kind="known_exact_official_source", published_sha256=record.get("published_sha256") or record.get("official_expected_sha256"))
    require_file_fields(row)
    if row["published_sha256"] and row["published_sha256"] != row["sha256"]:
        raise ValueError("Known observed SHA256 conflicts with known published SHA256")
    matches = [item for item in publisher_candidates if item["filename"] == row["filename"]]
    row["fresh_pypi_file_match"] = bool(matches)
    if matches:
        fresh = matches[0]
        if fresh["sha256"] != row["sha256"] or fresh["bytes"] != row["bytes"]:
            raise ValueError("Known exact source conflicts with fresh publisher digest/size")
        row.update(published_sha256=fresh["published_sha256"], published_bytes=fresh["bytes"],
                   fresh_pypi_url=fresh["url"], hash_source="Exact-version official PyPI JSON digest agrees with recorded wheel")
    return row


def resolve_package(name, installed, known, envelope, metadata_path, root):
    version = installed["version"]
    if not python_compatible(installed.get("requires_python")):
        raise ValueError("Installed Requires-Python excludes CPython 3.13.5")
    candidates, publisher_files = pypi_candidates(name, version, envelope)
    matches = []
    for record in known:
        record_name, record_version, _, compatible = wheel_identity(record["filename"])
        if record_name == name and exact_version(record_version, version) and compatible:
            matches.append(record)
    if name in KNOWN_PRECEDENCE:
        if not matches:
            raise ValueError("Required known exact source record is missing")
        chosen = known_candidate(matches[0], name, version, candidates)
        local, checked = cache_availability(chosen, known, root)
    elif candidates:
        chosen, local, checked = candidates[0], None, []
        checks = {}
        for candidate in candidates:
            available, cache_checks = cache_availability(candidate, known, root)
            checks[candidate["filename"]] = cache_checks
            if available:
                chosen, local, checked = candidate, available, cache_checks
                break
        if not local:
            checked = checks[chosen["filename"]]
    elif name == "antlr4-python3-runtime" and version == "4.9.3" and envelope["http_status"] == 200:
        chosen, local, checked, deferred = None, None, [], None
        for record in matches:
            origin = record.get("origin", {})
            origin_sha = origin.get("archive_info", {}).get("hashes", {}).get("sha256")
            if origin.get("url") != ANTLR_SOURCE_URL or origin_sha != ANTLR_SOURCE_SHA256 or not any(file.get("packagetype") == "sdist" and file.get("url") == ANTLR_SOURCE_URL and file.get("digests", {}).get("sha256") == ANTLR_SOURCE_SHA256 for file in publisher_files):
                raise ValueError("Historical ANTLR build origin does not match exact publisher archive")
            candidate = known_candidate(record, name, version, [])
            candidate.update(url=None, published_sha256=None, source_kind="historically_cached_built_wheel",
                             hash_source="Current cached wheel hash; source archive publisher hash is separately recorded",
                             publisher_wheel_available=False)
            available, cache_checks = cache_availability(candidate, known, root)
            checked.extend(cache_checks)
            if record["filename"] == "antlr4_python3_runtime-4.9.3-py3-none-any.whl" and record.get("sha256") == ANTLR_OLD_WHEEL_SHA256 and record.get("bytes") == 144577 and record.get("local_path") and cache_checks and all(not item["present_now"] for item in cache_checks):
                deferred = candidate
            if available:
                chosen, local = candidate, available
                break
        if chosen is None:
            if deferred is None:
                raise ValueError("ANTLR 4.9.3 has no verified cached wheel or absent pinned historical record; no generic source build is permitted")
            chosen = deferred
            chosen.update(availability="unavailable", source_build_status="deferred_pending_separately_approved_exact_source_build",
                          hash_source="Historical cached-wheel SHA256 only; unavailable bytes are not locally verified and this is not the new source-build wheel hash",
                          provenance_limit=chosen.get("provenance_limit", "") + " Historical wheel unavailable; only the separately approved exact ANTLR 4.9.3 source exception may produce its replacement in lock-r2. This resolver performs no build.")
    else:
        reason = "Exact installed version absent from PyPI" if envelope["http_status"] == 404 else "No compatible CPython 3.13 win_amd64 publisher wheel"
        raise ValueError(reason + "; no newer version, sdist or build selected")
    if envelope["http_status"] == 404 and name not in NON_PYPI_BUILDS:
        raise ValueError("Exact installed version absent from PyPI; no version substitution permitted")
    if local:
        chosen["local_path"] = local
    chosen.update(present_now=bool(local), cache_checks=checked, local_hash_status="current_streaming_sha256_verified" if local else "not_locally_verified",
                  metadata_url=envelope["request_url"], metadata_http_status=envelope["http_status"],
                  metadata_cache=public(metadata_path, root), metadata_observed_utc=envelope["fetched_utc"],
                  installed_metadata_sha256=installed["metadata_sha256"])
    return chosen


def resolve(root):
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    inventory = json.loads((output / "installed-metadata-inventory.json").read_text(encoding="utf-8"))
    known = json.loads((output / "cached-wheel-inventory.json").read_text(encoding="utf-8"))["wheels"]
    selected = inventory["selected"]
    if len(selected) != 90 or inventory.get("selected_declared_dependency_count") != 90:
        raise ValueError("Expected exactly 90 selected installed-version records")
    directory = root / "work/expanded-clean-runtime/metadata"
    attempt = directory / (datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S-") + str(os.getpid()) + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    packages, errors = [], []
    print(json.dumps({"phase": "metadata_resolution_started", "pid": os.getpid(), "packages": len(selected), "wheel_downloads": False}), flush=True)
    for name, installed in sorted(selected.items()):
        try:
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or canonical(installed["name"]) != name or not re.fullmatch(r"[A-Za-z0-9.!+_-]+", installed["version"]):
                raise ValueError("Invalid installed package identity")
            envelope, path = official_metadata(name, installed["version"], directory)
            row = resolve_package(name, installed, known, envelope, path, root)
            packages.append(row)
            partial = {"status": "resolved", "package": row}
        except Exception as error:
            failure = {"name": name, "version": installed.get("version"), "error_type": type(error).__name__, "error": public(error, root)}
            errors.append(failure)
            partial = {"status": "incomplete", "error": failure}
        save_new(attempt / (name + ".json"), partial)
        print(json.dumps({"name": name, "version": installed["version"], "status": partial["status"], "completed": len(packages) + len(errors), "total": len(selected)}), flush=True)
    result = {"status": "incomplete" if errors else "resolved", "resolved_utc": now(),
              "target": {"implementation": "CPython", "python": "3.13.5", "platform": "win_amd64", "free_threaded": False},
              "packages": packages, "errors": errors, "expected_package_count": 90,
              "metadata_attempt": public(attempt, root), "site_initialization_disabled": bool(sys.flags.no_site),
              "actions": {"metadata_network_permitted": True, "wheel_downloads": False, "pip": False, "builds": False, "framework_imports": False},
              "limitations": ["Known non-PyPI CUDA build URLs and publisher hashes retain their recorded provenance; HTTP 404 is not a PyPI verification of those files",
                              "Absent cached wheels are not opened or locally hash-verified", "ANTLR historical built-wheel hash is not a publisher-distributed wheel hash",
                              "Resolution records availability only; no wheel copying, download, build or installation occurs"]}
    staged = attempt / "wheel-resolution.json"
    save_new(staged, result)
    os.replace(staged, output / "wheel-resolution.json")
    print(json.dumps({"status": result["status"], "pid": os.getpid(), "resolved": len(packages), "errors": errors}), flush=True)
    return 1 if errors else 0


def selfcheck():
    global cache_availability
    assert wheel_identity("demo-1.2-cp313-cp313-win_amd64.whl")[3] == ["cp313-cp313-win_amd64"]
    assert wheel_identity("demo-1.2-cp37-abi3-win_amd64.whl")[3] == ["cp37-abi3-win_amd64"]
    assert wheel_identity("demo-1.2-py2.py3-none-any.whl")[3] == ["py3-none-any"]
    assert wheel_identity("demo-1.2-py3-none-win_amd64.whl")[3] == ["py3-none-win_amd64"]
    assert wheel_identity("demo-1.2-cp313-cp313t-win_amd64.whl")[3] == []
    assert wheel_identity("demo-1.2-cp314-abi3-win_amd64.whl")[3] == []
    assert wheel_identity("demo-1.2-cp313-cp313-manylinux_2_17_x86_64.whl")[3] == []
    assert wheel_identity("demo-1.2-cp313-cp313-win32.whl")[3] == []
    assert wheel_identity("demo-1.2-cp312-cp312-win_amd64.whl")[3] == []
    assert wheel_identity("demo-1.2-cp313-abi3-any.whl")[3] == []
    assert python_compatible(">=3.9,!=3.11.*,<4")
    assert python_compatible("~=3.13.0") and not python_compatible("~=3.12.0")
    assert not python_compatible(">=3.14") and not python_compatible("<3.13")
    assert exact_version("2.10.0+cu126", "2.10.0+cu126") and not exact_version("2.10.0", "2.10.0+cu126")
    url = "https://files.pythonhosted.org/packages/5d/22/1750a9651df15d81440f4a5ad7a7610c3f8cdbe7fd69245a9879b72da8b3/nvidia_physicsnemo-2.2.2-py3-none-any.whl"
    assert pip_http_path(url).name == "9539db41fd237aca378f99a587f85c4608c282b947f112399447f575.body"
    record = {"filename": "antlr4_python3_runtime-4.9.3-py3-none-any.whl", "bytes": 144577, "sha256": ANTLR_OLD_WHEEL_SHA256,
              "local_path": "${WORKSPACE}/mock-absent.whl", "origin": {"url": ANTLR_SOURCE_URL, "archive_info": {"hashes": {"sha256": ANTLR_SOURCE_SHA256}}}}
    envelope = {"http_status": 200, "request_url": "https://pypi.org/pypi/antlr4-python3-runtime/4.9.3/json", "fetched_utc": "mock",
                "document": {"info": {"name": "antlr4-python3-runtime", "version": "4.9.3"}, "urls": [{"packagetype": "sdist", "url": ANTLR_SOURCE_URL, "digests": {"sha256": ANTLR_SOURCE_SHA256}}]}}
    original_cache = cache_availability
    try:
        cache_availability = lambda *args: (None, [{"local_path": "${WORKSPACE}/mock-absent.whl", "present_now": False, "validation": "absent_not_opened"}])
        root = Path("mock-workspace").absolute()
        installed = {"version": "4.9.3", "metadata_sha256": "a" * 64}
        row = resolve_package("antlr4-python3-runtime", installed, [record], envelope, root / "metadata.json", root)
        assert not row["present_now"] and row["local_hash_status"] == "not_locally_verified" and row["sha256"] == ANTLR_OLD_WHEEL_SHA256
        assert row["source_build_status"] == "deferred_pending_separately_approved_exact_source_build" and row["published_sha256"] is None and row["url"] is None
        for invalid in ({**record, "sha256": "b" * 64}, {**record, "origin": {"url": ANTLR_SOURCE_URL, "archive_info": {"hashes": {"sha256": "b" * 64}}}}):
            try:
                resolve_package("antlr4-python3-runtime", installed, [invalid], envelope, root / "metadata.json", root)
            except ValueError:
                pass
            else:
                raise AssertionError("Unpinned absent ANTLR record accepted for deferral")
    finally:
        cache_availability = original_cache
    print(json.dumps({"status": "selfcheck_passed", "network": False, "writes": False, "site_initialization_disabled": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resolve-metadata", action="store_true", help="Permit bounded exact-version official PyPI JSON requests; no wheel downloads")
    mode.add_argument("--selfcheck", action="store_true", help="Check wheel tags and cache derivation; no network or writes")
    arguments = parser.parse_args()
    if arguments.selfcheck:
        selfcheck()
        return 0
    return resolve(arguments.workspace.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
