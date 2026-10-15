"""Stream exact resolved wheels; never install or import a framework."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request
import zipfile

HOSTS = {"files.pythonhosted.org", "download-r2.pytorch.org", "download.pytorch.org", "data.pyg.org"}
CHUNK = 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(CHUNK), b""):
            value.update(data)
    return value.hexdigest()


def stream_one(row, wheelhouse, root):
    filename = row["filename"]
    if Path(filename).name != filename or any(char in filename for char in ("/", "\\", ":")) or not filename.endswith(".whl"):
        raise ValueError("Unsafe/non-wheel filename")
    if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
        raise ValueError("Missing exact wheel hash/size")
    final, partial = wheelhouse / filename, wheelhouse / (filename + ".partial")
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in (wheelhouse, final, partial)):
        raise ValueError("Linked wheel destination refused")
    if final.exists():
        if final.stat().st_size != row["bytes"] or digest(final) != row["sha256"]:
            raise ValueError("Refusing conflicting wheel: " + filename)
        return {"filename": filename, "status": "already_verified", "bytes": row["bytes"], "sha256": row["sha256"]}
    count = partial.stat().st_size if partial.exists() else 0
    if count > row["bytes"]:
        raise ValueError("Oversize preserved partial")
    hasher = hashlib.sha256()
    if count:
        with partial.open("rb") as stream:
            for data in iter(lambda: stream.read(CHUNK), b""):
                hasher.update(data)
    cached = row.get("local_path")
    if cached:
        for token, value in (("${WORKSPACE}", str(root)), ("${USER_HOME}", str(Path.home())), ("${PYTHON_HOME}", sys.base_prefix)):
            cached = cached.replace(token, value)
        cached = Path(cached)
        if not cached.is_file():
            cached = None
    else:
        cached = None
    if cached is not None:
        source = cached.open("rb")
        source.seek(count)
        source_kind = "verified_cache_candidate"
    elif count == row["bytes"]:
        source = None
        source_kind = "complete_preserved_partial"
    else:
        url = row["url"]
        if urllib.parse.urlsplit(url).scheme != "https" or urllib.parse.urlsplit(url).hostname not in HOSTS:
            raise ValueError("Non-official wheel URL refused")
        headers = {"User-Agent": "physicsnemo-clean-runtime-source-preservation/1", "Accept-Encoding": "identity"}
        if count:
            headers["Range"] = "bytes=" + str(count) + "-"
        source = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
        if urllib.parse.urlsplit(source.geturl()).hostname not in HOSTS:
            source.close()
            raise ValueError("Unexpected wheel redirect host")
        if count and (source.status != 206 or not source.headers.get("Content-Range", "").startswith("bytes " + str(count) + "-")):
            source.close()
            raise ValueError("Server cannot resume preserved partial; prior bytes retained")
        source_kind = "official_https"
    if source is not None:
        try:
            with partial.open("ab" if count else "xb") as target:
                for data in iter(lambda: source.read(CHUNK), b""):
                    count += len(data)
                    if count > row["bytes"]:
                        raise ValueError("Wheel response larger than fixed metadata")
                    target.write(data)
                    hasher.update(data)
        finally:
            source.close()
    if count != row["bytes"] or hasher.hexdigest() != row["sha256"]:
        raise ValueError("Wheel bytes/hash mismatch; partial retained: " + filename)
    with zipfile.ZipFile(partial) as archive:
        if not any(name.endswith(".dist-info/WHEEL") for name in archive.namelist()):
            raise ValueError("Wheel metadata missing")
    partial.rename(final)
    return {"filename": filename, "status": "verified", "bytes": count, "sha256": hasher.hexdigest(), "source": source_kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--download-approved", action="store_true")
    args = parser.parse_args()
    assert sys.flags.no_site and args.download_approved, "Use Python -S and explicit approved stage"
    root = args.workspace.resolve()
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    wheelhouse = root / "work/expanded-clean-runtime/wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    resolution = json.loads((output / "wheel-resolution.json").read_text(encoding="utf-8"))
    if resolution["status"] != "resolved" or resolution.get("errors"):
        raise ValueError("Complete compatible exact-wheel resolution is required")
    rows = resolution["packages"]
    if len(rows) != len({row["name"].lower().replace("_", "-") for row in rows}):
        raise ValueError("Conflicting duplicate package selections")
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "files": [], "installation_performed": False}
    report = output / "wheel-download-manifest.json"
    for row in sorted(rows, key=lambda value: (value["bytes"], value["name"])):
        record["files"].append(stream_one(row, wheelhouse, root))
        report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record["files"][-1]), flush=True)
    record.update({"status": "complete_exact_wheels_verified", "finished_utc": datetime.now(timezone.utc).isoformat(), "bytes": sum(row["bytes"] for row in rows)})
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    pins = [row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"] for row in sorted(rows, key=lambda value: value["name"].lower())]
    (output / "requirements-lock.txt").write_text("# Exact resolved Windows CPython3.13 wheels; install offline with --require-hashes.\n" + "\n".join(pins) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "packages": len(rows), "bytes": record["bytes"], "installation_performed": False}), flush=True)


if __name__ == "__main__":
    main()
