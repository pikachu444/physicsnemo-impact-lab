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


def new_partial(original):
    return original.with_name(original.name.split(".whl", 1)[0] + ".whl.full-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".partial")


def choose_partial(row, wheelhouse):
    original = wheelhouse / (row["filename"] + ".partial")
    candidates = []
    for path in (original, *wheelhouse.glob(row["filename"] + ".full-*.partial")):
        if not path.is_file() or path.is_symlink():
            continue
        receipt = path.with_name(path.name + ".receipt.json")
        value = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
        good = value.get("url") == row.get("url") and value.get("sha256") == row["sha256"] and value.get("bytes") == row["bytes"] and bool(re.fullmatch(r'"[^"\r\n]*"', value.get("etag") or ""))
        size = path.stat().st_size
        if size == row["bytes"] or (good and size < row["bytes"]):
            candidates.append((size, path.name, path))
    return max(candidates)[2] if candidates else original


def open_http(row, original):
    """Resume only with a previously recorded strong ETag; retain every old partial."""
    url = row["url"]
    if urllib.parse.urlsplit(url).scheme != "https" or urllib.parse.urlsplit(url).hostname not in HOSTS:
        raise ValueError("Non-official wheel URL refused")
    partial = original
    count = partial.stat().st_size if partial.exists() else 0
    receipt_path = partial.with_name(partial.name + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    etag = receipt.get("etag") or ""
    can_resume = count and receipt.get("url") == url and receipt.get("sha256") == row["sha256"] and receipt.get("bytes") == row["bytes"] and bool(re.fullmatch(r'"[^"\r\n]*"', etag))
    retained = []
    if count and not can_resume:
        retained.append({"file": partial.name, "bytes": count, "reason": "No previously recorded matching strong ETag"})
        partial, count = new_partial(original), 0
    headers = {"User-Agent": "physicsnemo-clean-runtime-source-preservation/1", "Accept-Encoding": "identity"}
    if count:
        headers.update(Range="bytes=" + str(count) + "-", **{"If-Range": etag})
    response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)
    if urllib.parse.urlsplit(response.geturl()).hostname not in HOSTS:
        response.close()
        raise ValueError("Unexpected wheel redirect host")
    if count:
        content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        valid = response.status == 206 and response.headers.get("ETag") == etag and content_range and int(content_range[1]) == count and int(content_range[3]) == row["bytes"] and count <= int(content_range[2]) < row["bytes"]
        if not valid:
            response.close()
            retained.append({"file": partial.name, "bytes": count, "reason": "Range/ETag/Content-Range did not verify"})
            partial, count = new_partial(original), 0
            response = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": headers["User-Agent"], "Accept-Encoding": "identity"}), timeout=30)
            if urllib.parse.urlsplit(response.geturl()).hostname not in HOSTS or response.status != 200:
                response.close()
                raise ValueError("Whole-wheel response did not verify")
    elif response.status != 200:
        response.close()
        raise ValueError("Unexpected whole-wheel response status")
    receipt = {"url": url, "sha256": row["sha256"], "bytes": row["bytes"], "etag": response.headers.get("ETag"),
               "content_range": response.headers.get("Content-Range"), "accept_ranges": response.headers.get("Accept-Ranges"),
               "resume_bytes": count, "received_utc": datetime.now(timezone.utc).isoformat(), "previous_partials_retained": retained}
    receipt_path = partial.with_name(partial.name + ".receipt.json")
    staged = receipt_path.with_name(receipt_path.name + ".new")
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    staged.replace(receipt_path)
    return response, partial, count, retained


def stream_one(row, wheelhouse, root):
    filename = row["filename"]
    if Path(filename).name != filename or any(char in filename for char in ("/", "\\", ":")) or not filename.endswith(".whl"):
        raise ValueError("Unsafe/non-wheel filename")
    if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
        raise ValueError("Missing exact wheel hash/size")
    final, partial = wheelhouse / filename, choose_partial(row, wheelhouse)
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in (wheelhouse, final, partial)):
        raise ValueError("Linked wheel destination refused")
    if final.exists():
        if final.stat().st_size != row["bytes"] or digest(final) != row["sha256"]:
            raise ValueError("Refusing conflicting wheel: " + filename)
        return {"filename": filename, "status": "already_verified", "bytes": row["bytes"], "sha256": row["sha256"]}
    count = partial.stat().st_size if partial.exists() else 0
    if count > row["bytes"]:
        raise ValueError("Oversize preserved partial")
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
        retained = []
    elif count == row["bytes"]:
        source = None
        source_kind = "complete_preserved_partial"
        retained = []
    else:
        source, partial, count, retained = open_http(row, partial)
        source_kind = "official_https"
    resume_bytes = count
    hasher = hashlib.sha256()
    if count:
        with partial.open("rb") as stream:
            for data in iter(lambda: stream.read(CHUNK), b""):
                hasher.update(data)
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
    return {"filename": filename, "status": "verified", "bytes": count, "sha256": hasher.hexdigest(), "source": source_kind, "resumed_bytes": resume_bytes, "previous_partials_retained": retained}


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
