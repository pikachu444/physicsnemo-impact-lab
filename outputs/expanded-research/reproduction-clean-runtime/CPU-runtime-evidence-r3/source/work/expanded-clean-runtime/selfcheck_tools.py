"""Tiny stdlib copy/hash/path tests; no network or ML imports."""
import hashlib
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

from download_wheels import stream_one, defer_exact_antlr
import download_wheels


def main():
    assert sys.flags.no_site
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="small-wheel-selfcheck-", dir=own) as temporary:
        parent = Path(temporary).resolve()
        assert parent.is_relative_to(own.resolve())
        source = parent / "source.whl"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("demo-1.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        data = source.read_bytes()
        row = {"filename": "demo-1-py3-none-any.whl", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "local_path": str(source)}
        wheels = parent / "wheelhouse"
        wheels.mkdir()
        assert stream_one(row, wheels, own)["status"] == "verified"
        assert stream_one(row, wheels, own)["status"] == "already_verified"
        resume = parent / "resume"
        resume.mkdir()
        (resume / (row["filename"] + ".partial")).write_bytes(data[:50])
        assert stream_one(row, resume, own)["sha256"] == row["sha256"]
        remote_row = dict(row, url="https://files.pythonhosted.org/test/demo.whl")
        remote_row.pop("local_path")

        class Reply(io.BytesIO):
            def __init__(self, body, status, headers):
                super().__init__(body)
                self.status, self.headers = status, headers

            def geturl(self):
                return remote_row["url"]

        original_open = download_wheels.urllib.request.urlopen
        try:
            legacy = parent / "http-legacy"
            legacy.mkdir()
            old = legacy / (row["filename"] + ".partial")
            old.write_bytes(data[:50])
            requests = []

            def whole(request, timeout):
                requests.append(dict(request.header_items()))
                return Reply(data, 200, {"ETag": '"whole"', "Accept-Ranges": "bytes"})

            download_wheels.urllib.request.urlopen = whole
            result = stream_one(remote_row, legacy, own)
            assert result["resumed_bytes"] == 0 and result["previous_partials_retained"]
            assert old.read_bytes() == data[:50] and not any(key.lower() == "range" for key in requests[0])
            remote = parent / "http-resume"
            remote.mkdir()
            (remote / (row["filename"] + ".partial")).write_bytes(data[:10])
            part = remote / (row["filename"] + ".full-preserved.partial")
            part.write_bytes(data[:50])
            part.with_name(part.name + ".receipt.json").write_text(json.dumps({"url": remote_row["url"], "sha256": row["sha256"], "bytes": len(data), "etag": '"stable"'}))

            def ranged(request, timeout):
                assert request.get_header("Range") == "bytes=50-" and request.get_header("If-range") == '"stable"'
                return Reply(data[50:], 206, {"ETag": '"stable"', "Content-Range": "bytes 50-" + str(len(data) - 1) + "/" + str(len(data))})

            download_wheels.urllib.request.urlopen = ranged
            assert stream_one(remote_row, remote, own)["resumed_bytes"] == 50
        finally:
            download_wheels.urllib.request.urlopen = original_open
        antlr = dict(row, name="antlr4-python3-runtime", version="4.9.3", local_path="${WORKSPACE}/unavailable.whl", url=None, sha256=download_wheels.ANTLR_OLD_WHEEL_SHA256, origin={"archive_info": {"hashes": {"sha256": download_wheels.ANTLR_SOURCE_SHA256}}})
        assert defer_exact_antlr(antlr, parent / "absent", own)
        assert not defer_exact_antlr(dict(antlr, sha256="0" * 64), parent / "absent", own)
        assert not defer_exact_antlr(dict(antlr, name="other"), parent / "absent", own)
        assert not defer_exact_antlr(dict(antlr, version="4.9.4"), parent / "absent", own)
        assert not defer_exact_antlr(dict(antlr, url="https://files.pythonhosted.org/publisher.whl"), parent / "absent", own)
        assert not defer_exact_antlr(dict(antlr, local_path=str(source)), parent / "absent", own)
        assert not defer_exact_antlr(antlr, wheels, own)
        for filename in ("../escape.whl", "C:/escape.whl", "data.npy"):
            try:
                stream_one(dict(row, filename=filename), wheels, own)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe name accepted")
        (wheels / row["filename"]).write_bytes(b"conflict")
        try:
            stream_one(row, wheels, own)
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting wheel accepted")
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(), "checks": ["tiny_cached_wheel_copy", "exact_hash_size", "existing_verified_wheel_reuse", "preserved_cached_partial_resume", "legacy_HTTP_partial_retained_without_ETag", "mocked_HTTP_Range_ETag_ContentRange_resume", "only_exact_missing_ANTLR_cached_wheel_deferred", "unsafe_filename_refusals", "conflicting_wheel_refusal"], "network_used": False, "framework_imported": False}
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
