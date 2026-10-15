"""Inventory and stream only pinned public ANIM frames; source data stay in work/.

Run ``inventory --help`` first. Inventory never downloads FEM binary data.
Download is a separate explicit command and is limited to at most 6 GB.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "AIRBORNEPANDA/BumperBeamCrashExample"
REVISION = "78f1302fba4d75926c6fdd22d01d5133efc14742"
ROOT_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/RAW_DATA?recursive=true&expand=false&limit=1000"
MAX_BYTES = 6_000_000_000
CHUNK = 1024 * 1024
ANIM = re.compile(r"^(RAW_DATA/(TRAINING_DATA|VALIDATION_DATA)/(Run\d+))/([^/]+)A(\d{3})$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative source path: {value}")
    return path


def get_json(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "PhysicsNeMo-public-research/1.0"}), timeout=90) as response:
                return json.load(response), response.headers.get("Link", "")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_tree(destination):
    """Follow server Link cursors, including pages containing directory entries."""
    url, seen, pages, entries = TREE_URL, set(), [], []
    while url:
        if url in seen or not url.startswith(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/"):
            raise ValueError("Invalid or repeated pagination URL")
        seen.add(url)
        items, link = get_json(url)
        if not isinstance(items, list):
            raise ValueError("HF tree response must be a JSON list")
        number = len(pages) + 1
        save_json(destination / f"raw-tree-page-{number:03}.json", items)
        entries.extend(items)
        pages.append({"number": number, "url": url, "entry_count": len(items), "link": link})
        print(f"tree page {number}: {len(items)} entries", flush=True)
        matches = re.findall(r'<([^>]+)>;\s*rel="?next"?', link)
        if len(matches) > 1:
            raise ValueError("Ambiguous pagination links")
        url = matches[0] if matches else None
    paths = [item["path"] for item in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Tree pagination returned duplicate paths")
    save_json(destination / "raw-tree-all.json", entries)
    save_json(destination / "raw-tree-pages.json", {"dataset": REPO, "revision": REVISION, "fetched_utc": utc_now(), "pages": pages})
    return entries


def select_frames(entries):
    files, runs, extra = [], {}, []
    for item in entries:
        if item["type"] != "file":
            continue
        match = ANIM.fullmatch(item["path"])
        if not match:
            continue
        directory, split, run, stem, frame = match.groups()
        frame = int(frame)
        lfs = item.get("lfs") or {}
        digest = lfs.get("oid")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing LFS SHA256: {item['path']}")
        size = item["size"]
        if not isinstance(size, int) or size <= 0 or lfs.get("size") != size:
            raise ValueError(f"Invalid LFS size: {item['path']}")
        record = {"path": item["path"], "url": ROOT_URL + quote(item["path"], safe="/"), "run": run,
                  "source_split": split, "frame": frame, "bytes": size, "lfs_sha256": digest, "git_blob_oid": item.get("oid")}
        if 1 <= frame <= 11:
            files.append(record)
            case = runs.setdefault(run, {"run": run, "source_split": split, "source_directory": directory, "stem": stem, "frames": []})
            if case["source_directory"] != directory or case["stem"] != stem:
                raise ValueError(f"Ambiguous run source: {run}")
            case["frames"].append(frame)
        else:
            extra.append(record)
    for case in runs.values():
        case["frames"].sort()
        if case["frames"] != list(range(1, 12)):
            raise ValueError(f"Incomplete or duplicate 11-frame sequence: {case}")
    return sorted(files, key=lambda x: (int(x["run"][3:]), x["frame"])), sorted(runs.values(), key=lambda x: int(x["run"][3:])), extra


def inventory(args):
    destination = args.inventory_dir
    entries = fetch_tree(destination)
    files, runs, extra = select_frames(entries)
    siblings = json.loads(args.dataset_meta.read_text(encoding="utf-8"))
    if siblings["sha"] != REVISION:
        raise ValueError("Local dataset metadata does not match pinned revision")
    expected_raw = {x["rfilename"] for x in siblings["siblings"] if x["rfilename"].startswith("RAW_DATA/")}
    actual_raw = {x["path"] for x in entries if x["type"] == "file"}
    if actual_raw != expected_raw:
        raise ValueError(f"Raw tree differs from sibling inventory: missing={sorted(expected_raw-actual_raw)}, unexpected={sorted(actual_raw-expected_raw)}")
    split = json.loads(args.split_metrics.read_text(encoding="utf-8"))["split"]
    mapped = {run: name for name, group in split.items() for run in group}
    if sum(map(len, split.values())) != len(mapped) or set(mapped) != {x["run"] for x in runs}:
        raise ValueError("Existing train/validation/test split must partition every selected run exactly once")
    if {name: len(group) for name, group in split.items()} != {"train": 112, "validation": 12, "test": 7}:
        raise ValueError("Unexpected existing split counts")
    for case in runs:
        case["experiment_split"] = mapped[case["run"]]
        case["bytes"] = sum(x["bytes"] for x in files if x["run"] == case["run"])
    for item in files:
        item["experiment_split"] = mapped[item["run"]]
    total = sum(item["bytes"] for item in files)
    result = {"dataset": REPO, "revision": REVISION, "created_utc": utc_now(), "tree_complete_against_metadata_siblings": True,
              "raw_files_count": len(actual_raw), "raw_total_bytes": sum(x["size"] for x in entries if x["type"] == "file"),
              "selected_run_count": len(runs), "selected_file_count": len(files), "selected_bytes": total,
              "selected_decimal_GB": total / 1e9, "selected_GiB": total / (1024 ** 3), "approved_byte_ceiling": MAX_BYTES,
              "selected_frame_numbers": list(range(1, 12)), "source_extra_frames_excluded": extra,
              "experiment_split": split, "experiment_split_metrics_sha256": sha256_file(args.split_metrics),
              "runs": runs, "files": files}
    save_json(destination / "selected-anim-manifest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "runs", "experiment_split", "source_extra_frames_excluded"}}, indent=2), flush=True)
    print("excluded extra ANIM frames:", [x["path"] for x in extra], flush=True)
    if total > MAX_BYTES:
        raise ValueError("Selected downloads exceed approved 6 GB ceiling; no binaries were downloaded")
    if len(runs) != 131 or len(files) != 131 * 11:
        raise ValueError("Expected exactly 131 public cases, with A001..A011 each")


def download_file(item, destination):
    path = destination.joinpath(*valid_relative_path(item["path"]).parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size == item["bytes"] and sha256_file(path) == item["lfs_sha256"]:
            return {"path": item["path"], "bytes": item["bytes"], "sha256": item["lfs_sha256"], "status": "verified_existing"}
        raise ValueError(f"Existing source file differs from manifest; preserve and inspect: {item['path']}")
    partial = path.with_name(path.name + ".part")
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["bytes"]:
                raise ValueError(f"Oversized partial file: {item['path']}")
            if offset < item["bytes"]:
                headers = {"User-Agent": "PhysicsNeMo-public-research/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(item["url"], headers=headers), timeout=120) as response:
                    append = offset > 0 and response.status == 206
                    if response.status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected byte range response")
                    written = offset if append else 0
                    with partial.open("ab" if append else "wb") as stream:
                        for chunk in iter(lambda: response.read(CHUNK), b""):
                            written += len(chunk)
                            if written > item["bytes"]:
                                raise ValueError("Download exceeds manifest size")
                            stream.write(chunk)
            if partial.stat().st_size != item["bytes"] or sha256_file(partial) != item["lfs_sha256"]:
                raise ValueError(f"Downloaded size or LFS SHA256 differs: {item['path']}")
            partial.replace(path)
            return {"path": item["path"], "bytes": item["bytes"], "sha256": item["lfs_sha256"], "status": "downloaded_verified"}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def output_snapshot(directory):
    for name in ("nodal_targets.npy", "time_ms.npy", "manifest.json"):
        if not (directory / name).is_file():
            raise ValueError(f"Extractor did not produce required {name}")
    if json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("status") != "complete":
        raise ValueError("Extractor output manifest does not report complete")
    return [{"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(directory.rglob("*")) if path.is_file()]


def completed_extraction_matches(previous, manifest_hash, script_hash, arguments, output):
    extraction = previous.get("extraction", {})
    if previous.get("manifest_sha256") != manifest_hash or extraction.get("exit_code") != 0 or extraction.get("script_sha256") != script_hash or extraction.get("command_arguments") != arguments or not extraction.get("output_files"):
        return False
    try:
        return output_snapshot(output) == extraction["output_files"]
    except (OSError, ValueError):
        return False


def download(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest["files"]
    if manifest["dataset"] != REPO or manifest["revision"] != REVISION or sum(x["bytes"] for x in files) != manifest["selected_bytes"] or not 0 < manifest["selected_bytes"] <= MAX_BYTES:
        raise ValueError("Manifest revision, byte count or approved ceiling invalid")
    recorded_runs = {x["run"]: x for x in manifest["runs"]}
    by_run = {}
    for item in files:
        valid_relative_path(item["path"])
        match = ANIM.fullmatch(item["path"])
        if item["url"] != ROOT_URL + quote(item["path"], safe="/") or not match:
            raise ValueError("Manifest contains a source outside pinned ANIM paths")
        directory, source_split, run, stem, frame = match.groups()
        case = recorded_runs.get(run)
        if not case or item["run"] != run or item["source_split"] != source_split or item["frame"] != int(frame) or not isinstance(item["bytes"], int) or item["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item["lfs_sha256"]):
            raise ValueError("Manifest source metadata is inconsistent")
        if (case["source_directory"], case["stem"], case["source_split"], case["experiment_split"]) != (directory, stem, source_split, item["experiment_split"]):
            raise ValueError("Manifest run metadata is inconsistent")
        by_run.setdefault(run, []).append(int(frame))
    if len(files) != 1441 or len(recorded_runs) != 131 or len(manifest["runs"]) != 131 or set(by_run) != set(recorded_runs) or any(sorted(frames) != list(range(1, 12)) for frames in by_run.values()):
        raise ValueError("Manifest must contain each of 131 runs and A001..A011 exactly once")
    if manifest["selected_file_count"] != len(files) or manifest["selected_run_count"] != len(recorded_runs):
        raise ValueError("Manifest selection counts are inconsistent")
    if any(case["bytes"] != sum(x["bytes"] for x in files if x["run"] == run) for run, case in recorded_runs.items()):
        raise ValueError("Manifest per-run byte counts are inconsistent")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 bounded download workers")
    if args.extract_script and not args.cache_root:
        raise ValueError("--cache-root is required with --extract-script")
    selected_runs = args.runs or [case["run"] for case in manifest["runs"]]
    if len(set(selected_runs)) != len(selected_runs) or set(selected_runs) - {x["run"] for x in manifest["runs"]}:
        raise ValueError("Requested run list is duplicated or contains unknown cases")
    args.source_root.mkdir(parents=True, exist_ok=True)
    lock = args.source_root / ".bulk-download.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} started_utc={utc_now()}\n")
    start = time.perf_counter()
    try:
        save_json(args.source_root / "source-global-manifest.json", manifest)
        for number, run in enumerate(selected_runs, 1):
            case = next(x for x in manifest["runs"] if x["run"] == run)
            sources = [x for x in files if x["run"] == run]
            case_directory = args.source_root.joinpath(*valid_relative_path(case["source_directory"]).parts)
            receipt = case_directory / "source-manifest.json"
            previous = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else {}
            result = {"dataset": REPO, "revision": REVISION, "run": run, "source_split": case["source_split"],
                      "experiment_split": case["experiment_split"], "manifest_sha256": sha256_file(args.manifest), "files": [], "status": "downloading"}
            if not previous:
                save_json(receipt, result)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                result["files"] = list(executor.map(lambda item: download_file(item, args.source_root), sources))
            result.update({"status": "all_sources_size_and_sha256_verified", "verified_utc": utc_now()})
            if not args.extract_script:
                if previous.get("manifest_sha256") == result["manifest_sha256"] and "extraction" in previous:
                    result["extraction"] = previous["extraction"]
                save_json(receipt, result)
            print(f"case {number}/{len(selected_runs)} {run}: verified {len(sources)} frames, {case['bytes']} bytes", flush=True)
            if args.extract_script:
                output = args.cache_root / run
                command = [sys.executable, str(args.extract_script.resolve()), "--anim-dir", str(case_directory.resolve()), "--run-name", run,
                           "--output-dir", str(output.resolve()), *args.extract_extra_arg]
                arguments = ["--anim-dir", case["source_directory"], "--run-name", run, "--output-dir", run, *args.extract_extra_arg]
                script_hash = sha256_file(args.extract_script)
                if completed_extraction_matches(previous, result["manifest_sha256"], script_hash, arguments, output):
                    result["extraction"] = previous["extraction"]
                    result["extraction"]["reverified_utc"] = utc_now()
                    save_json(receipt, result)
                    print(f"{run}: prior extraction outputs size/hash reverified; skipped extraction", flush=True)
                    continue
                result["extraction"] = {"script_sha256": script_hash, "started_utc": utc_now(), "command_arguments": arguments}
                save_json(receipt, result)
                with (case_directory / "extract.stdout.log").open("w", encoding="utf-8") as stdout, (case_directory / "extract.stderr.log").open("w", encoding="utf-8") as stderr:
                    completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
                result["extraction"].update({"exit_code": completed.returncode, "finished_utc": utc_now()})
                if completed.returncode == 0:
                    result["extraction"]["output_files"] = output_snapshot(output)
                save_json(receipt, result)
                if completed.returncode:
                    raise RuntimeError(f"Extraction failed for {run}; see per-case logs")
        report = {"revision": REVISION, "finished_utc": utc_now(), "elapsed_seconds": time.perf_counter() - start,
                  "runs": selected_runs, "file_count": sum(x["run"] in selected_runs for x in files),
                  "bytes": sum(x["bytes"] for x in files if x["run"] in selected_runs), "download_workers": args.workers,
                  "read_chunk_bytes_per_worker": CHUNK, "manifest_sha256": sha256_file(args.manifest), "extraction_requested": bool(args.extract_script)}
        save_json(args.source_root / "download-reports" / f"completed-{time.time_ns()}.json", report)
        save_json(args.source_root / "completed-download-report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("inventory", help="Retrieve complete pinned metadata, select and size-check A001..A011; no binaries")
    check.add_argument("--inventory-dir", type=Path, required=True)
    check.add_argument("--dataset-meta", type=Path, required=True)
    check.add_argument("--split-metrics", type=Path, required=True)
    fetch = commands.add_parser("download", help="Explicitly stream and hash-verify selected ANIM files, run-by-run")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--source-root", type=Path, required=True)
    fetch.add_argument("--workers", type=int, default=2)
    fetch.add_argument("--runs", nargs="+", help="Optional bounded subset for smoke runs; default all manifest runs")
    fetch.add_argument("--extract-script", type=Path, help="Optional main-owned extract-anim-fields.py called after each verified run")
    fetch.add_argument("--cache-root", type=Path)
    fetch.add_argument("--extract-extra-arg", action="append", default=[], help="Pass-through extractor argument; use --extract-extra-arg=--flag")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory(args)
    else:
        download(args)


if __name__ == "__main__":
    main()
