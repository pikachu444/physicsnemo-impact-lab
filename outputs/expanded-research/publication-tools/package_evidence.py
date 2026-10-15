"""Copy reviewed text/figure evidence; keep raw arrays and executables local.

Public JSON/log copies redact local account paths. The manifest records both
the original and public byte hashes so redaction never masquerades as an exact
copy of the measured evidence. Executed Python source snapshots are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ALLOWED = {".md", ".json", ".jsonl", ".csv", ".py", ".ps1", ".txt", ".log",
           ".png", ".yaml", ".yml", ".toml", ".out"}
REDACT = {".md", ".json", ".jsonl", ".csv", ".txt", ".log", ".yaml", ".yml", ".out"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_local_paths(text: str, workspace: Path, home: Path) -> str:
    # Nested JSON log lines can escape each separator several times.
    for before, after in ((workspace, "<LOCAL_WORKSPACE>"), (home, "<LOCAL_ACCOUNT>")):
        parts = re.split(r"[\\/]+", str(before))
        pattern = r"[\\/]+".join(re.escape(part) for part in parts)
        text = re.sub(pattern, lambda _match: after, text, flags=re.IGNORECASE)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--select", nargs="+", required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    source = (workspace / "outputs/expanded-research").resolve(strict=True)
    repo = (workspace / "outputs/physicsnemo-impact-lab").resolve(strict=True)
    destination = repo / "outputs/expanded-research"
    # No filesystem removal or moves. Every output stays in the named repo.
    assert repo.is_relative_to(workspace / "outputs")
    destination.mkdir(parents=True, exist_ok=True)
    entries = {}
    withheld = {}
    manifest_path = destination / "publication-byte-manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = previous["files"]
        withheld = previous.get("withheld_exact_sources", {})
    copied = 0
    excluded = 0
    for selected in args.select:
        root = (source / selected).resolve(strict=True)
        assert root.is_relative_to(source)
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            # These two text metadata receipts describe the old inherited
            # environment. They are never installed/executed by this tool.
            metadata_area = source / "benchmarks/temporal_strategies/resource-calibration-v1/metadata"
            is_text_metadata = path.is_relative_to(metadata_area) and (path.suffix.lower() == ".pth" or path.name == "METADATA")
            if "__pycache__" in path.parts or (path.suffix.lower() not in ALLOWED and not is_text_metadata):
                excluded += 1
                continue
            original = path.read_bytes()
            rel = path.relative_to(source)
            if path.suffix.lower() in (".py", ".ps1"):
                original_text = original.decode("utf-8-sig")
                if redact_local_paths(original_text, workspace, Path.home()) != original_text:
                    # Never silently rewrite an executed source. A separate
                    # explicitly unexecuted portable derivative may be public.
                    if rel.as_posix() in entries or (destination / rel).exists():
                        raise ValueError("Previously packaged source now contains a local account path: " + rel.as_posix())
                    withheld[rel.as_posix()] = {"sha256": sha(original), "bytes": len(original),
                        "reason": "Exact local source contains workstation/account paths; preserved locally unchanged, not published or silently redacted."}
                    excluded += 1
                    continue
            public = original
            if path.suffix.lower() in REDACT or is_text_metadata:
                text = original.decode("utf-8-sig")
                text = redact_local_paths(text, workspace, Path.home())
                public = text.encode("utf-8")
            target = (destination / rel).resolve()
            assert target.is_relative_to(destination.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(public)
            entries[rel.as_posix()] = {
                "original_sha256": sha(original),
                "public_sha256": sha(public),
                "original_bytes": len(original),
                "public_bytes": len(public),
                "transformation": "local-path redaction and UTF-8 normalization" if original != public else "none",
            }
            copied += 1
    manifest = {
        "schema_version": 1,
        "scope": "Code, execution logs, source/data hashes, metrics and figures only.",
        "excluded": "Raw decks, FEM/VTK/TFRecord files, NPZ/NPY data and predictions, model weights, executables, libraries and bytecode remain local.",
        "original_evidence": "Local outputs/expanded-research retains original bytes. This manifest distinguishes any public path-redacted copies.",
        "withheld_exact_sources": dict(sorted(withheld.items())),
        "files": dict(sorted(entries.items())),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"copied": copied, "excluded": excluded, "withheld_exact_sources": len(withheld), "tracked_public_files": len(entries)}))


if __name__ == "__main__":
    main()
