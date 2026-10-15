"""Verify packaged bytes and parse text formats without importing ML runtimes."""
import ast
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads((root / "publication-byte-manifest.json").read_text(encoding="utf-8"))
errors = []
counts = {"files": 0, "json": 0, "jsonl": 0, "python": 0}
for rel, entry in manifest["files"].items():
    path = root / rel
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["public_sha256"]:
        errors.append([rel, "public byte hash mismatch"])
    try:
        if path.suffix == ".json":
            json.loads(data.decode("utf-8-sig"))
            counts["json"] += 1
        elif path.suffix == ".jsonl":
            for line in data.decode("utf-8-sig").splitlines():
                if line.strip():
                    json.loads(line)
            counts["jsonl"] += 1
        elif path.suffix == ".py":
            ast.parse(data.decode("utf-8-sig"), filename=rel)
            counts["python"] += 1
    except Exception as exc:
        errors.append([rel, str(exc)])
    counts["files"] += 1
print(json.dumps({"verified": counts, "errors": errors}))
raise SystemExit(bool(errors))
