"""Inspect wheel startup hooks before any installed Python startup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ALLOWED = "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = os.environ.get(var, 'local') == 'local'; enabled and __import__('_distutils_hack').add_shim();"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    assert sys.flags.no_site
    root = args.workspace.resolve()
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    selected = json.loads((output / "wheel-resolution.json").read_text(encoding="utf-8"))["packages"]
    hooks, findings = [], []
    for row in selected:
        path = root / "work/expanded-clean-runtime/wheelhouse" / row["filename"]
        assert path.is_file() and path.stat().st_size == row["bytes"]
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".pth"):
                    continue
                entry = archive.getinfo(name)
                if entry.file_size > 16384:
                    raise ValueError("Unexpectedly large startup hook")
                data = archive.read(name)
                text = data.decode("utf-8")
                accepted = row["name"] == "setuptools" and Path(name).name == "distutils-precedence.pth" and text.strip() == ALLOWED
                hooks.append({"wheel": row["filename"], "path": name, "sha256": hashlib.sha256(data).hexdigest(), "content": text,
                              "allowed_exact_setuptools_distutils_hook": accepted})
                if not accepted:
                    findings.append({"wheel": row["filename"], "path": name, "reason": "Unreviewed Python startup hook"})
    result = {"status": "passed_no_framework_startup_hooks" if not findings else "failed_unreviewed_startup_hook", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "wheels_checked": len(selected), "pth_files": hooks, "findings": findings, "framework_imported": False, "wheel_payloads_executed": False}
    (output / "wheel-startup-hook-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
