"""Resolve/download only the two pinned pure-Python packaging tools."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

TOOLS = {"setuptools": "80.9.0", "wheel": "0.45.1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--download-approved", action="store_true")
    args = parser.parse_args()
    assert args.download_approved and sys.flags.no_site
    root = args.workspace.resolve()
    own = root / "work/expanded-clean-runtime"
    output = root / "outputs/expanded-research/reproduction-clean-runtime"
    runtime = json.loads((output / "wheel-resolution.json").read_text(encoding="utf-8"))
    selected = []
    for name, version in TOOLS.items():
        url = "https://pypi.org/pypi/" + name + "/" + version + "/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(5 * 1024**2 + 1)
        if len(data) > 5 * 1024**2:
            raise ValueError("Build-tool metadata exceeds bound")
        release = json.loads(data)
        assert release["info"]["version"] == version
        candidates = [row for row in release["urls"] if row["packagetype"] == "bdist_wheel" and row["filename"].endswith("-py3-none-any.whl") and not row.get("yanked")]
        if len(candidates) != 1:
            raise ValueError("Expected one exact pure-Python build-tool wheel")
        row = candidates[0]
        filename, expected = row["filename"], row["digests"]["sha256"]
        path = own / "wheelhouse" / filename
        if path.exists():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected and path.stat().st_size == row["size"]
        else:
            if row["size"] > 5 * 1024**2 or not row["url"].startswith("https://files.pythonhosted.org/"):
                raise ValueError("Unexpected build-tool wheel source/size")
            with urllib.request.urlopen(row["url"], timeout=30) as response:
                binary = response.read(row["size"] + 1)
            assert len(binary) == row["size"] and hashlib.sha256(binary).hexdigest() == expected
            with path.open("xb") as stream:
                stream.write(binary)
        selected.append({"name": name, "version": version, "filename": filename, "url": row["url"], "bytes": row["size"], "sha256": expected,
                         "metadata_url": url, "metadata_sha256": hashlib.sha256(data).hexdigest(), "wheel_tags": ["py3-none-any"]})
    (output / "build-tools-lock.txt").write_text("\n".join(row["name"] + "==" + row["version"] + " --hash=sha256:" + row["sha256"] for row in selected) + "\n", encoding="utf-8")
    (output / "build-tools-resolution.json").write_text(json.dumps({"status": "exact_build_tools_wheels_verified", "packages": selected, "runtime_dependency_closure_changed": False, "tools_installed": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "exact_build_tools_wheels_verified", "tools": TOOLS, "installed": False}))


if __name__ == "__main__":
    main()
