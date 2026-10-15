"""Small stdlib verification of the captured sources; no experiment execution."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(package):
    specification = importlib.util.spec_from_file_location("addendum_restore_only", package / "bootstrap.py")
    bootstrap = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(bootstrap)
    manifest, records = bootstrap.verify()
    index = json.loads((package / "package-manifest.json").read_text(encoding="utf-8"))
    for item in index["files"]:
        data = (package / bootstrap.safe_relative(item["path"])).read_bytes()
        assert len(data) == item["bytes"] and sha(data) == item["sha256"], item["path"]
    python_count = 0
    for path in package.rglob("*.py"):
        ast.parse(path.read_bytes(), filename=path.relative_to(package).as_posix())
        python_count += 1
    restored, refused = {}, []
    with tempfile.TemporaryDirectory(prefix="research-source-selfcheck-") as temporary:
        parent = Path(temporary)
        for profile in ("current", *sorted(manifest["profiles"])):
            target = parent / profile
            dry = bootstrap.restore(target, profile, True)
            assert not target.exists() and dry["status"] == "dry_run"
            result = bootstrap.restore(target, profile)
            chosen = {records[key]["destination"]: records[key] for key in manifest["default_files"]}
            for key in manifest["profiles"].get(profile, {}).get("overrides", []):
                chosen[records[key]["destination"]] = records[key]
            for destination, record in chosen.items():
                assert sha((target / destination).read_bytes()) == record["sha256"], destination
            before = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
            try:
                bootstrap.restore(target, profile)
            except FileExistsError:
                refused.append(profile + ":overwrite")
            else:
                raise AssertionError("Overwrite was accepted")
            assert before == sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
            restored[profile] = {"files": result["source_files"], "bytes": result["bytes"], "hashes_verified": len(chosen)}
        for value in ("../outside.py", "/outside.py", "work/../outside.py", "work//outside.py", "work/venv/a.py", "work/cache/a.py", "C:/outside.py", "work/a.npy", "work\\outside.py"):
            try:
                bootstrap.safe_relative(value, "work")
            except ValueError:
                refused.append("path:" + value)
            else:
                raise AssertionError("Unsafe path accepted: " + value)
        clone = parent / "conflict-package"
        shutil.copytree(package, clone)
        altered = json.loads((clone / "source-manifest.json").read_text(encoding="utf-8"))
        altered["default_files"].append(altered["default_files"][0])
        (clone / "source-manifest.json").write_text(json.dumps(altered), encoding="utf-8")
        bootstrap.PACKAGE = clone
        try:
            bootstrap.verify()
        except ValueError as error:
            assert "Conflicting destination" in str(error)
            refused.append("manifest:conflicting-destination")
        else:
            raise AssertionError("Conflicting destination was accepted")
        finally:
            bootstrap.PACKAGE = package
    base = package.parents[0] / "reproduction/source-manifest.json"
    original_unchanged = None
    if base.exists():
        original_unchanged = sha(base.read_bytes()) == manifest["base_bundle_unchanged"]["manifest_sha256"]
        assert original_unchanged
    report = {"status": "passed_source_preservation_and_restore_only", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "capture_as_of_utc": manifest["created_utc"], "manifest_sha256": sha((package / "source-manifest.json").read_bytes()),
              "packaged_files_verified": len(records), "python_ast_files_verified": python_count, "profiles_restored_in_fresh_temporary_directories": restored,
              "refusals_verified": refused, "original_bundle_manifest_unchanged": original_unchanged,
              "experiment_sources_imported": False, "environment_installed": False, "framework_or_gpu_used": False,
              "arrays_or_weights_read": False, "training_inference_or_FEM_executed": False,
              "limits": ["Restoration only; not a clean environment installation or complete training reproduction", "External data, packages, solver and historical checkpoints are absent", "JSON tokens require explicit rebinding and new-run hash fixation"]}
    report_path = package / "bootstrap-acceptance.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.package.resolve()), indent=2, ensure_ascii=False))
