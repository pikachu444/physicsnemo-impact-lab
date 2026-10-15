"""Stdlib-only preparation check. No NumPy/Torch/CUDA imports or array reads."""
import ast
import json
from pathlib import Path
import sys
import tempfile

from run_batch_probe import DOMAINS, MAIN, parser
from train_probe_data import contained, file_hash, read_train_contract


def main():
    here = Path(__file__).resolve().parent
    checked = []
    for path in sorted(here.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in {"torch", "numpy", "psutil", "physicsnemo"}
                           for alias in node.names), f"Eager ML import: {path}"
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in {"torch", "numpy", "psutil", "physicsnemo"}
        checked.append(path.name)
    rows = []
    for domain, (folder, norm, manifest_sha, norm_sha) in DOMAINS.items():
        for batch_size in (1, 2, 5):
            contract = read_train_contract(MAIN/folder/"manifest.json", MAIN/norm, batch_size, manifest_sha, norm_sha)
            assert len(contract["cases"]) == batch_size
            train_targets = {contract["meta"]["runs"][case]["target"] for case in contract["cases"]}
            assert not train_targets.intersection(contract["forbidden_label_names"])
            assert set(contract["selected_names"]).issuperset(train_targets)
            rows.append({"domain": domain, "batch_size": batch_size, "train_cases": contract["cases"],
                         "selected_file_count": len(contract["selected_names"]), "array_files_opened": 0})
    # Aliased held-out labels and path traversal must be rejected before array loading.
    with tempfile.TemporaryDirectory(dir=here) as scratch:
        root = Path(scratch)
        try:
            contained(root, "../escape.npy")
        except ValueError:
            pass
        else:
            raise AssertionError("Path traversal was accepted")
        manifest, norm = root/"manifest.json", root/"normalization.json"
        payload = {"coordinates": "xyz.npy", "edge_index": "edges.npy", "split": {"train": ["train"], "test": ["test"]},
                   "runs": {"train": {"thickness": "scalar.npy", "target": "aliased.npy"},
                            "test": {"thickness": "scalar.npy", "target": "aliased.npy"}}}
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        norm.write_text("{}", encoding="utf-8")
        (root/"READY.json").write_text(json.dumps({"status": "ready", "writer_exit_code": 0,
                                                 "manifest_sha256": file_hash(manifest), "expected_cases": 2}), encoding="utf-8")
        try:
            read_train_contract(manifest, norm, 1, file_hash(manifest), file_hash(norm))
        except AssertionError as error:
            assert "aliases a held-out label" in str(error)
        else:
            raise AssertionError("Held-out label alias was accepted")
    args = parser().parse_args(["--domain", "phone", "--architecture", "geoflare", "--batch-size", "5", "--out", "unused"])
    assert not args.gpu_slot_authorized
    assert not any(name in sys.modules for name in ("torch", "numpy", "tensorflow", "physicsnemo"))
    print(json.dumps({"status": "static_checks_passed", "python_files": checked,
                      "training_metadata_checks": rows, "ML_modules_imported": False,
                      "array_files_opened": 0, "GPU_execution": False}, indent=2))


if __name__ == "__main__":
    main()
