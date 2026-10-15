"""Freeze small metadata for six future clean-runtime checks; never read arrays."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

MODELS = {"transolver": 43150, "flare": 38758, "geotransolver": 45360,
          "geoflare": 40968, "meshgraphnet": 81958, "figconvunet": 587494}
HELPERS = {"model_adapters.py": "fcfbfc72c4eb36793849410ba54e7e41c27a5261d7f0f3830800a774edd66e7d",
           "benchmark_data.py": "4888594826c9a517bb8fc876d2fa533daf7eb78d9f247c10af287821f70956c6",
           "run_benchmark.py": "6f11b318cc872b8e15401833db294f7bdfddc7a69ae78e0e8890dd20ece72eab"}
NORMALIZATION_SHA = "fc9574c541d0b3571abe995b5463c7ecdda3b1a007a24e4b36673383aa9b18b7"
MANIFEST_SHA = "08876879b2b003f8222b831bb51a15c63f8d69d30e6b83e077385f80850169b1"
GUARD_SHA = "45d6bfb697b95f8df6721d2d42146c53095dc81e062e885f9fa91d17b5fdda1a"
CORE_SHA = "fca03b395b84cdf23c875361c8a4dc3193cb6964ce009e809dbb7f7f480a5974"
COMMON = ("coordinates.npy", "edge_index.npy", "node_ids.npy", "thickness.npy")
TARGET = "runs/pilot_base.npy"


def relative(value):
    parts = PurePosixPath(value)
    if not value or "\\" in value or ":" in value or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Unsafe relative path")
    return parts


def small(root, value):
    path = root.joinpath(*relative(value).parts)
    if path.suffix not in (".json", ".py"):
        raise ValueError("Preparation reads only small source and JSON metadata")
    for parent in (path, *path.parents):
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Linked source refused")
        if parent == root:
            break
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("Metadata/source exceeds bound")
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def document(root, value):
    raw, digest = small(root, value)
    return json.loads(raw), digest


def source_record(root, origin, destination, role, expected=None):
    raw, digest = small(root, origin)
    if expected and digest != expected:
        raise ValueError("Frozen source changed: " + origin)
    return {"origin": origin, "scratch": destination, "sha256": digest,
            "bytes": len(raw), "role": role}


def prepare(root, output):
    if output.exists() or not output.parent.resolve().is_relative_to((root / "outputs/expanded-research/reproduction-clean-runtime").resolve()):
        raise ValueError("Choose a new absent clean-runtime prepared protocol path")
    cache = "work/expanded-structural-coupon/cache-initial13-nodal7"
    meta, manifest_sha = document(root, cache + "/manifest.json")
    ready, ready_sha = document(root, cache + "/READY.json")
    assert manifest_sha == MANIFEST_SHA == ready["manifest_sha256"]
    assert ready["status"] == "ready" and ready["writer_exit_code"] == 0
    assert ready["shape"] == [919, 10, 7]
    assert meta["split"]["train"][0] == "pilot_base" and len(meta["split"]["train"]) == 5
    assert meta["split"]["validation"][0] == "holdout_validation_01"
    assert len(meta["split"]["test"]) == 4
    assert meta["time_values"] == list(range(1, 11))
    assert meta["field_units"] == ["mm", "mm", "mm", "MPa", "1", "1", "1"]
    assert meta["runs"]["pilot_base"]["target"] == TARGET
    assert all(meta["runs"][case]["thickness"] == "thickness.npy" for case in ("pilot_base", "holdout_validation_01"))
    assert not any("coordinates" in meta["runs"][case] for case in ("pilot_base", "holdout_validation_01"))
    stats_path = "work/expanded-ml/coupon13-normalization.json"
    stats, stats_sha = document(root, stats_path)
    assert stats_sha == NORMALIZATION_SHA
    for key, size in (("field_mean", 7), ("field_std", 7), ("coordinate_min", 3), ("coordinate_span", 3), ("global_mean", 2), ("global_std", 2)):
        assert len(stats[key]) == size and all(math.isfinite(x) for x in stats[key])
    assert all(x > 0 for key in ("field_std", "coordinate_span", "global_std") for x in stats[key])
    core_origin = "outputs/expanded-research/reproduction-clean-runtime/core-source-contract.json"
    core, core_sha = document(root, core_origin)
    assert core_sha == CORE_SHA and len(core["files"]) == 40
    assert core["version"] == "2.2.2" and core["cached_wheel_sha256"] == "c447771384d92f5c31547e293f65b70e0947f8829981f34c7e8300daf1b8d2eb"
    core_hashes = {row["path"]: row["sha256"] for row in core["files"]}
    models = {}
    prefix = "outputs/expanded-research/benchmarks/structural_coupon/models/"
    for name, parameters in MODELS.items():
        origin = prefix + name + "/metrics.json"
        report, digest = document(root, origin)
        assert report["architecture"] == name and report["parameters"] == parameters
        assert report["source_sha256"] == HELPERS
        assert report["seed"] == 20261015 and report["contract"]["shape"] == [919, 10, 7]
        for filename, expected in HELPERS.items():
            assert small(root, prefix + name + "/source/" + filename)[1] == expected
        assert small(root, prefix + name + "/normalization.json")[1] == stats_sha
        model_source = report["model_source"]
        implementation = model_source["implementation_module"].replace(".", "/") + ".py"
        assert core_hashes[implementation] == model_source["source_sha256"]
        models[name] = {"config": report["model_config"], "expected_parameters": parameters,
                        "model_source": {key: model_source[key] for key in ("module", "class_name", "implementation_module", "source_sha256")},
                        "metrics_origin": origin, "metrics_sha256": digest}
    references = []
    for filename, digest in HELPERS.items():
        destination = ("source/" if filename == "model_adapters.py" else "reference/") + filename
        references.append(source_record(root, prefix + "transolver/source/" + filename, destination,
                                        "constructor_and_forward_only" if filename == "model_adapters.py" else "historical_reference_never_imported", digest))
    references += [source_record(root, stats_path, "reference/normalization.json", "frozen_five_train_statistics", stats_sha),
                   source_record(root, core_origin, "reference/core-source-contract.json", "clean_installed_core_contract", core_sha),
                   source_record(root, "work/expanded-clean-runtime/newfunctional_runtime.py", "driver/newfunctional_runtime.py", "prepared_future_runtime"),
                   source_record(root, "work/expanded-clean-runtime/run_guarded_r2.py", "driver/run_guarded_r2.py", "stdlib_job_supervisor_only", GUARD_SHA)]
    inventory, inventory_sha = document(root, "outputs/expanded-research/reproduction-clean-runtime/installed-metadata-inventory.json")
    assert len(inventory["selected"]) == 90
    files = {row["name"]: row for row in ready["files"]}
    arrays = {name: {"bytes": files[name]["bytes"], "sha256": files[name]["sha256"],
                     "phases": ["train"] if name == TARGET else ["train", "reload"]}
              for name in (*COMMON, TARGET)}
    protocol = {"schema": "clean-functional6-v1", "status": "prepared_not_executed", "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "models": models, "source_files": references,
                "runtime_source": "driver/newfunctional_runtime.py", "adapter_source": "source/model_adapters.py",
                "normalization_source": "reference/normalization.json",
                "core_contract": {"origin": core_origin, "sha256": core_sha, "files": core["files"]},
                "environment": {"relative_venv": "work/expanded-clean-runtime/venv", "python": "3.13.5",
                                "versions": {name: row["version"] for name, row in inventory["selected"].items()},
                                "historical_metadata_inventory_sha256": inventory_sha},
                "data": {"cache_root": cache, "manifest_sha256": manifest_sha, "ready_sha256": ready_sha,
                         "metadata_only_provenance_no_runtime_inventory_scan": True,
                         "shape": [919, 10, 7], "fields": meta["field_names"], "units": meta["field_units"], "times": meta["time_values"],
                         "coordinate_units": meta["coordinate_units"], "thickness_units": meta["thickness_units"], "time_units": meta["time_units"],
                         "support": meta["support"], "location": meta["location"], "basis": meta["basis"],
                         "C7_scope": "Raw physical-node displacement3 plus four native-cell invariants area-projected to the same physical919nodes; not wholly raw native nodal C7.",
                         "train_case": "pilot_base", "validation_case": "holdout_validation_01",
                         "train_globals": meta["runs"]["pilot_base"]["globals"], "validation_globals": meta["runs"]["holdout_validation_01"]["globals"],
                         "arrays": arrays, "normalization_training_case_count": 5,
                         "statistics_recomputed": False, "validation_targets_allowed": False, "test_payloads_or_array_hash_reads_allowed": False},
                "statistics": stats, "algorithm": {"updates": 3, "lr": 0.001, "seed": 20261015, "tolerance": 1e-5,
                                                     "optimizer": "Adam", "loss": "unweighted mean normalized squared error", "batch_size": 1},
                "allocation_required": {"status": "authorized_clean_functional6", "protocol_sha256": "external exact byte hash",
                                        "explicit_gpu_slot": True, "clean_install_pip_check_and_core40_audit_pass": True,
                                        "resource_caps": "Future root allocation supplies bounded ML familyRSS/wall caps; installer350MiB is not silently reused."},
                "claims": {"actual_executed": False, "functional_checks_planned": 6, "updates_per_check": 3,
                           "additional_full_fits_or_completed_models": 0, "fullfit_claimed": False,
                           "accuracy_comparison": False, "fresh_validation_reload_targetless": True,
                           "prepared_only_until_explicit_root_GPU_allocation": True}}
    data = (json.dumps(protocol, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with output.open("xb") as stream:
        stream.write(data)
    return {"status": "prepared_not_executed", "protocol": output.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(), "models": len(models), "selected_array_metadata_records": len(arrays),
            "arrays_read": False, "framework_imported": False, "runtime_executed": False}


def selfcheck():
    for value in ("../bad", "C:/bad", "/bad", "a\\bad", "a//bad", "a/./bad"):
        try:
            relative(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe path accepted")
    assert relative(TARGET).parts == ("runs", "pilot_base.npy")
    assert len(COMMON) == 4 and "holdout" not in TARGET
    return {"status": "passed_stdlib_preparation_guard_selfcheck", "unsafe_paths_rejected": 6, "arrays_or_frameworks_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    assert sys.flags.no_site, "Use Python -S"
    if args.selfcheck:
        print(json.dumps(selfcheck()))
    else:
        assert args.output is not None, "Choose a new prepared protocol file"
        print(json.dumps(prepare(args.workspace.resolve(), args.output.resolve()), indent=2))
