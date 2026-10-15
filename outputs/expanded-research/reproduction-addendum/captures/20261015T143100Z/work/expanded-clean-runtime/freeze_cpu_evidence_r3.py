"""Freeze small CPU-runtime evidence and resource-deferred functional disposition."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

assert sys.flags.no_site
root = Path(__file__).resolve().parents[2]
own = root / "work/expanded-clean-runtime"
out = root / "outputs/expanded-research/reproduction-clean-runtime"
bundle = out / "CPU-runtime-evidence-r3"
limit = 2 * 1024**2
now = datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def account_path(data):
    return re.search(rb"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", data, re.I)


def redact(value):
    if isinstance(value, str):
        for original in (str(root).replace("\\", "\\\\"), str(root), root.as_posix()):
            value = re.sub(re.escape(original), "${WORKSPACE}", value, flags=re.I)
        return re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "${USER_HOME}", value, flags=re.I)
    if isinstance(value, dict):
        return {redact(key): redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


closure = json.loads((out / "CPU-clean-runtime-closure-r3.json").read_text())
observations = closure["post_CPU_no_child_memory_observations"]
assert [row["available_bytes"] for row in observations] == [1976545280, 1553977344]
assert closure["status"] == "passed_independent_installation_and_static_audit_only"
assert closure["actual_functional_checks_completed"] == 0 and not closure["future_functional_1p5GiB_operating_admission_observations_pass"]
assert not (own / "newfunctional-runs").exists()
disposition = {"status": "NOT_EXECUTED_RESOURCE_ADMISSION_NOT_MET", "frozen_utc": now,
               "planned_functional_checks": 6, "planned_Adam_updates_per_check": 3,
               "actual_functional_checks_completed": 0, "failed_ML_checks": 0,
               "additional_full_fits_or_completed_models": 0,
               "operating_admission_bytes": 1610612736, "observations_seconds_apart": 10,
               "exact_observations_from_CPU_closure": observations,
               "functional_runtime_source_sha256": closure["functional_runtime_source_sha256"],
               "functional_protocol_sha256": closure["functional_protocol_sha256"],
               "CPU_closure_sha256": sha((out / "CPU-clean-runtime-closure-r3.json").read_bytes()),
               "GPU_currently_allocated": False, "framework_or_GPU_execution_performed": False,
               "data_arrays_or_checkpoints_read": False, "new_memory_or_process_query_performed": False,
               "automatic_retry_planned": False, "model_intrinsic_minimum_RAM_claimed": False,
               "authority": "Root explicitly closed the prepared checks using the two existing observations; no new probes.",
               "next_step_requires_user_resource_path_choice_and_new_explicit_allocation": True}
new(out / "newfunctional-disposition-v1.json", encoded(disposition))
assert not bundle.exists()
bundle.mkdir()
records, sources = [], []


def copy(source, destination, role):
    for path in (source, *source.parents):
        assert not path.is_symlink() and not getattr(path, "is_junction", lambda: False)()
        if path == root:
            break
    assert source.stat().st_size <= limit, source.name
    raw = source.read_bytes()
    if source.suffix == ".py":
        public, kind = raw, "none_exact_source_bytes"
    elif source.suffix == ".json":
        public, kind = encoded(redact(json.loads(raw))), "structured_metadata_path_tokens"
    else:
        public, kind = redact(raw.decode("utf-8")).encode("utf-8"), "text_path_tokens"
    assert not account_path(public), source.name
    assert len(public) <= limit
    new(bundle / destination, public)
    assert sha(source.read_bytes()) == sha(raw), "Input changed while freezing"
    record = {"path": destination, "source_origin": source.relative_to(root).as_posix(),
              "source_bytes": len(raw), "source_sha256": sha(raw), "bytes": len(public),
              "sha256": sha(public), "role": role, "redaction": kind, "public_bytes_changed": public != raw}
    records.append(record)
    if source.suffix == ".py":
        sources.append(record)


for source in sorted(out.iterdir()):
    if source.is_file() and source.suffix in (".json", ".txt", ".md"):
        copy(source, source.name, "small_public_metadata_lock_or_documentation")
for source in sorted([*own.glob("*.py"), *(own / "executed-sources").glob("*.py")]):
    copy(source, "source/" + source.relative_to(root).as_posix(),
         "exact_executed_source_snapshot" if source.parent.name == "executed-sources" else "current_prepared_or_preserved_source")
for source in sorted(own.glob("*.log")):
    copy(source, "logs/" + source.name, "public_redacted_log_original_retained")
resolver = own / "local-resolver-report.json"
if resolver.exists():
    copy(resolver, "metadata/local-resolver-report.json", "actual_offline_resolver_new_wheel_binding")
source_freeze = {"status": "exact_source_bytes_frozen", "frozen_utc": now, "files": sources,
                 "functional_disposition": disposition["status"], "new_ANTLR_build_attempts": 1,
                 "binary_wheels_environments_arrays_or_modelweights_included": False}
generated = {"source-freeze.json": encoded(source_freeze),
             "bundle-closure.json": encoded({"status": "public_ready_small_evidence_frozen", "frozen_utc": now,
                                              "CPU_install_static_audit": "PASS", "functional_disposition": disposition["status"],
                                              "functional_actual_completed": 0, "functional_failed_ML_checks": 0,
                                              "source_copy_count": len(sources), "copied_payload_files": len(records),
                                              "framework_GPU_array_or_checkpoint_execution_in_this_freeze": False,
                                              "new_hardware_or_process_queries": False,
                                              "prior_captures_sources_logs_and_one_build_proof_preserved": True})}
for name, data in generated.items():
    new(bundle / name, data)
    records.append({"path": name, "bytes": len(data), "sha256": sha(data), "role": "generated_small_freeze_metadata"})
index = {"status": "public_ready_small_evidence_index", "frozen_utc": now, "files": records,
         "index_self_hash_excluded": True, "all_input_files_at_most_bytes": limit,
         "exact_source_bytes_immutable": True, "metadata_and_logs_may_have_token_redaction": True,
         "original_log_files_modified": False, "wheel_environment_array_checkpoint_or_solver_binary_included": False}
index_bytes = encoded(index)
new(bundle / "evidence-index.json", index_bytes)
for row in records:
    data = (bundle / row["path"]).read_bytes()
    assert len(data) == row["bytes"] and sha(data) == row["sha256"] and not account_path(data)
assert not any(name in sys.modules for name in ("numpy", "torch", "physicsnemo"))
print(json.dumps({"status": "public_ready_small_evidence_frozen", "bundle": bundle.relative_to(root).as_posix(),
                  "files_including_index": len(records) + 1, "bytes": sum(row["bytes"] for row in records) + len(index_bytes),
                  "source_copies": len(sources), "index_sha256": sha(index_bytes),
                  "source_freeze_sha256": sha(generated["source-freeze.json"]),
                  "functional_disposition": disposition["status"], "actual_functional_checks": 0,
                  "new_hardware_process_or_ML_queries": False}))
