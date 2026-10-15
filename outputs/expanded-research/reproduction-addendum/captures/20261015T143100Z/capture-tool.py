"""Capture only small research sources/configuration; never import experiment code."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/expanded-research/reproduction"
AREA = ROOT / "outputs/expanded-research/reproduction-addendum"
LIMIT = 2 * 1024 * 1024
SOURCE_DIRS = ("expanded-batch", "expanded-drop", "expanded-plate", "expanded-temporal", "expanded-evaluation-audit", "expanded-elasticity-ddm", "expanded-clean-runtime")
ML_SOURCES = ("benchmark_data.py", "model_adapters.py", "budget_protocol.py", "budgeted_runtime.py", "prepare_budgeted_plan.py", "train_budgeted.py", "evaluate_budgeted.py", "run_budgeted_suite.py", "summarize_budgeted.py", "estimate_b1_training_cost.py", "prepare_phone.py", "prepare_multifield.py", "finalize_cache.py", "field_diagnostics.py", "clarify_phone_provenance.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def small(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Link source refused: " + str(path.relative_to(ROOT)))
    if path.stat().st_size > LIMIT:
        raise ValueError("Oversize source/config refused: " + str(path.relative_to(ROOT)))
    data = path.read_bytes()
    if len(data) > LIMIT:
        raise ValueError("Source grew beyond limit")
    return data


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def redact(value):
    if isinstance(value, str):
        # Preserve path meaning while removing the workstation account and root.
        for workspace in (str(ROOT).replace("\\", "\\\\"), str(ROOT)):
            value = re.sub(re.escape(workspace), "${WORKSPACE}", value, flags=re.I)
        value = re.sub(re.escape(ROOT.as_posix()), "${WORKSPACE}", value, flags=re.I)
        value = re.sub(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+", "${USER_HOME}", value, flags=re.I)
        return value
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {redact(key): redact(item) for key, item in value.items()}
    return value


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def redaction_selfcheck():
    for separator in ("/", "\\", "\\\\", "\\\\\\\\"):
        value = "C:" + separator + "Users" + separator + "sample-account" + separator + "AppData"
        assert "sample-account" not in redact(value) and "${USER_HOME}" in redact(value)
    for value in (str(ROOT), str(ROOT).replace("\\", "\\\\"), ROOT.as_posix()):
        assert redact(value + "/example") == "${WORKSPACE}/example"
    return {"status": "passed", "synthetic_account_separator_variants": 4, "workspace_path_variants": 3, "experiment_or_capture_executed": False}


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def capture(output, as_of, dry_run=False):
    redaction_selfcheck()
    if not output.resolve().is_relative_to(AREA.resolve()) or output.exists():
        raise ValueError("Choose a new, absent directory within reproduction-addendum")
    datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    files, defaults, profiles, observation = [], [], {}, []
    pending = []

    def add(source, destination, profile=None, role="current_source", expected=None, redact_text=False):
        raw = small(source)
        if expected and sha(raw) != expected:
            raise ValueError("Frozen source hash mismatch: " + relative(source))
        if source.suffix == ".py":
            ast.parse(raw, filename=relative(source))
        original_object = json.loads(raw) if source.suffix == ".json" else None
        public_object = redact(original_object) if source.suffix == ".json" else None
        data = encoded(public_object) if source.suffix == ".json" else raw
        if source.suffix == ".jsonl":
            original_object = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            public_object = redact(original_object)
            data = ("\n".join(json.dumps(line, ensure_ascii=False) for line in public_object) + "\n").encode("utf-8")
        if redact_text:
            data = redact(raw.decode("utf-8")).encode("utf-8")
        if re.search(rb"[A-Za-z]:[\\/]+Users[\\/]+(?!\$\{)[^\\/\s\"']+", data, flags=re.I):
            raise ValueError("Unredacted local account path: " + relative(source))
        if source.suffix in (".py", ".ps1", ".yaml", ".yml") and data != raw:
            raise ValueError("Executed source bytes must be immutable")
        packaged = ("recorded/" + profile + "/" if profile else "") + destination
        if any(record["packaged_path"] == packaged for record in files):
            raise ValueError("Duplicate packaged path: " + packaged)
        record = {"packaged_path": packaged, "destination": destination,
                  "bytes": len(data), "sha256": sha(data), "source_origin": relative(source),
                  "source_bytes": len(raw), "source_sha256": sha(raw), "role": role,
                  "captured_as_of_utc": as_of, "json_token_redacted": public_object != original_object,
                  "json_reencoded": source.suffix in (".json", ".jsonl") and data != raw, "text_token_redacted": redact_text and data != raw}
        files.append(record)
        pending.append((packaged, data, source, sha(raw)))
        if profile is None:
            defaults.append(packaged)
        return packaged

    def evidence(source, group, profile=None, expected=None):
        # The original restorer permits source/config/text suffixes only. Keep
        # log/CSV/JSONL content as named text evidence, with origin in manifest.
        name = source.name + ".txt" if source.suffix in (".log", ".csv", ".jsonl") else source.name
        return add(source, "work/expanded-coordination/addendum-evidence/" + group + "/" + name,
                   profile, "configuration_or_provenance_text", expected, source.suffix in (".md", ".log", ".csv"))

    for name in ML_SOURCES:
        add(ROOT / "work/expanded-ml" / name, "work/expanded-ml/" + name)
    for directory in SOURCE_DIRS:
        for source in sorted((ROOT / "work" / directory).iterdir()):
            if source.is_file() and source.suffix in (".py", ".ps1"):
                add(source, relative(source))
    for source in sorted((ROOT / "work/expanded-temporal/sources").rglob("*")):
        if source.is_file() and source.suffix in (".py", ".yaml", ".yml"):
            add(source, relative(source), role="pinned_official_temporal_dependency")
    for source in sorted((ROOT / "work/expanded-clean-runtime/executed-sources").glob("*.py")):
        add(source, relative(source), role="exact_executed_clean_runtime_acquisition_source_snapshot")
    ddm_directory = ROOT / "work/expanded-elasticity-ddm"
    ddm_adapter = json.loads(small(ddm_directory / "adapter-manifest.json"))
    for source in sorted(ddm_directory.glob("*.json")):
        add(source, relative(source), role="elasticity_DDM_configuration_or_provenance_json")
    for row in ddm_adapter["files"]:
        source = ddm_directory / "source/fpddm" / Path(row["path"]).name
        key = add(source, relative(source), role="pinned_official_elasticity_DDM_source", expected=row["executed_sha256"])
        next(record for record in files if record["packaged_path"] == key).update({"official_source_pin": ddm_adapter["official_pin"], "official_source_path": row["path"], "official_source_sha256": row["official_sha256"], "changes_from_official": row["changes"]})
    for name in ("phone-full-normalization.json", "bumper7-normalization.json", "budgeted-suite-plan.json"):
        add(ROOT / "work/expanded-ml" / name, "work/expanded-ml/" + name, role="current_configuration_json")

    config_paths = (
        "outputs/expanded-research/full-support-static-training-protocol.json",
        "outputs/expanded-research/benchmarks/deforming_plate/official32-validation-protocol.json",
        "outputs/expanded-research/benchmarks/deforming_plate/official32-completion-source-lock-v2.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/budget-v2-execution-plan.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-head-feasibility.md",
        "outputs/expanded-research/benchmarks/temporal_strategies/auxiliary-static-review.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-v3-protocol.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/temporal-multifield-current-plan-v3.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-v3-source-review.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/phone_full/budget_v2_complete_epoch/prepared-contract.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/bumper/budget_v2_complete_epoch/prepared-contract.json",
        "outputs/expanded-research/benchmarks/batching/batch-recommendations.json",
        "outputs/expanded-research/benchmarks/deforming_plate/final-split-contract.json",
        "outputs/expanded-research/benchmarks/drop_test/500us/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume02/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume03/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume03/source-control-and-restart-audit.json",
    )
    temporal_metadata = ROOT / "outputs/expanded-research/benchmarks/temporal_strategies"
    config_paths = (*config_paths, *(relative(path) for pattern in ("native-scalar-v3-*.json", "temporal-multifield-current-plan*.json") for path in temporal_metadata.glob(pattern)))
    for value in sorted(set(config_paths)):
        source = ROOT / value
        if source.exists():
            evidence(source, "current-" + source.parent.relative_to(ROOT).as_posix().replace("/", "__"))
    final_directory = ROOT / "outputs/expanded-research/benchmarks/drop_test/500us-final"
    if final_directory.exists():
        for source in sorted(final_directory.glob("*.json")):
            evidence(source, "current-phone-FEM-500us-final")
    ddm_results = ROOT / "outputs/expanded-research/benchmarks/elasticity_numerical/full-ddm"
    if ddm_results.exists():
        for source in sorted(ddm_results.rglob("*")):
            if not source.is_file() or not (source.suffix in (".json", ".md", ".log") or source.name in ("elasticity_metrics.csv", "stage-events.jsonl")):
                continue
            group = "current-elasticity-DDM-full-ddm" + ("__" + source.parent.relative_to(ddm_results).as_posix().replace("/", "__") if source.parent != ddm_results else "")
            evidence(source, group)
    runtime_results = ROOT / "outputs/expanded-research/reproduction-clean-runtime"
    if runtime_results.exists():
        for source in sorted(runtime_results.iterdir()):
            if source.is_file() and source.suffix in (".json", ".txt", ".md"):
                evidence(source, "current-clean-runtime-plan-and-acquisition")
        for name in ("evidence-index.json", "source-freeze.json"):
            source = runtime_results / "CPU-runtime-evidence-r3" / name
            if source.exists():
                evidence(source, "current-clean-runtime-CPU-evidence-index")
    for source in sorted((ROOT / "work/expanded-clean-runtime").glob("*.log")):
        evidence(source, "current-clean-runtime-stage-logs")

    # Preserve later closure evidence as sources/settings, without inventing a
    # completed replay profile or reading raw FEM/prediction/weight arrays.
    plate_results = ROOT / "outputs/expanded-research/benchmarks/deforming_plate"
    closure_groups = [plate_results / "official32_valpilot",
                      plate_results / "official32_fresh_final_corrected",
                      temporal_metadata / "resource-calibration-v1"]
    for folder in closure_groups:
        for source in sorted(folder.rglob("*")) if folder.exists() else ():
            if not source.is_file() or source.suffix not in (".json", ".jsonl", ".py", ".md", ".txt", ".log"):
                continue
            group = "later-closure-" + source.parent.relative_to(ROOT / "outputs/expanded-research").as_posix().replace("/", "__")
            evidence(source, group)
    for source in sorted((plate_results / "review").glob("official32-*")):
        if source.is_file() and source.suffix in (".json", ".py", ".md"):
            if source.name.endswith(".plot.py"):
                observation.append({"scope": "historical_exact_plot_source_local_only",
                                    "source_origin": relative(source), "source_sha256": sha(small(source)),
                                    "reason": "Exact source includes workstation-specific dependency paths. Keep unchanged locally; portable derivative is separately labeled unexecuted.",
                                    "completed_profile_created": False})
                continue
            evidence(source, "later-closure-official32-independent-review")

    for domain, folder in (("phone", "drop_test/models-full-c6"), ("bumper", "bumper/models-c7")):
        parent = ROOT / "outputs/expanded-research/benchmarks" / folder
        for run in sorted(parent.iterdir()) if parent.exists() else ():
            if not run.is_dir():
                continue
            train_path, test_path = run / "training_report.json", run / "test_report.json"
            if not (train_path.exists() and test_path.exists()):
                observation.append({"scope": domain + "/" + run.name, "status": "not_completed_as_of_capture", "path": relative(run)})
                continue
            train, test = json.loads(small(train_path)), json.loads(small(test_path))
            if test.get("status") != "completed" or test.get("test_performance_passes") != 1:
                observation.append({"scope": domain + "/" + run.name, "status": "test_not_completed_once", "path": relative(run)})
                continue
            profile = "fullsupport-" + domain + "-" + run.name
            overrides, source_checks = [], []
            if train["source_sha256"] != test["source_sha256"]:
                raise ValueError("Training/final source maps differ: " + relative(run))
            for name, expected in sorted(test["source_sha256"].items()):
                source = run / "source" / name
                destination = "work/" + ("expanded-batch/" if name == "batch_adapters.py" else "expanded-ml/") + name
                overrides.append(add(source, destination, profile, "exact_executed_source_snapshot", expected))
                source_checks.append({"name": name, "sha256": expected})
            for name in ("normalization.json", "prospective-training-protocol.json", "checkpoint-frozen.json", "train-command.json", "validation-command.json", "test-command.json", "training_report.json", "validation_fresh_process.json", "test_report.json", "data-provenance-clarification.json"):
                source = run / name
                if source.exists():
                    expected = test.get("frozen_evidence_sha256", {}).get(str(source))
                    overrides.append(evidence(source, profile, profile, expected))
            command = json.loads(small(run / "train-command.json"))["command"]
            cache_path = Path(command[command.index("--cache") + 1])
            overrides.append(evidence(cache_path, profile + "-cache", profile, command[command.index("--expected-manifest-sha256") + 1]))
            profiles[profile] = {"description": "Completed single final-test run; exact captured executed sources and token-redacted settings: " + domain + "/" + run.name,
                                 "status": "completed_actual_training_and_single_final_test", "result_origin": relative(run),
                                 "overrides": overrides, "executed_source_sha256": source_checks,
                                 "training_report_source_sha256": sha(small(train_path)), "test_report_source_sha256": sha(small(test_path)),
                                 "quality_gate_passed": test.get("quality_gates", {}).get("gate_passed"),
                                 "accuracy_or_convergence_claimed": False, "arrays_and_checkpoint_included": False}

    # These are source preservation groups, never completed experiment profiles.
    for scope, path in (("plate_completion", "outputs/expanded-research/benchmarks/deforming_plate/official32-completion-source-lock-v2.json"),
                        ("temporal_v2", "outputs/expanded-research/benchmarks/temporal_strategies/budget-v2-execution-plan.json")):
        source = ROOT / path
        value = json.loads(small(source))
        observation.append({"scope": scope, "status": value.get("status"), "evidence_origin": path,
                            "evidence_source_sha256": sha(small(source)), "completed_profile_created": False})
    v3_protocol = ROOT / "outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-v3-protocol.json"
    observation.append({"scope": "temporal_v3", "status": "protocol_present_execution_status_review_separately" if v3_protocol.exists() else "protocol_not_frozen_as_of_capture",
                        "expected_sources": ["work/expanded-temporal/" + name for name in ("native_scalar_rollout_v3.py", "run_budget_temporal_v3.py", "run_multifield_stage.py")],
                        "completed_profile_created": False})
    plate_closure = plate_results / "official32_fresh_final_corrected/closure-provenance.json"
    if plate_closure.exists():
        value = json.loads(small(plate_closure))
        observation.append({"scope": "plate_official32_fresh_closure",
                            "status": value["status"], "evidence_origin": relative(plate_closure),
                            "evidence_source_sha256": sha(small(plate_closure)),
                            "fresh_case_ids": value["fresh_ids"], "checkpoint_sha256": value["checkpoint_sha256"],
                            "metrics_sha256": value["metrics_sha256"],
                            "completed_profile_created": False,
                            "scope_limit": "Completed execution closure; accuracy/geometry failed. This source capture does not rerun training or inference."})
    calibration_summary = temporal_metadata / "resource-calibration-v1/SUMMARY.json"
    if calibration_summary.exists():
        value = json.loads(small(calibration_summary))
        observation.append({"scope": "native_scalar_v3_cold_resource_calibration",
                            "status": value["status"], "evidence_origin": relative(calibration_summary),
                            "evidence_source_sha256": sha(small(calibration_summary)),
                            "successful_functional_probes": value["successful_functional_probes"],
                            "successful_stream_profiles": value["successful_stream_profiles"],
                            "optimizer_updates": value["optimizer_updates"], "full_fits": value["full_fits"],
                            "completed_profile_created": False})
    final_summary = ROOT / "outputs/expanded-research/benchmarks/drop_test/500us-final/summary.json"
    fem_summary = json.loads(small(final_summary)) if final_summary.exists() else {}
    freeze_path = final_directory / "artifact-freeze.json"
    fem_freeze = json.loads(small(freeze_path)) if freeze_path.exists() else {}
    for name, frozen in fem_freeze.get("files", {}).items():
        if Path(name).suffix in (".py", ".ps1"):
            destination = "work/expanded-drop/" + name
            captured = next(record for record in files if record["destination"] == destination and record["packaged_path"] in defaults)
            if captured["source_sha256"] != frozen["sha256"]:
                raise ValueError("Current FEM helper differs from final frozen source: " + name)
            captured.update({"matches_fem_final_freeze_sha256": frozen["sha256"], "fem_final_snapshot_origin": relative(final_directory / name)})
    observation.append({"scope": "phone_FEM_0.5ms", "status": "final_summary_present_review_separately" if final_summary.exists() else "no_final_summary_as_of_capture",
                        "actual_FEM_completed_to_requested_time": fem_summary.get("completed_to_requested_time", False),
                        "actual_final_frame_s": fem_summary.get("last_actual_frame_s"), "frozen_at_utc": fem_freeze.get("freeze_utc"),
                        "final_checkpoint_reload_status": fem_freeze.get("final_checkpoint_reload_status", "not_observed"),
                        "final_checkpoint_same_time_bitwise_verified": fem_freeze.get("final_checkpoint_reload_same_time_all_native_fields_bitwise", False),
                        "completed_profile_created": False})
    ddm_receipt = ddm_results / "attempt-status.json"
    if not ddm_receipt.exists():
        ddm_receipt = ddm_results / "preflight-status.json"
    ddm_status = json.loads(small(ddm_receipt)) if ddm_receipt.exists() else {}
    observation.append({"scope": "elasticity_DDM_full_2x2", "status": ddm_status.get("status", "no_preflight_status_as_of_capture"),
                        "source_pin": ddm_adapter["official_pin"], "completed_profile_created": False,
                        "observed_receipt": relative(ddm_receipt) if ddm_receipt.exists() else None,
                        "actual_result_metadata_reviewed_for_completion": False})
    ddm_final = ddm_results / "attempt02"
    ddm_freeze_path = ddm_final / "record-freeze.json"
    if ddm_freeze_path.exists():
        frozen = json.loads(small(ddm_freeze_path))
        checks = json.loads(small(ddm_final / "independent-checks.json"))
        review = json.loads(small(ddm_final / "independent-review.json"))
        if frozen.get("raw_execution_context_correction_required") and not (ddm_final / "execution-context-correction.json").exists():
            raise ValueError("Completed DDM requires its preserved execution-context correction")
        for row in frozen["files"]:
            captured = next((record for record in files if record["source_origin"] == row["path"]), None)
            if captured and captured["source_sha256"] != row["sha256"]:
                raise ValueError("Captured DDM evidence differs from final freeze: " + row["path"])
        observation.append({"scope": "elasticity_DDM_full_2x2_attempt02", "status": frozen["status"],
                            "observed_receipt": relative(ddm_freeze_path), "actual_result_metadata_reviewed_for_completion": True,
                            "actual_completed_numerical_DDM": bool(frozen.get("completed_full_ddm") and checks.get("status") == "passed" and review.get("status") == "no_critical_issues_found"),
                            "independent_checks_passed": frozen.get("independent_checks_passed"), "independent_checks_total": frozen.get("independent_checks_total"),
                            "neural_models_trained": frozen.get("neural_models_trained"), "execution_context_correction_required": frozen.get("raw_execution_context_correction_required"),
                            "completed_profile_created": False, "raw_fields_array_included": False})
    runtime_audit = runtime_results / "environment-static-audit.json"
    runtime_status = json.loads(small(runtime_audit)) if runtime_audit.exists() else {}
    observation.append({"scope": "clean_runtime", "status": runtime_status.get("status", "plan_and_acquisition_only_no_environment_audit_as_of_capture"),
                        "completed_profile_created": False, "full_training_reproduction_claimed": False,
                        "framework_functional_or_validation_reload_passed": False})

    destinations = [next(record["destination"] for record in files if record["packaged_path"] == key) for key in defaults]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Conflicting default destinations")
    for profile in profiles.values():
        targets = [next(record["destination"] for record in files if record["packaged_path"] == key) for key in profile["overrides"]]
        if len(targets) != len(set(targets)):
            raise ValueError("Conflicting profile destinations")
    manifest = {"schema_version": 1, "created_utc": as_of, "capture_scope": "Small stdlib source/config preservation addendum; no experiment execution",
                "base_bundle_unchanged": {"manifest_path": "outputs/expanded-research/reproduction/source-manifest.json", "manifest_sha256": sha(small(BASE / "source-manifest.json"))},
                "files": sorted(files, key=lambda row: row["packaged_path"]), "default_files": sorted(defaults),
                "profiles": profiles, "observations_as_of_capture": observation,
                "redaction": {"json_strings_and_keys": "Local workspace root -> ${WORKSPACE}; remaining Windows user homes -> ${USER_HOME}",
                              "executed_sources": "Exact bytes, never redacted; refused if account paths present", "hashes": "sha256 is packaged/public bytes; source_sha256 is original bytes"}}
    result = {"status": "dry_run" if dry_run else "captured", "as_of_utc": as_of, "files": len(files), "profiles": sorted(profiles), "bytes": sum(row["bytes"] for row in files),
              "output": output.relative_to(ROOT).as_posix(), "experiment_code_executed": False, "write_performed": not dry_run}
    if dry_run:
        return result
    output.mkdir(parents=True)
    try:
        for name, data, source, expected in pending:
            if sha(small(source)) != expected:
                raise ValueError("Source changed during capture: " + relative(source))
            write_new(output / name, data)
        original_bootstrap = small(BASE / "bootstrap.py")
        marker = b"    return manifest,records\n"
        conflict_check = b'''    groups=[manifest["default_files"]]+[p["overrides"] for p in manifest["profiles"].values()]
    for keys in groups:
        destinations=[records[key]["destination"] for key in keys]
        if len(destinations)!=len(set(destinations)):
            raise ValueError("Conflicting destination within a manifest selection")
    return manifest,records
'''
        if original_bootstrap.count(marker) != 1:
            raise ValueError("Original restoration helper changed; review before reusing")
        write_new(output / "bootstrap.py", original_bootstrap.replace(marker, conflict_check))
        write_new(output / "source-manifest.json", encoded(manifest))
        write_new(output / "README.md", README.encode("utf-8"))
        write_new(output / "capture-tool.py", Path(__file__).read_bytes())
        write_new(output / "selfcheck-tool.py", small(ROOT / "work/expanded-coordination/addendum_selfcheck.py"))
        write_new(output / "licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt", small(BASE / "licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt"))
        index = []
        for path in sorted(output.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                index.append({"path": path.relative_to(output).as_posix(), "bytes": len(data), "sha256": sha(data)})
        write_new(output / "package-manifest.json", encoded({"schema_version": 1, "as_of_utc": as_of, "files": index,
                  "bootstrap_source_origin": "outputs/expanded-research/reproduction/bootstrap.py", "bootstrap_original_sha256": sha(original_bootstrap),
                  "bootstrap_change": "Added explicit duplicate destination rejection in each default/profile selection; original restoration safety logic retained",
                  "capture_tool_source_origin": relative(Path(__file__)), "manifest_self_hash_excluded": True,
                  "selfcheck_tool_source_origin": "work/expanded-coordination/addendum_selfcheck.py",
                  "license_source_origin": "outputs/expanded-research/reproduction/licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt",
                  "acceptance_report": "Generated separately by addendum_selfcheck.py after actual verification"}))
    except Exception:
        # Keep the partial capture for diagnosis; a later run must choose a new path.
        raise
    return result


README = """# 추가 실행 소스 복원

원본 `../reproduction/`(169 source files / 22 historical profiles)을 수정하지 않고 후속 full-support 정적 학습, 배치 검사, Phone 0.5ms FEM 제어·분석, Plate completion, Temporal V2/V3, 수치 탄성 DDM, 독립 수치 감사의 작은 실행 소스·설정을 보존했다. `source-manifest.json`의 `created_utc`가 관측 시점이다. `profiles`에 있는 완료 실행만 실제 학습과 최종 시험 완료로 분류한다. Plate·Temporal·FEM·DDM은 `observations_as_of_capture`를 읽는다. 완료 profile도 정확도·수렴 성공을 뜻하지 않는다.

## 별도 빈 workspace에서 복원

Python 표준 라이브러리만 필요하다. 이 폴더의 `bootstrap.py`는 원본 번들의 안전 복원 코드를 재사용한 독립 실행본이며, 같은 선택 내 복원 목적지가 충돌하면 명시적으로 거부하는 검사만 추가했다. 원본/public bootstrap hash와 변경 내용은 `package-manifest.json`에 기록했다.

이 README와 같은 capture 폴더를 현재 작업 폴더로 연 뒤 실행한다. 과거 capture의 bootstrap을 대신 실행하지 않는다.

```powershell
python bootstrap.py --verify-only
python bootstrap.py --list-profiles
python bootstrap.py --workspace <ABSOLUTE_EMPTY_DIRECTORY> --profile fullsupport-phone-transolver --dry-run
python bootstrap.py --workspace <ABSOLUTE_EMPTY_DIRECTORY> --profile fullsupport-phone-transolver
```

하나의 빈 폴더에 하나의 profile을 먼저 선택한다. `current`는 관측 시점의 소스 모음이며 완료 실험의 동결 실행본을 대신하지 않는다. 원본 번들과 같은 대상에 연속 복원하면 중복 파일 때문에 거부될 수 있으므로 각각 별도의 빈 폴더를 쓴다. 두 묶음을 합치는 자동 덮어쓰기 기능은 없다. 대상 파일은 내용이 같아도 덮어쓰지 않으며 경로 이탈·symbolic link·junction·환경/데이터 경로를 차단한다.

실행된 `.py`와 `.ps1`은 원본 바이트 그대로다. 모든 복사 경로, 원 위치, 대상 위치, 원본 SHA256, 공개 SHA256, profile 관계는 manifest에 기록한다. JSON 설정의 계정 경로는 `${WORKSPACE}`와 `${USER_HOME}` token으로 가렸으므로 설정은 기록용 template이다. 새 장비에서 해당 token과 로컬 환경/데이터 경로를 명시적으로 바인딩하고, 영향을 받는 설정 hash를 새 실행용으로 재고정해야 한다. 과거의 hash를 변경된 설정에 붙이면 안 된다. 동결 완료 profile은 당시 `source/` snapshot의 hash를 최종 `test_report.json`의 source hash와 맞춘다. `checkpoint-frozen.json`의 model-config/normalization/contract hash는 canonical JSON 객체 hash이며 파일 byte hash와 다를 수 있다. 공식 NVIDIA wrapper의 저작권 header와 `licenses/` Apache-2.0 고지는 유지했다. 코드 라이선스를 FEM 모델·데이터에 적용하지 않는다.

## 보존 범위와 제한

데이터·raw FEM deck·예측 배열·학습 checkpoint·solver binary·가상환경·패키지·비밀값은 포함하지 않는다. `work/`의 원래 상대 구조로 복원하지만 framework 설치·upstream 코드·데이터·solver·재시작 checkpoint는 각 연구 안내에 따라 별도로 확보해야 한다. 이 패키지는 신규 환경 설치부터 학습까지의 독립 재실행을 입증하지 않는다. GPU·framework import·데이터 로드·FEM·학습·추론은 packer/복원 검사에서 실행하지 않는다. FEM의 정상 0.5ms 적분 완료와 마지막 시점 checkpoint 재로딩 검증은 별도 상태다. 최종 metadata가 `NOT_VERIFIED_OPTIONAL_RELOAD`이면 그 한계를 유지한다. 수치 탄성 DDM 공식 source 7개는 adapter manifest의 원본/실행 hash로 확인하며 Schwarz의 사용하지 않는 import 2개 제거를 기록한다. DDM 결과 metadata를 직접 검토하기 전에는 완료 profile을 만들지 않는다.

`bootstrap-acceptance.json`은 source bytes/hash, 모든 Python AST, 빈 임시 폴더 복원, overwrite 거부와 경로 검사를 실제 수행한 결과다. 검증 시점 이후 새 결과는 이 capture에 자동으로 추가되지 않는다.

## 새 결과를 보존할 때

원 workspace에서 `work/expanded-coordination/addendum_pack.py --as-of <UTC_ISO_TIME> --output outputs/expanded-research/reproduction-addendum/captures/<NEW_NAME> --dry-run`으로 쓰기 없이 수집 목록·크기·새 경로를 확인한 뒤 `--dry-run`을 빼고 **없는 새 경로**에 생성한다. 같은 입력·같은 `--as-of`는 같은 manifest/바이트를 만든다. 기존 capture와 실험 source는 덮어쓰지 않는다. 새 Temporal V3 top-level Python source는 자동 수집하며 V3 protocol/현재계획/source-review도 존재하면 포함한다. FEM은 중간 live 결과 대신 `500us-final/`의 작은 JSON을 수집한다. 완료된 full-support `test_report.json`만 새 profile로 검출하고, snapshot hash가 맞지 않으면 거부한다. 생성 후 `addendum_selfcheck.py --package <NEW_CAPTURE>`를 실행한다. 공개 저장소에서는 보존된 `capture-tool.py`가 원 capture 코드를 설명하며 원 workspace의 입력 없이는 refresh할 수 없다.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Explicit ISO UTC capture time; held fixed for deterministic refresh")
    parser.add_argument("--output", type=Path, default=AREA)
    parser.add_argument("--dry-run", action="store_true", help="Validate and inventory sources without creating a capture")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(capture(output, args.as_of, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
