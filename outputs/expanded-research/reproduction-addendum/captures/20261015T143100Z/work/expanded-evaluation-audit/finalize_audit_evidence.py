"""Stdlib-only final metadata checks; never reads or recalculates model arrays."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import source_preservation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = ROOT / "outputs/expanded-research"
PIN = "5f9f578a2fdef265a0cefa920daf4704c7253b7ef4a85bd428a01adac9f11352"
SOURCES = source_preservation.sources(__file__, HERE / "source_preservation.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


latest_path = OUT / "full-support-latest-audit.json"
latest = read(latest_path)
old = read(OUT / "full-support-evaluation-review.json")
mean = read(OUT / "full-support-mean-baselines.json")
assert latest["status"] == "complete" and len(latest["models"]) == 12
assert latest["auditor"]["script_sha256"] == PIN
assert len(old["models"]) == 12
numeric_equality = {}
for key, row in latest["models"].items():
    assert row["status"] == "passed" and row["gates"]["quality_budget"] == "satisfied"
    assert row["auditor_script_sha256"] == PIN
    a = row["independent_recomputation"]
    b = old["models"][key]["independent_recomputation"]
    fields = ("metrics", "epsp", "inverse_normalization")
    checks = {field: a[field] == b[field] for field in fields}
    for value in (a, b):
        reference = value.get("phone_ballistic_reference")
        if reference:
            reference.pop("production_metric_paths", None)
    checks["ballistic_values_equal_ignoring_added_production_path_description"] = (
        a.get("phone_ballistic_reference") == b.get("phone_ballistic_reference"))
    assert all(checks.values()), (key, checks)
    numeric_equality[key] = checks

# Include preserved partial attempts: a later successful row does not erase an earlier peak.
records = [(latest_path, read(latest_path))]
seen = {str(latest_path.resolve())}
for path, record in records:
    for snap in record.get("prior_record_snapshots", []):
        item = Path(snap["snapshot"])
        if item.suffix == ".json" and item.name == latest_path.name and str(item.resolve()) not in seen:
            assert file_record(item)["sha256"] == snap["sha256"]
            seen.add(str(item.resolve()))
            records.append((item, read(item)))
peaks, frees, deferred = [], [], []
for path, record in records:
    header = record.get("auditor", {})
    peaks.append(header["peak_observed_RSS_bytes"])
    frees.append(header["minimum_observed_host_free_bytes"])
    for key, row in record.get("models", {}).items():
        resource = row.get("auditor_resources", {})
        if resource:
            peaks.append(resource["peak_RSS_bytes"])
            frees.append(resource["minimum_host_available_bytes"])
        if row.get("memory_deferral"):
            deferred.append({"model": key, "record": str(path), "reason": row["memory_deferral"]})

inventory = []
for path in sorted((HERE / "source-snapshots").glob("*/*.py")):
    item = file_record(path)
    assert item["sha256"] == path.parent.name
    inventory.append(item)
assert any(x["sha256"] == PIN and Path(x["path"]).name == "audit_full_support.py" for x in inventory)
mapping_path = HERE / "source-snapshots/source-record-map.json"
mapping = read(mapping_path)
for name in ("generator", "helper"):
    expected = mapping["phone_mean_original_record_sources"][name + "_sha256"]
    assert file_record(mapping["phone_mean_original_record_sources"][name + "_snapshot"])["sha256"] == expected
assert all(x["status"] == "computed_and_self_checked" for x in mean["domains"].values())
assert all(x["training_mean_independent_chunk_check"]["bitwise_equal"] for x in mean["domains"].values())
now = datetime.now(timezone.utc).isoformat()
summary = {
    "completed_utc": now, "scope": "Stdlib-only metadata finalization of saved-array audit; no new numeric array work, NN training, prediction, test pass, or selection",
    "completion_count": 12, "numeric_audit_passed": 12, "quality_budget_satisfied": 12,
    "frozen_numerical_auditor_sha256": PIN, "numeric_dictionaries_equal_to_prior_audit": numeric_equality,
    "resource_scope": "All latest-audit processes including immutable partial-attempt snapshots, not only final successful rows or final-process header",
    "peak_RSS_bytes_all_attempts": max(peaks), "minimum_host_available_bytes_all_attempts": min(frees),
    "RSS_limit_bytes": 209715200, "RSS_transient_excess_bytes": max(0, max(peaks) - 209715200),
    "resource_limit_status": "transient_RSS_excess_detected_then_deferred_and_restarted" if max(peaks) > 209715200 else "within_recorded_limits",
    "threads": 1, "GPU_used": False, "framework_imports": False,
    "partial_attempt_records": [file_record(path) for path, _ in records[1:]],
    "memory_deferrals_preserved": deferred,
    "mean_baseline_status": {k: v["status"] for k, v in mean["domains"].items()},
    "mean_claim_scope": "Computed and self-checked by baseline author; no third-party independent numerical acceptance verdict",
    "historical_source_limit": mapping["older_field_auditor_limitation"],
    "resolution": "Final saved predictions for all twelve models rechecked under exact frozen auditor bytes; historical missing revisions remain explicitly unrecovered",
    "finalizer_source_snapshots": SOURCES,
    "environment": {"python_executable": sys.executable, "python_version": sys.version, "numpy_installed_metadata_version": version("numpy"), "numpy_imported_by_finalizer": False},
    "unverified": ["Model and scalar-baseline figures", "Optimizer convergence", "Native-cell hotspot accuracy from projected nodal heads"]
}
# Preserve the original numerical output before adding clearly separated completion metadata.
preserved = [source_preservation.preserve(latest_path, record=True), source_preservation.preserve(OUT / "full-support-latest-audit.md", record=True)]
clean_latest = read(latest_path)
clean_latest["completion_metadata_only"] = summary
clean_latest["numeric_report_preserved_before_completion_metadata"] = preserved
write(latest_path, clean_latest)
md = OUT / "full-support-latest-audit.md"
with md.open("a", encoding="utf-8") as stream:
    stream.write("\n\n## 최종 완료·자원 이력\n\n12/12 saved-prediction 수치 감사와 예산 게이트가 통과했고, 기존 감사의 수치와 모두 일치했다. 새 NN 학습·추론·test는 실행하지 않았다.\n\n")
    stream.write("감사 코드 SHA-256은 `" + PIN + "`이며 exact source snapshot이 보존되어 있다. 이전 일부 감사 revision의 누락된 source bytes는 복구됐다고 주장하지 않는다.\n\n")
    stream.write(f"이전 partial 기록까지 포함한 실제 peak RSS는 **{max(peaks):,} bytes ({max(peaks)/1048576:.6f} MiB)**, 최소 host free는 **{min(frees):,} bytes ({min(frees)/1073741824:.6f} GiB)**였다. 200 MiB 상한을 {summary['RSS_transient_excess_bytes']/1048576:.6f} MiB 일시 초과한 것을 감지해 계산을 보류하고 새 프로세스로 이어 완료했다. 전 과정이 200 MiB 이하였다는 주장은 하지 않는다. CPU 1개, GPU·Torch·PhysicsNeMo 사용 없음.\n\n")
    stream.write("Mean baseline은 별도 산출자의 computed_and_self_checked 결과이며 제3자 독립 수치 PASS와 구분한다. 이 문단은 표준 라이브러리로 JSON 이력을 집계한 완료 메타데이터이며 수치를 다시 계산한 것이 아니다.\n")
summary["final_reports"] = [file_record(OUT / (stem + suffix)) for stem in ("full-support-evaluation-review", "full-support-latest-audit", "full-support-mean-baselines") for suffix in (".json", ".md")]
write(HERE / "final-saved-result-verification.json", summary)
mapping["resolution_status"] = "complete_saved_prediction_repeat_12_of_12"
mapping["resolution_evidence"] = {"path": str(HERE / "final-saved-result-verification.json"), "latest_report": file_record(latest_path), "numerical_auditor_sha256": PIN, "source_snapshot": str(HERE / "source-snapshots" / PIN / "audit_full_support.py"), "resource_limit_status": summary["resource_limit_status"]}
mapping["resolution_plan_historical"] = mapping.pop("resolution_plan", None)
write(mapping_path, mapping)
write(HERE / "source-snapshots/snapshot-payload-inventory.json", {"verified_utc": now, "scope": "Existing source snapshot payload hashes only; missing historical auditor revisions are not claimed recovered", "count": len(inventory), "all_match_content_address": True, "files": inventory})
print(json.dumps({k: summary[k] for k in ("completion_count", "numeric_audit_passed", "peak_RSS_bytes_all_attempts", "minimum_host_available_bytes_all_attempts", "RSS_transient_excess_bytes", "resource_limit_status")}, indent=2))
