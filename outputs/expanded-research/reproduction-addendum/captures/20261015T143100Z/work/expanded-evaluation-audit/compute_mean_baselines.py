"""Compute a train-only nodal time-field mean comparator from saved truth, CPU only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import audit_full_support as audit
import source_preservation

OUT = audit.ROOT / "outputs/expanded-research/full-support-mean-baselines.json"
EXECUTION_SOURCES = source_preservation.sources(__file__, audit.__file__, source_preservation.__file__)


def compute(row, plan):
    # Reserve the worker footprint before importing NumPy or mapping arrays.
    audit.array_headroom()
    import numpy as np
    audit.np = np
    audit.memory()
    meta, provenance = audit.inventory(row)
    run, cache = Path(row["out"]), Path(row["cache"]).parent
    training, test = audit.read(run/"training_report.json"), audit.read(run/"test_report.json")
    frozen, fresh = audit.read(run/"checkpoint-frozen.json"), audit.read(run/"validation_fresh_process.json")
    stats = audit.read(run/"normalization.json")
    audit.same(test["status"], "completed", "mean baseline completed truth source")
    audit.same(frozen["status"], "checkpoint_fixed_and_validation_roundtrip_passed", "mean source frozen")
    audit.same(training["contract"]["manifest_sha256"], row["manifest_sha256"], "mean source contract")
    audit.same(training["contract"]["split"], meta["split"], "mean split")
    audit.same(test["cases"], meta["split"]["test"], "saved truth case order")
    audit.same(test["field_names"], meta["field_names"], "mean field order")
    audit.same(test["field_units"], meta["field_units"], "mean physical units")
    audit.same(test["time_values"], meta["time_values"], "mean nominal times")
    audit.same(stats, audit.read(row["normalization"]), "common normalization")
    audit.same(audit.objsha(stats), frozen["normalization_sha256"], "frozen normalization")
    audit.same(audit.sha(run/"training_report.json"), frozen["training_report_sha256"], "mean training report freeze")
    audit.same(training["data_value_access"]["test"], {}, "mean training source test values")
    mean_path, truth_path = run/"training_mean_field.npy", run/"heldout/truth.npy"
    mean_hash, truth_hash = audit.sha(mean_path), audit.sha(truth_path)
    audit.same(mean_hash, test["frozen_evidence_sha256"][str(mean_path.resolve())], "test-frozen train mean hash")
    audit.same(mean_hash, fresh["frozen_evidence_sha256"][str(mean_path.resolve())], "fresh-frozen train mean hash")
    audit.same(truth_hash, test["output_sha256"]["truth.npy"], "saved truth output hash")
    if row["domain"] == "phone":
        audit.same(mean_hash, "b2596285b96916c8856e96dc13617680ef61e82c89fb7ae1ae0f6308dfa39085", "confirmed phone mean pin")
    shared_mean = []
    for item in plan["runs"]:
        directory = Path(item["out"])
        if item["domain"] == row["domain"] and (directory/"checkpoint-frozen.json").is_file():
            value = audit.sha(directory/"training_mean_field.npy")
            audit.same(value, mean_hash, "common frozen domain mean "+item["architecture"])
            shared_mean.append({"architecture": item["architecture"], "mean_sha256": value})
    mean = np.load(mean_path, mmap_mode="r")
    truth = np.load(truth_path, mmap_mode="r")
    try:
        audit.same(str(mean.dtype), "float32", "physical train mean dtype")
        audit.same(list(mean.shape), training["contract"]["shape"], "physical node-time-field mean shape")
        audit.same(list(truth.shape), [len(test["cases"]), *mean.shape], "truth support shape")
        audit.same(str(truth.dtype), "float32", "saved truth dtype")
        count = len(meta["split"]["train"])
        maximum_mean_delta = 0.
        for cut in audit.chunks(mean.shape[0]):
            accumulated = np.zeros((cut.stop-cut.start, *mean.shape[1:]), dtype=np.float64)
            for case in meta["split"]["train"]:
                target = np.load(cache/meta["runs"][case]["target"], mmap_mode="r")
                try:
                    audit.same(list(target.shape), list(mean.shape), "train response shape")
                    audit.same(str(target.dtype), "float32", "train response dtype")
                    values = target[cut]
                    assert np.isfinite(values).all(), "nonfinite train value"
                    accumulated += values
                finally:
                    audit.close(target)
            expected = (accumulated/count).astype(np.float32)
            maximum_mean_delta = max(maximum_mean_delta, float(np.abs(expected.astype(np.float64)-mean[cut]).max()))
            assert np.array_equal(expected, mean[cut]), "saved physical mean differs from training-case arithmetic mean"
        prediction = np.broadcast_to(mean, truth.shape)
        metrics, epsp = {}, {}
        for channel, name in enumerate(meta["field_names"]):
            metrics[name] = audit.scalar_metrics(prediction, truth, channel,
                any(token in name.lower() for token in ("vm", "von_mises", "epsp", "plastic", "frobenius", "magnitude")),
                test["cases"], meta["time_values"])
            metrics[name]["unit"] = meta["field_units"][channel]
            if any(token in name.lower() for token in ("epsp", "plastic")):
                epsp[name] = audit.epsp_metrics(prediction, truth, channel)
        metrics["displacement_vector"] = audit.vector_metrics(prediction, truth)
        metrics["displacement_vector"]["unit"] = meta["field_units"][0]
        audit.memory()
        return {"status": "computed_and_self_checked", "computed_utc": datetime.now(timezone.utc).isoformat(),
            "source_snapshots_before_array_work": EXECUTION_SOURCES,
            "domain": row["domain"], "definition": "Physical node x stored-time x channel arithmetic mean over training simulations only, accumulated in float64 then serialized float32; the identical nodal time-field is broadcast to every held-out case",
            "training_case_count": count, "training_cases": meta["split"]["train"], "test_cases": test["cases"],
            "shape": list(truth.shape), "field_names": meta["field_names"], "field_units": meta["field_units"],
            "time_values": meta["time_values"], "time_unit": meta.get("time_unit"), "provenance": provenance,
            "mean_source": str(mean_path), "mean_sha256": mean_hash, "saved_truth": str(truth_path), "saved_truth_sha256": truth_hash,
            "common_frozen_model_mean_hashes": shared_mean, "training_mean_independent_chunk_check": {
                "bitwise_equal": True, "maximum_physical_absolute_difference": maximum_mean_delta,
                "source_training_only": True, "count": count, "accumulation_dtype": "float64", "serialized_dtype": "float32"},
            "common_normalization_file_sha256": audit.sha(row["normalization"]), "physical_metrics": metrics,
            "epsp_diagnostics": epsp, "raw_predictions_clipped": False,
            "runtime": {"threads": 1, "GPU_used": False, "framework_imports": False, "peak_RSS_bytes": audit.peak_rss,
                "minimum_host_available_bytes": audit.minimum_free, "RSS_limit_bytes": audit.RSS_CAP},
            "validation_scope": "Baseline computation plus source/arithmetic/finite/shape self-checks; not a third-party independent acceptance verdict",
            "limitations": "Ignores geometry/material/wall/loading/thickness inputs and case variation. Node/time alignment and physical field definitions are inherited from this frozen domain only. No new NN prediction, test pass, checkpoint selection or fit; no coupon contract reuse. Projected nodal scalar errors do not establish native-cell hotspot accuracy."}
    finally:
        audit.close(mean); audit.close(truth)


def run_domain(domain):
    plan = audit.read(audit.PLAN)
    field_review = audit.read(audit.OUT)
    candidates = [row for row in plan["runs"] if row["domain"] == domain and
        field_review["models"].get(row["domain"]+"/"+row["architecture"], {}).get("status") in ("passed", "integrity_passed_insufficient_budget")]
    assert candidates, "A completed frozen saved-truth source is required"
    row = candidates[0]
    report = audit.read(OUT) if OUT.exists() else {"domains": {}}
    if report["domains"].get(domain, {}).get("status") == "computed_and_self_checked":
        print(json.dumps({"mean_baseline": domain, "status": "already_computed_no_repeat"}), flush=True)
        return
    prior_snapshots = [source_preservation.preserve(path, record=True) for path in (OUT, OUT.with_suffix(".md")) if path.exists()]
    try:
        report["domains"][domain] = compute(row, plan)
    except audit.DeferredMemory as error:
        report["domains"][domain] = {"status": "deferred_memory", "reason": str(error),
            "source_snapshots_before_array_work": EXECUTION_SOURCES}
    report.setdefault("script_sha256", EXECUTION_SOURCES[Path(__file__).name]["sha256"])
    report.setdefault("metric_script_sha256", EXECUTION_SOURCES[Path(audit.__file__).name]["sha256"])
    report.update({"updated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Separate train-mean comparator computation and self-checks; saved response metrics, no additional NN test pass",
        "legacy_top_level_source_hash_scope": "Initial report computation only; each new domain has its own start-of-execution source snapshots",
        "prior_record_snapshots": report.get("prior_record_snapshots", [])+prior_snapshots})
    temporary = OUT.with_suffix(".json.writing")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(OUT)
    lines = ["# Full-support train-mean 기준선", "", "학습 case의 physical node×time×C 산술평균을 모든 held-out case에 그대로 적용했다. 별도 기준선 산출과 자체 검산이며 제3자 독립 PASS 판정이 아니다.", "", "입력조건을 무시하므로 material·wall·loading·thickness 변화에 따른 case별 차이를 예측하지 않는다. NN 학습·test 예측·체크포인트 선택을 추가 실행하지 않았다.", "", "작거나 0인 변위 성분의 relative L2는 물리 단위 절대오차와 truth 분모를 함께 읽는다. 분모 0은 null이며 성분별 퍼센트 순위를 만들지 않는다."]
    for domain_name, result in report["domains"].items():
        lines.extend(["", f"## {domain_name}", "", f"상태: `{result['status']}`"])
        if result["status"] != "computed_and_self_checked":
            continue
        vector = result["physical_metrics"]["displacement_vector"]
        lines.extend(["", f"{result['training_case_count']}개 train case 평균을 float64로 독립 누적한 뒤 float32로 대조하여 bitwise 일치를 확인했다.", "",
            f"변위 벡터 relative L2: `{vector['relative_l2']:.14g}`; centroid 제거 L2: `{vector['centroid_removed_relative_l2']:.14g}`.", "",
            "|필드|단위|RMSE|relative L2|P95 / 최대 절대오차|음수 출력|", "|---|---|---:|---:|---:|---:|"])
        for name in result["field_names"]:
            item = result["physical_metrics"][name]
            l2 = "null" if item["relative_l2"] is None else f"{item['relative_l2']:.10g}"
            lines.append(f"|{name}|{item['unit']}|{item['rmse']:.10g}|{l2}|{item['absolute_error_p95']:.10g} / {item['absolute_error_max']:.10g}|{item['prediction_negative_count_for_nonnegative_field']}|")
        lines.extend(["", "EPSP 양수 support·0인 영역의 오차·strict 시간 감소·개별 peak/위치는 동명 JSON에 기록했다. Bumper VM의 GPa 오차는 MPa 표시 시 ×1000이다."])
    temporary = OUT.with_suffix(".md.writing")
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(OUT.with_suffix(".md"))
    print(json.dumps({"mean_baseline": domain, "status": report["domains"][domain]["status"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("phone", "bumper"), required=True)
    args = parser.parse_args()
    import msvcrt
    # OS byte-range locks release on exit/crash, including concurrent manual runs.
    with Path(__file__).with_name("mean-baselines.lock").open("a+b") as lock:
        if lock.seek(0, 2) == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print(json.dumps({"mean_baseline": args.domain, "status": "busy_existing_mean_computation"}), flush=True)
            return
        try:
            run_domain(args.domain)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
