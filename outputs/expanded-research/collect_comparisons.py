"""Index completed comparable field metrics without loading FEM or ML arrays."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def collect(root: Path) -> dict:
    runs, rows, other, reloads = [], [], [], []
    paths = set((root / "benchmarks").rglob("metrics.json"))
    paths.update((root / "benchmarks").rglob("test_report.json"))
    for path in sorted(paths):
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        budget_metadata = {}
        if path.name == "test_report.json":
            training_path = path.parent / "training_report.json"
            if (data.get("status") != "completed" or data.get("test_performance_passes") != 1
                    or "physical_metrics" not in data or not training_path.is_file()):
                other.append(path.relative_to(root).as_posix())
                continue
            training_bytes = training_path.read_bytes()
            training = json.loads(training_bytes.decode("utf-8-sig"))
            accounting = training["budget_accounting"]
            budget_metadata = {
                "training_report_sha256": hashlib.sha256(training_bytes).hexdigest(),
                "quality_gates": training["quality_gates"],
                "optimizer_updates": accounting["total_updates"],
                "case_observations": accounting["total_samples"],
                "training_total_wall_seconds": training["total_wall_seconds"],
                "test_total_wall_seconds": data["total_wall_seconds"],
                "separate_training_timers": training["separate_timers"],
                "monitoring_seconds": training["workloads"]["monitoring_seconds"],
                "timing_boundary": training["timing_boundary"],
            }
            data = {
                "status": "actual_budgeted_training_and_fresh_heldout_evaluation_completed",
                "mode": "budgeted_train_and_fresh_test",
                "contract": training["contract"],
                "metrics": {data["architecture"]: data["physical_metrics"]},
                "actual_epochs": len(accounting["completed_epochs"]),
                "best_epoch": training["best_epoch"],
                "best_validation_normalized_mse": training["best_validation_normalized_mse"],
                "selected_by_validation_only": True,
                "training_seconds": accounting["train_seconds"],
                "peak_sampled_rss_bytes": training["peak_sampled_rss_bytes_lower_bound"],
                "cuda_peak_allocated_bytes": training["training_cuda_allocator_peak_bytes"],
            }
        if str(data.get("status", "")).startswith("fresh_"):
            reloads.append({"evidence": path.relative_to(root).as_posix(),
                            "metrics_sha256": hashlib.sha256(raw).hexdigest(),
                            "status": data["status"]})
            continue
        container = data.get("test_metrics", data.get("metrics", {}))
        recognized = []
        if isinstance(container, dict):
            for model, fields in container.items():
                if not isinstance(fields, dict):
                    continue
                valid = {name: metric for name, metric in fields.items()
                         if isinstance(metric, dict) and "relative_l2" in metric
                         and ("rmse" in metric or "component_rmse" in metric)}
                if valid:
                    recognized.append((model, valid))
        relative = path.relative_to(root).as_posix()
        if not recognized:
            other.append(relative)
            continue
        contract = data.get("contract", {})
        provenance = {
            "evidence": relative,
            "metrics_sha256": hashlib.sha256(raw).hexdigest(),
            "status": data.get("status"),
            "mode": data.get("mode"),
            "cache_manifest_sha256": data.get("cache_manifest_sha256", contract.get("manifest_sha256")),
            "shape": contract.get("shape"),
            "split": contract.get("split"),
            "actual_epochs": data.get("actual_epochs"),
            "best_epoch": data.get("best_epoch"),
            "best_validation_normalized_mse": data.get("best_validation_normalized_mse"),
            "selected_by_validation_only": data.get("selected_by_validation_only"),
            "training_seconds": data.get("training_seconds"),
            "shared_fit_selection_timing_seconds": data.get("timing_seconds"),
            "total_seconds": data.get("total_seconds"),
            "peak_process_rss_bytes": data.get("peak_process_rss_bytes", data.get("peak_sampled_rss_bytes")),
            "cuda_peak_allocated_bytes": data.get("cuda_peak_allocated_bytes"),
            "models": [item[0] for item in recognized],
            **budget_metadata,
        }
        runs.append(provenance)
        for model, fields in recognized:
            for field, metric in fields.items():
                unit = metric.get("unit")
                factor = 1000.0 if unit == "GPa" else 1.0
                rows.append({
                    "evidence": relative,
                    "model": model,
                    "field": field,
                    "display_unit": "MPa" if unit == "GPa" else unit,
                    "relative_l2": metric["relative_l2"],
                    "relative_l2_percent": None if metric["relative_l2"] is None else 100 * metric["relative_l2"],
                    "centroid_removed_displacement_l2_percent": None if metric.get("centroid_removed_relative_l2") is None else 100 * metric["centroid_removed_relative_l2"],
                    "rmse": factor * metric.get("rmse", metric.get("component_rmse")),
                    "rmse_definition": "component RMSE" if "component_rmse" in metric else "scalar RMSE",
                    "p95_absolute_error": None if metric.get("absolute_error_p95", metric.get("vector_error_p95")) is None else factor * metric.get("absolute_error_p95", metric.get("vector_error_p95")),
                    "max_absolute_error": None if metric.get("absolute_error_max", metric.get("vector_error_max")) is None else factor * metric.get("absolute_error_max", metric.get("vector_error_max")),
                    "prediction_min": None if metric.get("prediction_min") is None else factor * metric["prediction_min"],
                    "negative_count_for_nonnegative_field": metric.get("prediction_negative_count_for_nonnegative_field"),
                    "prediction_nonfinite_count": metric.get("prediction_nonfinite_count"),
                    "count": metric.get("count"),
                    "truth_exact_nonzero_count": metric.get("truth_exact_nonzero_count"),
                })
    return {
        "scope": "Automatically indexed completed field-metric artifacts, not a count of independent architectures or physical-validation passes.",
        "comparison_rule": "Compare only matching field definitions, cache/split/support, time window and normalization. GPa scalar errors are displayed in MPa by multiplying by 1000; dimensionless strain is unchanged. Null relative L2 is undefined.",
        "timing_rule": "POD fitting/validation timings may be shared by several models. GPU allocator memory differs from total VRAM. Consult original evidence for inference and sampling scopes.",
        "not_covered_by_this_schema": other,
        "reloaded_inference_artifacts_not_additional_fits": reloads,
        "runs": runs,
        "field_rows": rows,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = collect(root)
    (root / "comparison-index.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = result["field_rows"]
    with (root / "comparison-fields.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["evidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"completed_metric_artifacts": len(result["runs"]), "field_rows": len(rows), "other_schemas": len(result["not_covered_by_this_schema"])}))
