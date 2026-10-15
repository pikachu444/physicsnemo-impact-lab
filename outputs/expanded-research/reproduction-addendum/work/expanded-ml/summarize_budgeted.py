"""Summarize the finished static comparison using JSON evidence only.

No framework, numerical package, checkpoint or FEM/prediction array is loaded.
The suite must have terminated all twelve configurations before any summary is
written. Failed stages and missing/invalid reports remain explicit evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ARCHITECTURES = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")
LABELS = {"transolver": "Transolver", "flare": "FLARE", "geotransolver": "GeoTransolver GALE",
          "geoflare": "GeoTransolver GALE_FA", "meshgraphnet": "MeshGraphNet", "figconvunet": "FIGConvUNet"}
DOMAIN_DIRS = {"phone": "outputs/expanded-research/benchmarks/drop_test/models-full-c6",
               "bumper": "outputs/expanded-research/benchmarks/bumper/models-c7"}
EXPECTED_TRAINING_STATUS = "training_completed_best_complete_epoch_checkpoint_fixed_no_test_evaluation"
REPORT_NAMES = {"training": "training_report.json", "validation": "validation_fresh_process.json", "test": "test_report.json"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(path.read_bytes())


def object_digest(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def read_required(path, inventory):
    raw = path.read_bytes()
    inventory[str(path.resolve())] = digest_bytes(raw)
    return json.loads(raw.decode("utf-8"))


def read_report(path, inventory):
    if not path.is_file():
        return None, {"state": "missing", "path": str(path)}
    raw = path.read_bytes()
    sha = digest_bytes(raw)
    inventory[str(path.resolve())] = sha
    try:
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("report must be a JSON object")
        return value, {"state": "present", "path": str(path), "sha256": sha,
                       "reported_status": value.get("status")}
    except (ValueError, UnicodeError) as error:
        return None, {"state": "invalid", "path": str(path), "sha256": sha, "error": str(error)}


def pick(value, keys):
    return {key: value.get(key) for key in keys}


def compact_metrics(metrics):
    scalar_keys = ("unit", "relative_l2", "rmse", "absolute_error_p95", "absolute_error_max",
                   "sum_squared_error", "truth_sum_square", "truth_l2_norm", "count",
                   "truth_exact_nonzero_count", "truth_nonfinite_count", "prediction_nonfinite_count",
                   "prediction_negative_count_for_nonnegative_field", "prediction_min", "prediction_max",
                   "per_timestep_rmse")
    result = {}
    for name, row in metrics.items():
        if not isinstance(row, dict):
            continue
        if name == "displacement_vector":
            result[name] = dict(row)
        else:
            result[name] = pick(row, scalar_keys)
            result[name]["per_case_peak_errors"] = row.get("per_case", [])
    return result


def compact_ballistic(value):
    if not value:
        return None
    result = pick(value, ("definition", "coordinate_unit", "source_time_unit", "time_conversion_to_seconds",
                          "case_initial_condition_evidence", "model_displacement_sse_over_ballistic_sse"))
    result["ballistic_physical_metrics"] = compact_metrics(value.get("ballistic_physical_metrics", {}))
    result["gravity_only_physical_metrics"] = compact_metrics(value.get("gravity_only_physical_metrics", {}))
    return result


def training_summary(value):
    budget = value.get("budget_accounting", {})
    selected = value.get("selected_epoch_accounting", {})
    validation = value.get("validation_history", [])
    epochs = value.get("epoch_history", [])
    training_losses = [row["training_normalized_mse"] for row in epochs
                       if isinstance(row.get("training_normalized_mse"), (int, float))]
    eligible = [row for row in validation if row.get("eligible_complete_epoch")]
    last = eligible[-1] if eligible else None
    return {"reported_status": value.get("status"), "architecture": value.get("architecture"),
        "seed": value.get("seed"), "batch_size": value.get("batch_size"), "parameters": value.get("parameters"),
        "updates": budget.get("total_updates"), "samples": budget.get("total_samples"),
        "complete_epoch_count": len(budget.get("completed_epochs", [])),
        "completed_epochs": budget.get("completed_epochs", []), "partial_epoch": budget.get("partial_epoch"),
        "selected_epoch": value.get("best_epoch"), "selected_epoch_accounting": selected,
        "compute_seconds_at_selection": selected.get("train_seconds"),
        "best_checkpoint_fraction_of_compute_budget": value.get("best_checkpoint_fraction_of_compute_budget"),
        "best_validation_normalized_mse": value.get("best_validation_normalized_mse"),
        "initial_validation_normalized_mse": validation[0].get("validation_normalized_mse") if validation else None,
        "minimum_complete_epoch_training_normalized_mse": min(training_losses) if training_losses else None,
        "last_complete_epoch_training_normalized_mse": epochs[-1].get("training_normalized_mse") if epochs else None,
        "last_eligible_validation": last, "last_validation_improved": value.get("last_validation_improved"),
        "convergence": value.get("convergence", "Unverified within this bounded comparison."),
        "compute_budget_seconds": budget.get("train_seconds_cap"), "optimizer_compute_seconds": budget.get("train_seconds"),
        "budget_reached": budget.get("budget_reached"), "budget_overshoot_seconds": budget.get("budget_overshoot_seconds"),
        "minimum_required_updates": budget.get("min_updates"), "min_updates_satisfied": budget.get("min_updates_satisfied"),
        "termination_eligibility": budget.get("termination_eligibility"), "stop_reason": value.get("stop_reason"),
        "quality_gates": value.get("quality_gates"), "quality_gate_role": "Report only; the declared compute cap was not extended.",
        "validation_count": budget.get("validation_count"), "partial_epoch_selected": value.get("partial_epoch_selected"),
        "total_wall_seconds": value.get("total_wall_seconds"), "setup_seconds": value.get("setup_seconds_including_framework_import"),
        "separate_timers": value.get("separate_timers"), "timing_boundary": value.get("timing_boundary"),
        "training_cuda_allocator_peak_bytes": value.get("training_cuda_allocator_peak_bytes"),
        "training_cuda_reserved_peak_bytes": value.get("training_cuda_reserved_peak_bytes"),
        "peak_sampled_rss_bytes_lower_bound": value.get("peak_sampled_rss_bytes_lower_bound"),
        "workloads": value.get("workloads"), "device_name": value.get("device_name"),
        "torch": value.get("torch"), "physicsnemo": value.get("physicsnemo"),
        "checkpoint_sha256": value.get("checkpoint_sha256"), "source_sha256": value.get("source_sha256"),
        "model_source": value.get("model_source"), "training_protocol_sha256": value.get("training_protocol_sha256"),
        "contract_sha256": object_digest(value["contract"]) if "contract" in value else None,
        "manifest_sha256": value.get("contract", {}).get("manifest_sha256"),
        "normalization_sha256": object_digest(value["normalization"]) if "normalization" in value else None,
        "data_value_access": value.get("data_value_access")}


def collect_model(root, domain, architecture, execution, inventory):
    directory = root/DOMAIN_DIRS[domain]/architecture
    reports, files = {}, {}
    for stage, name in REPORT_NAMES.items():
        reports[stage], files[stage] = read_report(directory/name, inventory)
    train, validation, test = (reports[name] for name in ("training", "validation", "test"))
    errors = []
    for stage, row in files.items():
        if row["state"] == "invalid":
            errors.append(stage+" report is invalid: "+row["error"])
    required = {"training": ("budget_accounting", "selected_epoch_accounting", "quality_gates", "model_config",
                              "best_epoch", "best_validation_normalized_mse", "checkpoint_sha256", "source_sha256", "contract", "normalization"),
                "validation": ("process_identity", "max_scale_normalized_difference", "tolerance_scale_normalized", "checkpoint_sha256", "source_sha256"),
                "test": ("process_identity", "checkpoint_sha256", "source_sha256", "quality_gates")}
    for stage, value in reports.items():
        if value is not None:
            for key in required[stage]:
                if key not in value:
                    errors.append(stage+" report is missing required key "+key)
    if train is not None and train.get("status") != EXPECTED_TRAINING_STATUS:
        errors.append("training report is not in its completed state")
    if validation is not None and (validation.get("status") != "passed" or validation.get("test_values_used") is not False):
        errors.append("fresh validation did not pass without test response access")
    if test is not None and test.get("status") != "completed":
        errors.append("test report status: "+str(test.get("status")))
    for stage, value in reports.items():
        if value is not None and value.get("architecture") != architecture:
            errors.append(stage+" architecture does not match its directory")
    if train is not None:
        for stage, value in (("validation", validation), ("test", test)):
            if value is None:
                continue
            if value.get("checkpoint_sha256") != train.get("checkpoint_sha256"):
                errors.append(stage+" checkpoint hash disagrees with training")
            if value.get("source_sha256") != train.get("source_sha256"):
                errors.append(stage+" source inventory disagrees with training")
            if value.get("quality_gates") != train.get("quality_gates"):
                errors.append(stage+" quality gate report disagrees with training")
        if train.get("data_value_access", {}).get("test"):
            errors.append("training reports test response access")
    if test is not None and test.get("status") == "completed":
        if test.get("test_performance_passes") != 1:
            errors.append("test does not report exactly one performance pass")
        cases = test.get("cases", [])
        if test.get("data_value_access", {}).get("test") != {case: 1 for case in cases}:
            errors.append("test source targets were not recorded exactly once per case")
        if not test.get("physical_metrics"):
            errors.append("completed test has no physical metrics")
    stages = execution.get("stages", [])
    failed_stages = [pick(row, ("stage", "exit_code", "status", "log", "finished_utc"))
                     for row in stages if row.get("status") == "failed" or row.get("exit_code") not in (None, 0)]
    missing = [name for name, row in files.items() if row["state"] == "missing"]
    if execution.get("status") == "failed" or failed_stages or errors:
        status = "failed"
    elif missing:
        status = "missing"
    else:
        status = "complete"
    row = {"architecture": architecture, "label": LABELS[architecture], "status": status,
           "directory": str(directory), "report_files": files, "missing_reports": missing,
           "evidence_errors": errors, "suite_result_status": execution.get("status"),
           "failed_stages": failed_stages, "training": training_summary(train) if train else None,
           "fresh_validation": pick(validation, ("status", "pid", "process_identity", "validation_anchor_case",
                "max_scale_normalized_difference", "tolerance_scale_normalized", "test_values_used",
                "forward_seconds_cuda_synchronized", "total_wall_seconds")) if validation else None}
    if test is not None:
        row["test"] = {**pick(test, ("status", "pid", "process_identity", "cases", "shape", "field_names", "field_units",
            "time_values", "time_unit", "test_performance_passes", "forward_seconds_cuda_synchronized_total",
            "metrics_and_figures_seconds", "total_wall_seconds", "cuda_allocator_peak_bytes", "cuda_reserved_peak_bytes",
            "peak_sampled_rss_bytes_lower_bound", "raw_predictions_clipped", "data_access_disclosure",
            "data_value_access", "versions", "output_sha256", "exception_type", "exception", "retry_allowed")),
            "physical_metrics": compact_metrics(test.get("physical_metrics", {})),
            "epsp_diagnostics": test.get("epsp_diagnostics"), "principal_order_diagnostics": test.get("principal_order_diagnostics"),
            "phone_ballistic_reference": compact_ballistic(test.get("phone_ballistic_reference")),
            "per_case_forward_seconds": [{"case": item.get("case"), "forward_seconds_cuda_synchronized":
                item.get("forward_seconds_cuda_synchronized")} for item in test.get("per_case", [])],
            "figures": test.get("figures")}
    else:
        row["test"] = None
    return row


def fmt(value, percentage=False):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nonfinite"
        return f"{100*value:.4g}%" if percentage else f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(summary):
    domain = summary["domain"]
    title = "Phone C6 full support" if domain == "phone" else "Bumper C7 full support"
    lines = ["# "+title+" — bounded static training comparison", "", summary["scope"], "",
             f"Suite status: **{summary['suite_status']}**. Model evidence: "+
             ", ".join(f"{count} {state}" for state, count in summary["model_status_counts"].items())+".", "",
             "Each configuration had a 240 s synchronized optimizer-compute budget and a 5,000-update maximum. Minimum 100 updates and two complete epochs are report-only quality gates. A final update may cross the time cap; its actual duration is retained.", "",
             "| Configuration | Evidence | Batch | Updates | Samples | Complete epochs | Selected epoch | Compute at selection (s) | Actual compute / cap (s) | Overshoot (s) | Quality gate | Best validation normalized MSE |", 
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in summary["models"]:
        train = row["training"] or {}
        gate = (train.get("quality_gates") or {}).get("gate_passed")
        cells = [row["label"], row["status"], train.get("batch_size"), train.get("updates"), train.get("samples"),
                 train.get("complete_epoch_count"), train.get("selected_epoch"), train.get("compute_seconds_at_selection"),
                 f"{fmt(train.get('optimizer_compute_seconds'))} / {fmt(train.get('compute_budget_seconds'))}",
                 train.get("budget_overshoot_seconds"), fmt(gate), train.get("best_validation_normalized_mse")]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "A normalized validation loss is used only for checkpoint selection within its own domain. It is not a physical accuracy metric and is not comparable between the two domains.", "",
                  "| Configuration | Selected updates | Selected case observations | Selected epoch | Selected compute (s) | Last complete training normalized MSE | Last eligible validation improved |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for row in summary["models"]:
        train = row["training"] or {}
        selected = train.get("selected_epoch_accounting") or {}
        cells = [row["label"], selected.get("optimizer_updates"), selected.get("samples"),
                 train.get("selected_epoch"), selected.get("train_seconds"),
                 train.get("last_complete_epoch_training_normalized_mse"), train.get("last_validation_improved")]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "| Configuration | Parameters | Training CUDA allocated peak (MiB) | CUDA reserved peak (MiB) | Observed RSS peak lower bound (MiB) | Minimum observed available host RAM (MiB) | Observed solver PIDs |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    def mib(value):
        return value/(1024**2) if isinstance(value, (int, float)) else None
    for row in summary["models"]:
        train = row["training"] or {}
        workloads = train.get("workloads") or {}
        pids = sorted({item.get("pid") for item in (workloads.get("observed_processes") or {}).values()
                       if item.get("pid") is not None})
        cells = [row["label"], train.get("parameters"), mib(train.get("training_cuda_allocator_peak_bytes")),
                 mib(train.get("training_cuda_reserved_peak_bytes")), mib(train.get("peak_sampled_rss_bytes_lower_bound")),
                 mib(workloads.get("minimum_sampled_host_available_bytes")), ", ".join(str(pid) for pid in pids) or "—"]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "| Configuration | Optimizer compute (s) | Training wall (s) | Setup including import (s) | Monitoring (s) | Test process wall (s) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in summary["models"]:
        train, test = row["training"] or {}, row["test"] or {}
        cells = [row["label"], train.get("optimizer_compute_seconds"), train.get("total_wall_seconds"),
                 train.get("setup_seconds"), (train.get("workloads") or {}).get("monitoring_seconds"),
                 test.get("total_wall_seconds")]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "| Configuration | Training read / target checks (s) | Normalization / CPU stacking (s) | Host-to-device (s) | CPU snapshots / checkpoint I/O (s) | Validation (s) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in summary["models"]:
        timers = (row["training"] or {}).get("separate_timers") or {}
        cells = [row["label"], timers.get("training_data_read_seconds"), timers.get("training_normalization_seconds"),
                 timers.get("host_device_transfer_seconds"), timers.get("checkpoint_io_seconds"), timers.get("validation_seconds")]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "These observed timing categories are not an exhaustive disjoint partition: setup includes initial monitoring, while end verification and other work are not all separately timed. The 240 s optimizer cap is not a wall-time cap. Monitoring and data preparation overhead are reported rather than attributed to model compute. Memory samples are lower bounds; concurrent solver intervals remain in SUMMARY.json.", "",
                  "| Configuration | Test state | Forward total (s) | Displacement relative L2 | Vector RMSE (mm) | Vector error P95 (mm) | Vector error max (mm) | Centroid-removed relative L2 | Fresh validation max difference |", 
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary["models"]:
        test = row["test"] or {}
        vector = test.get("physical_metrics", {}).get("displacement_vector", {})
        val = row["fresh_validation"] or {}
        cells = [row["label"], test.get("status", "missing"), test.get("forward_seconds_cuda_synchronized_total"),
                 fmt(vector.get("relative_l2"), True), vector.get("vector_rmse"), vector.get("vector_error_p95"),
                 vector.get("vector_error_max"), fmt(vector.get("centroid_removed_relative_l2"), True),
                 val.get("max_scale_normalized_difference")]
        lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "The forward time contains one synchronized batch-size-one forward per held-out case. It excludes input preparation, transfer, saved-output I/O and metric/figure generation; it is a shared-workstation observation, not an isolated system benchmark.", "",
                  "| Configuration | Field | Unit | Relative L2 | RMSE | Absolute error P95 | Absolute error max | Raw prediction negatives | Raw minimum | Truth exact nonzero / count |", 
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary["models"]:
        metrics = (row["test"] or {}).get("physical_metrics", {})
        for name, field in metrics.items():
            if name == "displacement_vector":
                continue
            cells = [row["label"], name, field.get("unit"), fmt(field.get("relative_l2"), True), field.get("rmse"),
                     field.get("absolute_error_p95"), field.get("absolute_error_max"),
                     field.get("prediction_negative_count_for_nonnegative_field"), field.get("prediction_min"),
                     f"{fmt(field.get('truth_exact_nonzero_count'))} / {fmt(field.get('count'))}"]
            lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "Relative L2 is undefined when the truth norm is zero (shown as —). Scalar values retain their physical units. Negative counts apply to declared nonnegative fields and refer to raw outputs; displacement sign is unrestricted.", "",
                  "| Configuration | EPSP field | Truth positive / exact zero / total | Prediction negatives | Prediction decreases / increments | Max prediction decrease | Truth decreases / increments | Exact support TP / FP / FN | Exact support IoU |", 
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary["models"]:
        for name, field in ((row["test"] or {}).get("epsp_diagnostics") or {}).items():
            history = field.get("history", {})
            pdelta, tdelta = history.get("prediction", {}), history.get("truth", {})
            exact = next((item for item in field.get("support", []) if item.get("threshold") == 0), {})
            cells = [row["label"], name,
                " / ".join(fmt(field.get(key)) for key in ("truth_exact_positive_count", "truth_exact_zero_count", "count")),
                field.get("prediction_negative_count"), f"{fmt(pdelta.get('negative_count'))} / {fmt(pdelta.get('count'))}",
                pdelta.get("maximum_decrease"), f"{fmt(tdelta.get('negative_count'))} / {fmt(tdelta.get('count'))}",
                " / ".join(fmt(exact.get(key)) for key in ("true_positive", "false_positive", "false_negative")),
                fmt(exact.get("iou"), True)]
            lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
    lines.extend(["", "EPSP support uses exact >0 on the projected nodal labels and raw predictions. Strict negative increments include roundoff; their magnitude distributions and the same truth statistics remain in SUMMARY.json. No clipping, support threshold tuning, monotonic correction or principal reordering is applied."])
    if domain == "phone":
        lines.extend(["", "| Configuration | Contact-free ballistic displacement L2 | Gravity-only displacement L2 | Model SSE / ballistic SSE |", "|---|---:|---:|---:|"])
        for row in summary["models"]:
            baseline = (row["test"] or {}).get("phone_ballistic_reference") or {}
            ballistic = baseline.get("ballistic_physical_metrics", {}).get("displacement_vector", {})
            gravity = baseline.get("gravity_only_physical_metrics", {}).get("displacement_vector", {})
            cells = [row["label"], fmt(ballistic.get("relative_l2"), True), fmt(gravity.get("relative_l2"), True),
                     baseline.get("model_displacement_sse_over_ballistic_sse")]
            lines.append("| "+" | ".join(fmt(value) for value in cells)+" |")
        lines.extend(["", "The ballistic comparator uses each case's hash-verified source-deck initial velocity and signed acceleration, with stored time converted to seconds: u=v0·t+0.5·a·t²=v0·t−0.5·g·t². It omits contact, deformation and constraints; its error is not a solver validation measure."])
    lines.extend(["", "## Evidence and interpretation", ""])
    for row in summary["models"]:
        train = row["training"] or {}
        lines.append(f"- **{row['label']}**: {row['status']}; stop={fmt(train.get('stop_reason'))}; minimum updates met={fmt(train.get('min_updates_satisfied'))}; last eligible validation improved={fmt(train.get('last_validation_improved'))}.")
        if row["missing_reports"]:
            lines.append("  Missing reports: "+", ".join(row["missing_reports"])+".")
        for error in row["evidence_errors"]:
            lines.append("  Evidence issue: "+fmt(error)+".")
        for stage in row["failed_stages"]:
            lines.append(f"  Failed stage: {stage.get('stage')} (exit {stage.get('exit_code')}); log={fmt(stage.get('log'))}.")
    lines.extend(["", *["- "+item for item in summary["limits"]], "",
        "This summary reads only the saved reports and suite metadata. It does not rerun inference or open checkpoint, FEM target, or saved prediction arrays. Input-report and summarizer hashes are recorded in SUMMARY.json.", ""])
    if summary["comparability_issues"]:
        lines.extend(["**Comparability issues:** "+"; ".join(summary["comparability_issues"]), ""])
    return "\n".join(lines)


def finished_suite(root, inventory):
    plan_path = root/"work/expanded-ml/budgeted-suite-plan.json"
    execution_path = root/"work/expanded-ml/budgeted-suite-execution.json"
    plan = read_required(plan_path, inventory)
    execution = read_required(execution_path, inventory)
    expected = {(domain, architecture) for domain in DOMAIN_DIRS for architecture in ARCHITECTURES}
    if execution.get("status") not in ("completed", "completed_with_failures"):
        raise RuntimeError("Suite is not finished; no summaries were written (status="+str(execution.get("status"))+")")
    planned = {(row["domain"], row["architecture"]): row for row in plan.get("runs", [])}
    results = {(row["domain"], row["architecture"]): row for row in execution.get("results", [])}
    if set(planned) != expected or len(plan.get("runs", [])) != 12:
        raise RuntimeError("Plan must contain exactly the expected phone6 and bumper6 configurations")
    if set(results) != expected or len(execution.get("results", [])) != 12:
        raise RuntimeError("All twelve configuration results must be terminal before summary writing")
    if execution.get("plan_sha256") != inventory[str(plan_path.resolve())]:
        raise RuntimeError("Execution's plan hash does not match the saved plan")
    for key, result in results.items():
        if result.get("status") not in ("passed", "failed"):
            raise RuntimeError("Nonterminal model result: "+str(key))
        if any(row.get("status") == "running" for row in result.get("stages", [])):
            raise RuntimeError("Model still has a running stage: "+str(key))
        expected_path = (root/DOMAIN_DIRS[key[0]]/key[1]).resolve()
        if Path(planned[key]["out"]).resolve() != expected_path or Path(result["out"]).resolve() != expected_path:
            raise RuntimeError("Refusing a model output outside its fixed domain directory: "+str(key))
    return plan, execution, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Workspace root containing work and outputs")
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = {}
    plan, execution, results = finished_suite(root, inventory)
    summaries = []
    for domain, relative in DOMAIN_DIRS.items():
        models = [collect_model(root, domain, architecture, results[(domain, architecture)], inventory)
                  for architecture in ARCHITECTURES]
        comparable = [row["training"] for row in models if row["training"]]
        issues = []
        for key in ("manifest_sha256", "contract_sha256", "normalization_sha256", "training_protocol_sha256"):
            observed = {row.get(key) for row in comparable}
            if len(observed) > 1:
                issues.append("Saved model reports disagree on "+key)
        counts = {state: sum(row["status"] == state for row in models) for state in ("complete", "failed", "missing")}
        summary = {"status": "finished_report_summary", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "domain": domain, "suite_status": execution["status"], "suite_finished_utc": execution.get("finished_utc"),
            "scope": "A bounded initial comparison of six configurations from five model families on a fixed full-support physical-field task; no universal ranking, convergence or design-use accuracy claim.",
            "model_status_counts": counts, "models": models, "comparability_issues": issues,
            "limits": [
                "A single fixed seed and small models under a short compute budget do not establish convergence or an optimized recipe; later improvements may remain possible.",
                "Different approved batch sizes change optimizer trajectories and observations per update. Compare the recorded compute, samples, complete epochs and quality gates together.",
                "Phone labels describe only the initial 2–20 µs on a fixed assembly and two condition axes with a 5/2/2 simulation split. Bumper uses its fixed 112/12/7 split. These splits were previously evaluated by separate ROM methods and are not globally unseen research holdouts.",
                "Fields keep their own physical units and denominators. Displacement accuracy can be dominated by rigid translation; inspect centroid-removed errors and each stress/EPSP field separately.",
                "Nodal projection and stored-time fields do not establish original integration-point peaks, damage, full rebound, failure prediction or new-geometry generalization.",
                "Fresh validation checks checkpoint serialization at the validation anchor; it is not an independent physical accuracy test. Held-out performance uses one frozen-checkpoint pass, with no test-driven selection.",
                "Observed GPU timings and sampled memory may include contention from concurrent local solver workloads; memory samples are lower bounds rather than continuous host-memory peaks."
            ],
            "source_metadata": {"summarizer_path": str(Path(__file__).resolve()),
                "summarizer_sha256": digest_file(Path(__file__)), "plan_source_sha256": plan.get("source_sha256"),
                "input_report_sha256": None,
                "array_or_checkpoint_values_read": False, "inference_performed": False}}
        summaries.append((root/relative, summary))
    # Validate everything before writing either domain. Inputs are hashed again
    # to prevent a live or concurrently changed report becoming final evidence.
    for path, sha in inventory.items():
        if digest_file(Path(path)) != sha:
            raise RuntimeError("Input report changed during summary preparation: "+path)
    prepared = []
    for directory, summary in summaries:
        summary["source_metadata"]["input_report_sha256"] = dict(inventory)
        if not directory.is_dir():
            raise RuntimeError("Expected domain output directory is missing: "+str(directory))
        for name in ("SUMMARY.json", "SUMMARY.md"):
            if (directory/name).exists():
                raise RuntimeError("Refusing to overwrite summary evidence: "+str(directory/name))
        prepared.append((directory, json.dumps(summary, indent=2, allow_nan=False), markdown(summary)))
    for directory, json_text, markdown_text in prepared:
        with (directory/"SUMMARY.json").open("x", encoding="utf-8") as stream:
            stream.write(json_text)
        with (directory/"SUMMARY.md").open("x", encoding="utf-8") as stream:
            stream.write(markdown_text)
    print(json.dumps({"status": "summaries_written", "domains": [str(path) for path, _ in summaries],
                      "counts": {summary["domain"]: summary["model_status_counts"] for _, summary in summaries}}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "summary_refused", "reason": str(error), "summaries_written": False}), file=sys.stderr)
        raise SystemExit(2)
