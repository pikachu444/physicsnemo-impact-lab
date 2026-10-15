"""Extrapolate measured update costs; these are proposals, not completed training."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parents[2]
names = ["transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet"]
datasets = [("phone-full-c6", "drop_test/probes-full-c6", 5), ("bumper-c7", "bumper/probes-c7", 112)]
destination = root / "outputs/expanded-research/benchmarks/bumper/probes-c7/b1-training-cost-estimates.json"
assert not destination.exists(), "Completed derived evidence is immutable"
results = []
for dataset, relative, count in datasets:
    rows = []
    for name in names:
        probe = json.loads((root / "outputs/expanded-research/benchmarks" / relative / name / "probe.json").read_text(encoding="utf-8"))
        warm = [v["seconds"] for v in probe["steps"][1:]]
        median = statistics.median(warm)
        rows.append({"architecture": name, "warm_update_median_seconds": median,
                     "observed_warm_update_seconds": warm,
                     "one_training_epoch_seconds_excluding_validation_and_case_io": median * count,
                     "one_training_epoch_observed_sample_scaled_interval_seconds": [min(warm)*count, max(warm)*count],
                     "1000_adam_updates_seconds": median * 1000,
                     "5000_adam_updates_seconds": median * 5000,
                     "cuda_allocator_peak_bytes": probe["cuda_peak_allocated_bytes"],
                     "sampled_rss_max_bytes_lower_bound": probe["peak_sampled_rss_bytes"]})
    results.append({"dataset": dataset, "batch_size": 1, "training_cases": count,
                    "adam_updates_per_epoch": count, "epochs_for_1000_updates": 1000/count,
                    "epochs_for_5000_updates": 5000/count, "models": rows})
report = {"status": "resource_based_training_budget_proposal; no full training or heldout evaluation performed",
          "optimizer_unit": "One FEM condition: predict all nodes and all stored10times/channels in one forward; one normalized mean loss and one Adam update. Time frames are flattened output channels, not separate samples or updates.",
          "limitations": "Based on only update2/3 from a single training condition. Their scaled interval is an observed sample range, not a statistical confidence interval or runtime guarantee. Excludes normalization/cache setup, per-case input/output copies and validation. Probe finite-gradient checks can add costs not present in the training loop. Actual epochs must be timed before a longer run. Does not establish convergence, generalization or an algorithm ranking.",
          "data_access": "Numerical train-only; all-split provenance hash access is separately disclosed in data-access-clarification.json.",
          "results": results}
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"written": str(destination), "results": results}))
