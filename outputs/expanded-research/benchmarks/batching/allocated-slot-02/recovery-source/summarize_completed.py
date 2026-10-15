"""Merge preserved JSON evidence only; never read any array or held-out label."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"outputs/expanded-research/benchmarks/batching"
MODELS = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), f"Do not overwrite evidence: {path}"
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    original = OUT/"allocated-slot-01/suite.json"
    recovery = read(OUT/"allocated-slot-02/suite.json")
    assert sha(original) == recovery["original_suite_sha256"]
    indexed = {(r["domain"], r["architecture"], r["batch_size"]): r for r in read(original)["results"]}
    indexed.update({(r["domain"], r["architecture"], r["batch_size"]): r for r in recovery["results"]})
    rows, source_checks, cases = [], [], {"bumper": {}, "phone": {}}
    for domain in ("bumper", "phone"):
        for model in MODELS:
            for b in (2, 5):
                key = (domain, model, b)
                row = {"domain": domain, "architecture": model, "batch_size": b}
                index = indexed.get(key)
                slot = "allocated-slot-02" if model == "figconvunet" and domain == "phone" else "allocated-slot-01"
                folder = OUT/slot/domain/model/f"B{b}"
                result_path = folder/"result.json"
                events = [json.loads(line) for line in (folder/"events.jsonl").read_text(encoding="utf-8").splitlines()]
                row.update(evidence_directory=str(folder), indexed_in_original_suite=key in {(r["domain"], r["architecture"], r["batch_size"]) for r in read(original)["results"]},
                           last_event=events[-1] if events else None)
                if result_path.exists():
                    result = read(result_path)
                    row.update(status=result["status"], result=str(result_path), total_seconds=result["total_seconds"],
                               timing=result.get("timing"), train_cases=result["train_cases"],
                               cuda_peak_allocated_bytes=result.get("cuda_peak_allocated_bytes"),
                               cuda_peak_reserved_bytes=result.get("cuda_peak_reserved_bytes"),
                               peak_sampled_rss_bytes=result.get("peak_sampled_rss_bytes"),
                               failed_stage=result.get("failed_stage"), exception_type=result.get("exception_type"),
                               manifest_sha256=result["manifest_sha256"], normalization_sha256=result["normalization_sha256"],
                               ready_sha256=result["ready_sha256"], opened_data_files=result["opened_data_files"],
                               static_feature_name=result["static_feature_name"], no_validation_or_test_label_file_opened=result["no_validation_or_test_label_file_opened"])
                    if key not in indexed:
                        row["recovery_note"] = "Orphan result existed after original suite last checkpoint; actual attempted CUDA OOM, not unattempted"
                    for name, expected in result["source_sha256"].items():
                        archived = folder/"source"/Path(name).name
                        actual = sha(archived)
                        assert actual == expected
                        source_checks.append({"file": str(archived), "sha256": actual, "matched_recorded": True})
                    if result["status"] == "passed":
                        assert result["timing_repeats"] == 2 and result["adam"]["optimizer_updates"] == 3
                        assert result["gradient_check"]["passed"]
                        assert all(v["passed"] for v in result["output_checks"].values() if isinstance(v, dict) and "passed" in v)
                        row["exposure"] = {"complete": True, "optimizer_updates": 3, "adam_simulation_observations": 3*b,
                                           "label_loss_case_observations": 9*b, "forward_case_evaluations_including_perturbed_globals": 17*b+1,
                                           "per_case": []}
                        for i, case in enumerate(result["train_cases"]):
                            exposure = {"case": case, "forward_evaluations": 17+(i == b-1),
                                        "perturbed_global_forward_evaluations": 2 if i == b-1 else 0,
                                        "label_loss_observations": 9, "adam_observations": 3}
                            row["exposure"]["per_case"].append(exposure)
                            total = cases[domain].setdefault(case, Counter())
                            total.update({k: v for k, v in exposure.items() if k != "case"})
                    else:
                        row.update(cuda_peak_allocated_bytes_at_failure=result.get("cuda_peak_allocated_bytes_at_failure"),
                                   cuda_peak_reserved_bytes_at_failure=result.get("cuda_peak_reserved_bytes_at_failure"))
                else:
                    assert index is not None and index["status"] == "resource_guard_stop"
                    row.update(status=index["status"], resource_guard=index["stop_sample"], resource_summary=index["resource_summary"],
                               resource_log=index["resource_log"], total_seconds=index["seconds"], result=None,
                               train_cases=read(OUT/slot/domain/model/"B2/result.json")["train_cases"][:2])
                    # B5 selected the first five train cases, shown in existing B5 metadata from another architecture.
                    row["train_cases"] = read(OUT/"allocated-slot-01"/domain/"transolver/B5/result.json")["train_cases"]
                    row["source_sha256_from_archived_snapshot"] = {p.name: sha(p) for p in (folder/"source").iterdir()}
                if row["status"] != "passed":
                    complete = {e["stage"] for e in events if e["stage"].endswith(":complete")}
                    assert "warm_original_B1_mean_gradient:complete" in complete
                    assert "warm_native_batchmean_gradient:complete" not in complete
                    row["exposure"] = {"complete": False, "optimizer_updates": 0, "adam_simulation_observations": 0,
                                       "label_loss_case_observations_completed_lower_bound": b,
                                       "forward_case_evaluations_completed_lower_bound": 5*b+1,
                                       "per_selected_case_completed_lower_bound": {"forward_evaluations": 5, "label_loss_observations": 1, "adam_observations": 0},
                                       "last_selected_case_extra_completed_forward_evaluations": 1,
                                       "note": "Output check prefix and sequential mean-gradient finished; native mean-gradient began but did not finish. No full native correctness, timing, or Adam success inferred."}
                rows.append(row)
    recommendations = []
    for domain in ("bumper", "phone"):
        for model in MODELS:
            options = [r for r in rows if r["domain"] == domain and r["architecture"] == model and r["status"] == "passed"]
            eligible = [r for r in options if r["timing"]["forward_backward"]["native_faster_in_every_paired_repeat"]
                        and r["cuda_peak_reserved_bytes"] <= 0.8*6441926656]
            baseline = ROOT/"outputs/expanded-research/benchmarks"/("bumper/probes-c7" if domain == "bumper" else "drop_test/probes-full-c6")/model/"probe.json"
            b1 = read(baseline)
            assert b1["status"] == "actual_FEM_forward_backward_optimizer_steps_completed" and len(b1["steps"]) == 3
            recommendation = {"domain": domain, "architecture": model, "baseline_B1_evidence": str(baseline), "optimizer_budget_warning": "A batch changes updates per epoch. Do not infer equal convergence from this 3-update functionality and paired timing probe."}
            if eligible:
                chosen = max(eligible, key=lambda r: r["timing"]["forward_backward"]["native_batch_simulations_per_second"])
                recommendation.update(recommended_batch_size=chosen["batch_size"], evidence=chosen["result"],
                                      paired_forward_backward_speedup=chosen["timing"]["forward_backward"]["measured_speedup"],
                                      native_simulations_per_second=chosen["timing"]["forward_backward"]["native_batch_simulations_per_second"],
                                      native_faster_in_both_repeats=True, cuda_peak_allocated_bytes=chosen["cuda_peak_allocated_bytes"],
                                      cuda_peak_reserved_bytes=chosen["cuda_peak_reserved_bytes"],
                                      nominal_gpu_capacity_minus_reserved_bytes=6441926656-chosen["cuda_peak_reserved_bytes"],
                                      reason="Fastest passing tested candidate with both paired backward repeats faster and allocator reserve <=80% nominal GPU capacity")
            else:
                assert domain == "phone" and model == "meshgraphnet"
                recommendation.update(recommended_batch_size=1, evidence=str(baseline),
                                      cuda_peak_allocated_bytes=b1["cuda_peak_allocated_bytes"],
                                      cuda_peak_reserved_bytes=b1["cuda_peak_reserved_bytes"],
                                      nominal_gpu_capacity_minus_reserved_bytes=6441926656-b1["cuda_peak_reserved_bytes"],
                                      reason="B2 native forward/backward is about 10x slower, exceeds physical capacity in allocator reserve, and B5 has an actual archived CUDA OOM")
            recommendations.append(recommendation)
    counts = dict(Counter(r["status"] for r in rows))
    assert len(rows) == 24 and counts == {"passed": 22, "oom": 1, "resource_guard_stop": 1}
    assert len(recommendations) == 12 and len(source_checks) == 23*6
    payload = {"status": "completed", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "original_suite_preserved": {"path": str(original), "sha256": sha(original), "checkpoint_results": 21},
               "recovery_suite": str(OUT/"allocated-slot-02/suite.json"), "counts": counts, "rows": rows,
               "source_snapshot_checks": source_checks,
               "heldout_access": "This merger only reads result/suite/event JSON, archived Python source bytes, and preceding B1 probe JSON. It opens/hashes no dataset arrays. Strict batch loader hashes only selected train data files; held-out target arrays are never opened or hashed.",
               "phone_static_feature_provenance_correction": "Actual scalar is reference_mean_incident_corner_cell_volume_cuberoot, as recorded by top-level static_feature_name and frozen cache. Nested model_source.input_information inherits the generic nodal_thickness display label; executed source/snapshots/results are preserved. This is a display metadata defect, not a different model input.",
               "rss_note": "Recovery OS resource logs sample the venv launcher process RSS (about4MiB), not the actual Torch interpreter. Use each actual probe result/events phase-boundary RSS and Windows peak working set for actual process memory; OS available physical/commit samples remain system-wide and valid.",
               "exposure_accounting": {"scope": "Completed strict B2/B5 probes only; excludes main B1 probes and failed prefixes shown separately per row",
                                       "per_completed_probe": "Each selected case has9 labeled loss/backward observations,3 Adam observations,17 forward evaluations (last case18, two with perturbed globals). B2 totals35/18/6; B5 totals86/45/15 for forwards/labeled-loss/Adam.",
                                       "complete_probe_aggregate_per_case": cases,
                                       "failed_prefix_note": "Both failed B5 rows expose all first five training cases to data loading and completed at least one labeled sequential backward each, with zero Adam updates. Incomplete native forward may add work; exact failed-operation exposure is not claimed."},
               "recommendations": recommendations,
               "limits": "Two timing repeats on an active Windows workstation; allocator reserve is not total driver VRAM. No main training or held-out evaluation was performed by this recovery task."}
    write(OUT/"merged-summary.json", payload)
    write(OUT/"batch-recommendations.json", {"status": "completed", "generated_utc": payload["generated_utc"],
                                            "selection_policy": "Passing correctness+3Adam; faster in both paired backward repeats; reserve<=80%capacity; then highest native simulations/s. B1 fallback with explicit evidence.",
                                            "main_training_still_requires_preflight_OS_memory": True,
                                            "recommendations": recommendations})
    print(json.dumps({"counts": counts, "recommendations": [{"domain": r["domain"], "architecture": r["architecture"], "batch": r["recommended_batch_size"]} for r in recommendations], "source_snapshot_checks": len(source_checks)}, indent=2))


if __name__ == "__main__":
    main()
