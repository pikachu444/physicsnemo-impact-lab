"""Independent, CPU-only audit of sealed full-support results; never infer again."""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "work/expanded-ml/budgeted-suite-plan.json"
OUT = ROOT / "outputs/expanded-research/full-support-evaluation-review.json"
RSS_CAP = 200 * 1024**2
FREE_MIN = 1024**3
CHUNK = 2048
np = None
peak_rss = 0
minimum_free = None
domain_records = {}


class DeferredMemory(RuntimeError):
    pass


def memory():
    global peak_rss, minimum_free
    # Windows working set is the requested RSS boundary, including mapped pages.
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("peak", "rss", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
    status, counters = Status(), Counters()
    status.length = ctypes.sizeof(status)
    counters.cb = ctypes.sizeof(counters)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)), "host memory query failed"
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    query.restype = ctypes.c_int
    assert query(get_process(), ctypes.byref(counters), counters.cb), "RSS query failed"
    rss, free = counters.rss, status.available
    peak_rss = max(peak_rss, rss, counters.peak)
    minimum_free = free if minimum_free is None else min(minimum_free, free)
    if peak_rss > RSS_CAP or free < FREE_MIN:
        raise DeferredMemory(f"array work deferred: RSS={rss}, peak working set={peak_rss}, available={free}")
    return {"rss_bytes": rss, "host_available_bytes": free}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def array_headroom():
    status = memory()
    if status["host_available_bytes"] < FREE_MIN+RSS_CAP:
        raise DeferredMemory(f"array audit waiting for startup headroom: available={status['host_available_bytes']}, required={FREE_MIN+RSS_CAP}; reserves the audit footprint while preserving 1GiB host availability")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(262144), b""):
            digest.update(block)
    return digest.hexdigest()


def objsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def same(actual, expected, label):
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=1e-16), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def close(array):
    if getattr(array, "_mmap", None) is not None:
        array._mmap.close()


def inventory(row):
    path = Path(row["cache"])
    meta, ready = read(path), read(path.with_name("READY.json"))
    same(sha(path), row["manifest_sha256"], "manifest pin")
    same(ready["manifest_sha256"], row["manifest_sha256"], "READY manifest")
    same(ready["status"], "ready", "READY status")
    same(ready["writer_exit_code"], 0, "READY writer")
    same(ready["expected_cases"], len(meta["runs"]), "READY case count")
    all_cases = sum(meta["split"].values(), [])
    same(len(set(all_cases)), len(all_cases), "split disjointness")
    same(set(all_cases), set(meta["runs"]), "split exhaustiveness")
    cache_path = Path(__file__).with_name("domain-cache-"+row["domain"]+".json")
    signature = {"manifest_sha256": sha(path), "ready_sha256": sha(path.with_name("READY.json")),
        "normalization_file_sha256": sha(row["normalization"]),
        "file_stat": {item["name"]: [int((path.parent/item["name"]).stat().st_size),
            int((path.parent/item["name"]).stat().st_mtime_ns)] for item in ready["files"]}}
    old = read(cache_path) if cache_path.exists() else {}
    reused = old.get("signature") == signature
    if reused:
        domain_records[row["cache"]] = old
    else:
        domain_records[row["cache"]] = {"signature": signature, "audited_utc": datetime.now(timezone.utc).isoformat()}
    for item in ready["files"]:
        file = path.parent / item["name"]
        same(file.stat().st_size, item["bytes"], str(file)+" size")
        if not reused:
            same(sha(file), item["sha256"], str(file)+" hash")
    for case in meta["runs"].values():
        for bracket in case.get("target_time_brackets", []):
            lo, hi, nominal, weight = (bracket[k] for k in ("lower_actual_time_s", "upper_actual_time_s", "nominal_time_s", "upper_weight"))
            assert lo <= nominal <= hi and 0 <= weight <= 1
            same(weight, (nominal-lo)/(hi-lo), "interpolation bracket weight")
    cache_path.write_text(json.dumps(domain_records[row["cache"]], indent=2), encoding="utf-8")
    return meta, {"manifest_sha256": sha(path), "ready_sha256": sha(path.with_name("READY.json")),
        "verified_files": len(ready["files"]), "all_split_hash_audit_reused_from_domain_cache": reused,
        "domain_cache": str(cache_path), "cache_invalidation": "manifest/READY/normalization SHA or any inventoried file size/mtime_ns change",
        "all_split_hashing_is_provenance_not_response_selection": True}


def metadata(row, meta):
    run = Path(row["out"])
    train, frozen = read(run/"training_report.json"), read(run/"checkpoint-frozen.json")
    fresh, test = read(run/"validation_fresh_process.json"), read(run/"test_report.json")
    sentinel = read(run/"test-performance-pass.started.json")
    stats, protocol = read(run/"normalization.json"), read(run/"prospective-training-protocol.json")
    same(train["architecture"], row["architecture"], "plan architecture")
    same(train["batch_size"], row["batch_size"], "plan batch size")
    same(test["architecture"], row["architecture"], "test architecture")
    same(train["training_protocol_sha256"], read(PLAN)["protocol_sha256"], "plan protocol hash")
    same(train["status"], "training_completed_best_complete_epoch_checkpoint_fixed_no_test_evaluation", "training status")
    same(frozen["status"], "checkpoint_fixed_and_validation_roundtrip_passed", "freeze status")
    same(fresh["status"], "passed", "fresh validation status")
    same(test["status"], "completed", "test status")
    same(sentinel["status"], "started_irrevocably", "sentinel")
    same(frozen["test_performance_passes_so_far"], 0, "frozen pass count")
    same(test["test_performance_passes"], 1, "test pass count")
    same(test["raw_predictions_clipped"], False, "raw output clipping")
    same(train["partial_epoch_selected"], False, "partial selection")
    same(train["data_value_access"]["test"], {}, "training test-value access")
    same(fresh["data_value_access"]["test"], {}, "fresh validation test-value access")
    same(fresh["test_values_used"], False, "fresh validation test-value flag")
    same(test["cases"], meta["split"]["test"], "test case order")
    same(test["data_value_access"]["test"], {case: 1 for case in test["cases"]}, "one target read per test case")
    same(test["field_names"], meta["field_names"], "fields")
    same(test["field_units"], meta["field_units"], "units")
    same(test["time_values"], meta["time_values"], "time order")
    same(train["contract"]["manifest_sha256"], row["manifest_sha256"], "contract manifest")
    same(train["contract"]["split"], meta["split"], "contract split")
    same(train["contract"]["fields"], meta["field_names"], "contract fields")
    same(train["contract"]["times"], meta["time_values"], "contract times")
    same(stats, read(row["normalization"]), "shared normalization")
    same(stats, train["normalization"], "training normalization")
    same(sha(run/"training_report.json"), frozen["training_report_sha256"], "frozen report")
    same(objsha(stats), frozen["normalization_sha256"], "frozen normalization")
    same(objsha(train["contract"]), frozen["contract_sha256"], "frozen contract")
    same(objsha(train["model_config"]), frozen["best_model_config_sha256"], "frozen model config")
    same(sha(run/"validation_reference.npy"), frozen["validation_reference_sha256"], "validation reference")
    same(sha(run/"prospective-training-protocol.json"), train["training_protocol_sha256"], "training protocol")
    same(train["training_protocol_sha256"], frozen["training_protocol_sha256"], "frozen protocol")
    same(protocol["training"]["synchronized_optimizer_compute_budget_seconds"], 240, "declared cap")
    checkpoint_hash = sha(run/"best_checkpoint.pt")
    for record in (train, frozen, fresh, test, sentinel):
        same(record["checkpoint_sha256"], checkpoint_hash, "same checkpoint across phases")
    same(sentinel["validation_report_sha256"], sha(run/"validation_fresh_process.json"), "sentinel validation hash")
    training_identity = {"pid": train["training_process_pid"], "created_unix_seconds": train["training_process_created_epoch"]}
    assert training_identity != fresh["process_identity"] != test["process_identity"] != training_identity
    same(sentinel["process_identity"], test["process_identity"], "sentinel process identity")
    assert frozen["frozen_utc"] <= fresh["completed_utc"] <= sentinel["utc"] <= test["completed_utc"], "phase chronology"
    for record in (fresh, test):
        same(record["source_sha256"], frozen["source_sha256"], "frozen source inventory")
        same(record["quality_gates"], train["quality_gates"], "quality disclosure")
        for path, expected in record["frozen_evidence_sha256"].items():
            same(sha(path), expected, "frozen evidence "+path)
    same(train["source_sha256"], frozen["source_sha256"], "source inventory")
    for name, expected in frozen["source_sha256"].items():
        same(sha(run/"source"/name), expected, "archived source "+name)
    for filename, expected in test["output_sha256"].items():
        same(sha(run/"heldout"/filename), expected, "saved output "+filename)
    for record in (train["validation_roundtrip"], read(run/"validation_serialization_roundtrip.json"), fresh):
        assert record["max_scale_normalized_difference"] <= record["tolerance_scale_normalized"] <= 1e-5
        same(record["test_values_used"], False, "roundtrip test value use")
    candidates = [v for v in train["validation_history"] if v["eligible_complete_epoch"]]
    best = min(candidates, key=lambda v: v["validation_normalized_mse"])
    same(train["best_epoch"], best["epoch"], "validation-only best epoch")
    assert best["epoch"] >= 1
    same(train["best_validation_normalized_mse"], best["validation_normalized_mse"], "best validation loss")
    same(train["selected_epoch_accounting"], best["state_accounting"], "selected state accounting")
    with (run/"update_history.csv").open(encoding="utf-8", newline="") as stream:
        updates = list(csv.DictReader(stream))
    total_seconds, total_samples, seen, by_epoch = 0., 0, {}, {}
    rng = random.Random(train["seed"])
    expected_cases = {}
    for index, item in enumerate(updates, 1):
        epoch, samples, seconds = int(item["epoch"]), int(item["batch_samples"]), float(item["update_seconds"])
        assert math.isfinite(seconds) and seconds > 0
        assert total_seconds < 240 and index <= 5000
        if epoch not in expected_cases:
            cases = list(meta["split"]["train"]); rng.shuffle(cases); expected_cases[epoch] = cases
        prior = seen.get(epoch, 0)
        batch_cases = json.loads(item["cases"])
        same(batch_cases, expected_cases[epoch][prior:prior+train["batch_size"]], "shuffled training exposure")
        same(samples, len(batch_cases), "batch samples")
        seen[epoch] = prior+samples
        total_seconds += seconds; total_samples += samples
        same(int(item["update"]), index, "update sequence")
        same(float(item["learning_rate"]), .001, "Adam constant LR")
        same(float(item["cumulative_train_seconds"]), total_seconds, "compute ledger")
        same(int(item["cumulative_samples"]), total_samples, "sample ledger")
        entry = by_epoch.setdefault(epoch, {"updates": 0, "samples": 0, "seconds": 0.})
        entry["updates"] += 1; entry["samples"] += samples; entry["seconds"] += seconds
    budget = train["budget_accounting"]
    same(len(updates), budget["total_updates"], "total updates")
    same(total_samples, budget["total_samples"], "total observations")
    same(total_seconds, budget["train_seconds"], "total optimizer compute")
    same(max(0., total_seconds-240), budget["budget_overshoot_seconds"], "last-update overshoot")
    for epoch in budget["completed_epochs"]:
        entry = by_epoch[epoch["epoch"]]
        same(entry["samples"], len(meta["split"]["train"]), "complete epoch observations")
        same(entry["updates"], epoch["updates"], "complete epoch updates")
        same(entry["seconds"], epoch["train_seconds"], "complete epoch compute")
    complete_numbers = [entry["epoch"] for entry in budget["completed_epochs"]]
    same(complete_numbers, list(range(1, len(complete_numbers)+1)), "complete epoch sequence")
    assert train["best_epoch"] in complete_numbers, "selected epoch must be complete"
    selected_ledger = [item for item in updates if int(item["epoch"]) <= train["best_epoch"]]
    same(train["selected_epoch_accounting"]["optimizer_updates"], len(selected_ledger), "selected update ledger")
    same(train["selected_epoch_accounting"]["samples"], sum(int(item["batch_samples"]) for item in selected_ledger), "selected sample ledger")
    same(train["selected_epoch_accounting"]["train_seconds"], sum(float(item["update_seconds"]) for item in selected_ledger), "selected compute ledger")
    unfinished = [item for item in updates if int(item["epoch"]) not in complete_numbers]
    same(budget["partial_epoch"]["updates"], len(unfinished), "partial update ledger")
    same(budget["partial_epoch"]["samples"], sum(int(item["batch_samples"]) for item in unfinished), "partial sample ledger")
    same(budget["partial_epoch"]["train_seconds"], sum(float(item["update_seconds"]) for item in unfinished), "partial compute ledger")
    assert total_seconds >= 240 or len(updates) == 5000, "declared stopping boundary not reached"
    complete_count = len(budget["completed_epochs"])
    quality = len(updates) >= 100 and complete_count >= 2
    same(train["quality_gates"]["minimum_updates_satisfied"], len(updates) >= 100, "update quality gate")
    same(train["quality_gates"]["minimum_two_complete_epochs_satisfied"], complete_count >= 2, "epoch quality gate")
    same(train["quality_gates"]["gate_passed"], quality, "combined quality gate")
    assert complete_count >= 1
    return train, test, stats, {"integrity": "passed", "quality_budget": "satisfied" if quality else "insufficient_budget",
        "optimizer_updates": len(updates), "case_observations": total_samples, "complete_epochs": complete_count,
        "optimizer_compute_seconds": total_seconds, "last_update_overshoot_seconds": max(0., total_seconds-240),
        "best_epoch": train["best_epoch"], "last_validation_improved": train["last_validation_improved"],
        "convergence": "unverified", "timing_isolated": False,
        "phase_timing": {"training_wall_seconds": train["total_wall_seconds"], "training_excluded_timers": train["separate_timers"],
                         "setup_seconds_including_framework_import": train["setup_seconds_including_framework_import"],
                         "setup_components": train["setup_components"],
                         "source_preparation_seconds": train["source_preparation_seconds"],
                         "monitoring_seconds": train["workloads"]["monitoring_seconds"],
                         "monitoring_note": "Process-enumeration/host monitoring overhead is outside optimizer compute; setup-boundary samples can overlap setup timing. These components must not be assumed mutually exclusive.",
                         "test_wall_seconds": test["total_wall_seconds"], "test_forward_seconds": test["forward_seconds_cuda_synchronized_total"]},
        "resources": {"training_peak_sampled_rss_bytes_lower_bound": train["peak_sampled_rss_bytes_lower_bound"],
                      "training_cuda_allocator_peak_bytes": train["training_cuda_allocator_peak_bytes"],
                      "workload_observed_processes": train["workloads"]["observed_processes"]}}


def chunks(nodes):
    for start in range(0, nodes, CHUNK):
        memory()
        yield slice(start, min(start+CHUNK, nodes))


def data_sha(array):
    digest = hashlib.sha256()
    flat = array.reshape(-1)
    for start in range(0, flat.size, 65536):
        memory()
        digest.update(flat[start:start+65536].tobytes())
    return digest.hexdigest()


def training_normalization(row, meta, stats, contract):
    cache = Path(row["cache"]).parent
    record = domain_records[row["cache"]]
    norm_pin = {"normalization_object_sha256": objsha(stats), "coordinate_sha256": contract["coordinate_sha256"],
        "edge_sha256": contract["edge_sha256"]}
    if record.get("normalization_pin") == norm_pin and "train_only_normalization" in record:
        return {**record["train_only_normalization"], "reused_from_unchanged_domain_audit": True}
    for key, expected in (("coordinates", "coordinate_sha256"), ("edge_index", "edge_sha256")):
        array = np.load(cache/meta[key], mmap_mode="r")
        try:
            same(data_sha(array), contract[expected], "array order hash "+key)
        finally:
            close(array)
    count = 0
    mean, m2 = np.zeros(len(meta["field_names"])), np.zeros(len(meta["field_names"]))
    for case in meta["split"]["train"]:
        array = np.load(cache/meta["runs"][case]["target"], mmap_mode="r")
        try:
            for cut in chunks(array.shape[0]):
                values = array[cut].astype(np.float64)
                n = values.shape[0]*values.shape[1]
                local_mean, local_m2 = values.mean(axis=(0, 1)), values.var(axis=(0, 1))*n
                delta = local_mean-mean
                m2 += local_m2 + delta*delta*count*n/(count+n)
                mean += delta*n/(count+n)
                count += n
        finally:
            close(array)
    std = np.sqrt(m2/count)
    std[std == 0] = 1.
    std[:3] = math.sqrt(float(m2[:3].sum()/count/3)) or 1.
    for index, (actual, claimed) in enumerate(zip(mean, stats["field_mean"])):
        same(float(actual), claimed, f"independent train-only field mean {index}")
    for index, (actual, claimed) in enumerate(zip(std, stats["field_std"])):
        same(float(actual), claimed, f"independent train-only field std {index}")
    result = {"field_mean": mean.tolist(), "field_std": std.tolist(), "node_time_observations": count,
            "all_displacement_components_share_train_covariance_scale": True, "coordinate_and_graph_array_order_hashes_match": True}
    record.update({"normalization_pin": norm_pin, "train_only_normalization": result})
    Path(__file__).with_name("domain-cache-"+row["domain"]+".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {**result, "reused_from_unchanged_domain_audit": False}


def compare_metrics(actual, claimed, path="metric"):
    for key, value in actual.items():
        if key not in claimed:
            assert path == "displacement vector" and key in ("centroid_removed_sum_squared_error", "centroid_removed_truth_sum_square", "truth_centroid_translation_fraction"), "missing required claimed metric: "+path+"."+key
            continue
        if isinstance(value, dict):
            compare_metrics(value, claimed[key], path+"."+key)
        elif isinstance(value, list):
            same(len(value), len(claimed[key]), path+"."+key+" length")
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    compare_metrics(item, claimed[key][index], path+f".{key}[{index}]")
                else:
                    same(claimed[key][index], item, path+f".{key}[{index}]")
        else:
            same(claimed[key], value, path+"."+key)


def scalar_metrics(p, t, channel, nonnegative, case_ids=None, time_values=None):
    cases, nodes, times, fields = p.shape
    errors = np.empty(cases*nodes*times, dtype=np.float64)
    sse = den = 0.; nonzero = pneg = pnfinite = tnfinite = 0
    low, high = math.inf, -math.inf
    per_time = np.zeros(times, dtype=np.float64)
    per_case, offset = [], 0
    for case in range(cases):
        csse = cden = 0.
        truth_peak = predicted_peak = -1.
        truth_location = predicted_location = None
        for cut in chunks(nodes):
            pv, tv = p[case, cut, :, channel].astype(np.float64), t[case, cut, :, channel].astype(np.float64)
            pnfinite += int((~np.isfinite(pv)).sum()); tnfinite += int((~np.isfinite(tv)).sum())
            assert np.isfinite(pv).all() and np.isfinite(tv).all(), "nonfinite saved output or FEM"
            error = pv-tv; square = error*error
            count = error.size; errors[offset:offset+count] = np.abs(error).ravel(); offset += count
            csse += float(square.sum()); cden += float((tv*tv).sum())
            per_time += square.sum(axis=0)
            nonzero += int(np.count_nonzero(tv)); pneg += int((pv < 0).sum())
            low = min(low, float(pv.min())); high = max(high, float(pv.max()))
            ti, pi = int(np.abs(tv).argmax()), int(np.abs(pv).argmax())
            local_tpeak, local_ppeak = float(np.abs(tv).flat[ti]), float(np.abs(pv).flat[pi])
            if local_tpeak > truth_peak:
                truth_peak, truth_location = local_tpeak, (cut.start+ti//times, ti%times)
            if local_ppeak > predicted_peak:
                predicted_peak, predicted_location = local_ppeak, (cut.start+pi//times, pi%times)
        sse += csse; den += cden
        row = {"sse": csse, "truth_sum_square": cden, "count": nodes*times,
            "truth_absolute_peak": truth_peak, "prediction_absolute_peak": predicted_peak,
            "absolute_peak_error": abs(predicted_peak-truth_peak),
            "truth_peak_node_index": truth_location[0], "truth_peak_time_index": truth_location[1],
            "prediction_at_truth_peak": float(p[case, truth_location[0], truth_location[1], channel]),
            "absolute_error_at_truth_peak": float(abs(float(p[case, truth_location[0], truth_location[1], channel])-float(t[case, truth_location[0], truth_location[1], channel]))),
            "predicted_peak_node_index": predicted_location[0], "predicted_peak_time_index": predicted_location[1]}
        if case_ids is not None: row["case"] = case_ids[case]
        if time_values is not None: row["truth_peak_time"] = time_values[truth_location[1]]
        per_case.append(row)
    maximum = float(errors.max())
    p95 = float(np.quantile(errors, .95, overwrite_input=True))
    del errors
    return {"sum_squared_error": sse, "truth_sum_square": den, "truth_l2_norm": math.sqrt(den),
        "relative_l2": math.sqrt(sse/den) if den else None, "rmse": math.sqrt(sse/(cases*nodes*times)),
        "absolute_error_p95": p95, "absolute_error_max": maximum, "truth_exact_nonzero_count": nonzero,
        "count": cases*nodes*times, "prediction_nonfinite_count": pnfinite, "truth_nonfinite_count": tnfinite,
        "prediction_negative_count_for_nonnegative_field": pneg if nonnegative else None,
        "prediction_min": low, "prediction_max": high, "per_timestep_rmse": np.sqrt(per_time/(cases*nodes)).tolist(), "per_case": per_case}


def vector_metrics(p, t):
    cases, nodes, times, _ = p.shape
    norms = np.empty(cases*nodes*times, dtype=np.float64)
    sse = den = centered_sse = centered_den = 0.; offset = 0
    for case in range(cases):
        pmean, tmean = np.zeros((times, 3)), np.zeros((times, 3))
        for cut in chunks(nodes):
            pmean += p[case, cut, :, :3].sum(axis=0, dtype=np.float64)
            tmean += t[case, cut, :, :3].sum(axis=0, dtype=np.float64)
        pmean /= nodes; tmean /= nodes
        for cut in chunks(nodes):
            pv, tv = p[case, cut, :, :3].astype(np.float64), t[case, cut, :, :3].astype(np.float64)
            square = (pv-tv)**2
            sse += float(square.sum()); den += float((tv*tv).sum())
            centered_sse += float(((pv-pmean-tv+tmean)**2).sum())
            centered_den += float(((tv-tmean)**2).sum())
            local = np.sqrt(square.sum(axis=-1)); count = local.size
            norms[offset:offset+count] = local.ravel(); offset += count
    maximum, p95 = float(norms.max()), float(np.quantile(norms, .95, overwrite_input=True))
    del norms
    return {"sum_squared_error": sse, "truth_sum_square": den, "relative_l2": math.sqrt(sse/den) if den else None,
        "component_rmse": math.sqrt(sse/(cases*nodes*times*3)), "vector_rmse": math.sqrt(sse/(cases*nodes*times)),
        "vector_error_p95": p95, "vector_error_max": maximum,
        "centroid_removed_relative_l2": math.sqrt(centered_sse/centered_den) if centered_den else None,
        "centroid_removed_sum_squared_error": centered_sse, "centroid_removed_truth_sum_square": centered_den,
        "truth_centroid_translation_fraction": 1-centered_den/den if den else None}


def history(p, channel):
    cases, nodes, times, _ = p.shape
    decreases = np.empty(cases*nodes*(times-1), dtype=np.float64)
    low, high, negative, zero, offset = math.inf, -math.inf, 0, 0, 0
    for case in range(cases):
        for cut in chunks(nodes):
            delta = np.diff(p[case, cut, :, channel].astype(np.float64), axis=1)
            value = -delta[delta < 0]; count = value.size
            decreases[offset:offset+count] = value; offset += count
            negative += count; zero += int((delta == 0).sum())
            low = min(low, float(delta.min())); high = max(high, float(delta.max()))
    quantiles = {str(q): float(np.quantile(decreases[:offset], q)) for q in (.5, .9, .95, .99, 1.)} if offset else None
    result = {"minimum_delta": low, "maximum_delta": high, "negative_count": negative, "zero_count": zero,
        "count": cases*nodes*(times-1), "maximum_decrease": float(decreases[:offset].max()) if offset else 0.,
        "decrease_magnitude_quantiles": quantiles}
    del decreases
    return result


def epsp_metrics(p, t, channel):
    cases, nodes, times, _ = p.shape
    tp = fp = fn = tn = pos = zeros = negative = pnegative = 0
    pos_sse = zero_sse = zero_peak = 0.
    tcount, pcount = np.zeros(times, dtype=np.int64), np.zeros(times, dtype=np.int64)
    for case in range(cases):
        for cut in chunks(nodes):
            pv, tv = p[case, cut, :, channel].astype(np.float64), t[case, cut, :, channel].astype(np.float64)
            reference, predicted, zero = tv > 0, pv > 0, tv == 0
            tp += int((reference & predicted).sum()); fp += int((~reference & predicted).sum())
            fn += int((reference & ~predicted).sum()); tn += int((~reference & ~predicted).sum())
            pos += int(reference.sum()); zeros += int(zero.sum())
            negative += int((tv < 0).sum()); pnegative += int((pv < 0).sum())
            error = pv-tv
            pos_sse += float((error[reference]**2).sum()); zero_sse += float((error[zero]**2).sum())
            if zero.any(): zero_peak = max(zero_peak, float(np.abs(pv[zero]).max()))
            tcount += reference.sum(axis=0); pcount += predicted.sum(axis=0)
    return {"truth_exact_positive_count": pos, "truth_exact_zero_count": zeros, "truth_negative_count": negative,
        "prediction_negative_count": pnegative, "truth_positive_support_per_timestep": tcount.tolist(),
        "prediction_positive_support_per_timestep": pcount.tolist(),
        "rmse_on_reference_positive_support": math.sqrt(pos_sse/pos) if pos else None,
        "rmse_on_reference_exact_zero_support": math.sqrt(zero_sse/zeros) if zeros else None,
        "prediction_absolute_peak_on_reference_exact_zero_support": zero_peak if zeros else None,
        "support": [{"threshold": 0., "reference_count": pos, "prediction_count": tp+fp, "true_positive": tp,
            "false_positive": fp, "false_negative": fn, "true_negative": tn, "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None, "iou": tp/(tp+fp+fn) if tp+fp+fn else None}],
        "history": {"prediction": history(p, channel), "truth": history(t, channel)}}


def phone_cards(path):
    """Read only the numeric rows of the source initial-condition cards."""
    velocities, gravity, points, current = [], {}, [], None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            value = line.strip()
            if value.startswith("/"):
                current = value
                continue
            if not value or value.startswith("#"):
                continue
            fields = value.split()
            if current and current.startswith("/INIVEL/TRA/") and len(fields) == 5:
                try:
                    row = [float(x) for x in fields]
                except ValueError:
                    continue
                assert row[4] == 0 and row[3].is_integer(), "unsupported initial velocity frame/group"
                velocities.append((row[:3], int(row[3])))
            elif current and current.startswith("/GRAV/") and len(fields) == 7 and fields[1] in ("X", "Y", "Z"):
                assert fields[0] == "2" and fields[2:4] == ["0", "0"] and float(fields[5]) == 0
                assert fields[1] not in gravity
                gravity[fields[1]] = (float(fields[6]), int(fields[4]))
            elif current == "/FUNCT/2" and len(fields) == 2:
                try:
                    points.append([float(x) for x in fields])
                except ValueError:
                    continue
    assert len(velocities) == 1 and set(gravity) == set("XYZ")
    assert len(set(gravity[x][1] for x in "XYZ")) == 1 and len(points) >= 2
    assert all(point[1] == 1. for point in points), "gravity function must be constant one"
    return {"velocity_mm_per_s": velocities[0][0], "velocity_node_group": velocities[0][1],
        "acceleration_mm_per_s_squared": [gravity[x][0] for x in "XYZ"], "gravity_node_group": gravity["X"][1],
        "constant_gravity_function_domain_s": [min(point[0] for point in points), max(point[0] for point in points)]}


def phone_baseline(row, meta, test, truth, model_vector):
    if row["domain"] != "phone":
        return None
    cache = Path(row["cache"]).parent
    manifest_path = cache/meta["source_case_manifest"]
    source = {item["run_id"]: item for item in read(manifest_path)["runs"]}
    claimed = test["phone_ballistic_reference"]
    factor = {"s": 1., "ms": 1e-3, "us": 1e-6}[meta["time_unit"]]
    same(factor, claimed["time_conversion_to_seconds"], "baseline time conversion")
    times = np.asarray(meta["time_values"], dtype=np.float64)*factor
    pure = np.empty((len(test["cases"]), 1, len(times), 3), dtype=np.float64)
    correction = np.empty_like(pure)
    evidence, card_cache = [], domain_records[row["cache"]].setdefault("ballistic_source_cards", {})
    for index, case in enumerate(test["cases"]):
        source_row = source[case]
        path = Path(claimed["case_initial_condition_evidence"][index]["starter_path"])
        same(path.parent.resolve(), (ROOT/source_row["raw_run_dir"]).resolve(), "source deck directory")
        stat = path.stat()
        signature = {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "expected_sha256": source_row["starter_sha256"]}
        cached = card_cache.get(case, {})
        if cached.get("signature") == signature:
            cards = cached["cards"]
        else:
            same(sha(path), source_row["starter_sha256"], "baseline source starter hash")
            cards = phone_cards(path)
            card_cache[case] = {"signature": signature, "cards": cards}
        for key, value in cards.items():
            same(value, claimed["case_initial_condition_evidence"][index][key], "source baseline card "+key)
        low, high = cards["constant_gravity_function_domain_s"]
        assert low <= times.min() <= times.max() <= high
        pure[index, 0] = times[:, None]*np.asarray(cards["velocity_mm_per_s"])
        correction[index, 0] = .5*times[:, None]**2*np.asarray(cards["acceleration_mm_per_s_squared"])
        evidence.append({"case": case, **signature, **cards})
    Path(__file__).with_name("domain-cache-phone.json").write_text(json.dumps(domain_records[row["cache"]], indent=2), encoding="utf-8")
    shape = (*truth.shape[:3], 3)
    references = {"v0_times_nominal_t": pure, "v0_times_nominal_t_plus_constant_gravity": pure+correction,
        "constant_gravity_only": correction}
    metrics = {key: vector_metrics(np.broadcast_to(value, shape), truth) for key, value in references.items()}
    compare_metrics(metrics["v0_times_nominal_t_plus_constant_gravity"], claimed["ballistic_physical_metrics"]["displacement_vector"], "displacement vector")
    compare_metrics(metrics["constant_gravity_only"], claimed["gravity_only_physical_metrics"]["displacement_vector"], "displacement vector")
    ratio = model_vector["sum_squared_error"]/metrics["v0_times_nominal_t_plus_constant_gravity"]["sum_squared_error"]
    same(ratio, claimed["model_displacement_sse_over_ballistic_sse"], "model over baseline displacement SSE")
    return {"status": "passed", "source_cards": evidence, "time_conversion_to_seconds": factor,
        "nominal_times_seconds": times.tolist(), "displacement_vector_metrics": metrics,
        "model_displacement_sse_over_ballistic_sse": ratio,
        "production_metric_paths": {"gravity_included": "phone_ballistic_reference.ballistic_physical_metrics.displacement_vector",
            "gravity_only": "phone_ballistic_reference.gravity_only_physical_metrics.displacement_vector",
            "pure_v0_times_t": "Independent supplemental comparator; no production metric for this exact gravity-free formula"},
        "scope": "Independent raw-deck v0*t and v0*t+0.5*a*t^2 uniform contact-free displacement vector metrics on saved complete support. No contact/deformation/constraint model; scalar baseline channels/figures are not separately audited."}


def baseline_only(row, meta, existing):
    """Supplement a preserved successful field audit without repeating its metrics."""
    global np
    array_headroom()
    import numpy as np
    memory()
    run = Path(row["out"])
    test = read(run/"test_report.json")
    same(test["status"], "completed", "supplement completed source")
    same(test["cases"], meta["split"]["test"], "supplement case order")
    for name in ("predictions.npy", "truth.npy"):
        same(sha(run/"heldout"/name), test["output_sha256"][name], "supplement saved array hash "+name)
    p, t = np.load(run/"heldout/predictions.npy", mmap_mode="r"), np.load(run/"heldout/truth.npy", mmap_mode="r")
    try:
        same(list(p.shape), test["shape"], "supplement prediction shape")
        same(list(t.shape), test["shape"], "supplement truth shape")
        sse = den = 0.
        for case in range(len(test["cases"])):
            for cut in chunks(p.shape[1]):
                pv, tv = p[case, cut, :, :3].astype(np.float64), t[case, cut, :, :3].astype(np.float64)
                sse += float(((pv-tv)**2).sum()); den += float((tv*tv).sum())
        original = existing["independent_recomputation"]["metrics"]["displacement_vector"]
        same(sse, original["sum_squared_error"], "preserved original independent model SSE")
        same(den, original["truth_sum_square"], "preserved original truth norm")
        reference = phone_baseline(row, meta, test, t, {"sum_squared_error": sse})
        memory()
        return reference, {"status": "passed", "audited_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Supplemental source-card baseline displacement-vector audit only; original successful field metrics and ledger are preserved",
            "saved_model_displacement_sse_rechecked": sse, "peak_RSS_bytes": peak_rss,
            "minimum_host_available_bytes": minimum_free, "script_sha256": sha(__file__)}
    finally:
        close(p); close(t)


def arrays(row, meta, train, test, stats):
    global np
    array_headroom()
    import numpy as np
    memory()
    norm_audit = training_normalization(row, meta, stats, train["contract"])
    run, cache = Path(row["out"]), Path(row["cache"]).parent
    p, t = np.load(run/"heldout/predictions.npy", mmap_mode="r"), np.load(run/"heldout/truth.npy", mmap_mode="r")
    try:
        same(list(p.shape), test["shape"], "prediction shape")
        same(list(t.shape), test["shape"], "truth shape")
        same(list(p.shape), [len(test["cases"]), *train["contract"]["shape"]], "complete support shape")
        same(str(p.dtype), "float32", "prediction dtype"); same(str(t.dtype), "float32", "truth dtype")
        mean, std = np.asarray(stats["field_mean"], dtype=np.float32), np.asarray(stats["field_std"], dtype=np.float32)
        inverse_bitwise, inverse_ulp = True, 0
        inverse_abs, inverse_normalized = np.zeros(len(mean)), np.zeros(len(mean))
        for index, case in enumerate(test["cases"]):
            original = np.load(cache/meta["runs"][case]["target"], mmap_mode="r")
            normalized = np.load(run/"heldout"/(case+".normalized.npy"), mmap_mode="r")
            physical = np.load(run/"heldout"/(case+".physical.npy"), mmap_mode="r")
            try:
                for cut in chunks(p.shape[1]):
                    assert np.array_equal(t[index, cut], original[cut]), "saved truth differs from original FEM / node order"
                    assert np.array_equal(p[index, cut], physical[cut]), "aggregate differs from raw physical prediction"
                    expected = normalized[cut]*std+mean
                    stored = physical[cut]
                    same(str(expected.dtype), "float32", "CPU inverse dtype")
                    same(str(stored.dtype), "float32", "saved physical dtype")
                    difference = np.abs(stored.astype(np.float64)-expected)
                    inverse_abs = np.maximum(inverse_abs, difference.max(axis=(0, 1)))
                    inverse_normalized = np.maximum(inverse_normalized, (difference/std).max(axis=(0, 1)))
                    inverse_bitwise = inverse_bitwise and np.array_equal(stored, expected)
                    a, b = stored.view(np.uint32), expected.view(np.uint32)
                    a = np.where(a & 0x80000000, ~a, a | 0x80000000).astype(np.int64)
                    b = np.where(b & 0x80000000, ~b, b | 0x80000000).astype(np.int64)
                    inverse_ulp = max(inverse_ulp, int(np.abs(a-b).max()))
                    assert inverse_ulp <= 1, "inverse normalization differs by more than one float32 ULP"
            finally:
                close(original); close(normalized); close(physical)
        metrics, epsp = {}, {}
        for channel, name in enumerate(meta["field_names"]):
            metrics[name] = scalar_metrics(p, t, channel, any(token in name.lower() for token in ("vm", "von_mises", "epsp", "plastic", "frobenius", "magnitude")), test["cases"], meta["time_values"])
            compare_metrics(metrics[name], test["physical_metrics"][name], name)
            if any(token in name.lower() for token in ("epsp", "plastic")):
                epsp[name] = epsp_metrics(p, t, channel)
                compare_metrics(epsp[name], test["epsp_diagnostics"][name], name+" EPSP")
        metrics["displacement_vector"] = vector_metrics(p, t)
        compare_metrics(metrics["displacement_vector"], test["physical_metrics"]["displacement_vector"], "displacement vector")
        ballistic = phone_baseline(row, meta, test, t, metrics["displacement_vector"])
        memory()
        return {"status": "passed", "saved_truth_matches_original_FEM_bitwise": True,
            "raw_physical_aggregate_matches_bitwise": True, "metrics": metrics, "epsp": epsp,
            "inverse_normalization": {"bitwise_equal": inverse_bitwise, "maximum_float32_ULP_difference": inverse_ulp,
                "maximum_physical_absolute_difference_per_field": inverse_abs.tolist(),
                "maximum_scale_normalized_difference_per_field": inverse_normalized.tolist(),
                "formula": "float32 normalized * float32 field_std + float32 field_mean, saved node/time/channel order; at most one ULP roundoff is distinguished from an actual mismatch"},
            "train_only_normalization": norm_audit,
            "relative_l2_zero_denominator_policy": "null; physical RMSE remains reported", "raw_output_clipping": False,
            "phone_ballistic_reference": ballistic,
            "scope_exclusion": "Model figures and scalar baseline channels/figures are not separately audited. Baseline displacement vector metrics are audited where present."}
    finally:
        close(p); close(t)


def self_check():
    global np
    memory()
    import numpy as np
    t = np.ones((2, 3, 4, 1), dtype=np.float32)
    m = scalar_metrics(2*t, t, 0, False)
    assert m["rmse"] == m["relative_l2"] == m["absolute_error_p95"] == 1.
    z = np.zeros_like(t)
    assert scalar_metrics(z, z, 0, False)["relative_l2"] is None
    rigid = np.ones((1, 3, 2, 3), dtype=np.float32)
    assert vector_metrics(rigid, rigid)["centroid_removed_relative_l2"] is None
    assert history(rigid, 0)["negative_count"] == 0
    damaged = np.array([0., 1., .5], dtype=np.float32).reshape(1, 1, 3, 1)
    assert history(damaged, 0)["negative_count"] == 1
    assert history(damaged, 0)["maximum_decrease"] == .5
    print("independent CPU metric self-check passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--domain")
    parser.add_argument("--architecture")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    plan = read(PLAN)
    same(sha(plan["protocol"]), plan["protocol_sha256"], "prospective plan protocol")
    for path, expected in plan["source_sha256"].items():
        same(sha(path), expected, "prospective plan source "+path)
    previous = read(OUT) if OUT.exists() else {"models": {}}
    results, inventories = previous.get("models", {}), {}
    backup = None
    if args.baseline_only:
        assert args.domain == "phone" and args.architecture
        existing_backups = [item.get("original_passed_report_backup") for item in results[args.domain+"/"+args.architecture].get("supplementary_audits", []) if item.get("original_passed_report_backup")]
        if existing_backups:
            backup = Path(existing_backups[0])
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            backup = Path(__file__).with_name("review-before-"+args.architecture+"-baseline-"+stamp+".json")
            backup.write_text(OUT.read_text(encoding="utf-8"), encoding="utf-8")
            if OUT.with_suffix(".md").exists():
                backup.with_suffix(".md").write_text(OUT.with_suffix(".md").read_text(encoding="utf-8"), encoding="utf-8")
    for row in plan["runs"]:
        if args.domain and row["domain"] != args.domain: continue
        if args.architecture and row["architecture"] != args.architecture: continue
        key = row["domain"]+"/"+row["architecture"]
        run = Path(row["out"])
        base = {"domain": row["domain"], "architecture": row["architecture"], "run": str(run),
                "audited_utc": datetime.now(timezone.utc).isoformat()}
        if not (run/"test_report.json").exists():
            results[key] = {**base, "status": "pending_actual_completed_test", "present_evidence": [p.name for p in run.iterdir() if p.is_file()] if run.exists() else []}
            continue
        if args.baseline_only:
            existing = results[key]
            assert existing["status"] in ("passed", "integrity_passed_insufficient_budget")
            meta, provenance = inventory(row)
            supplement = {"status": "deferred_memory", "original_passed_report_backup": str(backup)}
            try:
                reference, supplement = baseline_only(row, meta, existing)
                supplement["original_passed_report_backup"] = str(backup)
                existing["independent_recomputation"]["phone_ballistic_reference"] = reference
                existing["independent_recomputation"]["scope_exclusion"] = "Model figures and scalar baseline channels/figures are not separately audited. Supplemental source-card baseline vector metrics are passed."
            except DeferredMemory as error:
                supplement["memory_deferral"] = str(error)
            history = existing.setdefault("supplementary_audits", [])
            if history and history[-1]["status"] == supplement["status"] == "deferred_memory":
                history[-1] = supplement
            else:
                history.append(supplement)
            print(json.dumps({"model": key, "baseline_supplement": supplement["status"]}), flush=True)
            continue
        try:
            if row["cache"] not in inventories:
                inventories[row["cache"]] = inventory(row)
            meta, provenance = inventories[row["cache"]]
            train, test, stats, gates = metadata(row, meta)
            result = {**base, "status": "metadata_passed_metrics_pending", "provenance": provenance, "gates": gates}
            try:
                result["independent_recomputation"] = arrays(row, meta, train, test, stats)
                result["status"] = "passed" if gates["quality_budget"] == "satisfied" else "integrity_passed_insufficient_budget"
            except DeferredMemory as error:
                result["status"] = "metadata_passed_metrics_deferred_memory"
                result["memory_deferral"] = str(error)
            result["auditor_resources"] = {"peak_RSS_bytes": peak_rss, "minimum_host_available_bytes": minimum_free,
                "RSS_limit_bytes": RSS_CAP, "minimum_host_free_bytes": FREE_MIN, "threads": 1, "GPU_used": False}
            results[key] = result
        except Exception as error:
            results[key] = {**base, "status": "failed", "error": repr(error)}
        print(json.dumps({"model": key, "status": results[key]["status"]}), flush=True)
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(), "scope": "Only prospective phone full C6 and bumper C7 supervised GPU results; saved-array CPU audit, no model execution",
        "status": "complete" if len(results) == len(plan["runs"]) and all(r["status"] in ("passed", "integrity_passed_insufficient_budget") and not (r.get("supplementary_audits") and r["supplementary_audits"][-1]["status"] == "deferred_memory") for r in results.values()) else "in_progress_or_findings",
        "auditor": {"framework_imports": False, "GPU_used": False, "threads": 1, "RSS_limit_bytes": RSS_CAP, "minimum_host_free_bytes": FREE_MIN,
            "array_startup_free_threshold_bytes_including_audit_reserve": FREE_MIN+RSS_CAP,
            "peak_observed_RSS_bytes": peak_rss, "minimum_observed_host_free_bytes": minimum_free, "script_sha256": sha(__file__)},
        "interpretation": ["Quality budget and integrity are separate. At least one complete frozen epoch permits exactly one held-out pass; fewer than 100 updates or two complete epochs is insufficient_budget.",
            "All-split file hashing verifies provenance and does not itself use held-out response values to select weights.",
            "Raw values are not clipped; negative nominally nonnegative fields and EPSP history decreases are reported.",
            "Phone 2..20us displacement is dominated by common translation. Centroid-removed error and tiny EPSP support must accompany scalar/vector scores.",
            "Phone GPStress-derived VM and projected total-strain Frobenius are distinct fields; bumper upper/lower VM in GPa and EPSP are not total strain. GPa RMSE converts to MPa by multiplying 1000.",
            "Node- and timestep-weighted projected nodal metrics do not equal native-cell hotspot errors; near-zero component relative percentages are not a ranking.",
            "Workstation compute times are concurrent-workload observations, and convergence remains unverified."], "models": results}
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lines = ["# Full-support 실행결과 독립 감사", "", f"갱신: {report['updated_utc']}", "", "저장된 raw 예측과 FEM 배열을 CPU 1개로 독립 검산한다. 아직 완료되지 않은 실행은 pending이며 통과로 세지 않는다. passed는 실행 계약·저장 결과·독립 재계산의 일치 판정이다. 공학적 예측 품질은 동명 JSON의 실제 오차와 기준선 비교로 별도 평가한다.", "", "|도메인 / 설정|감사 상태|업데이트 / 완료 epoch|선택 epoch|", "|---|---|---:|---:|"]
    for key, result in results.items():
        gates = result.get("gates", {})
        lines.append(f"|{key}|{result['status']}|{gates.get('optimizer_updates', '—')} / {gates.get('complete_epochs', '—')}|{gates.get('best_epoch', '—')}|")
        if result.get("error"): lines.extend(["", f"- {key}: `{result['error']}`"])
    for key, result in results.items():
        if result.get("supplementary_audits"):
            lines.extend(["", f"{key} 추가 기준선 감사: `{result['supplementary_audits'][-1]['status']}`. 최초 필드 감사의 passed 기록과 백업을 보존한 별도 추가 감사다."])
    references = [result.get("independent_recomputation", {}).get("phone_ballistic_reference") for result in results.values()]
    reference = next((item for item in references if item and item.get("status") == "passed"), None)
    if reference:
        values = reference["displacement_vector_metrics"]
        lines.extend(["", "원 RAD 초기속도와 nominal 저장 시점으로 계산한 접촉 없는 균일 이동 기준선도 변위 벡터 지표에 한해 독립 검산했다.", "",
            f"- 순수 `v0×t`: relative L2 `{values['v0_times_nominal_t']['relative_l2']:.14g}`.",
            f"- 중력 포함 `v0×t + 0.5×a×t²`: relative L2 `{values['v0_times_nominal_t_plus_constant_gravity']['relative_l2']:.14g}`.",
            "", "production의 정확한 중력 포함 지표 경로는 `phone_ballistic_reference.ballistic_physical_metrics.displacement_vector`이다. `gravity_only_physical_metrics.displacement_vector`는 초기속도를 제외한 중력만의 별도 기준선이다. 균일 이동 기준선은 접촉·변형·구속을 계산하지 않으며 scalar 기준선 지표와 그림은 별도 검산 범위에 포함하지 않는다."])
    lines.extend(["", "무결성과 학습량 충족을 분리한다. 100 updates / 완료 epoch 2개 미달은 insufficient_budget이며 수렴·알고리즘 우열의 근거로 쓰지 않는다. 완료 epoch가 1개 이상이고 checkpoint가 고정됐으면 단 한 번의 test는 허용된다.", "", "Phone full C6는 76,065 corner nodes, 10개 nominal 2..20us 시점, 5/2/2 split이다. 큰 강체 병진 비중 때문에 centroid 제거 오차를 함께 읽어야 한다. Bumper C7의 upper/lower VM 단위는 GPa(표시 MPa는 ×1000)이며 EPSP와 total strain을 혼동하지 않는다.", "", "Raw 음수/비유한 값, EPSP 양수 support와 시간 감소는 보정 없이 센다. 분모가 0인 relative L2는 null이며 RMSE는 유지한다. 투영된 nodal head와 native cell hotspot은 같은 필드·support가 아니다.", "", "검산 세부 수치와 해시는 동명의 JSON에 있다. GPU·Torch·PhysicsNeMo를 사용하지 않았고 예측이나 학습을 다시 실행하지 않았다. 미완료 결과의 정확도와 수렴은 확인하지 않았다."])
    OUT.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
