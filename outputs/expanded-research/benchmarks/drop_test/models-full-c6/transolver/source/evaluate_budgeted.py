"""Fresh-process validation and a single, sealed held-out performance pass.

Only standard-library modules are imported before GPU-slot authorization. Test
values are opened only after the exclusive sentinel has been durably created;
even a failed pass cannot be restarted by this executable.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

from budgeted_runtime import (TensorData, filehash, forward, load_runtime,
                              make_cache, objecthash, sync, verify_ready, write_json)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def freeze_snapshot(run, expected_manifest):
    """Check immutable evidence without loading arrays or a framework."""
    frozen = read_json(run/"checkpoint-frozen.json")
    report = read_json(run/"training_report.json")
    stats = read_json(run/"normalization.json")
    require(frozen["status"] == "checkpoint_fixed_and_validation_roundtrip_passed", "Checkpoint is not frozen")
    require(report["status"] == "training_completed_best_complete_epoch_checkpoint_fixed_no_test_evaluation", "Unexpected training report")
    require(frozen["test_performance_passes_so_far"] == 0, "Frozen checkpoint already records a test pass")
    require(frozen["training_report_sha256"] == filehash(run/"training_report.json"), "Training report changed")
    require(frozen["checkpoint_sha256"] == report["checkpoint_sha256"] == filehash(run/"best_checkpoint.pt"), "Checkpoint changed")
    require(frozen["validation_reference_sha256"] == filehash(run/"validation_reference.npy"), "Validation anchor changed")
    require(frozen["best_model_config_sha256"] == objecthash(report["model_config"]), "Model config changed")
    require(frozen["normalization_sha256"] == objecthash(stats) == objecthash(report["normalization"]), "Normalization changed")
    require(frozen["contract_sha256"] == objecthash(report["contract"]), "Data contract changed")
    require(report["contract"]["manifest_sha256"] == expected_manifest, "Unexpected data manifest")
    require(frozen["source_sha256"] == report["source_sha256"], "Source inventories disagree")
    require(frozen["quality_gates"] == report["quality_gates"], "Quality gates changed")
    require(report["partial_epoch_selected"] is False, "Partial epoch cannot be selected")
    require(not report["data_value_access"]["test"], "Training report records test response access")
    require(frozen["training_protocol_sha256"] == report["training_protocol_sha256"] == filehash(run/"prospective-training-protocol.json"), "Protocol changed")
    protocol = read_json(run/"prospective-training-protocol.json")
    require(protocol["training"]["synchronized_optimizer_compute_budget_seconds"] == 240, "Unexpected budget")
    require(protocol["training"]["maximum_optimizer_updates"] == 5000, "Unexpected update cap")
    paths = [run/name for name in ("checkpoint-frozen.json", "training_report.json", "best_checkpoint.pt",
             "validation_reference.npy", "validation_serialization_roundtrip.json", "normalization.json",
             "training_mean_field.npy", "prospective-training-protocol.json")]
    for name, digest in frozen["source_sha256"].items():
        archived = run/"source"/name
        require(filehash(archived) == digest, "Archived source changed: "+name)
        current = Path(__file__).with_name(name)
        if name == "batch_adapters.py" and not current.is_file():
            current = Path(__file__).resolve().parent.parent/"expanded-batch"/name
        require(current.is_file() and filehash(current) == digest, "Executing source changed: "+name)
        paths.extend([archived, current])
    require("evaluate_budgeted.py" in frozen["source_sha256"], "Evaluator was not frozen before training")
    implementation = Path(report["model_source"]["source_file"])
    require(filehash(implementation) == report["model_source"]["source_sha256"], "Installed model implementation changed")
    paths.append(implementation)
    if (run/"data-provenance-clarification.json").exists():
        paths.append(run/"data-provenance-clarification.json")
    return frozen, report, stats, {str(path.resolve()): filehash(path) for path in paths}


def verify_snapshot(snapshot):
    for path, digest in snapshot.items():
        require(filehash(path) == digest, "Frozen evidence changed during evaluation: "+path)


def process_identity(psutil):
    return {"pid": os.getpid(), "created_unix_seconds": psutil.Process().create_time()}


def fresh_from_training(identity, report):
    # A PID may be recycled; creation time makes the distinction unambiguous.
    training = report.get("training_process_identity")
    if training:
        require(identity != training, "Evaluation must run in a fresh process")
    elif "training_process_pid" in report:
        created = report.get("training_process_created_epoch", report.get("training_process_created_unix_seconds"))
        require(identity["pid"] != report["training_process_pid"] or
                (created is not None and identity["created_unix_seconds"] != created),
                "Evaluation must run in a fresh process")


def validate_checkpoint(payload, frozen, report, stats, cache, policy):
    for key in ("architecture", "model_config", "seed", "parameters", "best_epoch",
                "best_validation_normalized_mse", "source_sha256", "training_protocol_sha256",
                "selected_epoch_accounting", "batch_size"):
        require(objecthash(payload[key]) == objecthash(report[key]), "Checkpoint/report mismatch: "+key)
    require(payload["normalization"] == stats, "Checkpoint normalization mismatch")
    require(payload["contract"] == cache.contract == report["contract"], "Checkpoint data contract mismatch")
    require(objecthash(cache.contract) == frozen["contract_sha256"], "Frozen contract mismatch")
    require(payload["training_policy"] == policy, "Checkpoint policy mismatch")


def runtime_versions(runtime):
    versions = {"python": sys.version, "numpy": str(runtime.np.__version__),
                "torch": str(runtime.torch.__version__), "cuda": str(runtime.torch.version.cuda),
                "cudnn": runtime.torch.backends.cudnn.version()}
    for name in ("nvidia-physicsnemo", "torch-geometric", "torch-scatter", "warp-lang", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def delta_summary(np, value):
    decreases = -value[value < 0]
    return {"minimum_delta": float(value.min()), "maximum_delta": float(value.max()),
            "negative_count": int(decreases.size), "zero_count": int(np.count_nonzero(value == 0)),
            "count": int(value.size), "maximum_decrease": float(decreases.max()) if decreases.size else 0.,
            "decrease_magnitude_quantiles": {str(q): float(np.quantile(decreases, q))
                for q in (.5, .9, .95, .99, 1.)} if decreases.size else None}


def epsp_diagnostics(np, prediction, truth, names, cases):
    fields = {}
    for channel, name in enumerate(names):
        if not any(token in name.lower() for token in ("epsp", "plastic")):
            continue
        p = prediction[..., channel].astype(np.float64)
        t = truth[..., channel].astype(np.float64)
        error = p-t
        positive, zero = t > 0, t == 0
        support = []
        for threshold in (0.,):
            reference, predicted = t > threshold, p > threshold
            tp = int(np.count_nonzero(reference & predicted))
            fp = int(np.count_nonzero(~reference & predicted))
            fn = int(np.count_nonzero(reference & ~predicted))
            support.append({"threshold": threshold, "reference_count": int(reference.sum()),
                "prediction_count": int(predicted.sum()), "true_positive": tp, "false_positive": fp,
                "false_negative": fn, "true_negative": int(np.count_nonzero(~reference & ~predicted)),
                "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
                "iou": tp/(tp+fp+fn) if tp+fp+fn else None})
        fields[name] = {"support_definition": "Raw EPSP strictly greater than the exact dimensionless zero threshold; no fitted threshold or clipping. Projected nodal support is not a raw-cell damage criterion.",
            "truth_exact_nonzero_count": int(np.count_nonzero(t)), "truth_exact_positive_count": int(positive.sum()),
            "truth_exact_zero_count": int(zero.sum()), "truth_negative_count": int(np.count_nonzero(t < 0)),
            "prediction_exact_nonzero_count": int(np.count_nonzero(p)),
            "prediction_negative_count": int(np.count_nonzero(p < 0)), "prediction_min": float(p.min()),
            "prediction_max": float(p.max()), "count": int(t.size), "support": support,
            "truth_positive_support_per_timestep": positive.sum((0, 1)).tolist(),
            "prediction_positive_support_per_timestep": (p > 0).sum((0, 1)).tolist(),
            "rmse_on_reference_positive_support": float(np.sqrt(np.mean(error[positive]**2))) if positive.any() else None,
            "rmse_on_reference_exact_zero_support": float(np.sqrt(np.mean(error[zero]**2))) if zero.any() else None,
            "prediction_absolute_peak_on_reference_exact_zero_support": float(np.abs(p[zero]).max()) if zero.any() else None,
            "history": {"definition": "Strict time increments at fixed support; raw cumulative EPSP is expected not to decrease. Counts include roundoff. Reference increments are reported separately without corrections.",
                "prediction": delta_summary(np, np.diff(p, axis=2)),
                "truth": delta_summary(np, np.diff(t, axis=2)),
                "stored_prediction_dtype": str(prediction.dtype), "stored_truth_dtype": str(truth.dtype),
                "float32_machine_epsilon": float(np.finfo(np.float32).eps)},
            "per_case": [{"case": case, "truth_positive_count": int(positive[index].sum()),
                "truth_exact_zero_count": int(zero[index].sum()),
                "prediction_negative_count": int(np.count_nonzero(p[index] < 0)),
                "truth_peak": float(t[index].max()), "prediction_peak": float(p[index].max())}
                for index, case in enumerate(cases)]}
    return fields


def principal_diagnostics(np, prediction, truth, names):
    major = [i for i, name in enumerate(names) if "principal_major" in name]
    minor = [i for i, name in enumerate(names) if "principal_minor" in name]
    if len(major) != 1 or len(minor) != 1:
        return None
    p = prediction[..., minor[0]].astype(np.float64)-prediction[..., major[0]]
    t = truth[..., minor[0]].astype(np.float64)-truth[..., major[0]]
    return {"prediction_minor_greater_than_major_count": int(np.count_nonzero(p > 0)),
            "truth_minor_greater_than_major_count": int(np.count_nonzero(t > 0)),
            "prediction_maximum_order_violation": float(max(0., p.max())),
            "definition": "Projected major principal strain must be >= paired minor strain; no reordering."}


def parse_phone_cards(path):
    """Read the source-deck constant translation/gravity cards, never responses."""
    cards, current = {}, None
    with path.open(encoding="utf-8", errors="strict") as stream:
        for line in stream:
            value = line.strip()
            if value.startswith("/"):
                current = value if value.startswith(("/INIVEL/TRA/", "/GRAV/", "/FUNCT/2")) else None
                if current is not None:
                    cards[current] = []
            elif current is not None and value and not value.startswith("#"):
                cards[current].append(value)
    velocities = [rows for key, rows in cards.items() if key.startswith("/INIVEL/TRA/")]
    require(len(velocities) == 1, "Ballistic reference requires one uniform translation card")
    row = velocities[0][1].split()
    require(len(row) == 5 and int(row[4]) == 0, "Non-global initial velocity is unsupported")
    velocity = [float(x) for x in row[:3]]
    gravity, groups = {}, []
    for key, rows in cards.items():
        if not key.startswith("/GRAV/"):
            continue
        values = rows[1].split()
        require(len(values) == 7 and values[0] == "2" and values[2:4] == ["0", "0"] and float(values[5]) == 0., "Unsupported gravity card")
        require(values[1] in "XYZ" and values[1] not in gravity, "Ambiguous gravity direction")
        gravity[values[1]] = float(values[6]); groups.append(int(values[4]))
    require(set(gravity) == set("XYZ") and len(set(groups)) == 1, "Gravity vector is incomplete")
    function = cards.get("/FUNCT/2", [])
    require(len(function) >= 3, "Gravity time-function is missing")
    points = [[float(x) for x in row.split()] for row in function[1:]]
    require(all(len(point) == 2 and point[1] == 1. for point in points), "Gravity is not constant")
    return {"velocity_mm_per_s": velocity, "acceleration_mm_per_s_squared": [gravity[x] for x in "XYZ"],
            "velocity_node_group": int(row[3]), "gravity_node_group": groups[0],
            "constant_gravity_function_domain_s": [min(x[0] for x in points), max(x[0] for x in points)]}


def ballistic_reference(runtime, cache, prediction, truth, cases):
    """A contact-free analytic comparator, pinned to each case's actual deck."""
    np = runtime.np
    if "phone" not in str(cache.meta.get("dataset", "")).lower():
        return None, {}
    require(cache.meta.get("coordinate_unit") == "mm" and cache.units[:3] == ["mm"]*3,
            "Phone ballistic baseline requires declared mm coordinates/displacements")
    factor = {"s": 1., "ms": 1e-3, "us": 1e-6, "µs": 1e-6}.get(cache.meta.get("time_unit"))
    require(factor is not None, "Phone time unit is not declared")
    times = np.asarray(cache.meta["time_values"], dtype=np.float64)*factor
    source_path = cache.root/cache.meta["source_case_manifest"]
    source = read_json(source_path)
    by_case = {row["run_id"]: row for row in source["runs"]}
    baseline = np.empty((len(cases), 1, len(times), 3), dtype=np.float64)
    gravity_only = np.empty_like(baseline)
    evidence, hashes = [], {str(source_path.resolve()): filehash(source_path)}
    root = Path(__file__).resolve().parents[2]
    for index, case in enumerate(cases):
        row = by_case[case]
        directory = root/row["raw_run_dir"]
        candidates = [path for path in directory.glob("*_0000.rad") if filehash(path) == row["starter_sha256"]]
        require(len(candidates) == 1, "Cannot verify case initial conditions against pinned starter: "+case)
        path = candidates[0]
        cards = parse_phone_cards(path)
        low, high = cards["constant_gravity_function_domain_s"]
        require(low <= float(times.min()) <= float(times.max()) <= high, "Ballistic time is outside constant gravity function")
        velocity = np.asarray(cards["velocity_mm_per_s"])
        acceleration = np.asarray(cards["acceleration_mm_per_s_squared"])
        gravity_only[index, 0] = .5*times[:, None]**2*acceleration
        baseline[index, 0] = times[:, None]*velocity + gravity_only[index, 0]
        hashes[str(path.resolve())] = row["starter_sha256"]
        evidence.append({"case": case, "starter_sha256": row["starter_sha256"],
                         "starter_path": str(path.resolve()), "initial_speed_mm_per_s": float(np.linalg.norm(velocity)), **cards})
    baseline = np.broadcast_to(baseline, truth[..., :3].shape)
    gravity_only = np.broadcast_to(gravity_only, truth[..., :3].shape)
    metrics = runtime.physical_metrics(baseline, truth[..., :3], cache.names[:3], cache.units[:3], cases, cache.meta["time_values"])
    gravity_metrics = runtime.physical_metrics(gravity_only, truth[..., :3], cache.names[:3], cache.units[:3], cases, cache.meta["time_values"])
    error_den = metrics["displacement_vector"]["sum_squared_error"]
    model_sse = float(np.sum((prediction[..., :3].astype(np.float64)-truth[..., :3])**2))
    return {"definition": "Uniform contact-free displacement v0*t + 0.5*a*t^2 = v0*t - 0.5*g*t^2 with signed g=-a from each hash-verified source deck. Actual source cards and nominal stored times are used; contact, deformation and constraints are intentionally absent from this analytic comparator.",
            "coordinate_unit": "mm", "source_time_unit": cache.meta["time_unit"], "time_conversion_to_seconds": factor,
            "case_initial_condition_evidence": evidence, "ballistic_physical_metrics": metrics,
            "gravity_only_physical_metrics": gravity_metrics,
            "model_displacement_sse_over_ballistic_sse": model_sse/error_den if error_den else None}, hashes


def make_figures(runtime, cache, result, prediction, truth, cases):
    # matplotlib is deliberately lazy and reached only inside an authorized pass.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np = runtime.np
    case = cases[0]  # Fixed split order, chosen before looking at responses/errors.
    xyz = cache.coordinates(case)
    selected = [("displacement_magnitude", cache.units[0],
                 np.linalg.norm(prediction[0, :, -1, :3].astype(np.float64), axis=-1),
                 np.linalg.norm(truth[0, :, -1, :3].astype(np.float64), axis=-1))]
    selected.extend((name, cache.units[c], prediction[0, :, -1, c], truth[0, :, -1, c])
                    for c, name in enumerate(cache.names)
                    if any(token in name.lower() for token in ("vm", "von_mises", "frobenius", "epsp", "plastic")))
    output = []
    for name, unit, p, t in selected:
        p, t = np.asarray(p, dtype=np.float64), np.asarray(t, dtype=np.float64)
        error = p-t
        low, high = float(min(p.min(), t.min())), float(max(p.max(), t.max()))
        if high == low:
            high = low+max(abs(low)*.01, 1e-12)
        limit = float(np.abs(error).max()) or 1e-12
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        for axis, values, title, cmap, vmin, vmax in (
                (axes[0], t, "FEM", "viridis", low, high),
                (axes[1], p, "Prediction (raw)", "viridis", low, high),
                (axes[2], error, "Prediction - FEM", "coolwarm", -limit, limit)):
            scatter = axis.scatter(xyz[:, 0], xyz[:, 1], c=values, s=2, linewidths=0,
                                   cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            axis.set(title=title, xlabel="Reference x [mm]", ylabel="Reference y [mm]", aspect="equal")
            fig.colorbar(scatter, ax=axis, label=unit, shrink=.8)
        fig.suptitle(f"{case} | {name} | t={cache.meta['time_values'][-1]:.6g} {cache.meta.get('time_unit', 'unspecified')} | full {len(xyz):,} support")
        path = result/(re.sub(r"[^a-zA-Z0-9_]+", "_", name)+"_final_FEM_prediction_error.png")
        fig.savefig(path, dpi=170); plt.close(fig)
        output.append({"path": path.name, "sha256": filehash(path), "case": case, "field": name,
                       "fem_prediction_common_color_bounds": [low, high], "error_symmetric_bounds": [-limit, limit]})
    return {"case_selection": "First held-out case in frozen split order, final stored time; no error-based selection",
            "geometry": "All support points at reference XY; overlapping assembly parts can hide each other. No response clipping, nodal subsampling or deformed-geometry exaggeration.", "figures": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--mode", choices=("validation", "test"), required=True)
    parser.add_argument("--gpu-slot-authorized", action="store_true")
    args = parser.parse_args()
    require(args.gpu_slot_authorized, "Root GPU allocation is required before importing the framework")
    wall = time.perf_counter()
    run = args.run.resolve()
    frozen, training, stats, snapshot = freeze_snapshot(run, args.expected_manifest_sha256)
    validation_path = run/"validation_fresh_process.json"
    if args.mode == "validation":
        require(not validation_path.exists(), "Fresh validation evidence already exists; no overwrite")
        require(not (run/"test-performance-pass.started.json").exists(), "Test pass already started")
    else:
        validation = read_json(validation_path)
        require(validation["status"] == "passed" and validation["test_values_used"] is False, "Fresh validation must pass before held-out access")
        require(validation["checkpoint_sha256"] == frozen["checkpoint_sha256"], "Fresh validation used another checkpoint")
        require(validation["frozen_evidence_sha256"] == snapshot, "Fresh validation used different frozen evidence")
        require(training["best_epoch"] >= 1, "An eligible complete-epoch checkpoint is required")
        snapshot[str(validation_path)] = filehash(validation_path)
        require(not (run/"test_report.json").exists(), "Test report already exists")
        require(not (run/"test-performance-pass.started.json").exists(), "One held-out pass was already started, including partial failure")
    os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    runtime = load_runtime()
    np, torch, psutil = runtime.np, runtime.torch, runtime.psutil
    identity = process_identity(psutil)
    fresh_from_training(identity, training)
    if args.mode == "test":
        require(identity != validation["process_identity"], "Held-out evaluation must be a fresh process after fresh validation")
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    random.seed(training["seed"]); np.random.seed(training["seed"]); torch.manual_seed(training["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "CUDA is required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    ready = verify_ready(args.cache, args.expected_manifest_sha256)
    # Construction opens the first training case only; the target guard remains
    # closed until the sentinel. Inference preparation never requests targets.
    cache = make_cache(args.cache, allow_test_values=False)
    payload = torch.load(run/"best_checkpoint.pt", map_location="cpu", weights_only=True)
    validate_checkpoint(payload, frozen, training, stats, cache,
                        read_json(run/"prospective-training-protocol.json")["training"])
    model, config, model_source = runtime.build_model(training["architecture"], cache.shape[1]*cache.shape[2],
        config=payload["model_config"], global_dim=len(stats["global_mean"]))
    model_source["input_information"][1] = cache.meta.get("static_feature_name", "nodal_thickness")
    require(objecthash(config) == frozen["best_model_config_sha256"], "Rebuilt config differs")
    require(model_source["source_sha256"] == training["model_source"]["source_sha256"], "Model implementation differs")
    require(str(torch.__version__) == training["torch"] and
            importlib.metadata.version("nvidia-physicsnemo") == training["physicsnemo"], "Framework version changed")
    model.load_state_dict(payload["model_state_dict"]); model.to(device).eval()
    data = TensorData(cache, stats, device)
    setup_seconds = time.perf_counter()-wall
    common = {"process_identity": identity, "pid": os.getpid(), "architecture": training["architecture"],
        "checkpoint_sha256": frozen["checkpoint_sha256"], "versions": runtime_versions(runtime),
        "device_name": torch.cuda.get_device_name(), "setup_seconds": setup_seconds,
        "quality_gates": frozen["quality_gates"], "frozen_evidence_sha256": dict(snapshot),
        "source_sha256": frozen["source_sha256"], "model_source": model_source}
    if args.mode == "validation":
        anchor = frozen["validation_anchor_case"]
        require(anchor == cache.split["validation"][0], "Validation anchor differs")
        batch, _, preparation = data.prepare([anchor], with_targets=False)
        sync(torch, device); tick = time.perf_counter()
        with torch.inference_mode():
            output = forward(training["architecture"], model, batch)
        sync(torch, device); seconds = time.perf_counter()-tick
        prediction = output.detach().cpu().numpy()
        reference = np.load(run/"validation_reference.npy", mmap_mode="r")
        require(prediction.shape == reference.shape and np.isfinite(prediction).all(), "Invalid fresh validation prediction")
        maximum = float(np.max(np.abs(prediction-reference)))
        require(maximum <= 1e-5, "Fresh-process serialization error exceeds 1e-5")
        final_ready = verify_ready(args.cache, args.expected_manifest_sha256)
        require(ready == final_ready, "Frozen cache changed")
        verify_snapshot(snapshot)
        require(not cache.target_value_access_log["test"], "Validation touched test values")
        report = {**common, "status": "passed", "completed_utc": utc(), "validation_anchor_case": anchor,
            "max_scale_normalized_difference": maximum, "tolerance_scale_normalized": 1e-5,
            "test_values_used": False, "forward_seconds_cuda_synchronized": seconds,
            "preparation": preparation, "data_value_access": cache.target_value_access_log,
            "frozen_data_verification": ready, "frozen_data_end_verification": final_ready,
            "total_wall_seconds": time.perf_counter()-wall,
            "definition": "Fresh executable process reconstructs model from fixed serialized checkpoint and compares its validation anchor with training's saved normalized reference; no test response values."}
        with validation_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({"status": "fresh_validation_passed", "pid": os.getpid(), "maximum_difference": maximum}), flush=True)
        return
    # The exclusive durable sentinel precedes permission to open any held-out
    # target, prediction forward, output array or performance result directory.
    sentinel = run/"test-performance-pass.started.json"
    with sentinel.open("x", encoding="utf-8") as stream:
        json.dump({"status": "started_irrevocably", "utc": utc(), "process_identity": identity,
            "checkpoint_sha256": frozen["checkpoint_sha256"], "validation_report_sha256": filehash(validation_path),
            "rule": "One pass only, including partial failure. Any further metrics must use saved predictions."}, stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    cache.allow_test_values = True
    result = run/"heldout"
    result.mkdir(exist_ok=False)
    report = {**common, "status": "started", "started_utc": utc(), "test_performance_passes": 1}
    try:
        cases = list(cache.split["test"])
        shape = (len(cases), *cache.shape)
        prediction = np.lib.format.open_memmap(result/"predictions.npy", mode="w+", dtype=np.float32, shape=shape)
        truth = np.lib.format.open_memmap(result/"truth.npy", mode="w+", dtype=np.float32, shape=shape)
        mean, std = np.asarray(stats["field_mean"], dtype=np.float32), np.asarray(stats["field_std"], dtype=np.float32)
        rows, peak_rss = [], psutil.Process().memory_info().rss
        for index, case in enumerate(cases):
            safe = re.sub(r"[^a-zA-Z0-9_\-]+", "_", case)
            require(safe == case, "Unsafe case filename")
            batch, _, preparation = data.prepare([case], with_targets=False)
            sync(torch, device); tick = time.perf_counter()
            with torch.inference_mode():
                output = forward(training["architecture"], model, batch)
            sync(torch, device); elapsed = time.perf_counter()-tick
            tick = time.perf_counter()
            normalized = output.detach().cpu().numpy().reshape(cache.shape)
            require(np.isfinite(normalized).all(), "Nonfinite raw prediction")
            physical = normalized*std+mean
            require(np.isfinite(physical).all(), "Nonfinite physical prediction")
            np.save(result/(safe+".normalized.npy"), normalized)
            np.save(result/(safe+".physical.npy"), physical)
            prediction[index] = physical
            # Exactly one source target read per held-out case. All subsequent
            # metrics and plots read the saved disk array instead of source FEM.
            target = cache.target(case)
            require(target.shape == cache.shape, "Held-out target shape changed")
            truth[index] = target
            prediction.flush(); truth.flush()
            rows.append({"case": case, "forward_seconds_cuda_synchronized": elapsed, "preparation": preparation,
                         "device_to_host_unscale_save_truth_read_seconds": time.perf_counter()-tick,
                         "normalized_prediction": safe+".normalized.npy", "physical_prediction": safe+".physical.npy"})
            del output, normalized, physical, target, batch
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        require(cache.target_value_access_log["test"] == {case: 1 for case in cases}, "Test values were not read exactly once per case")
        # Saved aggregate arrays are memory mapped; these established metrics
        # allocate one physical channel at a time, never a full host tensor stack.
        tick = time.perf_counter()
        metrics = runtime.physical_metrics(prediction, truth, cache.names, cache.units, cases, cache.meta["time_values"])
        epsp = epsp_diagnostics(np, prediction, truth, cache.names, cases)
        principal = principal_diagnostics(np, prediction, truth, cache.names)
        ballistic, baseline_hashes = ballistic_reference(runtime, cache, prediction, truth, cases)
        figures = make_figures(runtime, cache, result, prediction, truth, cases)
        metric_seconds = time.perf_counter()-tick
        final_ready = verify_ready(args.cache, args.expected_manifest_sha256)
        require(ready == final_ready, "Frozen cache changed")
        verify_snapshot(snapshot); verify_snapshot(baseline_hashes)
        output_hashes = {path.name: filehash(path) for path in result.iterdir() if path.is_file()}
        report.update({"status": "completed", "completed_utc": utc(), "cases": cases, "shape": list(shape),
            "field_names": cache.names, "field_units": cache.units, "time_values": cache.meta["time_values"],
            "time_unit": cache.meta.get("time_unit"), "physical_metrics": metrics, "epsp_diagnostics": epsp,
            "principal_order_diagnostics": principal, "phone_ballistic_reference": ballistic, "figures": figures,
            "per_case": rows, "forward_seconds_cuda_synchronized_total": sum(row["forward_seconds_cuda_synchronized"] for row in rows),
            "metrics_and_figures_seconds": metric_seconds, "output_sha256": output_hashes,
            "baseline_source_sha256": baseline_hashes, "frozen_data_verification": ready,
            "frozen_data_end_verification": final_ready, "raw_predictions_clipped": False,
            "data_access_disclosure": "READY/contract hashing covers every split. Each held-out FEM target is opened once after sentinel creation; metrics/figures then reuse saved memory-mapped truth and predictions. One B1 forward per held-out case, with no warm-up or repeated performance pass.",
            "metric_memory_scope": "Disk-backed aggregate float32 arrays; physical_metrics temporarily allocates channel-wise float64 fields and displacement vectors.",
            "peak_sampled_rss_bytes_lower_bound": max(peak_rss, psutil.Process().memory_info().rss)})
    except BaseException as error:
        report.update({"status": "failed_pass_consumed", "failure_utc": utc(), "exception_type": type(error).__name__,
                       "exception": str(error), "traceback": traceback.format_exc(),
                       "retry_allowed": False, "remediation": "Preserve all partial evidence. Do not rerun a held-out prediction pass; inspect any saved predictions only."})
        raise
    finally:
        report.update({"data_value_access": cache.target_value_access_log,
            "cuda_allocator_peak_bytes": torch.cuda.max_memory_allocated(),
            "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            "total_wall_seconds": time.perf_counter()-wall})
        with (run/"test_report.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "architecture": training["architecture"], "cases": cases,
                      "test_performance_passes": 1}), flush=True)


if __name__ == "__main__":
    main()
