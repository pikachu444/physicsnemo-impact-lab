"""Prospective fixed-compute supervised comparison; never evaluates test responses.

This is a separate experiment from the immutable coupon epoch-budget runs.
Only standard-library imports occur before explicit GPU-slot authorization.
"""
import argparse
import csv
from datetime import datetime, timezone
import difflib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from budget_protocol import BudgetTracker, CompletedEpochSelector
from budgeted_runtime import (TensorData, filehash, forward, load_batch_adapter,
                              load_runtime, make_cache, objecthash, sync, verify_ready, write_json)


def utc():
    return datetime.now(timezone.utc).isoformat()


class WorkloadMonitor:
    def __init__(self, psutil, explicit_pids):
        self.psutil, self.explicit_pids = psutil, set(explicit_pids)
        self.samples, self.observed_processes = [], {}
        self.seconds, self.last_tick = 0., -1e30

    def sample(self, stage, force=False):
        tick = time.perf_counter()
        if not force and tick-self.last_tick < 5.:
            return
        self.last_tick = tick
        stamp = utc()
        processes, denied = [], 0
        for process in self.psutil.process_iter(["pid", "name", "create_time", "memory_info"]):
            try:
                item = process.info
                name = (item["name"] or "").lower()
                if item["pid"] not in self.explicit_pids and not any(
                    key in name for key in ("engine_win64", "starter_win64", "opencourant")):
                    continue
                row = {"pid": item["pid"], "name": item["name"],
                       "rss_bytes": item["memory_info"].rss,
                       "process_created_utc": datetime.fromtimestamp(item["create_time"], timezone.utc).isoformat()}
                processes.append(row)
                key = str(item["pid"])+":"+str(item["create_time"])
                observed = self.observed_processes.setdefault(key, {**row, "first_observed_utc": stamp})
                observed["last_observed_utc"] = stamp
                observed["max_observed_rss_bytes"] = max(observed.get("max_observed_rss_bytes", 0), row["rss_bytes"])
            except (self.psutil.AccessDenied, self.psutil.NoSuchProcess, TypeError):
                denied += 1
        memory = self.psutil.virtual_memory()
        self.samples.append({"utc": stamp, "stage": stage, "host_available_bytes": memory.available,
                             "own_rss_bytes": self.psutil.Process().memory_info().rss,
                             "solver_processes": processes, "inaccessible_or_exited_process_count": denied})
        self.seconds += time.perf_counter()-tick

    def report(self):
        return {"samples": self.samples, "observed_processes": self.observed_processes,
                "monitoring_seconds": self.seconds,
                "minimum_sampled_host_available_bytes": min(x["host_available_bytes"] for x in self.samples),
                "note": "Observed solver PID/time intervals and boundary RSS samples; not exact process exit times or continuous peaks. Shared workstation timings may include CPU/host-memory contention. No claim of isolated GPU system performance."}


def main():
    if not __debug__:
        raise RuntimeError("Optimized Python disables required experiment gates")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=["transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet"], required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 5], required=True)
    parser.add_argument("--batch-adapter", type=Path)
    parser.add_argument("--batch-approval", type=Path, action="append", default=[])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--data-clarification", type=Path)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--concurrent-pid", type=int, action="append", default=[])
    parser.add_argument("--gpu-slot-authorized", action="store_true")
    args = parser.parse_args()
    wall_start = time.perf_counter()
    assert args.gpu_slot_authorized, "Root GPU allocation is required before importing the framework"
    assert filehash(args.protocol) == args.protocol_sha256
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    policy = protocol["training"]
    assert math.isfinite(float(policy["synchronized_optimizer_compute_budget_seconds"]))
    assert policy["synchronized_optimizer_compute_budget_seconds"] == 240
    assert policy["maximum_optimizer_updates"] == 5000
    assert policy["minimum_updates_quality_gate"] == 100 and policy["minimum_complete_epochs_quality_gate"] == 2
    assert not args.out.exists() or (args.out.is_dir() and not any(args.out.iterdir())), "Training requires a new, empty output directory; no retraining or overwrite of final evidence"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"source").mkdir(exist_ok=True)
    source_files = ["train_budgeted.py", "evaluate_budgeted.py", "budgeted_runtime.py", "budget_protocol.py", "benchmark_data.py", "model_adapters.py"]
    source_hashes = {}
    for name in source_files:
        path = Path(__file__).with_name(name)
        source_hashes[name] = filehash(path)
        shutil.copyfile(path, args.out/"source"/name)
    if args.batch_adapter:
        source_hashes["batch_adapters.py"] = filehash(args.batch_adapter)
        shutil.copyfile(args.batch_adapter, args.out/"source/batch_adapters.py")
    shutil.copyfile(args.protocol, args.out/"prospective-training-protocol.json")
    if args.data_clarification:
        shutil.copyfile(args.data_clarification, args.out/"data-provenance-clarification.json")
    legacy = Path.cwd()/"outputs/expanded-research/benchmarks/structural_coupon/models/transolver/source/run_benchmark.py"
    (args.out/"source_diff_from_coupon.patch").write_text("".join(difflib.unified_diff(
        legacy.read_text(encoding="utf-8").splitlines(True), Path(__file__).read_text(encoding="utf-8").splitlines(True),
        fromfile="immutable_coupon/run_benchmark.py", tofile="prospective/train_budgeted.py")), encoding="utf-8")
    source_preparation_seconds = time.perf_counter()-wall_start
    start = time.perf_counter()
    os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    runtime = load_runtime()
    np, torch, psutil = runtime.np, runtime.torch, runtime.psutil
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    monitor = WorkloadMonitor(psutil, args.concurrent_pid)
    monitor.sample("setup_start", force=True)
    component_start = time.perf_counter()
    ready = verify_ready(args.cache, args.expected_manifest_sha256)
    verification_seconds = time.perf_counter()-component_start
    component_start = time.perf_counter()
    cache = make_cache(args.cache, allow_test_values=False)
    cache_setup_seconds = time.perf_counter()-component_start
    component_start = time.perf_counter()
    stats, mean_field = cache.statistics()
    training_statistics_seconds = time.perf_counter()-component_start
    assert stats == json.loads(args.normalization.read_text(encoding="utf-8")), "Common training normalization changed"
    write_json(args.out/"normalization.json", stats)
    np.save(args.out/"training_mean_field.npy", mean_field)
    del mean_field
    data = TensorData(cache, stats, device)
    globals_count = len(cache.meta["runs"][cache.split["train"][0]]["globals"])
    model, config, model_source = runtime.build_model(args.architecture, cache.shape[1]*cache.shape[2], global_dim=globals_count)
    model_source["input_information"][1] = cache.meta.get("static_feature_name", "nodal_thickness")
    model.to(device)
    train_cases = list(cache.split["train"])
    remainder = len(train_cases) % args.batch_size
    actual_batch_sizes = {args.batch_size} | ({remainder} if remainder else set())
    adapter, approvals = load_batch_adapter(args.batch_adapter, args.batch_approval,
        architecture=args.architecture, batch_sizes=actual_batch_sizes,
        manifest_sha256=cache.contract["manifest_sha256"], normalization_sha256=filehash(args.normalization), model_config=config)
    for index, row in enumerate(approvals):
        shutil.copyfile(row["path"], args.out/f"batch-approval-{index}.json")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    tracker = BudgetTracker(240., 15., 100)
    selector = CompletedEpochSelector()
    rng = random.Random(args.seed)  # Independent of architecture initialization.
    timers = {"training_data_read_seconds": 0., "training_normalization_seconds": 0.,
              "data_read_normalize_seconds": 0., "host_device_transfer_seconds": 0.,
              "validation_seconds": 0., "checkpoint_io_seconds": 0.}
    validation_rows, epoch_rows = [], []
    last_state, last_accounting = None, None
    peak_rss = psutil.Process().memory_info().rss
    setup_seconds = time.perf_counter()-start

    def cpu_state():
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def validation(epoch, state, accounting, reason):
        tick = time.perf_counter()
        model.eval()
        losses = []
        with torch.inference_mode():
            for case in cache.split["validation"]:
                batch, target, _ = data.prepare([case], with_targets=True)
                prediction = forward(args.architecture, model, batch)
                loss = torch.mean((prediction-target)**2)
                assert bool(torch.isfinite(loss))
                losses.append(float(loss.item()))
        sync(torch, device)
        score = sum(losses)/len(losses)
        improved = selector.consider(epoch, score, completed=True) if epoch else False
        row = {"epoch": epoch, "eligible_complete_epoch": epoch > 0,
               "validation_normalized_mse": score, "improved": improved, "reason": reason,
               "state_accounting": accounting, "accounted_train_seconds_at_evaluation": tracker.train_seconds}
        validation_rows.append(row)
        timers["validation_seconds"] += time.perf_counter()-tick
        if improved:
            payload = {"architecture": args.architecture, "model_config": config, "model_state_dict": state,
                       "normalization": stats, "contract": cache.contract, "seed": args.seed,
                       "parameters": sum(p.numel() for p in model.parameters()), "source_sha256": source_hashes,
                       "best_epoch": epoch, "best_validation_normalized_mse": score,
                       "training_policy": policy, "training_protocol_sha256": args.protocol_sha256,
                       "selected_epoch_accounting": accounting, "batch_size": args.batch_size}
            tick = time.perf_counter(); torch.save(payload, args.out/"best_checkpoint.pt")
            timers["checkpoint_io_seconds"] += time.perf_counter()-tick
        write_json(args.out/"validation_history.json", validation_rows)
        print(json.dumps({"event": "validation", **row}), flush=True)
        model.train()

    validation(0, None, {"optimizer_updates": 0, "samples": 0, "train_seconds": 0.}, "initial_state_not_eligible")
    monitor.sample("initial_validation_finished", force=True)
    stop_reason = None
    update_path = args.out/"update_history.csv"
    with update_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "epoch", "batch_samples", "cumulative_samples", "update_seconds", "cumulative_train_seconds", "normalized_mse", "learning_rate", "cases"])
        writer.writeheader()
        epoch = 0
        while stop_reason is None:
            epoch += 1
            cases = train_cases.copy(); rng.shuffle(cases)
            epoch_update_count, epoch_samples, weighted_loss = 0, 0, 0.
            epoch_wall_start = time.perf_counter()
            for offset in range(0, len(cases), args.batch_size):
                current_cases = cases[offset:offset+args.batch_size]
                batch, target, preparation = data.prepare(current_cases, with_targets=True)
                timers["training_data_read_seconds"] += preparation["read_seconds"]
                timers["training_normalization_seconds"] += preparation["normalize_seconds"]
                timers["data_read_normalize_seconds"] += preparation["read_normalize_seconds"]
                timers["host_device_transfer_seconds"] += preparation["host_device_transfer_seconds"]
                sync(torch, device)
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(args.architecture, model, batch, adapter)
                assert prediction.shape == target.shape
                loss = torch.mean((prediction-target)**2)
                assert bool(torch.isfinite(loss))
                loss.backward()
                finite_grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                assert finite_grads and bool(torch.stack(finite_grads).all())
                optimizer.step()
                sync(torch, device)
                duration = time.perf_counter()-tick
                samples, loss_value = len(current_cases), float(loss.item())
                tracker.record_update(duration, samples)
                epoch_update_count += 1; epoch_samples += samples; weighted_loss += loss_value*samples
                writer.writerow({"update": tracker.total_updates, "epoch": epoch, "batch_samples": samples,
                    "cumulative_samples": tracker.total_samples, "update_seconds": duration,
                    "cumulative_train_seconds": tracker.train_seconds, "normalized_mse": loss_value,
                    "learning_rate": .001, "cases": json.dumps(current_cases)})
                stream.flush()
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                if tracker.budget_reached:
                    stop_reason = "synchronized_optimizer_time_cap"
                elif tracker.total_updates >= 5000:
                    stop_reason = "maximum_optimizer_updates"
                if offset+samples == len(cases):
                    tracker.completed_epoch(epoch, epoch_update_count, epoch_samples)
                    tick = time.perf_counter(); last_state = cpu_state()
                    timers["checkpoint_io_seconds"] += time.perf_counter()-tick
                    last_accounting = {"optimizer_updates": tracker.total_updates, "samples": tracker.total_samples,
                                       "train_seconds": tracker.train_seconds}
                    epoch_row = {"epoch": epoch, "updates": epoch_update_count, "samples": epoch_samples,
                        "training_normalized_mse": weighted_loss/epoch_samples,
                        "optimizer_compute_seconds": tracker.summary()["completed_epochs"][-1]["train_seconds"],
                        "epoch_wall_seconds_excluding_validation": time.perf_counter()-epoch_wall_start,
                        "cumulative_train_seconds": tracker.train_seconds}
                    epoch_rows.append(epoch_row)
                    if epoch <= 3 or epoch in (5,10) or epoch % 10 == 0:
                        print(json.dumps({"event": "completed_epoch", **epoch_row}), flush=True)
                    if tracker.validation_due or stop_reason:
                        validation(epoch, last_state, last_accounting, "periodic_or_budget_complete_epoch")
                        tracker.mark_validation()
                    monitor.sample("completed_epoch")
                if stop_reason:
                    break
    training_allocator_peak = torch.cuda.max_memory_allocated()
    training_reserved_peak = torch.cuda.max_memory_reserved()
    partial = tracker.partial_epoch_summary()
    if partial["present"]:
        tick = time.perf_counter()
        torch.save({"model_state_dict": cpu_state(), "optimizer_state_dict": optimizer.state_dict(),
                    "budget_accounting": tracker.summary(), "eligible_for_selection": False,
                    "reason": "Stopped during an incomplete epoch; not an eligible validation candidate"}, args.out/"partial_stopped_state.pt")
        timers["checkpoint_io_seconds"] += time.perf_counter()-tick
    if last_state is None:
        raise RuntimeError("Budget ended before an eligible complete epoch; no test evaluation is permitted")
    model.load_state_dict(last_state)
    tick = time.perf_counter()
    torch.save({"model_state_dict": last_state, "epoch": epoch_rows[-1]["epoch"],
                "accounting": last_accounting, "eligible_complete_epoch": True}, args.out/"last_complete_state.pt")
    timers["checkpoint_io_seconds"] += time.perf_counter()-tick
    if validation_rows[-1]["epoch"] != epoch_rows[-1]["epoch"]:
        validation(epoch_rows[-1]["epoch"], last_state, last_accounting, "final_restored_complete_epoch")
    assert selector.best_epoch is not None and not cache.target_value_access_log["test"]
    checkpoint = args.out/"best_checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    anchor = cache.split["validation"][0]
    batch, _, _ = data.prepare([anchor], with_targets=False)
    with torch.inference_mode():
        reference = forward(args.architecture, model, batch).detach().cpu().numpy()
    np.save(args.out/"validation_reference.npy", reference)
    restored, _, _ = runtime.build_model(args.architecture, cache.shape[1]*cache.shape[2], config=config, global_dim=globals_count)
    restored.load_state_dict(payload["model_state_dict"]); restored.to(device).eval()
    with torch.inference_mode():
        reloaded = forward(args.architecture, restored, batch).detach().cpu().numpy()
    maximum = float(np.max(np.abs(reloaded-reference)))
    assert maximum <= 1e-5
    roundtrip = {"validation_anchor_case": anchor, "max_scale_normalized_difference": maximum,
                 "tolerance_scale_normalized": 1e-5, "test_values_used": False,
                 "definition": "New model constructed from serialized checkpoint; same validation input only"}
    write_json(args.out/"validation_serialization_roundtrip.json", roundtrip)
    quality = {"minimum_updates_satisfied": tracker.min_updates_satisfied,
               "minimum_two_complete_epochs_satisfied": len(epoch_rows) >= 2,
               "gate_passed": tracker.min_updates_satisfied and len(epoch_rows) >= 2,
               "failure_action": "Report insufficient budget; do not silently extend cap"}
    final_ready = verify_ready(args.cache, args.expected_manifest_sha256)
    assert ready == final_ready
    monitor.sample("training_and_validation_roundtrip_finished", force=True)
    best_accounting = payload["selected_epoch_accounting"]
    report = {"status": "training_completed_best_complete_epoch_checkpoint_fixed_no_test_evaluation",
        "training_process_pid": os.getpid(), "training_process_created_epoch": psutil.Process().create_time(),
        "architecture": args.architecture, "model_config": config, "model_source": model_source,
        "parameters": payload["parameters"], "seed": args.seed, "batch_size": args.batch_size,
        "batch_approvals": approvals, "normalization": stats, "contract": cache.contract,
        "source_sha256": source_hashes, "training_protocol_sha256": args.protocol_sha256,
        "budget_accounting": tracker.summary(), "stop_reason": stop_reason, "quality_gates": quality,
        "best_epoch": selector.best_epoch, "best_validation_normalized_mse": selector.best_score,
        "selected_epoch_accounting": best_accounting, "epoch_history": epoch_rows,
        "validation_history": validation_rows, "partial_epoch_selected": False,
        "validation_event_count_including_initial_and_restored_final": len(validation_rows),
        "budget_tracker_validation_count_definition": "Interval-reset marks at the currently completed training epoch; excludes initial and a restored final state after partial work. All actual validation events are counted separately.",
        "last_validation_improved": validation_rows[-1]["improved"],
        "best_checkpoint_fraction_of_compute_budget": best_accounting["train_seconds"]/240.,
        "convergence": "Unverified. A bounded initial comparison, not an optimized recipe; inspect the recorded validation curve and improvement at the last eligible state.",
        "optimizer": "Adam constant learning_rate0.001; mean normalized squared error over B,N,time,channel",
        "training_unit": "One case observation contains all nodes and stored10times. Native batches average the complete case losses; the last smaller batch and actual observations are explicitly counted.",
        "timing_boundary": "CUDA synchronized wall time around zero_grad, forward, loss, finite-loss/gradient checks, backward, Adam step. Inputs and labels are prepared/copied before the timer. File preparation, validation, CPU snapshots/checkpoint I/O and monitoring are excluded from the compute cap.",
        "setup_seconds_including_framework_import": setup_seconds, "separate_timers": timers,
        "source_preparation_seconds": source_preparation_seconds,
        "setup_components": {"ready_hash_verification_seconds": verification_seconds,
                             "cache_contract_setup_seconds": cache_setup_seconds,
                             "training_statistics_seconds": training_statistics_seconds},
        "workloads": monitor.report(), "peak_sampled_rss_bytes_lower_bound": peak_rss,
        "training_cuda_allocator_peak_bytes": training_allocator_peak,
        "training_cuda_reserved_peak_bytes": training_reserved_peak,
        "cuda_peak_after_validation_roundtrip_bytes": torch.cuda.max_memory_allocated(),
        "total_wall_seconds": time.perf_counter()-wall_start,
        "data_value_access": cache.target_value_access_log,
        "data_access_disclosure": "No test response values used in training or checkpoint selection; READY/contract provenance verification hashes files from every split.",
        "frozen_data_verification": ready, "frozen_data_end_verification": final_ready,
        "checkpoint_sha256": filehash(checkpoint), "torch": str(torch.__version__),
        "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"),
        "device_name": torch.cuda.get_device_name(), "validation_roundtrip": roundtrip}
    write_json(args.out/"training_report.json", report)
    write_json(args.out/"checkpoint-frozen.json", {
        "status": "checkpoint_fixed_and_validation_roundtrip_passed", "frozen_utc": utc(),
        "training_process_pid": os.getpid(), "training_process_created_epoch": psutil.Process().create_time(),
        "checkpoint_sha256": filehash(checkpoint), "validation_reference_sha256": filehash(args.out/"validation_reference.npy"),
        "validation_anchor_case": anchor, "tolerance_scale_normalized": 1e-5,
        "source_sha256": source_hashes, "training_protocol_sha256": args.protocol_sha256,
        "best_model_config_sha256": objecthash(config), "normalization_sha256": objecthash(stats),
        "contract_sha256": objecthash(cache.contract), "training_report_sha256": filehash(args.out/"training_report.json"),
        "quality_gates": quality, "test_performance_passes_so_far": 0})
    print(json.dumps({"status": report["status"], "architecture": args.architecture,
                      "budget": tracker.summary(), "best_epoch": selector.best_epoch, "quality_gates": quality}), flush=True)


if __name__ == "__main__":
    main()
