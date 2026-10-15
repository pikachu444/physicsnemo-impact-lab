"""One explicitly allocated CUDA subprocess, one domain/model/batch, train cases only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import random
import sys
import time
import traceback

from train_probe_data import TrainProbeData, file_hash, read_train_contract

WORKSPACE = Path(__file__).resolve().parents[2]
MAIN = WORKSPACE / "work/expanded-ml"
OUTPUT_ROOT = WORKSPACE / "outputs/expanded-research/benchmarks/batching"
ARCHITECTURES = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")
DOMAINS = {
    "bumper": ("multifield7-cache", "bumper7-normalization.json",
               "af937bc0798d9362e7e5298e5387099ac6aa305dccbee31b123a103c03c35e41",
               "213b946707404130ed9e0f72513629cf158c4d118ef79574fe79c2af5bd9aba3"),
    "phone": ("phone-full-cache-v2", "phone-full-normalization.json",
              "aa0848d5da1ce3ab103a03bcad6ae356896869357d5eec00cde82bb8d1ef1dca",
              "a51410b8b2ed7066e650e94e0f704e5a9195273886da1f0399d07a70a6d8574d"),
}
OUTPUT_ATOL, OUTPUT_RTOL = 1e-5, 1e-4
GRAD_ATOL, GRAD_RTOL = 1e-5, 5e-4


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--domain", choices=DOMAINS, required=True)
    value.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    value.add_argument("--batch-size", type=int, choices=(1, 2, 5), required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--timing-repeats", type=int, default=2)
    value.add_argument("--seed", type=int, default=20261015)
    value.add_argument("--gpu-slot-authorized", action="store_true",
                       help="Required explicit scheduler allocation; checked before ML imports")
    return value


def execute(args, result, event):
    os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import psutil
    import torch
    import numpy as np

    sys.path.insert(0, str(MAIN))
    from model_adapters import build_model, forward_model, validate_batch
    from batch_adapters import forward_batched, single, subset

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available(), "Allocated CUDA is unavailable"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    result.update(torch=torch.__version__, device_name=torch.cuda.get_device_name(),
                  gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  tf32=False, dtype="float32", cpu_threads=4,
                  determinism={"seed": args.seed, "cudnn_deterministic": True, "cudnn_benchmark": False,
                               "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                               "note": "Original native CUDA scatter reductions may differ by float32 rounding. Numerical equivalence uses the declared tolerances; bitwise repeatability is not claimed. Strict deterministic-algorithm mode is not forced because unsupported native scatter would test a different execution restriction."})

    def memory():
        info = process.memory_info()
        row = {"rss_bytes": info.rss, "windows_process_peak_wset_bytes": getattr(info, "peak_wset", None),
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        result["peak_sampled_rss_bytes"] = max(result.get("peak_sampled_rss_bytes", 0), info.rss)
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            result[key] = max(result.get(key, 0), row[key])
        return row

    def timed(label, operation):
        event(label+":start")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        row = {"seconds": time.perf_counter()-tick, **memory()}
        event(label+":complete", **row)
        return value, row

    folder, norm, expected_manifest, expected_norm = DOMAINS[args.domain]
    event("train_data:start")
    contract = read_train_contract(MAIN/folder/"manifest.json", MAIN/norm, args.batch_size,
                                   expected_manifest, expected_norm)
    data = TrainProbeData(contract)
    batch, target = data.to_device(torch.device("cuda"))
    result.update(train_cases=data.cases, train_count=len(contract["meta"]["split"]["train"]),
                  manifest_sha256=expected_manifest, normalization_sha256=expected_norm,
                  ready_sha256=contract["ready_sha256"], opened_data_files=data.opened_files,
                  no_validation_or_test_label_file_opened=True,
                  shape=list(target.shape), global_dim=batch["globals"].shape[-1],
                  real_directed_edges_per_case=int(batch["edge_index"].shape[1]),
                  static_feature_name=data.meta.get("static_feature_name", "nodal_thickness"))
    model, config, source = build_model(args.architecture, data.out_dim,
                                       global_dim=batch["globals"].shape[-1])
    model = model.cuda().eval()
    result.update(model_config=config, model_source=source,
                  parameters=sum(p.numel() for p in model.parameters()))
    for i in range(args.batch_size):
        validate_batch(single(batch, i), require_edges=args.architecture == "meshgraphnet")
    event("train_data:complete", **memory())

    def weight_hash():
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def compare(left, right, atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL):
        assert left.shape == right.shape
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        if not finite:
            return {"passed": False, "max_absolute_difference": None, "relative_l2_difference": None,
                    "atol": atol, "rtol": rtol, "all_finite": False}
        difference = (left-right).abs()
        denominator = torch.sum(right*right, dtype=torch.float64).sqrt().item()
        return {"passed": finite and bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
                "max_absolute_difference": difference.max().item(),
                "relative_l2_difference": torch.sum(difference*difference, dtype=torch.float64).sqrt().item()/max(denominator, 1e-30),
                "atol": atol, "rtol": rtol, "all_finite": finite}

    initial_hash = weight_hash()
    result["initial_weight_sha256"] = initial_hash
    b = args.batch_size
    with torch.inference_mode():
        reference, _ = timed("warm_original_B1_outputs", lambda: torch.cat([
            forward_model(args.architecture, model, single(batch, i)).cpu() for i in range(b)], 0))
        native, _ = timed("warm_native_batch_output", lambda: forward_batched(args.architecture, model, batch).cpu())
        correctness = {"original_B1_vs_native_batch": compare(native, reference)}
        if b > 1:
            order = list(reversed(range(b)))
            permuted, _ = timed("case_order_independence", lambda: forward_batched(
                args.architecture, model, subset(batch, order)).cpu())
            correctness["case_order_independence"] = compare(permuted[order], native)
            del permuted
            changed = dict(batch)
            changed["globals"] = batch["globals"].clone()
            changed["globals"][-1, 0] += 0.375
            altered, _ = timed("other_case_globals_independence", lambda: forward_batched(
                args.architecture, model, changed).cpu())
            correctness["other_case_globals_independence"] = compare(altered[:-1], native[:-1])
            changed_original = forward_model(args.architecture, model, single(changed, b-1)).cpu()
            correctness["changed_case_original_B1_vs_batch"] = compare(altered[-1:], changed_original)
            correctness["changed_case_output_max_delta"] = (altered[-1]-native[-1]).abs().max().item()
            correctness["normalized_global_delta"] = 0.375
            del changed, altered, changed_original
        else:
            correctness["cross_case_checks"] = "not_applicable_to_B1; B2/B5 explicitly test independence"
        del reference, native
    result["output_checks"] = correctness

    def sequential_backward(capture_loss=False):
        model.zero_grad(set_to_none=True)
        total = 0.
        for i in range(b):
            prediction = forward_model(args.architecture, model, single(batch, i))
            loss = torch.mean((prediction-target[i:i+1])**2)
            (loss/b).backward()
            if capture_loss:
                total += loss.detach().item()/b
        return total

    def native_backward(capture_loss=False):
        model.zero_grad(set_to_none=True)
        prediction = forward_batched(args.architecture, model, batch)
        loss = torch.mean((prediction-target)**2)
        loss.backward()
        return loss.detach().item() if capture_loss else None

    # These correctness backward passes are also the backward warmup; no optimizer updates.
    sequential_loss, _ = timed("warm_original_B1_mean_gradient", lambda: sequential_backward(True))
    gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                 for name, p in model.named_parameters()}
    native_loss, _ = timed("warm_native_batchmean_gradient", lambda: native_backward(True))
    assert math.isfinite(sequential_loss) and math.isfinite(native_loss), "Nonfinite normalized loss"
    gradient_rows = {}
    for name, parameter in model.named_parameters():
        baseline = gradients[name]
        if baseline is None or parameter.grad is None:
            gradient_rows[name] = {"passed": baseline is None and parameter.grad is None,
                                   "both_none": baseline is None and parameter.grad is None}
        else:
            gradient_rows[name] = compare(parameter.grad.detach().cpu(), baseline, GRAD_ATOL, GRAD_RTOL)
    result["gradient_check"] = {
        "passed": all(row["passed"] for row in gradient_rows.values()),
        "definition": "gradient(mean_B mean_NTC MSE) versus sum_B gradient(B1 MSE)/B, same eval weights",
        "original_B1_mean_loss": sequential_loss, "native_batchmean_loss": native_loss,
        "loss_absolute_difference": abs(sequential_loss-native_loss),
        "atol": GRAD_ATOL, "rtol": GRAD_RTOL, "parameters": gradient_rows}
    del gradients
    model.zero_grad(set_to_none=True)

    def sequential_forward():
        # Keep all B outputs live through synchronization/memory sampling, like native B.
        return [forward_model(args.architecture, model, single(batch, i)) for i in range(b)]

    measurements = {"forward": {"original_B1_x_B": [], "native_batch": []},
                    "forward_backward": {"original_B1_x_B": [], "native_batch": []}}
    for repeat in range(args.timing_repeats):
        # Alternate ordering to reduce systematic temperature/clock ordering effects.
        modes = ("original_B1_x_B", "native_batch") if repeat % 2 == 0 else ("native_batch", "original_B1_x_B")
        for mode in modes:
            with torch.inference_mode():
                operation = sequential_forward if mode == "original_B1_x_B" else lambda: forward_batched(args.architecture, model, batch)
                value, row = timed(f"forward:{repeat+1}:{mode}", operation)
                del value
            measurements["forward"][mode].append(row)
        for mode in modes:
            _, row = timed(f"forward_backward:{repeat+1}:{mode}", sequential_backward if mode == "original_B1_x_B" else native_backward)
            measurements["forward_backward"][mode].append(row)
    model.zero_grad(set_to_none=True)
    import statistics
    timing = {}
    for phase, modes in measurements.items():
        baseline = statistics.median(row["seconds"] for row in modes["original_B1_x_B"])
        native = statistics.median(row["seconds"] for row in modes["native_batch"])
        timing[phase] = {"measurements": modes, "original_B1_x_B_median_seconds": baseline,
                         "native_batch_median_seconds": native,
                         "original_B1_simulations_per_second": b/baseline,
                         "native_batch_simulations_per_second": b/native,
                         "measured_speedup": baseline/native,
                         "native_faster_in_every_paired_repeat": all(
                             modes["native_batch"][i]["seconds"] < modes["original_B1_x_B"][i]["seconds"]
                             for i in range(args.timing_repeats))}
    result["timing"] = timing
    assert weight_hash() == initial_hash, "Eval/gradient/timing unexpectedly changed weights or buffers"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    updates = []
    for step in range(3):
        def update():
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_batched(args.architecture, model, batch)
            loss = torch.mean((prediction-target)**2)
            loss.backward()
            optimizer.step()
            return loss.detach()
        loss, row = timed(f"actual_Adam_update:{step+1}", update)
        assert bool(torch.isfinite(loss))
        assert all(bool(torch.isfinite(p).all()) and (p.grad is None or bool(torch.isfinite(p.grad).all()))
                   for p in model.parameters())
        updates.append({"update": step+1, "normalized_mse": loss.item(), **row})
    result["adam"] = {"updates": updates, "optimizer_updates": 3, "simulation_observations": 3*b,
                      "learning_rate": 1e-3, "final_weight_sha256": weight_hash()}
    assert result["adam"]["final_weight_sha256"] != initial_hash, "Adam never changed the model"
    result["budget_note"] = (
        f"Timing compares equal {b}-simulation workloads: B separate original B1 calls versus one native batch. "
        "Forward-backward timing uses one accumulated mean gradient and no optimizer update in either path. "
        f"The separate Adam probe performs exactly 3 updates and {3*b} simulation observations. "
        "B1's 3 updates observe 3 simulations; matching epochs with batch B reduces updates to ceil(train_count/B). "
        "Timing benefit alone does not establish equal training convergence or equal optimizer-update budgets.")
    result["timing_note"] = (
        "One correctness forward and one correctness backward warm each path before two default timed repetitions. "
        "CUDA synchronization brackets each timing. Inputs/labels already reside on GPU; adapter layout/graph preparation "
        "is included, disk reading and transfer excluded. Allocator peaks cover phase work and live tensors, not driver VRAM. "
        "Both forward paths retain all B simulation outputs until after synchronization/memory sampling. "
        "RSS is a phase-boundary snapshot; windows_process_peak_wset_bytes is process-wide context, not a per-phase CPU peak. "
        "No test labels or held-out cases are opened.")
    checks = [value["passed"] for value in correctness.values() if isinstance(value, dict) and "passed" in value]
    result["status"] = "passed" if all(checks) and result["gradient_check"]["passed"] else "validation_failed"
    result["memory_final"] = memory()
    del optimizer, model, batch, target, data
    torch.cuda.empty_cache()
    return 0 if result["status"] == "passed" else 21


def main():
    args = parser().parse_args()
    if not args.gpu_slot_authorized:
        raise SystemExit("GPU slot flag missing: no Torch/NumPy/PhysicsNeMo import was performed.")
    assert args.timing_repeats >= 2
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT_ROOT.resolve()), "Evidence must stay under owned batching output directory"
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "domain": args.domain, "architecture": args.architecture,
              "batch_size": args.batch_size, "seed": args.seed, "timing_repeats": args.timing_repeats,
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "python_executable": sys.executable, "stage": "source_snapshot"}
    tick = time.perf_counter()
    source_files = [Path(__file__), Path(__file__).with_name("batch_adapters.py"),
                    Path(__file__).with_name("train_probe_data.py"), MAIN/"model_adapters.py",
                    MAIN/"benchmark_data.py", MAIN/"run_benchmark.py"]
    (out/"source").mkdir()
    result["source_sha256"] = {str(path): file_hash(path) for path in source_files}
    for path in source_files:
        shutil.copyfile(path, out/"source"/path.name)

    def event(stage, **payload):
        result["stage"] = stage
        row = {"utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-tick,
               "stage": stage, **payload}
        with (out/"events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        code = execute(args, result, event)
        assert all(file_hash(path) == result["source_sha256"][str(path)] for path in source_files), "A source changed during the probe"
    except Exception as error:
        detail = traceback.format_exc()
        (out/"failure_traceback.txt").write_text(detail, encoding="utf-8")
        print(detail, file=sys.stderr, flush=True)
        is_oom = "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"
        result.update(status="oom" if is_oom else "failed", exception_type=type(error).__name__,
                      exception_message=str(error), failed_stage=result["stage"],
                      release_note="This isolated process exits immediately; no larger batch success is inferred.")
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            result["cuda_peak_allocated_bytes_at_failure"] = torch.cuda.max_memory_allocated()
            result["cuda_peak_reserved_bytes_at_failure"] = torch.cuda.max_memory_reserved()
        code = 20 if is_oom else 1
    result["total_seconds"] = time.perf_counter()-tick
    write_json(out/"result.json", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
