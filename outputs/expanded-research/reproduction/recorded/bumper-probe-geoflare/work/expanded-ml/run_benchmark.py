"""Resource probes and streamed, supervised multi-field PhysicsNeMo benchmarks.

Each architecture uses the same real FEM cases, support, fields, split and loss.
This custom benchmark is distinct from reproducing NVIDIA's full training recipe.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time

os.environ.setdefault("PHYSICSNEMO_FORCE_TE", "False")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import numpy as np
import psutil
import torch
from benchmark_data import Cache, physical_metrics
from model_adapters import build_model, forward_model, validate_batch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=["transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet"])
    parser.add_argument("--mode", choices=["probe", "train", "infer"], default="probe")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, help="Prior physical-unit predictions.npy for fresh-process equivalence check")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        torch.cuda.reset_peak_memory_stats()
    cache = Cache(args.cache)
    source = {}
    for name in ["run_benchmark.py", "benchmark_data.py", "model_adapters.py"]:
        p = Path(__file__).with_name(name)
        source[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copyfile(p, args.out / ("executed_"+name))
    payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        assert payload["contract"] == cache.contract, "Input mesh/field/split contract differs from checkpoint"
        stats = payload["normalization"]
        architecture = payload["architecture"]
        saved_config = payload["model_config"]
        mean_field = None
    else:
        assert args.architecture is not None
        architecture, saved_config = args.architecture, None
        stats, mean_field = cache.statistics()
        np.save(args.out / "training_mean_field.npy", mean_field)
    write_json(args.out / "normalization.json", stats)
    xyz = (cache.xyz-np.asarray(stats["coordinate_min"]))/np.asarray(stats["coordinate_span"])
    static = {"xyz": torch.tensor(xyz, dtype=torch.float32, device=device),
              "edge_index": torch.tensor(np.asarray(cache.edges), dtype=torch.long, device=device)}
    fmean = np.asarray(stats["field_mean"], dtype=np.float32)
    fstd = np.asarray(stats["field_std"], dtype=np.float32)

    def batch(case):
        thickness, globals_ = cache.inputs(case)
        x = dict(static)
        x["thickness"] = torch.tensor((thickness-stats["thickness_mean"])/stats["thickness_std"], dtype=torch.float32, device=device)
        x["globals"] = torch.tensor((globals_-np.asarray(stats["global_mean"]))/np.asarray(stats["global_std"]), dtype=torch.float32, device=device)
        return x

    def target(case):
        y = ((cache.target(case)-fmean)/fstd).reshape(len(cache.xyz), -1)
        return torch.tensor(y, dtype=torch.float32, device=device)[None]

    model, model_config, model_source = build_model(architecture, cache.shape[1]*cache.shape[2], config=saved_config)
    model = model.to(device)
    validate_batch(batch(cache.split["train"][0]), require_edges=architecture == "meshgraphnet")
    if payload:
        model.load_state_dict(payload["model_state_dict"])
    base = {"mode": args.mode, "architecture": architecture, "model_config": model_config, "model_source": model_source,
            "parameters": sum(p.numel() for p in model.parameters()), "source_sha256": source,
            "contract": cache.contract, "seed": args.seed, "epochs_budget": args.epochs,
            "python": platform.python_version(), "torch": torch.__version__, "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"),
            "device": str(device), "device_name": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
            "normalization_note": "Train cases only; one global mean/std per physical channel across nodes and time. Constant channels use scale1. Equal normalized squared error weights per channel and time; all nodes uniformly weighted.",
            "input_information": "Initial coordinates, nodal thickness, three global parameters; no target result is an input. Architecture adapters differ only in API/layout/graph use.",
            "prediction_scope": "One-shot all stored nonzero times. Evaluation cases already have FEM reference results; no novel appliance or geometry generalization is tested.",
            "memory_note": "RSS sampled at step/epoch boundaries; lower bound on true process peak. CUDA allocator peak measured by PyTorch, not total driver VRAM.",
            "data_manifest": str(args.cache.resolve()), "load_and_setup_seconds": time.perf_counter()-start}
    sampled_peak = process.memory_info().rss

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    if args.mode == "probe":
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = cache.split["train"][0]
        x, y = batch(first), target(first)
        rows = []
        for i in range(args.probe_steps):
            sync()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            p = forward_model(architecture, model, x)
            assert p.shape == y.shape and torch.isfinite(p).all(), (p.shape, y.shape)
            loss = torch.mean((p-y)**2)
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            optimizer.step()
            sync()
            rows.append({"step": i+1, "seconds": time.perf_counter()-tick, "normalized_mse": loss.item()})
            sampled_peak = max(sampled_peak, process.memory_info().rss)
            print(json.dumps(rows[-1]), flush=True)
        base.update(status="actual_FEM_forward_backward_optimizer_steps_completed", steps=rows,
                    estimated_60_epoch_training_seconds_excluding_validation=float(np.median([r["seconds"] for r in rows[1:] or rows])*len(cache.split["train"])*60))
        write_json(args.out / "probe.json", {**base, "peak_sampled_rss_bytes": sampled_peak,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None})
        return

    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        best, best_epoch = float("inf"), 0
        history = []
        training_start = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            tick = time.perf_counter()
            model.train()
            train_loss = 0.
            ids = cache.split["train"]
            for j in torch.randperm(len(ids)).tolist():
                case = ids[j]
                optimizer.zero_grad(set_to_none=True)
                p = forward_model(architecture, model, batch(case))
                y = target(case)
                assert p.shape == y.shape and torch.isfinite(p).all(), case
                loss = torch.mean((p-y)**2)
                assert torch.isfinite(loss), case
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()
            scheduler.step()
            model.eval()
            with torch.inference_mode():
                val = np.mean([torch.mean((forward_model(architecture, model, batch(case))-target(case))**2).item() for case in cache.split["validation"]])
            if val < best:
                best, best_epoch = float(val), epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({"architecture": architecture, "model_config": model_config,
                            "model_state_dict": best_state, "normalization": stats, "contract": cache.contract,
                            "best_epoch": best_epoch, "validation_normalized_mse": best,
                            "seed": args.seed, "source_sha256": source}, args.out / "best_checkpoint.pt")
            row = {"epoch": epoch, "training_normalized_mse": train_loss/len(ids),
                   "validation_normalized_mse": float(val), "seconds": time.perf_counter()-tick}
            history.append(row)
            with (args.out / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
            sampled_peak = max(sampled_peak, process.memory_info().rss)
            if epoch == 1 or epoch % 5 == 0:
                print(json.dumps(row), flush=True)
        model.load_state_dict(best_state)
        base.update(training_seconds=time.perf_counter()-training_start, best_epoch=best_epoch,
                    best_validation_normalized_mse=best, status="actual_supervised_training_and_heldout_evaluation_completed")
        del optimizer, best_state
    else:
        assert payload is not None, "infer mode requires checkpoint"
        base.update(status="fresh_checkpoint_load_and_heldout_inference_completed", checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())

    model.eval()
    test_ids = cache.split["test"]
    output_shape = (len(test_ids), *cache.shape)
    prediction = np.lib.format.open_memmap(args.out / "predictions.npy", mode="w+", dtype=np.float32, shape=output_shape)
    truth = np.lib.format.open_memmap(args.out / "truth.npy", mode="w+", dtype=np.float32, shape=output_shape)
    inference_forward_seconds, inference_with_input_seconds = 0., 0.
    with torch.inference_mode():
        x = batch(test_ids[0])
        for _ in range(3):
            forward_model(architecture, model, x)
        sync()
        for i, case in enumerate(test_ids):
            tick = time.perf_counter()
            x = batch(case)
            sync()
            forward_start = time.perf_counter()
            normalized = forward_model(architecture, model, x)
            sync()
            inference_forward_seconds += time.perf_counter()-forward_start
            physical = normalized[0].cpu().numpy().reshape(cache.shape)*fstd+fmean
            prediction[i] = physical
            truth[i] = cache.target(case)
            inference_with_input_seconds += time.perf_counter()-tick
    prediction.flush(); truth.flush()
    write_json(args.out / "predictions_contract.json", cache.contract)
    evaluated = physical_metrics(prediction, truth, cache.names, cache.units, test_ids, cache.meta["time_values"])
    comparison = {architecture: evaluated}
    if mean_field is not None:
        comparison["training_mean_field"] = physical_metrics(np.broadcast_to(mean_field, output_shape), truth, cache.names, cache.units, test_ids, cache.meta["time_values"])
    comparison["zero_field"] = physical_metrics(np.zeros(output_shape, dtype=np.float32), truth, cache.names, cache.units, test_ids, cache.meta["time_values"])
    if args.reference:
        reference = np.load(args.reference, mmap_mode="r")
        assert reference.shape == prediction.shape, "Reference prediction shape differs"
        reference_contract = args.reference.with_name("predictions_contract.json")
        assert reference_contract.exists() and json.loads(reference_contract.read_text(encoding="utf-8")) == cache.contract, "Reference prediction data/case-order contract differs"
        difference = float(np.max(np.abs(prediction-reference)))
        scaled_difference = float(np.max(np.abs(prediction-reference)/fstd))
        # GPU scatter reductions may differ by roundoff between fresh processes.
        assert scaled_difference <= 1e-5, f"Fresh checkpoint inference differs in normalized units: {scaled_difference}"
        base["fresh_checkpoint_prediction_max_absolute_difference"] = difference
        base["fresh_checkpoint_prediction_max_absolute_difference_by_field"] = np.max(np.abs(prediction-reference),axis=(0,1,2)).tolist()
        base["fresh_checkpoint_prediction_max_scale_normalized_difference"] = scaled_difference
        base["fresh_checkpoint_prediction_tolerance_in_scale_normalized_units"] = 1e-5
    base.update(metrics=comparison, test_inference_forward_seconds=inference_forward_seconds,
                test_inference_with_input_transfer_and_output_copy_seconds=inference_with_input_seconds,
                inference_timing_note="Warmup excluded. Forward-only CUDA-synchronized time excludes input transfer and output copy; second time includes per-case input and output copies/cache I/O, excludes metric calculation.",
                peak_sampled_rss_bytes=max(sampled_peak, process.memory_info().rss),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                total_seconds=time.perf_counter()-start)
    write_json(args.out / "metrics.json", base)
    print(json.dumps({"status": base["status"], "architecture": architecture, "seconds": base["total_seconds"],
                      "fields": {k:{a:b for a,b in v.items() if a in ["unit", "relative_l2", "rmse", "component_rmse"]} for k,v in evaluated.items()}}), flush=True)


if __name__ == "__main__":
    main()
