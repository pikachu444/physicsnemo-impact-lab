"""Bounded actual PhysicsNeMo HybridMeshGraphNet training and held-out rollout.

Uses the NVIDIA graph/target contract with lazy records; does not reproduce the
8-H100 full-data recipe. Derived strain is kinematic, not a COMSOL strain label.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "deps"))

import numpy as np
import psutil
import torch
from torch_geometric.data import Data
from physicsnemo.models.meshgraphnet import HybridMeshGraphNet

from plate_data import build_statistics, deformation_strain, edge_values, load_trajectory, world_edges
from plate_evaluation import evaluate as evaluate_fields
from selection_policy import comparison_key, parse_selection_epochs, rollout_selection_record, validate_diagnostic_names


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--processor", type=int, default=3)
    parser.add_argument("--checkpoint-segments", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--frames-per-trajectory", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.003)
    parser.add_argument("--probe", type=int, default=0)
    parser.add_argument("--probe-trajectory")
    parser.add_argument("--probe-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--rollout-steps", type=int, default=399)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--valid-count", type=int)
    parser.add_argument("--test-start", type=int, default=0)
    parser.add_argument("--test-stop", type=int)
    parser.add_argument("--evaluate-validation", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--evaluation-label", choices=["development", "fresh_final", "exploratory"], default="exploratory")
    parser.add_argument("--checkpoint-selection", choices=["teacher", "rollout"], default="teacher")
    parser.add_argument("--selection-epochs", default="")
    parser.add_argument("--selection-trajectories", default="valid_0000,valid_0005")
    parser.add_argument("--evaluation-mode", choices=["ar_only", "ar_and_teacher"], default="ar_and_teacher")
    parser.add_argument("--lr-decay-per-update", type=float)
    parser.add_argument("--protocol", type=pathlib.Path)
    return parser.parse_args()


def architecture(args):
    return {"input_dim_nodes": 3, "input_dim_edges": 8, "output_dim": 4,
            "processor_size": args.processor, "hidden_dim_processor": args.hidden,
            "hidden_dim_node_encoder": args.hidden, "hidden_dim_edge_encoder": args.hidden,
            "hidden_dim_node_decoder": args.hidden,
            "num_processor_checkpoint_segments": args.checkpoint_segments}


def prepare_graph(record, positions, stats, device):
    mesh = record["mesh_edges"]
    world = world_edges(positions, mesh)
    em, es = np.asarray(stats["edge_mean"], dtype=np.float32), np.asarray(stats["edge_std"], dtype=np.float32)
    fixed = (record["mesh_edge_values"] - em) / es
    current = (edge_values(positions, mesh) - em) / es
    world_values = (edge_values(positions, world) - em) / es
    graph = Data(edge_index=torch.as_tensor(np.concatenate([mesh, world], axis=1), device=device),
                 num_nodes=len(positions))
    mapping = np.select([record["node_type"][:, 0] == 0, record["node_type"][:, 0] == 1], [0, 1], 2)
    nodes = torch.as_tensor(np.eye(3, dtype=np.float32)[mapping], device=device)
    mesh_features = torch.as_tensor(np.concatenate([fixed, current], axis=1), device=device)
    world_features = torch.as_tensor(np.tile(world_values, (1, 2)), device=device)
    return nodes, mesh_features, world_features, graph


def loss_for(model, record, frame, stats, device, noise_std=0, return_prediction=False):
    current = record["world_pos"][frame].copy()
    target = np.concatenate([record["world_pos"][frame + 1] - current, record["stress"][frame + 1]], axis=1)
    if noise_std:
        noise = np.random.normal(0, noise_std, current.shape).astype(np.float32)
        noise[record["node_type"][:, 0] != 0] = 0
        current += noise
        target[:, :3] -= noise
    prediction = model(*prepare_graph(record, current, stats, device))
    target = torch.as_tensor((target - np.asarray(stats["target_mean"], dtype=np.float32)) /
                             np.asarray(stats["target_std"], dtype=np.float32), device=device)
    loss = (prediction - target).square().mean()
    return (loss, prediction) if return_prediction else loss


def metric(prediction, truth):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    error = p - t
    norm = np.linalg.norm(t)
    return {"relative_l2": float(np.linalg.norm(error) / norm) if norm else None,
            "component_rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_component_error": float(np.abs(error).max()),
            "truth_l2": float(norm), "values": int(t.size)}


@torch.inference_mode()
def teacher_validation(model, paths, frames, stats, device, detailed=False):
    model.eval()
    losses = []
    sums = {field: {"squared_error": 0.0, "squared_truth": 0.0, "values": 0,
                    "max_abs_component_error": 0.0} for field in ["free_increment", "free_stress"]}
    means = np.asarray(stats["target_mean"], dtype=np.float32)
    stds = np.asarray(stats["target_std"], dtype=np.float32)
    for path in paths:
        record = load_trajectory(path)
        for frame in frames:
            loss, prediction = loss_for(model, record, int(frame), stats, device, return_prediction=True)
            losses.append(float(loss))
            if detailed:
                raw = prediction.cpu().numpy() * stds + means
                free = record["node_type"][:, 0] == 0
                truth_increment = record["world_pos"][frame + 1] - record["world_pos"][frame]
                for field, pred, truth in [("free_increment", raw[free, :3], truth_increment[free]),
                                           ("free_stress", raw[free, 3:4], record["stress"][frame + 1, free])]:
                    m = metric(pred, truth)
                    sums[field]["squared_error"] += m["component_rmse"] ** 2 * m["values"]
                    sums[field]["squared_truth"] += m["truth_l2"] ** 2
                    sums[field]["values"] += m["values"]
                    sums[field]["max_abs_component_error"] = max(sums[field]["max_abs_component_error"], m["max_abs_component_error"])
    result = {"normalized_mse": float(np.mean(losses)), "anchors_per_trajectory": len(frames),
              "trajectories": len(paths), "scope": "Teacher-forced anchors; free-node physical-unit metrics use unnoised inputs"}
    for field, total in sums.items():
        if detailed:
            result[field] = {"relative_l2": (total["squared_error"] / total["squared_truth"]) ** 0.5 if total["squared_truth"] else None,
                             "component_rmse": (total["squared_error"] / total["values"]) ** 0.5,
                             "truth_l2": total["squared_truth"] ** 0.5,
                             "values": total["values"], "max_abs_component_error": total["max_abs_component_error"]}
    return result if detailed else result["normalized_mse"]


def evaluate(model, paths, stats, args, output, device, *, prefix="", compute_teacher=None):
    if compute_teacher is None:
        compute_teacher = args.evaluation_mode == "ar_and_teacher"
    return evaluate_fields(model, paths, stats, args, output, device,
                           prepare_graph=prepare_graph, metric=metric,
                           compute_teacher=compute_teacher, prefix=prefix)


def pooled_metrics(results):
    if not results or any(r.get("status") == "failed" for r in results):
        return {"status": "unavailable_incomplete_evaluation", "requested_cases": len(results or []),
                "reason": "A missing or failed trajectory cannot be silently removed from a pooled result"}
    names = [k for k, value in results[0].items() if isinstance(value, dict) and "relative_l2" in value]
    pooled = {}
    for name in names:
        n = sum(r[name]["values"] for r in results)
        squared_error = sum(r[name]["component_rmse"] ** 2 * r[name]["values"] for r in results)
        squared_truth = sum(r[name]["truth_l2"] ** 2 for r in results)
        pooled[name] = {"relative_l2": float(np.sqrt(squared_error / squared_truth)) if squared_truth else None,
                        "component_rmse": float(np.sqrt(squared_error / n)),
                        "max_abs_component_error": max(r[name]["max_abs_component_error"] for r in results),
                        "values": n}
    return pooled


def main():
    args = arguments()
    if args.checkpoint_selection == "rollout":
        assert args.validation_only and not args.probe and not args.checkpoint
        assert args.rollout_steps == 399, "Checkpoint diagnostics require the complete 399-step rollout"
        assert args.protocol, "Rollout checkpoint selection requires a frozen pretraining protocol"
    assert args.epochs > 0 and 1 <= args.frames_per_trajectory <= 399
    assert args.lr > 0 and args.noise >= 0
    assert args.lr_decay_per_update is None or 0 < args.lr_decay_per_update <= 1
    output = args.output.resolve()
    if (output / "metrics.json").exists() or (args.probe and (output / "probe.json").exists()):
        raise FileExistsError(f"Completed evidence already exists: {output}; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "executed_source.py").write_bytes(pathlib.Path(__file__).read_bytes())
    (output / "plate_data_executed.py").write_bytes((ROOT / "plate_data.py").read_bytes())
    for helper in ["plate_evaluation.py", "selection_policy.py"]:
        (output / f"{pathlib.Path(helper).stem}_executed.py").write_bytes((ROOT / helper).read_bytes())
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    # Val-only runs do not even glob or hash source-test NPZ files.
    paths = {split: sorted((ROOT / "data").glob(f"{split}_*.npz")) for split in ["train", "valid"]}
    paths["test"] = [] if args.validation_only else sorted((ROOT / "data").glob("test_*.npz"))[args.test_start:args.test_stop]
    if args.train_count:
        assert len(paths["train"]) >= args.train_count
        paths["train"] = paths["train"][:args.train_count]
    if args.valid_count:
        assert len(paths["valid"]) >= args.valid_count
        paths["valid"] = paths["valid"][:args.valid_count]
    assert paths["train"] and paths["valid"] and (args.validation_only or paths["test"])
    dataset_hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for ps in paths.values() for p in ps}
    selection_epochs = parse_selection_epochs(args.selection_epochs, args.epochs)
    diagnostic_paths = []
    if args.checkpoint_selection == "rollout":
        names = validate_diagnostic_names(args.selection_trajectories.split(","), [p.stem for p in paths["valid"]])
        diagnostic_paths = [next(p for p in paths["valid"] if p.stem == n) for n in names]
    protocol = None
    if args.protocol:
        protocol = json.loads(args.protocol.read_text(encoding="utf8"))
        assert protocol["training"] == [p.stem for p in paths["train"]]
        assert protocol["validation"] == [p.stem for p in paths["valid"]]
        assert all(protocol["dataset_npz_sha256"][n] == h for n, h in dataset_hashes.items())
        for key, value in protocol["required_options"].items():
            assert getattr(args, key) == value, f"Frozen protocol option differs: {key}"
        assert protocol["selection_epochs"] == selection_epochs
        assert protocol["selection_trajectories"] == [p.stem for p in diagnostic_paths]
        assert args.validation_only and not paths["test"]
        (output / "pretraining_protocol.json").write_bytes(args.protocol.read_bytes())
    meta = json.loads((ROOT / "meta.json").read_text())
    assert meta["collision_radius"] == 0.03 and meta["trajectory_length"] == 400
    stats = build_statistics(paths["train"])
    # The pinned example computes normalization after input/target noise. Use its
    # analytic equal-trajectory variance for freshly sampled per-update noise.
    if args.noise:
        free_fraction = np.mean([np.mean(load_trajectory(p)["node_type"] == 0) for p in paths["train"]])
        st = np.asarray(stats["target_std"])
        st[:3] = np.sqrt(st[:3] ** 2 + free_fraction * args.noise ** 2)
        stats["target_std"] = st.tolist()
        stats["noise_normalization"] = "Analytic variance of fresh zero-mean independent noise; equal trajectory free-node fractions"
    model = HybridMeshGraphNet(**architecture(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
    config.update({"architecture": architecture(args), "parameters": sum(p.numel() for p in model.parameters()),
                   "splits": {s: [p.stem for p in ps] for s, ps in paths.items()},
                   "dataset_npz_sha256": dataset_hashes,
                   "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"), "torch": torch.__version__,
                   "torch_geometric": importlib.metadata.version("torch-geometric"),
                   "torch_scatter": importlib.metadata.version("torch-scatter"),
                   "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(device),
                   "training_contract": "Official node_type onehot, bidirectional tetra mesh edges, radius .03 world edges excluding mesh edges; increments xyz + next nodal scalar stress",
                   "world_edge_backend": "scipy.spatial.cKDTree on CPU, complete radius graph with no neighbor cap; same geometric criterion as the official helper",
                   "code_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                   "helper_sha256": hashlib.sha256((ROOT / "plate_data.py").read_bytes()).hexdigest(),
                   "evaluator_sha256": hashlib.sha256((ROOT / "plate_evaluation.py").read_bytes()).hexdigest(),
                   "selection_policy_sha256": hashlib.sha256((ROOT / "selection_policy.py").read_bytes()).hexdigest(),
                   "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest() if args.protocol else None,
                   "test_npz_opened_or_hashed": not args.validation_only,
                   "selection_description": "Two fixed validation 399-step AR cases; failure then any inversion then worst per-case displacement/stress relative-L2 minimax" if args.checkpoint_selection == "rollout" else "Minimum teacher-forced validation normalized MSE",
                   "learning_rate_recipe": "Per-update official decay" if args.lr_decay_per_update else "Bounded prior pilot decay .98 per epoch",
                   "recipe_deviations": ["32 or fewer independent train trajectories and sparse adjacent-step anchors rather than full-data training", "Fresh-noise analytic target variance using train-only statistics rather than original fixed noise draw", "CPU SciPy complete-radius geometric graph rather than original neighbor-search backend"]})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    (output / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        stats = saved["normalization"]
        (output / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    if args.probe:
        probe_path = next((p for p in paths["train"] if p.stem == args.probe_trajectory), None) if args.probe_trajectory else paths["train"][0]
        assert probe_path is not None, "Probe trajectory must belong to the selected training split"
        record = load_trajectory(probe_path)
        assert 0 <= args.probe_frame <= 398
        times = []
        for step in range(args.probe + 2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, record, min(args.probe_frame + step, 398), stats, device, args.noise)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            peak_rss = max(peak_rss, process.memory_info().rss)
        model.eval()
        inference_times = []
        forward_times = []
        probe_positions = record["world_pos"][min(args.probe_frame + args.probe + 1, 398)]
        with torch.inference_mode():
            for step in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                inputs = prepare_graph(record, probe_positions, stats, device)
                model(*inputs).cpu().numpy()
                torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - started)
            for step in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model(*inputs)
                torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - started)
        peak_rss = max(peak_rss, process.memory_info().rss)
        result = {"measured_steps": args.probe, "warmup_steps": 2, "mean_step_seconds": float(np.mean(times[2:])),
                  "allocated_peak_bytes": torch.cuda.max_memory_allocated(), "rss_sampled_peak_bytes": peak_rss,
                  "loss": float(loss.detach()), "parameters": config["parameters"], "host_available_bytes": psutil.virtual_memory().available,
                  "timing_scope": "CPU geometry graph preparation + CUDA forward/backward/Adam, synchronized; excludes imports and data loading"}
        result.update({"timed_inference_steps": 3, "inference_warmup_steps": 2,
                       "inference_graph_forward_output_mean_seconds": float(np.mean(inference_times[2:])),
                       "forward_only_mean_seconds": float(np.mean(forward_times[2:])),
                       "inference_scope": "Training-geometry resource timing only: graph preparation/H2D + CUDA forward + output transfer; excludes file I/O and accuracy/strain evaluation. Pure forward uses prebuilt GPU inputs.",
                       "resource_model_not_accuracy_trained": True})
        result.update({"trajectory": record["name"], "starting_frame": args.probe_frame,
                       "nodes": len(record["mesh_pos"]), "mesh_edges": record["mesh_edges"].shape[1],
                       "world_edges_at_start": world_edges(record["world_pos"][args.probe_frame], record["mesh_edges"]).shape[1]})
        (output / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        print(json.dumps(result), flush=True)
        return
    history, selection_history = [], []
    started = time.perf_counter()
    best = saved["validation_loss"] if args.checkpoint else float("inf")
    best_epoch = saved["epoch"] if args.checkpoint else None
    best_rank, selected_record = None, None
    frames = np.unique(np.linspace(0, 398, args.frames_per_trajectory, dtype=int))
    valid_frames = np.unique(np.linspace(0, 398, 12, dtype=int))
    config["training_anchor_current_frames"] = frames.tolist()
    config["teacher_validation_current_frames"] = valid_frames.tolist()
    config["selection_epochs"] = selection_epochs
    config["selection_trajectories"] = [p.stem for p in diagnostic_paths]
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False), encoding="utf8")
    if not args.checkpoint:
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            model.train()
            losses = []
            shuffled_paths = paths["train"].copy()
            random.shuffle(shuffled_paths)
            for index, path in enumerate(shuffled_paths):
                record = load_trajectory(path)
                shuffled_frames = frames.copy()
                np.random.shuffle(shuffled_frames)
                for frame in shuffled_frames:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_for(model, record, int(frame), stats, device, args.noise)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss epoch={epoch + 1} case={path.stem} frame={frame}")
                    loss.backward()
                    optimizer.step()
                    if args.lr_decay_per_update is not None:
                        for group in optimizer.param_groups:
                            group["lr"] *= args.lr_decay_per_update
                    losses.append(float(loss.detach()))
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 8 == 0 or index + 1 == len(shuffled_paths):
                    print(json.dumps({"stage": "training_progress", "epoch": epoch + 1,
                                      "completed_trajectories": index + 1, "updates": len(losses),
                                      "epoch_seconds_so_far": time.perf_counter() - epoch_start}), flush=True)
            del record
            optimizer.zero_grad(set_to_none=True)
            val = teacher_validation(model, paths["valid"], valid_frames, stats, device, detailed=True)
            row = {"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val["normalized_mse"],
                   "teacher_validation": val, "learning_rate_after_updates": optimizer.param_groups[0]["lr"],
                   "host_available_bytes": psutil.virtual_memory().available}
            state = {"model": model.state_dict(), "normalization": stats, "config": config,
                     "epoch": epoch + 1, "validation_loss": val["normalized_mse"]}
            if args.checkpoint_selection == "teacher":
                if val["normalized_mse"] < best:
                    best, best_epoch = val["normalized_mse"], epoch + 1
                    torch.save(state, output / "checkpoint.pt")
            elif epoch + 1 in selection_epochs:
                candidate_path = output / f"candidate_epoch_{epoch + 1:03d}.pt"
                torch.save(state, candidate_path)
                candidate_results = evaluate(model, diagnostic_paths, stats, args, output, device,
                                             compute_teacher=False, prefix=f"selection_epoch_{epoch + 1:03d}__")
                selection = rollout_selection_record(candidate_results)
                selection.update({"epoch": epoch + 1, "checkpoint_file": candidate_path.name,
                                  "checkpoint_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                                  "teacher_validation": val})
                selection_history.append(selection)
                rank = comparison_key(selection)
                # Equal scores preserve the earlier epoch: the rule is frozen before training.
                if best_rank is None or rank < best_rank:
                    best_rank, selected_record = rank, selection
                    best, best_epoch = val["normalized_mse"], epoch + 1
                    state["rollout_selection"] = selection
                    torch.save(state, output / "checkpoint.pt")
                row["rollout_selection"] = selection
                (output / "selection_history.json").write_text(json.dumps(selection_history, indent=2, allow_nan=False), encoding="utf8")
            row["seconds"] = time.perf_counter() - epoch_start
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf8")
            print(json.dumps(row, allow_nan=False), flush=True)
            if args.lr_decay_per_update is None:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.98
        # Preserve last-epoch optimizer/RNG state before loading the selected model.
        # Saving this state does not claim that a resume command has been implemented.
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "normalization": stats,
                    "config": config, "epoch": args.epochs, "random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(), "torch_random_state": torch.get_rng_state(),
                    "cuda_random_state": torch.cuda.get_rng_state()}, output / "last_training_state.pt")
        saved = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        selected_record = saved.get("rollout_selection")
    training_seconds = time.perf_counter() - started
    validation_results = evaluate(model, paths["valid"], stats, args, output, device) if args.evaluate_validation or args.validation_only else None
    results = validation_results if args.validation_only else evaluate(model, paths["test"], stats, args, output, device)
    final_checkpoint = args.checkpoint if args.checkpoint else output / "checkpoint.pt"
    summary = {"status": "completed_actual_gpu_training_and_validation_only" if args.validation_only and not args.checkpoint else ("completed_actual_gpu_training_and_heldout_evaluation" if not args.checkpoint else "completed_checkpoint_reuse_inference"),
               "checkpoint_sha256": hashlib.sha256(final_checkpoint.read_bytes()).hexdigest(),
               "train_seconds": training_seconds,
               "train_timing_scope": "Training, per-epoch teacher validation, scheduled AR checkpoint diagnostics and checkpoint I/O; excludes final full-validation rollout",
               "best_epoch": best_epoch, "selected_checkpoint_teacher_validation_mse": best,
               "minimum_observed_teacher_validation_mse": min((r["validation_mse"] for r in history), default=None),
               "checkpoint_selection": args.checkpoint_selection, "selected_rollout_ranking": selected_record,
               "selected_candidate_geometry_admissible": (not selected_record["failure_flag"] and not selected_record["geometry_invalid_flag"]) if selected_record else None,
               "accuracy_qualified": False,
               "accuracy_statement": "Actual validation diagnostics only; suitability requires reviewing per-field error, baselines, geometry and generalization. No automatic accuracy qualification.",
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
               "sampled_peak_rss_bytes": max(peak_rss, process.memory_info().rss),
               "results": results, "pooled_metrics": pooled_metrics(results),
               "evaluation_completed_all_cases": all(r.get("status") == "completed" for r in results),
               "evaluation_failed_case_count": sum(r.get("status") != "completed" for r in results),
               "evaluation_split": "valid" if args.validation_only else "test",
               "evaluation_role": args.evaluation_label,
               "fresh_test_evaluated": not args.validation_only and args.evaluation_label == "fresh_final",
               "test_npz_opened_or_hashed": not args.validation_only,
               "validation_results": validation_results,
               "validation_pooled_metrics": pooled_metrics(validation_results) if validation_results else None,
               "limitations": ["Bounded subset of independent official trajectories, not the full paper recipe.",
                   "Future boundary motion is prescribed, as in official autoregressive inference.",
                   "Original scalar stress invariant and physical units require primary-source interpretation; no SI units are inferred.",
                   "No original COMSOL strain label exists; GL strain is derived from coordinates."]}
    if not summary["evaluation_completed_all_cases"]:
        summary["status"] = "training_or_checkpoint_load_completed_with_evaluation_failures"
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf8")


if __name__ == "__main__":
    main()
