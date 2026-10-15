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


def loss_for(model, record, frame, stats, device, noise_std=0):
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
    return (prediction - target).square().mean()


def metric(prediction, truth):
    p, t = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    error = p - t
    norm = np.linalg.norm(t)
    return {"relative_l2": float(np.linalg.norm(error) / norm) if norm else None,
            "component_rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_abs_component_error": float(np.abs(error).max()),
            "truth_l2": float(norm), "values": int(t.size)}


@torch.inference_mode()
def teacher_validation(model, paths, frames, stats, device):
    model.eval()
    losses = []
    for path in paths:
        record = load_trajectory(path)
        for frame in frames:
            losses.append(float(loss_for(model, record, int(frame), stats, device)))
    return float(np.mean(losses))


@torch.inference_mode()
def evaluate(model, paths, stats, args, output, device):
    model.eval()
    results = []
    np.random.seed(args.seed)
    for path in paths:
        record = load_trajectory(path)
        free = record["node_type"][:, 0] == 0
        steps = min(args.rollout_steps, 399)
        current = record["world_pos"][0].copy()
        positions, stresses, zero_positions = [], [], []
        teacher_positions, teacher_stresses = [], []
        means = np.asarray(stats["target_mean"], dtype=np.float32)
        stds = np.asarray(stats["target_std"], dtype=np.float32)
        started = time.perf_counter()
        for frame in range(steps):
            raw = model(*prepare_graph(record, current, stats, device)).cpu().numpy() * stds + means
            following = current + raw[:, :3]
            following[~free] = record["world_pos"][frame + 1, ~free]
            positions.append(following.copy())
            stresses.append(raw[:, 3:4])
            baseline = record["world_pos"][0].copy()
            baseline[~free] = record["world_pos"][frame + 1, ~free]
            zero_positions.append(baseline)
            # Teacher-forced and autoregressive accuracy are reported separately.
            tf = model(*prepare_graph(record, record["world_pos"][frame], stats, device)).cpu().numpy() * stds + means
            teacher_positions.append(record["world_pos"][frame] + tf[:, :3])
            teacher_stresses.append(tf[:, 3:4])
            current = following
        elapsed = time.perf_counter() - started
        positions = np.stack(positions)
        stresses = np.stack(stresses)
        truth_pos = record["world_pos"][1:steps + 1]
        truth_stress = record["stress"][1:steps + 1]
        reference = record["world_pos"][0]
        derived_frames = sorted(set([steps - 1] + list(range(0, steps, 20))))
        pred_strain, true_strain = [], []
        # Clamp nodes belong to the plate; actuator-body tetrahedra do not.
        cell_mask = (record["node_type"][:, 0][record["cells"]] != 1).all(axis=1)
        assert cell_mask.any()
        cells = record["cells"][cell_mask]
        predicted_inversions = []
        truth_inversions = []
        ref_tet = record["mesh_pos"][cells]
        ref_det = np.linalg.det((ref_tet[:, 1:] - ref_tet[:, :1]).transpose(0, 2, 1))
        for frame in derived_frames:
            pred_strain.append(deformation_strain(record["mesh_pos"], positions[frame], cells)[0])
            true_strain.append(deformation_strain(record["mesh_pos"], truth_pos[frame], cells)[0])
            for coordinates, counts in [(positions[frame], predicted_inversions), (truth_pos[frame], truth_inversions)]:
                tet = coordinates[cells]
                det = np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1))
                counts.append(int((det / ref_det <= 0).sum()))
        pred_strain = np.stack(pred_strain)
        true_strain = np.stack(true_strain)
        final = {
            "trajectory": record["name"], "nodes": len(reference), "free_nodes": int(free.sum()),
            "steps": steps, "boundary": "type 1 actuator and type 3 clamp follow prescribed FEM boundary positions at each step",
            "rollout_free_displacement": metric(positions[:, free] - reference[free], truth_pos[:, free] - reference[free]),
            "rollout_free_stress": metric(stresses[:, free], truth_stress[:, free]),
            "predicted_negative_von_mises_values": int((stresses[:, free] < 0).sum()),
            "predicted_von_mises_minimum": float(stresses[:, free].min()),
            "truth_von_mises_maximum": float(truth_stress[:, free].max()),
            "teacher_forced_free_displacement_increment": metric(np.stack(teacher_positions)[:, free] - record["world_pos"][:steps, free], np.diff(record["world_pos"][:steps + 1], axis=0)[:, free]),
            "teacher_forced_free_stress": metric(np.stack(teacher_stresses)[:, free], truth_stress[:, free]),
            "zero_increment_baseline_free_displacement": metric(np.stack(zero_positions)[:, free] - reference[free], truth_pos[:, free] - reference[free]),
            "zero_stress_baseline": metric(np.zeros_like(truth_stress[:, free]), truth_stress[:, free]),
            "training_mean_constant_stress_baseline": metric(np.full_like(truth_stress[:, free], means[3]), truth_stress[:, free]),
            "derived_GL_strain": metric(pred_strain, true_strain),
            "derived_GL_strain_frames": [i + 1 for i in derived_frames], "derived_GL_strain_tetrahedra": int(cell_mask.sum()),
            "derived_GL_strain_component_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
            "derived_GL_strain_weighting": "Unweighted per sampled tetrahedron/component; tensor shear; selected frames only",
            "predicted_inverted_tetrahedra_per_sampled_frame": predicted_inversions,
            "truth_inverted_tetrahedra_per_sampled_frame": truth_inversions,
            "strain_label_caveat": "Both reference and prediction are kinematically derived from FEM/AI nodal positions; no original COMSOL strain tensor is available.",
            "coordinate_units": "dataset coordinate units (not confirmed SI)", "stress_units": "dataset stress units (not confirmed Pa/MPa)",
            "evaluation_seconds": elapsed,
            "evaluation_timing_scope": "Combined autoregressive + teacher-forced forward, CPU graph preparation and output transfer; excludes strain calculation and file writing",
        }
        assert all(np.isfinite(x).all() for x in [positions, stresses, pred_strain, true_strain])
        np.savez_compressed(output / f"{record['name']}_evaluation.npz", prediction_positions=positions, truth_positions=truth_pos,
                            prediction_stress=stresses, truth_stress=truth_stress, reference_positions=reference,
                            teacher_forced_positions=np.stack(teacher_positions), teacher_forced_stress=np.stack(teacher_stresses),
                            free_mask=free, predicted_derived_strain=pred_strain, truth_derived_strain=true_strain,
                            strain_frames=np.asarray(derived_frames), cells=cells)
        results.append(final)
        print(json.dumps(final), flush=True)
    return results


def pooled_metrics(results):
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
    output = args.output.resolve()
    if (output / "metrics.json").exists() or (args.probe and (output / "probe.json").exists()):
        raise FileExistsError(f"Completed evidence already exists: {output}; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "executed_source.py").write_bytes(pathlib.Path(__file__).read_bytes())
    (output / "plate_data_executed.py").write_bytes((ROOT / "plate_data.py").read_bytes())
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
    paths = {split: sorted((ROOT / "data").glob(f"{split}_*.npz")) for split in ["train", "valid", "test"]}
    if args.train_count:
        assert len(paths["train"]) >= args.train_count
        paths["train"] = paths["train"][:args.train_count]
    if args.valid_count:
        assert len(paths["valid"]) >= args.valid_count
        paths["valid"] = paths["valid"][:args.valid_count]
    paths["test"] = paths["test"][args.test_start:args.test_stop]
    assert all(paths.values())
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
                   "dataset_npz_sha256": {p.stem: hashlib.sha256(p.read_bytes()).hexdigest() for ps in paths.values() for p in ps},
                   "physicsnemo": importlib.metadata.version("nvidia-physicsnemo"), "torch": torch.__version__,
                   "torch_geometric": importlib.metadata.version("torch-geometric"),
                   "training_contract": "Official node_type onehot, bidirectional tetra mesh edges, radius .03 world edges excluding mesh edges; increments xyz + next nodal von Mises stress",
                   "world_edge_backend": "scipy.spatial.cKDTree on CPU, complete radius graph with no neighbor cap; same geometric criterion as the official helper",
                   "code_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                   "helper_sha256": hashlib.sha256((ROOT / "plate_data.py").read_bytes()).hexdigest(),
                   "checkpoint_selection": "Minimum teacher-forced validation normalized MSE; not a long-rollout convergence selection"})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    (output / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        stats = saved["normalization"]
        (output / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf8")
    if args.probe:
        record = load_trajectory(paths["train"][0])
        times = []
        for step in range(args.probe + 2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, record, step, stats, device, args.noise)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
            peak_rss = max(peak_rss, process.memory_info().rss)
        result = {"measured_steps": args.probe, "warmup_steps": 2, "mean_step_seconds": float(np.mean(times[2:])),
                  "allocated_peak_bytes": torch.cuda.max_memory_allocated(), "rss_sampled_peak_bytes": peak_rss,
                  "loss": float(loss.detach()), "parameters": config["parameters"], "host_available_bytes": psutil.virtual_memory().available,
                  "timing_scope": "CPU geometry graph preparation + CUDA forward/backward/Adam, synchronized; excludes imports and data loading"}
        (output / "probe.json").write_text(json.dumps(result, indent=2), encoding="utf8")
        print(json.dumps(result), flush=True)
        return
    history = []
    started = time.perf_counter()
    best = saved["validation_loss"] if args.checkpoint else float("inf")
    best_epoch = saved["epoch"] if args.checkpoint else None
    frames = np.unique(np.linspace(0, 398, args.frames_per_trajectory, dtype=int))
    valid_frames = np.unique(np.linspace(0, 398, 12, dtype=int))
    if not args.checkpoint:
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            model.train()
            losses = []
            shuffled_paths = paths["train"].copy()
            random.shuffle(shuffled_paths)
            for path in shuffled_paths:
                record = load_trajectory(path)
                shuffled_frames = frames.copy()
                np.random.shuffle(shuffled_frames)
                for frame in shuffled_frames:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_for(model, record, int(frame), stats, device, args.noise)
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                peak_rss = max(peak_rss, process.memory_info().rss)
            val = teacher_validation(model, paths["valid"], valid_frames, stats, device)
            if val < best:
                best, best_epoch = val, epoch + 1
                torch.save({"model": model.state_dict(), "normalization": stats, "config": config,
                            "epoch": best_epoch, "validation_loss": best}, output / "checkpoint.pt")
            row = {"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val,
                   "seconds": time.perf_counter() - epoch_start, "host_available_bytes": psutil.virtual_memory().available}
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
            print(json.dumps(row), flush=True)
            optimizer.param_groups[0]["lr"] *= 0.98
        saved = torch.load(output / "checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
    training_seconds = time.perf_counter() - started
    validation_results = evaluate(model, paths["valid"], stats, args, output, device) if args.evaluate_validation or args.validation_only else None
    results = validation_results if args.validation_only else evaluate(model, paths["test"], stats, args, output, device)
    final_checkpoint = args.checkpoint if args.checkpoint else output / "checkpoint.pt"
    summary = {"status": "completed_actual_gpu_training_and_heldout_evaluation" if not args.checkpoint else "completed_checkpoint_reuse_inference",
               "checkpoint_sha256": hashlib.sha256(final_checkpoint.read_bytes()).hexdigest(),
               "train_seconds": training_seconds, "best_epoch": best_epoch, "best_validation_mse": best,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "sampled_peak_rss_bytes": max(peak_rss, process.memory_info().rss),
               "results": results, "pooled_metrics": pooled_metrics(results),
               "evaluation_split": "valid" if args.validation_only else "test",
               "evaluation_role": args.evaluation_label,
               "validation_results": validation_results,
               "validation_pooled_metrics": pooled_metrics(validation_results) if validation_results else None,
               "limitations": ["Bounded subset of independent official trajectories, not the full paper recipe.",
                   "Boundary future motion is prescribed, as in the official autoregressive inference.",
                   "No original COMSOL strain label exists; GL strain is derived from coordinates."]}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
