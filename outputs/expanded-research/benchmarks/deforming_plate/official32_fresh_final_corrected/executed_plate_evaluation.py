"""Actual plate rollout fields and geometry checks; no held-out split selection."""
import json
import time
import traceback

import numpy as np
import torch

from plate_data import deformation_strain, load_trajectory


def _inversions(coordinates, cells, reference_det):
    tet = np.asarray(coordinates, dtype=np.float64)[cells]
    det = np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1)) / reference_det
    return int((det <= 0).sum()), float(det.min())


def _one(model, path, stats, args, output, device, prepare_graph, metric,
         compute_teacher, prefix, context):
    record = load_trajectory(path)
    free = record["node_type"][:, 0] == 0
    steps = min(args.rollout_steps, 399)
    assert steps > 0 and free.any()
    reference = record["world_pos"][0]
    current = reference.copy()
    means = np.asarray(stats["target_mean"], dtype=np.float32)
    stds = np.asarray(stats["target_std"], dtype=np.float32)
    cell_mask = (record["node_type"][:, 0][record["cells"]] != 1).all(axis=1)
    cells = record["cells"][cell_mask]
    assert len(cells)
    ref_tet = record["mesh_pos"].astype(np.float64)[cells]
    ref_det = np.linalg.det((ref_tet[:, 1:] - ref_tet[:, :1]).transpose(0, 2, 1))
    assert np.all(abs(ref_det) > 1e-15)
    positions, stresses, teacher_positions, teacher_stresses = [], [], [], []
    all_pred_inv, all_true_inv, all_pred_det, all_true_det = [], [], [], []
    graph_sizes, graph_seconds, forward_seconds = [], [], []
    context["completed_AR_graph_sizes"] = graph_sizes
    started = time.perf_counter()
    for frame in range(steps):
        context.update({"step": frame + 1, "stage": "autoregressive_graph_or_forward"})
        tick = time.perf_counter()
        inputs = prepare_graph(record, current, stats, device, graph_context=context)
        graph_seconds.append(time.perf_counter() - tick)
        graph_sizes.append({k: context[k] for k in ["graph_nodes", "mesh_directed_edges", "world_directed_edges", "total_directed_edges"]})
        tick = time.perf_counter()
        raw = model(*inputs).cpu().numpy() * stds + means
        forward_seconds.append(time.perf_counter() - tick)
        if not np.isfinite(raw).all():
            raise FloatingPointError("Nonfinite autoregressive output")
        following = current + raw[:, :3]
        following[~free] = record["world_pos"][frame + 1, ~free]
        if not np.isfinite(following).all():
            raise FloatingPointError("Nonfinite integrated coordinates")
        positions.append(following.copy())
        stresses.append(raw[:, 3:4])
        inv, det = _inversions(following, cells, ref_det)
        all_pred_inv.append(inv)
        all_pred_det.append(det)
        inv, det = _inversions(record["world_pos"][frame + 1], cells, ref_det)
        all_true_inv.append(inv)
        all_true_det.append(det)
        if compute_teacher:
            context["stage"] = "teacher_forced_graph_or_forward"
            tf = model(*prepare_graph(record, record["world_pos"][frame], stats, device)).cpu().numpy() * stds + means
            if not np.isfinite(tf).all():
                raise FloatingPointError("Nonfinite teacher-forced output")
            teacher_positions.append(record["world_pos"][frame] + tf[:, :3])
            teacher_stresses.append(tf[:, 3:4])
        current = following
        if (frame + 1) % 100 == 0:
            print(json.dumps({"stage": "rollout_progress", "trajectory": record["name"],
                              "prefix": prefix, "completed_steps": frame + 1,
                              "seconds_so_far": time.perf_counter() - started}), flush=True)
    loop_seconds = time.perf_counter() - started
    positions, stresses = np.stack(positions), np.stack(stresses)
    truth_pos = record["world_pos"][1:steps + 1]
    truth_stress = record["stress"][1:steps + 1]
    derived_frames = sorted(set([steps - 1] + list(range(0, steps, 20))))
    pred_strain, true_strain = [], []
    context["stage"] = "derived_strain_and_metrics"
    strain_started = time.perf_counter()
    for frame in derived_frames:
        pred_strain.append(deformation_strain(record["mesh_pos"], positions[frame], cells)[0])
        true_strain.append(deformation_strain(record["mesh_pos"], truth_pos[frame], cells)[0])
    pred_strain, true_strain = np.stack(pred_strain), np.stack(true_strain)
    strain_seconds = time.perf_counter() - strain_started
    assert all(np.isfinite(a).all() for a in [positions, stresses, pred_strain, true_strain])
    result = {
        "status": "completed", "trajectory": record["name"], "nodes": len(reference),
        "free_nodes": int(free.sum()), "steps": steps,
        "boundary": "Actuator type1 and clamp type3 follow prescribed next boundary positions; free type0 is predicted",
        "rollout_free_displacement": metric(positions[:, free] - reference[free], truth_pos[:, free] - reference[free]),
        "rollout_free_stress": metric(stresses[:, free], truth_stress[:, free]),
        "predicted_negative_von_mises_values": int((stresses[:, free] < 0).sum()),
        "predicted_von_mises_minimum": float(stresses[:, free].min()),
        "truth_negative_von_mises_values": int((truth_stress[:, free] < 0).sum()),
        "truth_von_mises_maximum": float(truth_stress[:, free].max()),
        "zero_increment_baseline_free_displacement": metric(np.zeros_like(truth_pos[:, free]), truth_pos[:, free] - reference[free]),
        "zero_stress_baseline": metric(np.zeros_like(truth_stress[:, free]), truth_stress[:, free]),
        "training_mean_constant_stress_baseline": metric(np.full_like(truth_stress[:, free], means[3]), truth_stress[:, free]),
        "derived_GL_strain": metric(pred_strain, true_strain),
        "derived_GL_strain_frames": [f + 1 for f in derived_frames],
        "derived_GL_strain_tetrahedra": len(cells),
        "derived_GL_strain_component_order": ["xx", "yy", "zz", "xy", "yz", "xz"],
        "derived_GL_strain_weighting": "Unweighted per sampled tetrahedron/component; tensor shear; selected strain frames only",
        "all_frame_predicted_inverted_tetrahedra": all_pred_inv,
        "all_frame_truth_inverted_tetrahedra": all_true_inv,
        "all_frame_predicted_minimum_detF": all_pred_det,
        "all_frame_truth_minimum_detF": all_true_det,
        "predicted_inverted_tetrahedra_per_sampled_frame": [all_pred_inv[f] for f in derived_frames],
        "truth_inverted_tetrahedra_per_sampled_frame": [all_true_inv[f] for f in derived_frames],
        "strain_label_caveat": "Kinematically derived from FEM/AI positions, not an original COMSOL strain output",
        "coordinate_units": "unconfirmed dataset coordinate units", "stress_units": "unconfirmed dataset stress units",
        "teacher_forcing_computed": compute_teacher, "evaluation_seconds": loop_seconds,
        "all_frame_graph_sizes": graph_sizes,
        "maximum_AR_total_directed_edges": max(g["total_directed_edges"] for g in graph_sizes),
        "AR_graph_preparation_seconds": sum(graph_seconds),
        "AR_forward_and_output_transfer_seconds": sum(forward_seconds),
        "derived_strain_seconds": strain_seconds,
        "timing_detail": "Graph preparation includes CPU geometry/features and H2D allocations; forward includes synchronized output transfer. These are local sequential wall measurements, not isolated CUDA kernel timings.",
        "evaluation_timing_scope": "AR graph/forward/output plus every-frame determinant checks, and teacher forcing only when enabled; excludes strain postprocessing and file writing",
    }
    arrays = {"prediction_positions": positions, "truth_positions": truth_pos, "prediction_stress": stresses,
              "truth_stress": truth_stress, "reference_positions": reference, "free_mask": free,
              "predicted_derived_strain": pred_strain, "truth_derived_strain": true_strain,
              "strain_frames": np.asarray(derived_frames), "cells": cells,
              "original_array_tetrahedron_indices_not_solver_ids": np.flatnonzero(cell_mask)}
    if compute_teacher:
        teacher_positions, teacher_stresses = np.stack(teacher_positions), np.stack(teacher_stresses)
        result["teacher_forced_free_displacement_increment"] = metric(teacher_positions[:, free] - record["world_pos"][:steps, free], np.diff(record["world_pos"][:steps + 1], axis=0)[:, free])
        result["teacher_forced_free_stress"] = metric(teacher_stresses[:, free], truth_stress[:, free])
        arrays.update({"teacher_forced_positions": teacher_positions, "teacher_forced_stress": teacher_stresses})
    field_path = output / f"{prefix}{record['name']}_evaluation.npz"
    if field_path.exists():
        raise FileExistsError(f"Preserved field file already exists: {field_path}")
    write_started = time.perf_counter()
    np.savez_compressed(field_path, **arrays)
    result["field_write_seconds"] = time.perf_counter() - write_started
    result["field_file"] = field_path.name
    return result


@torch.inference_mode()
def evaluate(model, paths, stats, args, output, device, *, prepare_graph, metric,
             compute_teacher=True, prefix=""):
    model.eval()
    results = []
    for path in paths:
        for suffix in ["evaluation.npz", "failure.txt", "metrics.json"]:
            if (output / f"{prefix}{path.stem}_{suffix}").exists():
                raise FileExistsError("Evaluation evidence exists; use a fresh directory/prefix")
    for path in paths:
        context = {"trajectory": path.stem, "step": None, "stage": "load_record"}
        failed = False
        try:
            result = _one(model, path, stats, args, output, device, prepare_graph, metric,
                          compute_teacher, prefix, context)
        except Exception as exc:
            failed = True
            failure_path = output / f"{prefix}{path.stem}_failure.txt"
            with failure_path.open("x", encoding="utf8") as handle:
                handle.write(traceback.format_exc())
            result = {"status": "failed", "trajectory": path.stem, "failure_context": context.copy(),
                      "exception_type": type(exc).__name__, "exception": str(exc),
                      "failure_traceback_file": failure_path.name, "steps": min(args.rollout_steps, 399)}
        if failed:
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_error:
                result["cache_cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        result_path = output / f"{prefix}{path.stem}_metrics.json"
        with result_path.open("x", encoding="utf8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    return results
