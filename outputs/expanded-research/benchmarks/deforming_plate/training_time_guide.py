"""Project candidate budgets only from a completed 128/15 resource measurement.

Standard-library only. Does not import frameworks, construct a model, read FEM
field labels, or open the fresh-final-test predictions.
"""
import argparse
import hashlib
import json
import math
import pathlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=pathlib.Path, required=True)
    parser.add_argument("--split-contract", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    attempt_path = args.probe_dir / "resource-attempt.json"
    attempt = json.loads(attempt_path.read_text())
    if attempt.get("status") != "completed_resource_probe":
        raise ValueError("Do not project a training budget from a failed/incomplete resource attempt")
    config_path = args.probe_dir / "config.json"
    probe_path = args.probe_dir / "probe.json"
    cfg = json.loads(config_path.read_text())
    architecture = cfg["architecture"]
    width_keys = ["hidden_dim_processor", "hidden_dim_node_encoder", "hidden_dim_edge_encoder", "hidden_dim_node_decoder"]
    if any(architecture[k] != 128 for k in width_keys) or architecture["processor_size"] != 15:
        raise ValueError("This guide requires an actual width-128 / 15-block measurement, not small-model extrapolation")
    measured = attempt["measurement"]
    if measured != json.loads(probe_path.read_text()):
        raise ValueError("Attempt measurement does not match preserved probe.json")
    for key, path in [("config_sha256", config_path), ("probe_sha256", probe_path)]:
        if attempt.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Resource artifact hash mismatch: {path}")
    if not 1 <= measured["measured_steps"] <= 10 or measured["warmup_steps"] != 2:
        raise ValueError("Unexpected resource update counts")
    update = measured["mean_step_seconds"]
    inference = measured["inference_graph_forward_output_mean_seconds"]
    if any(not math.isfinite(v) or v <= 0 for v in [update, inference, measured["forward_only_mean_seconds"]]):
        raise ValueError("Resource timings must be finite positive measured values")
    contract = json.loads(args.split_contract.read_text())
    ntrain, nvalid, ntest = [len(contract[k]) for k in ["training", "validation", "fresh_final_test"]]
    if (ntrain, nvalid, ntest) != (32, 6, 6):
        raise ValueError("The resource guide must match the frozen 32/6/6 expanded split")
    groups = [contract[k] for k in ["training", "validation", "fresh_final_test"]]
    if any(len(g) != len(set(g)) for g in groups) or len(set().union(*groups)) != ntrain + nvalid + ntest:
        raise ValueError("The frozen split contains duplicate/overlapping trajectory names")
    candidates = []
    final_evaluation_calls = (nvalid + ntest) * 399 * 2
    final_eval_proxy = final_evaluation_calls * inference
    for anchors in [8, 16, 32]:
        for epochs in [5, 10, 20]:
            updates = ntrain * anchors * epochs
            per_epoch_validation_calls = nvalid * 12
            training_seconds = updates * update
            teacher_validation_seconds = epochs * per_epoch_validation_calls * inference
            candidates.append({
                "candidate_training_step_anchors_per_trajectory": anchors,
                "candidate_epochs": epochs, "optimizer_updates": updates,
                "training_proxy_seconds": training_seconds,
                "teacher_validation_proxy_seconds": teacher_validation_seconds,
                "post_training_validation_and_fresh_test_proxy_seconds": final_eval_proxy,
                "combined_proxy_minutes": (training_seconds + teacher_validation_seconds + final_eval_proxy) / 60,
            })
    result = {
        "status": "measurement_based_candidate_time_guide_not_selected_training_configuration",
        "resource_attempt_sha256": hashlib.sha256(attempt_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "time_guide_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "split_contract_sha256": hashlib.sha256(args.split_contract.read_bytes()).hexdigest(),
        "architecture": architecture, "parameters": cfg["parameters"],
        "measured_update_seconds": update,
        "measured_graph_forward_output_seconds": inference,
        "measured_forward_only_seconds": measured["forward_only_mean_seconds"],
        "measured_resource_geometry": {k: measured[k] for k in ["trajectory", "starting_frame", "nodes", "mesh_edges", "world_edges_at_start"]},
        "actual_resource_memory": {k: attempt.get(k) for k in ["memory", "cuda"]},
        "checkpoint_segments": cfg["checkpoint_segments"],
        "candidates": candidates,
        "limitations": [
            "Projections use one sampled high-edge training geometry; they are not guaranteed upper bounds or wall-clock forecasts.",
            "File I/O, repeated trajectory decompression, model checkpoint serialization, strain calculations, plots and OS paging are not measured by these proxies.",
            "Per-epoch validation uses 12 teacher-forced anchors per validation trajectory. Post-training evaluation models 399 AR steps and 399 separate teacher-forced calls per validation/final-test trajectory.",
            "The final fresh-test inference must run only after architecture, training cap and checkpoint selection have been locked using training/validation information.",
            "No candidate is selected by this script. Actual first-epoch duration and validation learning curves must update the cap decision.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as handle:
        handle.write(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
