"""Reload the validation-selected checkpoint in a new process; one fresh test attempt.

Stdlib gates run before any optional/framework import or fresh-test file access.
The training directory receives a permanent exclusive attempt receipt. There is
no training, model selection, normalization fitting, or retry in this program.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import traceback


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def json_equal(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    training = args.training_run.resolve()
    output = args.output.resolve()
    assert training != output
    output.mkdir(parents=True, exist_ok=False)
    attempt = {"status": "gating", "started_at_utc": now(), "process_id": os.getpid(),
               "command_arguments": sys.argv[1:], "fresh_test_content_access_started": False,
               "prediction_attempt_count_per_case": 1, "teacher_forcing_computed": False}
    attempt_path = output / "final-attempt.json"
    started = time.perf_counter()
    stop, sampler, process = threading.Event(), None, None
    peak_rss, available_min, samples = 0, None, 0
    failure = None
    try:
        lock_path = training / "preexecution_source_lock.json"
        lock = read(lock_path)
        assert lock["protocol_sha256"] == "a71497614937762d2ca7434d8b5c6e2f91d3fecb4c750ab9efd97192e776bf33"
        assert sha(training / "pretraining_protocol.json") == lock["protocol_sha256"]
        assert all(sha(root / name) == value for name, value in lock["source_sha256"].items())
        config, norm = read(training / "config.json"), read(training / "normalization.json")
        summary, train_attempt = read(training / "metrics.json"), read(training / "training-attempt.json")
        contract = read(args.contract)
        protocol = read(training / "pretraining_protocol.json")
        contract_input = next(item for item in protocol["provenance"]["metadata_inputs"]
                              if pathlib.PurePosixPath(item["path"]).name == "final-split-contract.json")
        assert sha(args.contract) == contract_input["sha256"], "Split contract differs from the frozen pretraining input"
        history = read(training / "selection_history.json")
        assert train_attempt["status"] in {"completed_actual_validation_experiment", "completed_training_with_incomplete_validation"}
        assert train_attempt["process_id"] != os.getpid(), "Final inference must use a new process"
        assert train_attempt["source_lock_sha256"] == sha(lock_path)
        assert train_attempt["metrics_sha256"] == sha(training / "metrics.json")
        for name in ["config.json", "normalization.json", "checkpoint.pt"]:
            assert train_attempt[f"{name.replace('.', '_')}_sha256"] == sha(training / name)
        assert not summary["test_npz_opened_or_hashed"] and summary["evaluation_split"] == "valid"
        assert [row["trajectory"] for row in summary["results"]] == protocol["validation"]
        assert contract["training"] == protocol["training"] == config["splits"]["train"]
        assert contract["validation"] == protocol["validation"] == config["splits"]["valid"]
        assert contract["development"] == ["test_0000", "test_0001"]
        fresh = [f"test_{i:04d}" for i in range(2, 8)]
        assert contract["fresh_final_test"] == protocol["fresh_final_test"] == fresh
        assert contract["fresh_final_test_accuracy_evaluated"] is False
        assert not config["splits"]["test"] and not config["test_npz_opened_or_hashed"]
        assert norm["fit_trajectories"] == protocol["training"]
        assert config["parameters"] == 3605636
        assert config["selection_epochs"] == protocol["selection_epochs"] == [1, 3, 5]
        assert config["selection_trajectories"] == protocol["selection_trajectories"] == ["valid_0000", "valid_0005"]
        for key, value in protocol["required_options"].items():
            assert config[key] == value, f"Frozen configuration differs: {key}"
        for key, name in [("code_sha256", "train_plate.py"), ("helper_sha256", "plate_data.py"),
                          ("evaluator_sha256", "plate_evaluation.py"), ("selection_policy_sha256", "selection_policy.py")]:
            assert config[key] == lock["source_sha256"][name]
        assert config["protocol_sha256"] == lock["protocol_sha256"]
        assert [row["epoch"] for row in history] == [1, 3, 5]
        # Infinity is used only internally, exactly as in the frozen policy.
        rank = lambda row: (*[float("inf") if v is None else v for v in row["selection_key"]], row["epoch"])
        selected = min(history, key=rank)
        assert selected["epoch"] == summary["best_epoch"]
        assert json_equal(selected, summary["selected_rollout_ranking"])
        checkpoint = training / "checkpoint.pt"
        assert sha(checkpoint) == summary["checkpoint_sha256"]
        for row in history:
            assert sha(training / row["checkpoint_file"]) == row["checkpoint_sha256"]
        # This lock is saved before opening/hashing fresh arrays or importing frameworks.
        gate = {"frozen_at_utc": now(), "training_run": str(training), "output": str(output),
                "checkpoint_sha256": sha(checkpoint), "selected_epoch": selected["epoch"],
                "selected_validation_rank": selected, "config_sha256": sha(training / "config.json"),
                "normalization_sha256": sha(training / "normalization.json"),
                "source_lock_sha256": sha(lock_path), "source_sha256": lock["source_sha256"],
                "split_contract_sha256": sha(args.contract), "fresh_final_test": fresh,
                "fresh_expected_npz_sha256": {n: contract["hashes"][n] for n in fresh},
                "test_use": "One fixed-checkpoint AR399 prediction attempt per case; no test-driven retuning",
                "original_paper_recipe_reproduced": False,
                "strain_label_caveat": "Derived Green-Lagrange from source/AI coordinates, not native COMSOL strain labels"}
        marker = training / "fresh-final-access.json"
        with marker.open("x", encoding="utf8") as stream:
            json.dump(gate, stream, indent=2, allow_nan=False)
        (output / "pre-evaluation-lock.json").write_bytes(marker.read_bytes())
        for name in lock["source_sha256"]:
            (output / f"executed_{name}").write_bytes((root / name).read_bytes())
        attempt.update({"status": "starting_fixed_inference", "pre_evaluation_lock_sha256": sha(marker),
                        "checkpoint_sha256": gate["checkpoint_sha256"], "selected_epoch": selected["epoch"]})
        attempt_path.write_text(json.dumps(attempt, indent=2, allow_nan=False), encoding="utf8")
        paths = [root / "data" / f"{name}.npz" for name in fresh]
        attempt["fresh_test_content_access_started"] = True
        assert all(sha(path) == gate["fresh_expected_npz_sha256"][path.stem] for path in paths)
        import psutil
        process = psutil.Process()

        def sample():
            nonlocal peak_rss, available_min, samples
            while not stop.is_set():
                peak_rss = max(peak_rss, process.memory_info().rss)
                available = psutil.virtual_memory().available
                available_min = available if available_min is None else min(available_min, available)
                samples += 1
                stop.wait(0.2)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        import train_plate as trainer
        torch = trainer.torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        model_args = argparse.Namespace(**config)
        model = trainer.HybridMeshGraphNet(**trainer.architecture(model_args)).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        assert json_equal(saved["config"], config)
        assert json_equal(saved["normalization"], norm)
        assert saved["epoch"] == selected["epoch"]
        assert json_equal(saved["rollout_selection"], selected)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        results = trainer.evaluate(model, paths, norm, model_args, output, device, compute_teacher=False)
        metrics = {"status": "completed_fixed_checkpoint_fresh_test" if all(r["status"] == "completed" for r in results) else "fixed_checkpoint_fresh_test_with_failed_cases",
                   "evaluation_role": "fresh_final", "evaluation_split": "test", "fresh_test_evaluated": True,
                   "checkpoint_sha256": gate["checkpoint_sha256"], "best_epoch": selected["epoch"],
                   "results": results, "pooled_metrics": trainer.pooled_metrics(results),
                   "evaluation_completed_all_cases": all(r["status"] == "completed" for r in results),
                   "evaluation_failed_case_count": sum(r["status"] != "completed" for r in results),
                   "teacher_forcing_computed": False, "accuracy_qualified": False,
                   "accuracy_statement": "Fixed-checkpoint independent test errors; no automatic deployment qualification",
                   "original_paper_recipe_reproduced": False,
                   "strain_label_caveat": gate["strain_label_caveat"], "stress_units": "Unconfirmed dataset units; no MPa conversion",
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        with (output / "metrics.json").open("x", encoding="utf8") as stream:
            json.dump(metrics, stream, indent=2, allow_nan=False)
        assert sha(checkpoint) == gate["checkpoint_sha256"]
        assert sha(training / "config.json") == gate["config_sha256"]
        assert sha(training / "normalization.json") == gate["normalization_sha256"]
        attempt.update({"status": metrics["status"], "metrics_sha256": sha(output / "metrics.json"),
                        "evaluation_completed_all_cases": metrics["evaluation_completed_all_cases"]})
    except BaseException as exc:
        failure = exc
        attempt.update({"status": "failed", "exception_type": type(exc).__name__, "exception": str(exc),
                        "is_cuda_out_of_memory": "outofmemory" in type(exc).__name__.lower() or "cuda out of memory" in str(exc).lower()})
        (output / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        attempt.update({"finished_at_utc": now(), "elapsed_wall_seconds_including_imports_gates_io": time.perf_counter() - started,
                        "memory": {"sampled_rss_peak_bytes": peak_rss, "sampled_host_available_minimum_bytes": available_min,
                                   "sampling_interval_seconds": 0.2, "samples": samples}})
        if process is not None:
            info = process.memory_info()
            attempt["memory"].update({"windows_peak_working_set_bytes": getattr(info, "peak_wset", None),
                                      "windows_peak_commit_bytes_not_physical_rss": getattr(info, "peak_pagefile", None)})
        attempt_path.write_text(json.dumps(attempt, indent=2, allow_nan=False), encoding="utf8")
        print(json.dumps(attempt, allow_nan=False), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
