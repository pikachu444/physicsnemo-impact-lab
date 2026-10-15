"""Run a fresh, frozen validation-only experiment; preserves failure and resources.

Execute only after the parent assigns the exclusive GPU slot. Parsing, source
hashing and attempt creation precede all optional/framework imports.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--protocol", type=pathlib.Path, required=True)
    parser.add_argument("--source-lock", type=pathlib.Path, required=True)
    parser.add_argument("--source-lock-sha256", required=True)
    parser.add_argument("--validation-only", action="store_true", required=True)
    args, remaining = parser.parse_known_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = pathlib.Path(__file__).resolve().parent
    sources = ["run_validation_training.py", "train_plate.py", "plate_data.py",
               "plate_evaluation.py", "selection_policy.py"]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    source_lock = json.loads(args.source_lock.read_text(encoding="utf8"))
    assert hashlib.sha256(args.source_lock.read_bytes()).hexdigest() == args.source_lock_sha256
    assert source_lock["protocol_sha256"] == hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    assert source_lock["protocol_sha256"] == "a71497614937762d2ca7434d8b5c6e2f91d3fecb4c750ab9efd97192e776bf33"
    assert all(source_lock["source_sha256"][name] == value for name, value in source_hashes.items())
    (output / "preexecution_source_lock.json").write_bytes(args.source_lock.read_bytes())
    for name in sources:
        (output / f"attempt_{name}").write_bytes((root / name).read_bytes())
    attempt = {"status": "running", "started_at_utc": now(), "process_id": os.getpid(), "command_arguments": sys.argv[1:],
               "source_sha256": source_hashes, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
               "source_lock_sha256": hashlib.sha256(args.source_lock.read_bytes()).hexdigest(),
               "test_npz_opened_or_hashed": False,
               "interpretation": "Actual validation-only training/evaluation attempt, not a fresh-final-test result"}
    path = output / "training-attempt.json"
    path.write_text(json.dumps(attempt, indent=2, allow_nan=False), encoding="utf8")
    stop, sampler = threading.Event(), None
    peak_rss, available_min, samples = 0, None, 0
    sampling_errors = []
    process, failure = None, None
    started = time.perf_counter()
    try:
        import psutil
        process = psutil.Process()

        def sample_memory():
            nonlocal peak_rss, available_min, samples
            while not stop.is_set():
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                    available = psutil.virtual_memory().available
                    available_min = available if available_min is None else min(available_min, available)
                    samples += 1
                except Exception as exc:
                    sampling_errors.append(str(exc))
                    return
                stop.wait(0.2)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        import train_plate
        # The wrapper owns this option; the trainer receives the original frozen recipe.
        sys.argv = [sys.argv[0], "--output", str(args.output), "--protocol", str(args.protocol),
                    "--validation-only", *remaining]
        train_plate.main()
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf8"))
        assert metrics["evaluation_split"] == "valid" and not metrics["test_npz_opened_or_hashed"]
        attempt["status"] = ("completed_actual_validation_experiment" if metrics["evaluation_completed_all_cases"]
                             else "completed_training_with_incomplete_validation")
        attempt["evaluation_completed_all_cases"] = metrics["evaluation_completed_all_cases"]
        attempt["evaluation_failed_case_count"] = metrics["evaluation_failed_case_count"]
        attempt["metrics_sha256"] = hashlib.sha256((output / "metrics.json").read_bytes()).hexdigest()
        for name in ["config.json", "normalization.json", "checkpoint.pt"]:
            attempt[f"{name.replace('.', '_')}_sha256"] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        attempt["accuracy_qualified"] = metrics["accuracy_qualified"]
    except BaseException as exc:
        failure = exc
        attempt.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "exception_type": type(exc).__name__, "exception": str(exc),
                        "is_cuda_out_of_memory": "outofmemory" in type(exc).__name__.lower() or "cuda out of memory" in str(exc).lower()})
        (output / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        attempt.update({"finished_at_utc": now(), "elapsed_wall_seconds_including_imports_setup_and_io": time.perf_counter() - started,
                        "memory": {"sampled_rss_peak_bytes": peak_rss, "sampled_host_available_minimum_bytes": available_min,
                                   "requested_sampling_interval_seconds": 0.2, "samples": samples, "sampling_errors": sampling_errors}})
        if process is not None:
            try:
                info = process.memory_info()
                attempt["memory"].update({"windows_peak_working_set_bytes": getattr(info, "peak_wset", None),
                                          "windows_peak_commit_bytes_not_physical_rss": getattr(info, "peak_pagefile", None)})
            except Exception as exc:
                attempt["memory"]["final_measurement_error"] = str(exc)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    attempt["cuda"] = {"peak_tensor_allocated_bytes": torch.cuda.max_memory_allocated(),
                                       "peak_allocator_reserved_bytes": torch.cuda.max_memory_reserved()}
            except Exception as exc:
                attempt["cuda_measurement_error"] = str(exc)
        path.write_text(json.dumps(attempt, indent=2, allow_nan=False), encoding="utf8")
        print(json.dumps(attempt, allow_nan=False), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
