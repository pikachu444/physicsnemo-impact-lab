"""Run one immutable GPU resource attempt, preserving even import failures/OOM.

Only execute this file during the exclusive GPU slot. It imports the frameworks
after starting CPU memory sampling and writing the attempt's source provenance.
"""
import argparse
import datetime
import hashlib
import json
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
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--processor", type=int, default=15)
    parser.add_argument("--checkpoint-segments", type=int, default=0)
    parser.add_argument("--probe", type=int, default=3)
    parser.add_argument("--probe-trajectory", default="train_0011")
    parser.add_argument("--probe-frame", type=int, default=319)
    args, _ = parser.parse_known_args()
    if not 1 <= args.probe <= 10:
        parser.error("A resource attempt must request 1..10 measured updates")
    original_arguments = sys.argv[1:].copy()
    effective_arguments = original_arguments.copy()
    for flag, value in {
        "--hidden": args.hidden, "--processor": args.processor,
        "--checkpoint-segments": args.checkpoint_segments, "--probe": args.probe,
        "--probe-trajectory": args.probe_trajectory, "--probe-frame": args.probe_frame,
    }.items():
        if not any(a == flag or a.startswith(flag + "=") for a in effective_arguments):
            effective_arguments.extend([flag, str(value)])
    sys.argv = [sys.argv[0], *effective_arguments]
    output = args.output.resolve()
    attempt_path = output / "resource-attempt.json"
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}; choose a new directory")
    output.mkdir(parents=True, exist_ok=False)
    source_root = pathlib.Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                     for name in ["run_resource_probe.py", "train_plate.py", "plate_data.py"]}
    for name in source_hashes:
        (output / f"resource_{name}").write_bytes((source_root / name).read_bytes())
    attempt = {
        "status": "running", "started_at_utc": now(), "original_command_arguments": original_arguments,
        "effective_command_arguments": effective_arguments,
        "source_sha256": source_hashes, "configuration": vars(args) | {"output": str(args.output)},
        "official_capacity": {"hidden_width": 128, "processor_blocks": 15},
        "activation_checkpointing": {
            "segments": args.checkpoint_segments,
            "changed_from_official_default": args.checkpoint_segments != 0,
            "note": "Activation checkpointing is a separate memory optimization attempt, not silently mixed into the default measurement.",
        },
        "counting": "--probe counts timed Adam updates; two additional warm-up Adam updates are performed.",
        "interpretation": "Resource probe only. It is not completed accuracy training and never evaluates fresh-final-test predictions.",
    }
    attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf8")
    stop = threading.Event()
    sampled_rss = 0
    available_min = None
    samples = 0
    memory_errors = []
    process = None
    sampler = None
    started = time.perf_counter()
    failure = None
    try:
        import psutil
        process = psutil.Process()

        def memory_loop():
            nonlocal sampled_rss, available_min, samples
            while not stop.is_set():
                try:
                    sampled_rss = max(sampled_rss, process.memory_info().rss)
                    available = psutil.virtual_memory().available
                    available_min = available if available_min is None else min(available_min, available)
                    samples += 1
                except Exception as exc:
                    memory_errors.append(str(exc))
                    return
                stop.wait(0.02)

        sampler = threading.Thread(target=memory_loop, daemon=True)
        sampler.start()
        import train_plate
        train_plate.main()
        assert (output / "probe.json").exists(), "Resource entry point did not produce probe.json"
        measurement = json.loads((output / "probe.json").read_text())
        attempt["measurement"] = measurement
        attempt["probe_sha256"] = hashlib.sha256((output / "probe.json").read_bytes()).hexdigest()
        attempt["status"] = "completed_resource_probe"
    except BaseException as exc:
        failure = exc
        attempt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        attempt["exception_type"] = type(exc).__name__
        attempt["exception"] = str(exc)
        attempt["is_cuda_out_of_memory"] = "outofmemory" in type(exc).__name__.lower() or "cuda out of memory" in str(exc).lower()
        (output / "failure-traceback.txt").write_text(traceback.format_exc(), encoding="utf8")
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        attempt["elapsed_wall_seconds_including_imports_and_setup"] = time.perf_counter() - started
        attempt["finished_at_utc"] = now()
        attempt["memory"] = {
            "sampled_rss_peak_bytes": sampled_rss,
            "sampling_interval_seconds": 0.02,
            "sample_count": samples,
            "sampled_host_available_minimum_bytes": available_min,
            "sampling_errors": memory_errors,
            "note": "The sampled RSS peak is a lower bound, not a continuous OS high-water measurement.",
        }
        if process is not None:
            try:
                info = process.memory_info()
                attempt["memory"]["windows_peak_working_set_bytes"] = getattr(info, "peak_wset", None)
                attempt["memory"]["windows_peak_pagefile_bytes"] = getattr(info, "peak_pagefile", None)
            except Exception as exc:
                attempt["memory"]["final_process_memory_error"] = str(exc)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                if torch.cuda.is_initialized():
                    attempt["cuda"] = {
                        "max_tensor_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "max_allocator_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "note": "PyTorch allocator values exclude unrelated processes and some CUDA driver/library allocations.",
                    }
            except Exception as exc:
                attempt["cuda_measurement_error"] = str(exc)
        config_path = output / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                attempt["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
                attempt["trainable_parameters"] = cfg.get("parameters")
                attempt["runtime_versions"] = {k: cfg.get(k) for k in ["physicsnemo", "torch", "torch_geometric", "torch_scatter", "cuda_runtime", "device"]}
            except Exception as exc:
                attempt["config_read_error"] = str(exc)
        attempt_path.write_text(json.dumps(attempt, indent=2), encoding="utf8")
        print(json.dumps(attempt), flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
