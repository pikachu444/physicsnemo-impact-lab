"""Five distinct training-case updates with separately measured input phases.

Importing this module is stdlib-only. Call ``measure_stream`` only after the
caller's allocation/source gates, on a fresh model separate from joint probes.
The caller owns immutable receipt creation; this module never writes results.
"""

import ast
import hashlib
import json
from pathlib import Path
import sys


SEED = 20261015
CASE_COUNT = 5


def measure_stream(data, strategy, model, dp, channels, device, source,
                   protocol_sha256, time_indices=None):
    """Perform five batch-one Adam updates using training targets exclusively.

    Each CPU-prepared sample is compared bitwise to frozen ``Domain.sample``
    outside timing. Shared coordinates/statistics remain resident, as in that
    sampler; H2D includes CPU-tensor construction and actual transfers of full
    targets, static features, global scalars and optional TC time tensors.
    """
    import gc
    import inspect
    import math
    import statistics
    import time
    import numpy as np
    import torch

    try:
        import psutil
    except ImportError:
        psutil = None

    assert strategy in ("time_conditional", "ar_rollout_training", "one_step_teacher_forcing")
    assert device.type == "cuda" and data.device == device
    assert len(protocol_sha256) == 64 and all(c in "0123456789abcdef" for c in protocol_sha256)
    assert source.get("protocol_sha256", protocol_sha256) == protocol_sha256
    assert data.t == 10 and channels == data.c, "Stream profiles require every native target channel"
    train_cases = list(data.cache.split["train"])
    assert len(train_cases) >= CASE_COUNT and len(train_cases) == len(set(train_cases))
    if data.name == "phone_full":
        assert len(train_cases) == CASE_COUNT, "Phone profile must include all five training cases"
        cases = list(train_cases)
        selection = "all five frozen phone training cases in split order"
    else:
        assert data.name == "bumper"
        selected = np.random.default_rng(SEED).choice(len(train_cases), size=CASE_COUNT, replace=False)
        cases = [train_cases[int(index)] for index in selected]
        selection = "five frozen bumper training cases sampled without replacement"
    assert len(cases) == CASE_COUNT and len(set(cases)) == CASE_COUNT
    assert all(case in train_cases for case in cases)

    phase_offsets = None
    if strategy == "time_conditional":
        offsets = np.resize(np.arange(data.t), len(train_cases))
        np.random.default_rng(SEED).shuffle(offsets)
        phase_offsets = {case: int(value) for case, value in zip(train_cases, offsets)}
        if time_indices is None:
            indices = {case: phase_offsets[case] for case in cases}
        else:
            indices = {case: time_indices[case] for case in cases}
        assert all(isinstance(value, (int, np.integer)) and 0 <= value < data.t for value in indices.values())
        indices = {case: int(value) for case, value in indices.items()}
        assert all(indices[case] == phase_offsets[case] for case in cases), "TC stream must use the seeded first-epoch V2 phase"
    else:
        assert time_indices is None, "AR/TF profiles always use all ten future frames"
        indices = {case: None for case in cases}

    def prepare_cpu(case):
        assert case in train_cases
        y = data.cache.target(case)
        normalized = np.empty((data.n, data.t, channels), np.float32)
        normalized[..., :3] = data.coords[:, None, :] + y[..., :3] / data.length
        if channels > 3:
            normalized[..., 3:] = (y[..., 3:] - data.mean[3:]) / data.std[3:]
        static, global_values = data.cache.inputs(case)
        static = (np.asarray(static) - data.norm["thickness_mean"]) / data.norm["thickness_std"]
        gv = (global_values - np.asarray(data.norm["global_mean"])) / np.asarray(data.norm["global_std"])
        return normalized, static.astype(np.float32), gv

    def transfer(cpu, time_index):
        normalized, static, gv = cpu
        target = torch.from_numpy(normalized).to(device)
        inputs = {"coords": data.coords_t, "features": torch.from_numpy(static).to(device)}
        globals_t = {key: torch.tensor(float(value), device=device)
                     for key, value in zip(data.cache.meta["global_keys"], gv)}
        if time_index is not None:
            inputs["time"] = torch.tensor(time_index / data.t, device=device)
            target = target[:, time_index]
        return dp.SimSample(node_features=inputs, node_target=target, global_features=globals_t)

    def timed_cuda(action):
        torch.cuda.synchronize(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record(torch.cuda.current_stream(device))
        result = action()
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        event = begin.elapsed_time(end) / 1000.0
        assert math.isfinite(wall) and wall > 0 and math.isfinite(event) and event > 0
        return result, {"wall_seconds": wall, "cuda_event_seconds": event}

    def same_bits(actual, reference):
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        assert actual.device == reference.device
        assert torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                           reference.contiguous().reshape(-1).view(torch.uint8))

    process = psutil.Process() if psutil is not None else None
    observed_rss = []

    def observe_rss():
        if process is not None:
            observed_rss.append(process.memory_info().rss)
            return observed_rss[-1]
        return None

    head_name, head_weight = next((name, value) for name, value in model.named_parameters()
                                  if name.startswith("ln_mlp_out.") and value.ndim == 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    data.current = None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    observe_rss()
    rows = []
    reference_peak_allocated = 0
    for case in cases:
        assert case in train_cases
        time_index = indices[case]
        head_before = head_weight.detach().clone()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        cpu = prepare_cpu(case)
        cpu_seconds = time.perf_counter() - started
        assert math.isfinite(cpu_seconds) and cpu_seconds > 0
        observe_rss()
        sample, h2d = timed_cuda(lambda: transfer(cpu, time_index))
        del cpu

        # Equality may temporarily hold two full targets; it is outside timings.
        reference_started = time.perf_counter()
        data.current = None
        reference = data.sample(dp, case, channels, time_index=time_index)
        assert set(sample.node_features) == set(reference.node_features)
        for key in sample.node_features:
            same_bits(sample.node_features[key], reference.node_features[key])
        assert set(sample.global_features) == set(reference.global_features)
        for key in sample.global_features:
            same_bits(sample.global_features[key], reference.global_features[key])
        same_bits(sample.node_target, reference.node_target)
        torch.cuda.synchronize(device)
        reference_peak_allocated = max(reference_peak_allocated, torch.cuda.memory_allocated(device))
        reference_seconds = time.perf_counter() - reference_started
        observe_rss()
        data.current = None
        del reference
        gc.collect()
        torch.cuda.synchronize(device)
        assert data.current is None

        compute_begin = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_started = time.perf_counter()
        compute_begin.record(torch.cuda.current_stream(device))
        optimizer.zero_grad(set_to_none=True)

        def forward_loss():
            pred = model(sample, data.stats_t)
            expected_shape = (data.n, channels) if time_index is not None else (data.n, data.t, channels)
            assert tuple(pred.shape) == expected_shape == tuple(sample.node_target.shape)
            assert torch.isfinite(pred).all()
            loss = data.loss(pred, sample.node_target)
            assert torch.isfinite(loss)
            return pred, loss

        (pred, loss), forward = timed_cuda(forward_loss)
        _, backward = timed_cuda(loss.backward)
        gradient_norm, clip = timed_cuda(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        assert torch.isfinite(gradient_norm)
        _, adam = timed_cuda(optimizer.step)
        compute_end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        compute_wall = time.perf_counter() - compute_started
        compute_event = compute_begin.elapsed_time(compute_end) / 1000.0
        assert math.isfinite(compute_wall) and compute_wall > 0
        assert math.isfinite(compute_event) and compute_event > 0
        changed = not torch.equal(head_before, head_weight.detach())
        delta = float(torch.linalg.vector_norm(head_weight.detach() - head_before))
        assert changed and math.isfinite(delta) and delta > 0, "Stream update did not change the output head"
        observed = observe_rss()
        rows.append({
            "case": case, "time_index": time_index, "batch_size": 1,
            "cpu_prepare_seconds": cpu_seconds,
            "h2d_wall_seconds": h2d["wall_seconds"],
            "h2d_cuda_event_seconds": h2d["cuda_event_seconds"],
            "compute_wall_seconds": compute_wall,
            "compute_cuda_event_seconds": compute_event,
            "measured_update_wall_seconds": cpu_seconds + h2d["wall_seconds"] + compute_wall,
            "phase_seconds": {"forward_loss": forward, "backward": backward, "gradient_clip": clip, "adam_step": adam},
            "input_equivalence_bitwise": True,
            "reference_equivalence_seconds_excluded": reference_seconds,
            "head_weight_changed": changed, "head_weight_delta_l2": delta,
            "loss": float(loss.detach()), "unclipped_prediction_finite": True,
            "gradient_norm_before_clip": float(gradient_norm),
            "prediction_shape": list(pred.shape), "rss_observed_bytes": observed,
        })
        del pred, loss, gradient_norm, sample, head_before
        gc.collect()
        torch.cuda.synchronize(device)

    assert data.current is None
    updates = [row["measured_update_wall_seconds"] for row in rows]
    warm_median = statistics.median(updates[1:])
    warm_max = max(updates[1:])
    required = max(100, 2 * len(train_cases))
    helper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    sample_file = Path(inspect.getfile(type(data).sample)).resolve()
    report_source = {**source, "protocol_sha256": protocol_sha256,
                     "profile_helper_sha256": helper_hash,
                     "reference_sample_file": str(sample_file),
                     "reference_sample_file_sha256": hashlib.sha256(sample_file.read_bytes()).hexdigest()}
    return {
        "status": "actual_five_distinct_train_case_optimizer_stream_profile",
        "scope": "train_only_resource_stream", "domain": data.name, "strategy": strategy,
        "profile": "budget_v2_complete_epoch" if strategy == "time_conditional" else "multifield_v3_native_scalar_heads",
        "protocol_sha256": protocol_sha256, "profile_helper_sha256": helper_hash, "source": report_source,
        "batch_size": 1, "channels": channels, "full_nodes": data.n, "future_frames": data.t,
        "train_case_ids": train_cases, "train_case_count": len(train_cases),
        "case_ids": cases, "case_selection": selection, "case_selection_seed": SEED,
        "first_case_train_only": True, "selected_cases_train_only": True,
        "validation_target_values_accessed": False, "test_target_values_accessed": False,
        "test_predicted": False, "updates": len(rows),
        "changed_parameter_updates": sum(row["head_weight_changed"] for row in rows),
        "changed_parameter_evidence": {"parameter": head_name, "rule": "bitwise change and positive finite L2 delta after each actual Adam step"},
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "gradient_clip_max_norm": 1.0,
                      "output_clipping": False, "optimizer_state": "fresh Adam; five sequential updates on caller-provided fresh model"},
        "steps": rows,
        "projection": {
            "required_updates": required, "minimum_optimizer_updates": 100, "minimum_case_epochs": 2,
            "warm_updates": len(rows) - 1, "cold_first_update_wall_seconds": updates[0],
            "warm_median_update_wall_seconds": warm_median, "warm_max_update_wall_seconds": warm_max,
            "all_measured_max_update_wall_seconds": max(updates),
            "median_estimate_seconds": required * warm_median,
            "max_estimate_seconds": required * max(updates),
            "cold_plus_warm_estimate_seconds": updates[0] + (required - 1) * warm_median,
            "two_case_epochs_warm_median_estimate_seconds": 2 * len(train_cases) * warm_median,
            "estimates_only": True,
            "definition": "Required work=max(100 optimizer updates, two complete case epochs). Each estimate multiplies measured split-phase wall sums; validation/setup/reference checks are excluded. Cold first update is retained, warm median excludes it. Five cases do not establish convergence or bound future runtime.",
        },
        "time_sampling": {
            "indices": indices, "v2_phase_offsets": phase_offsets,
            "definition": "TC: one future frame per case using first sampled-epoch V2 balanced seeded phase; AR/TF: all ten future frames per case.",
            "coverage_interpretation": "Ten complete TC sampled case epochs expose each case-time pair once; one TC sampled epoch is not a full-time epoch. Weights change between updates, so this is not an identical optimization trajectory. AR/TF updates expose full trajectories.",
            "custom_tc_indices_supplied": time_indices is not None,
        },
        "timing_definition": {
            "cpu_prepare": "Train target mmap access/finite checks, position and scalar normalization, static/global normalization and static float32 conversion; shared coordinates/statistics were prepared before the profile.",
            "h2d": "Synchronized wall and CUDA-event stream span include CPU tensor construction plus full target/static/global/TC-time device transfers and SimSample construction; this is not a pure DMA timing. Fixed coords/stats/loss scales remain resident, exactly as in Domain.sample.",
            "compute": "Synchronized forward+loss, backward, gradient clip and Adam measured separately; total includes zero_grad, assertions, phase/event/synchronization overhead and GPU idle spans caused by host work. No output clipping.",
            "update_sum": "CPU prepare + H2D wall + compute wall; reference equality, head-change checks, cleanup, model/data setup and validation are excluded.",
            "cold_scope": "First optimizer update in this fresh-model stream; caller setup and reference comparisons already occurred, so this is not a process/import/storage cold-start measurement.",
        },
        "resources": {
            "rss_observed_peak_bytes": max(observed_rss) if observed_rss else None,
            "rss_available": process is not None, "cuda_baseline_allocated_bytes": baseline_allocated,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "reference_comparison_observed_allocated_bytes": reference_peak_allocated,
            "peak_scope": "Includes profiling events, output-head snapshot, optimizer state and temporary comparison sample/cache. Equality temporarily retains two target tensors; reference and data.current are released before compute, allocator reservations may remain. RSS is sampled, not an OS high-water mark.",
        },
        "fresh_model_requirement": "Caller must construct this model in a separate fresh stream-probe process; these five optimizer updates never become fitted checkpoint evidence or reuse the separate joint-gradient probe.",
    }


def source_selfcheck():
    """Compile/inspect source only, without importing data or ML libraries."""
    forbidden = {"numpy", "torch", "physicsnemo", "psutil"}
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)
    path = Path(__file__)
    text = path.read_text(encoding="utf8")
    tree = ast.parse(text, filename=str(path))
    compile(tree, str(path), "exec")
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(item.name.split(".")[0] in forbidden for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module.split(".")[0] not in forbidden
    measure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure_stream")
    assert [arg.arg for arg in measure.args.args] == [
        "data", "strategy", "model", "dp", "channels", "device", "source", "protocol_sha256", "time_indices"]
    code = ast.unparse(measure)
    for expected in ["assert case in train_cases", "data.cache.target(case)", "data.cache.inputs(case)",
                     "data.sample(dp, case, channels, time_index=time_index)", "data.current = None",
                     "actual.contiguous().reshape(-1).view(torch.uint8)", "required = max(100, 2 * len(train_cases))",
                     "torch.optim.Adam(model.parameters(), lr=0.001)", "timed_cuda(loss.backward)"]:
        assert expected in code, expected
    assert "data.cache.split['validation']" not in code and "data.cache.split['test']" not in code
    assert not any(isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                   and node.value.id == "data" and node.attr == "sample" and isinstance(node.ctx, ast.Store)
                   for node in ast.walk(measure))
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)
    print(json.dumps({"status": "stdlib_source_selfcheck_pass", "profile_helper_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "framework_imported": False, "cache_arrays_created": False, "gpu_executed": False,
                      "not_executed": ["input bitwise comparison", "actual optimizer steps", "timing and resource measurements"]}))


if __name__ == "__main__":
    source_selfcheck()
