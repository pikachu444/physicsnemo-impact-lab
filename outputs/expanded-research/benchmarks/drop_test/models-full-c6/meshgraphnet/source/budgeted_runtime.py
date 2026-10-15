"""Shared lazy runtime for prospective, time-budgeted train/fresh-test runs.

Importing this module uses the standard library only. Framework imports happen
only after the executable checks its explicit GPU-slot authorization flag.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

_runtime = None


def filehash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def objecthash(value):
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_runtime():
    global _runtime
    if _runtime is None:
        import numpy as np
        import torch
        import psutil
        from benchmark_data import Cache, physical_metrics, verify_ready
        from model_adapters import build_model, forward_model
        _runtime = SimpleNamespace(np=np, torch=torch, psutil=psutil, Cache=Cache,
                                   physical_metrics=physical_metrics, verify_ready=verify_ready,
                                   build_model=build_model, forward_model=forward_model)
    return _runtime


def sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def verify_ready(manifest, expected_sha256=None):
    return load_runtime().verify_ready(manifest, expected_sha256)


def make_cache(manifest, allow_test_values=False):
    runtime = load_runtime()

    class GuardedCache(runtime.Cache):
        def __init__(self, path):
            self.target_value_access_log = {"train": {}, "validation": {}, "test": {}}
            self.allow_test_values = allow_test_values
            super().__init__(path)

        def target(self, case):
            group = next(k for k, rows in self.split.items() if case in rows)
            if group == "test" and not self.allow_test_values:
                raise AssertionError("Training/selection may not read test response values")
            counts = self.target_value_access_log[group]
            counts[case] = counts.get(case, 0) + 1
            return super().target(case)

    return GuardedCache(manifest)


class TensorData:
    """Stream physical cases, measuring CPU preparation and CUDA copies separately."""
    def __init__(self, cache, stats, device):
        self.runtime = runtime = load_runtime()
        self.cache, self.stats, self.device = cache, stats, device
        np, torch = runtime.np, runtime.torch
        self.mean = np.asarray(stats["field_mean"], dtype=np.float32)
        self.std = np.asarray(stats["field_std"], dtype=np.float32)
        self.edges = torch.tensor(np.asarray(cache.edges), dtype=torch.long, device=device)

    def prepare(self, cases, with_targets=False):
        np, torch = self.runtime.np, self.runtime.torch
        tick = time.perf_counter()
        xyz, scalar, global_rows, targets = [], [], [], []
        read_seconds, normalize_seconds = 0., 0.
        for case in cases:
            section = time.perf_counter()
            feature, global_values = self.cache.inputs(case)
            coordinates = self.cache.coordinates(case)
            physical_target = self.cache.target(case) if with_targets else None
            read_seconds += time.perf_counter()-section
            section = time.perf_counter()
            xyz.append(((coordinates - self.stats["coordinate_min"])
                        / self.stats["coordinate_span"]).astype(np.float32))
            scalar.append(((feature-self.stats["thickness_mean"])/self.stats["thickness_std"]).astype(np.float32))
            global_rows.append(((global_values-self.stats["global_mean"])/self.stats["global_std"]).astype(np.float32))
            if with_targets:
                targets.append(((physical_target-self.mean)/self.std).reshape(len(self.cache.xyz), -1))
            normalize_seconds += time.perf_counter()-section
        section = time.perf_counter()
        arrays = {"xyz": np.stack(xyz), "scalar": np.stack(scalar), "globals": np.stack(global_rows)}
        target_array = np.stack(targets) if with_targets else None
        normalize_seconds += time.perf_counter()-section
        read_normalize = time.perf_counter()-tick
        sync(torch, self.device)
        tick = time.perf_counter()
        batch = {key: torch.tensor(value, dtype=torch.float32, device=self.device)
                 for key, value in arrays.items()}
        batch["edge_index"] = self.edges
        target = torch.tensor(target_array, dtype=torch.float32, device=self.device) if with_targets else None
        sync(torch, self.device)
        return batch, target, {"read_normalize_seconds": read_normalize,
                               "read_seconds": read_seconds, "normalize_seconds": normalize_seconds,
                               "host_device_transfer_seconds": time.perf_counter()-tick}


def forward(architecture, model, batch, batch_module=None):
    if batch["xyz"].shape[0] == 1:
        original = {"xyz": batch["xyz"][0], "thickness": batch["scalar"][0],
                    "globals": batch["globals"][0], "edge_index": batch["edge_index"]}
        return load_runtime().forward_model(architecture, model, original)
    assert batch_module is not None, "Native batches require their independently approved adapter"
    return batch_module.forward_batched(architecture, model, batch)


def load_batch_adapter(path, approval_paths, *, architecture, batch_sizes,
                       manifest_sha256, normalization_sha256, model_config):
    """Verify output/gradient/independence/Adam evidence before importing the adapter."""
    required = set(batch_sizes)-{1}
    if not required:
        return None, []
    assert path is not None
    adapter_hash = filehash(path)
    approvals = []
    for approval_path in approval_paths:
        row = json.loads(Path(approval_path).read_text(encoding="utf-8"))
        if row["batch_size"] not in required:
            continue
        assert row["status"] == "passed" and row["architecture"] == architecture
        assert row["manifest_sha256"] == manifest_sha256
        assert row["normalization_sha256"] == normalization_sha256
        assert row["model_config"] == json.loads(json.dumps(model_config))
        assert all(row["output_checks"][key]["passed"] for key in (
            "original_B1_vs_native_batch", "case_order_independence",
            "other_case_globals_independence", "changed_case_original_B1_vs_batch"))
        assert row["gradient_check"]["passed"] and row["adam"]["optimizer_updates"] == 3
        assert row["timing"]["forward_backward"]["native_faster_in_every_paired_repeat"], "Selected native batch lacks a repeatable measured throughput benefit"
        source_hashes = [value for key, value in row["source_sha256"].items()
                         if Path(key).name == "batch_adapters.py"]
        assert source_hashes == [adapter_hash], "Adapter changed after independent numerical approval"
        approvals.append({"batch_size": row["batch_size"], "path": str(approval_path),
                          "sha256": filehash(approval_path), "adapter_sha256": adapter_hash})
    assert {row["batch_size"] for row in approvals} == required, "Each actual batch size, including remainder, needs passing evidence"
    spec = importlib.util.spec_from_file_location("approved_native_batch_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, approvals
