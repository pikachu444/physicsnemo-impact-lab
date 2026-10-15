"""Fixed clean-environment functional checks; execution requires a separately hashed GPU allocation."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import site
import subprocess
import sys
import time

MODELS = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")
ADAPTER_SHA = "fcfbfc72c4eb36793849410ba54e7e41c27a5261d7f0f3830800a774edd66e7d"
GUARD_SHA = "45d6bfb697b95f8df6721d2d42146c53095dc81e062e885f9fa91d17b5fdda1a"
COMMON = ("coordinates.npy", "edge_index.npy", "node_ids.npy", "thickness.npy")
TARGET = "runs/pilot_base.npy"
FIELDS = ("displacement_x", "displacement_y", "displacement_z", "von_mises_shell_scalar_projected", "equivalent_plastic_strain_projected", "total_strain_principal_major_inplane_projected", "total_strain_principal_minor_inplane_projected")
UNITS = ("mm", "mm", "mm", "MPa", "1", "1", "1")
FORBIDDEN_MODULES = {"numpy", "torch", "physicsnemo", "torch_geometric", "torch_scatter", "warp"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_relative(value):
    require(isinstance(value, str) and value and not value.startswith(("/", "\\")), "Relative path required")
    parts = value.split("/")
    require(all(part and part not in {".", ".."} and not re.search(r'[\\:\x00-\x1f<>"|?*]', part) and not part.endswith((".", " ")) for part in parts), "Unsafe relative path")
    return Path(*parts)


def plain(path, root, exists=True):
    path, root = path.absolute(), root.resolve()
    require(path.is_relative_to(root) and path.resolve().is_relative_to(root), "Path escaped its root")
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, "is_junction", lambda: False)(), "Linked path refused")
        if part == root:
            break
    require(not exists or path.exists(), "Required path is absent: " + path.name)
    return path


def relative_path(root, value):
    return plain(root / safe_relative(value.removeprefix("${WORKSPACE}/")), root)


def hash_file(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino), "File changed during hash")
    return digest.hexdigest()


def read_json(path, expected=None):
    require(path.stat().st_size <= 5 * 1024**2, "Metadata exceeds bound")
    data = path.read_bytes()
    require(expected is None or sha(data) == expected, "Metadata SHA256 mismatch")
    return json.loads(data.decode("utf-8")), data


def save_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_contract(protocol, allocation, protocol_sha):
    require(protocol.get("schema") == "clean-functional6-v1" and protocol.get("status") == "prepared_not_executed", "Prepared fixed protocol required")
    require(set(protocol["models"]) == set(MODELS) and len(protocol["models"]) == 6, "Exactly six fixed models required")
    fixed = {"updates": 3, "lr": 0.001, "seed": 20261015, "tolerance": 1e-5}
    information = {"optimizer": "Adam", "loss": "unweighted mean normalized squared error", "batch_size": 1}
    require(all(protocol["algorithm"].get(key) == value for key, value in fixed.items()) and set(protocol["algorithm"]) <= fixed.keys() | information.keys() and all(protocol["algorithm"][key] == value for key, value in information.items() if key in protocol["algorithm"]), "Fixed algorithm differs")
    require(protocol["claims"].get("actual_executed") is False and protocol["claims"].get("fullfit_claimed") is False, "Protocol claims differ")
    data = protocol["data"]
    require(data["cache_root"] == "work/expanded-structural-coupon/cache-initial13-nodal7" and data["shape"] == [919, 10, 7], "Native support contract differs")
    require(tuple(data["fields"]) == FIELDS and tuple(data["units"]) == UNITS and data["times"] == list(range(1, 11)), "C7 fields/units/times differ")
    require(data["train_case"] == "pilot_base" and data["validation_case"] == "holdout_validation_01" and data["train_globals"] == [1, 1], "Fixed cases differ")
    require(set(data["arrays"]) == {*COMMON, TARGET}, "Only five named arrays may be declared")
    for name, record in data["arrays"].items():
        require(record["phases"] == (["train"] if name == TARGET else ["train", "reload"]), "Array phase whitelist differs")
        require(isinstance(record["bytes"], int) and record["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Exact array bytes/hash required")
    stats = protocol["statistics"]
    for key, length in (("field_mean", 7), ("field_std", 7), ("global_mean", 2), ("global_std", 2), ("coordinate_min", 3), ("coordinate_span", 3)):
        require(len(stats[key]) == length and all(isinstance(v, (int, float)) and math.isfinite(v) for v in stats[key]), "Invalid frozen statistics")
        if key.endswith(("std", "span")):
            require(all(v > 0 for v in stats[key]), "Nonpositive frozen scale")
    require(math.isfinite(stats["thickness_mean"]) and stats["thickness_std"] > 0, "Invalid thickness statistics")
    require(protocol["environment"]["relative_venv"] == "work/expanded-clean-runtime/venv" and protocol["environment"]["python"] == "3.13.5" and len(protocol["environment"]["versions"]) == 90, "Exact clean environment contract required")
    require(len(protocol["core_contract"]["files"]) == 40, "Forty pinned core source files required")
    require(protocol["runtime_source"] == "driver/newfunctional_runtime.py" and protocol["adapter_source"] == "source/model_adapters.py" and protocol["normalization_source"] == "reference/normalization.json", "Frozen source layout differs")
    sources = {row["scratch"]: row for row in protocol["source_files"]}
    require(len(sources) == len(protocol["source_files"]) and protocol["runtime_source"] in sources and protocol["normalization_source"] in sources and sources[protocol["adapter_source"]]["sha256"] == ADAPTER_SHA, "Exact restored runtime/adapter/normalization required")
    require(sum(row["sha256"] == GUARD_SHA for row in sources.values()) == 1, "Exact resource guard source required")
    for row in sources.values():
        safe_relative(row["origin"]); safe_relative(row["scratch"])
        require(Path(row["scratch"]).suffix in {".py", ".json", ".md", ".txt"}, "Sources must contain code/metadata only")
    require(all(isinstance(protocol["models"][key]["expected_parameters"], int) and protocol["models"][key]["expected_parameters"] > 0 for key in MODELS), "Exact parameter counts required")
    require(allocation.get("status") == "authorized_clean_functional6" and allocation.get("protocol_sha256") == protocol_sha and allocation.get("explicit_gpu_slot") is True, "Explicit protocol-bound GPU allocation required")
    require(set(allocation["allowed_models"]) == set(MODELS) and len(allocation["allowed_models"]) == 6, "Allocation model set differs")
    limits = allocation["resource_limits"]
    require(all(isinstance(limits[key], int) and not isinstance(limits[key], bool) for key in ("preflight_min_available_bytes", "host_floor_bytes", "family_RSS_cap_bytes", "disk_floor_bytes", "wall_seconds")), "Integer resource limits required")
    require(limits["preflight_min_available_bytes"] >= int(1.5 * 1024**3) and limits["host_floor_bytes"] >= 512 * 1024**2 and limits["disk_floor_bytes"] >= 5 * 1024**3 and limits["family_RSS_cap_bytes"] > 350 * 1024**2 and limits["wall_seconds"] > 0, "Explicit ML resource limits required")


def context(args):
    require(not (FORBIDDEN_MODULES & set(sys.modules)), "Framework loaded before authorization gate")
    root = args.workspace.resolve()
    protocol, protocol_bytes = read_json(plain(args.protocol.absolute(), root), args.protocol_sha256)
    allocation, allocation_bytes = read_json(plain(args.allocation.absolute(), root), args.allocation_sha256)
    validate_contract(protocol, allocation, args.protocol_sha256)
    receipts = {}
    for key in ("pip_check_receipt", "static_audit_receipt"):
        record = allocation[key]
        receipts[key], _ = read_json(relative_path(root, record["path"]), record["sha256"])
    check, audit = receipts["pip_check_receipt"], receipts["static_audit_receipt"]
    require(check.get("stage") == "check" and check.get("status") == "completed" and check.get("exit_code") == 0 and check.get("child_reaped") is True and check.get("descendants_reaped", True) is True, "Successful reaped pip check required")
    require(audit.get("status") == "passed_static_clean_environment_audit_only" and audit.get("isolated_interpreter") is True and audit.get("site_packages_inheritance") is False and audit.get("user_site_enabled") is False, "Passed isolated static audit required")
    require(len(audit["packages"]) == 90 and {row["name"]: row["version"] for row in audit["packages"]} == protocol["environment"]["versions"], "Static audit exact package set differs")
    core = protocol["core_contract"]["files"]
    audited = audit["core_source_files_checked_against_wheel_and_RECORD"]
    require(len(audited) == 40 and all(row["wheel_and_RECORD_equal"] is True for row in audited) and {row["path"]: row["sha256"] for row in audited} == {row["path"]: row["sha256"] for row in core}, "Static audit core40 differs")
    environment = relative_path(root, protocol["environment"]["relative_venv"])
    require("include-system-site-packages = false" in (environment / "pyvenv.cfg").read_text(encoding="utf-8"), "Inherited environment refused")
    for row in core:
        path = plain(environment / "Lib/site-packages" / safe_relative(row["path"]), environment)
        require(hash_file(path) == row["sha256"], "New-environment core source mismatch")
    return root, protocol, allocation, protocol_bytes, allocation_bytes


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, "Frozen module cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def source_check(protocol, scratch):
    for row in protocol["source_files"]:
        require(hash_file(plain(scratch / safe_relative(row["scratch"]), scratch)) == row["sha256"], "Frozen source bytes changed")
    normalization, _ = read_json(scratch / protocol["normalization_source"])
    require(canonical(normalization) == canonical(protocol["statistics"]), "Frozen normalization differs")
    for row in protocol["core_contract"]["files"]:
        require(hash_file(plain(scratch / "core" / safe_relative(row["path"]), scratch)) == row["sha256"], "Frozen core evidence bytes changed")


def module_origins(environment, frozen):
    base, origins = Path(sys.base_prefix).resolve(), {}
    for name, module in list(sys.modules.items()):
        value = getattr(module, "__dict__", {}).get("__file__")
        if not value or str(value).startswith("<"):
            continue
        path = Path(value).resolve()
        standard = path.is_relative_to(base) and "site-packages" not in {part.lower() for part in path.parts}
        require(path.is_relative_to(environment) or standard or path in frozen, "Module escaped clean environment/frozen sources: " + name)
        origins[name] = str(path)
    return origins


class AccessJournal:
    def __init__(self, root, cache, allowed, phase_dir, frozen, artifacts, environment):
        self.root, self.cache, self.allowed, self.phase_dir = root, cache, allowed, phase_dir
        self.frozen, self.artifacts, self.environment = frozen, artifacts, environment
        self.base = Path(sys.base_prefix).resolve()
        self.observed, self.denied, self.busy = set(), [], False
        self.stream = (phase_dir / "access.jsonl").open("x", encoding="utf-8")

    def record(self, value):
        self.busy = True
        try:
            self.stream.write(json.dumps(value) + "\n"); self.stream.flush()
        finally:
            self.busy = False

    def hook(self, event, arguments):
        if self.busy:
            return
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn"} or event.startswith("socket."):
            raise PermissionError("Child process/network access is outside the functional protocol")
        if event in {"os.listdir", "os.scandir"} and arguments and isinstance(arguments[0], (str, bytes, os.PathLike)):
            require(not Path(os.fsdecode(arguments[0])).absolute().is_relative_to(self.cache), "Cache enumeration refused")
        if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        raw, mode, flags = os.fsdecode(arguments[0]), arguments[1], arguments[2]
        if raw.lower() in {os.devnull.lower(), "nul"}:
            return
        lexical = Path(raw).absolute()
        require(".." not in lexical.parts, "Traversal in audited open path")
        for component in (lexical, *lexical.parents):
            require(not component.is_symlink() and not getattr(component, "is_junction", lambda: False)(), "Linked audited open path")
        path = lexical.resolve()
        writing = isinstance(mode, str) and any(token in mode for token in "wax+") or isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        allowed, scope = False, "other"
        if path.is_relative_to(self.cache):
            relative = path.relative_to(self.cache).as_posix(); scope = "cache_payload"
            allowed = not writing and relative in self.allowed
            if allowed:
                self.observed.add(relative)
        elif writing:
            allowed = path.is_relative_to(self.phase_dir); scope = "new_phase_output"
        else:
            standard = path.is_relative_to(self.base) and "site-packages" not in {part.lower() for part in path.parts}
            allowed = path.is_relative_to(self.environment) or standard or path in self.frozen or path in self.artifacts or path.is_relative_to(self.phase_dir)
            scope = "frozen_source" if path in self.frozen else "own_artifact" if path in self.artifacts else scope
            if path.suffix.lower() in {".npy", ".npz", ".pt", ".ckpt", ".safetensors"} and path not in self.artifacts:
                allowed = False
        if scope != "other" or not allowed:
            self.record({"event": "open", "path": str(path), "scope": scope, "write": bool(writing), "allowed": bool(allowed)})
        if not allowed:
            self.denied.append(str(path))
            raise PermissionError("Access outside phase whitelist: " + str(path))


def child(args):
    require(sys.flags.isolated and not site.ENABLE_USER_SITE, "Child requires isolated -I interpreter")
    root, protocol, allocation, _, _ = context(args)
    run = plain(args.run_dir.absolute(), root / "work/expanded-clean-runtime/newfunctional-runs")
    environment = relative_path(root, protocol["environment"]["relative_venv"])
    require(Path(sys.prefix).resolve() == environment and tuple(sys.version_info[:3]) == (3, 13, 5), "Wrong clean interpreter")
    base = Path(sys.base_prefix).resolve()
    require(all(value and (Path(value).resolve().is_relative_to(environment) or Path(value).resolve().is_relative_to(base) and "site-packages" not in {part.lower() for part in Path(value).parts}) for value in sys.path), "External import/deps path")
    for name, version in protocol["environment"]["versions"].items():
        distribution = importlib.metadata.distribution(name)
        require(distribution.version == version and Path(distribution.locate_file("")).resolve().is_relative_to(environment), "Current distribution escaped exact clean environment")
    inside = ctypes.c_int()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    require(bool(kernel.IsProcessInJob(ctypes.c_void_p(-1), None, ctypes.byref(inside))) and inside.value, "Child must already be Job-contained")
    scratch, phase = run / "scratch", args.child_phase
    phase_dir = plain(run / args.model / phase, run)
    launch, _ = read_json(phase_dir / "launch.json")
    require(launch["pid"] == os.getpid() and launch["protocol_sha256"] == args.protocol_sha256 and launch["allocation_sha256"] == args.allocation_sha256, "Child launch binding differs")
    frozen = {plain(scratch / safe_relative(row["scratch"]), scratch) for row in protocol["source_files"]}
    artifacts, binding = ({phase_dir / "checkpoint.pt", phase_dir / "validation-anchor.pt"} if phase == "train" else set()), None
    if phase == "reload":
        binding, _ = read_json(phase_dir / "artifact-binding.json", args.binding_sha256)
        require(binding["model"] == args.model and binding["protocol_sha256"] == args.protocol_sha256 and binding["allocation_sha256"] == args.allocation_sha256, "Reload artifact binding differs")
        for key, filename in (("checkpoint", "checkpoint.pt"), ("validation_anchor", "validation-anchor.pt")):
            path = plain(run / args.model / "train" / filename, run)
            require(binding[key]["path"] == path.relative_to(run).as_posix() and path.stat().st_size == binding[key]["bytes"] and hash_file(path) == binding[key]["sha256"], "Reload artifact bytes changed")
            artifacts.add(path)
    allowed = {*COMMON, *([TARGET] if phase == "train" else [])}
    cache = relative_path(root, protocol["data"]["cache_root"])
    core_evidence = {plain(scratch / "core" / safe_relative(row["path"]), scratch) for row in protocol["core_contract"]["files"]}
    journal = AccessJournal(root, cache, allowed, phase_dir, frozen | core_evidence, artifacts, environment)
    sys.addaudithook(journal.hook)
    source_check(protocol, scratch)
    for name in sorted(allowed):
        path = plain(cache / safe_relative(name), cache)
        record = protocol["data"]["arrays"][name]
        require(path.stat().st_size == record["bytes"] and hash_file(path) == record["sha256"], "Allowed array bytes differ")
    # Every ML import is below allocation, receipt, isolation, source, and data gates.
    import numpy as np
    import torch

    require(torch.cuda.is_available(), "Allocated CUDA device is unavailable")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    seed = protocol["algorithm"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    arrays = {name: np.load(cache / name, allow_pickle=False, mmap_mode="r") for name in sorted(allowed)}
    require(all(not value.flags.writeable for value in arrays.values()), "Input arrays must remain read-only")
    xyz, edges, ids, thickness = (arrays[name] for name in COMMON)
    require(xyz.shape == (919, 3) and ids.shape == (919,) and thickness.shape in {(919,), (919, 1)}, "Native input shape differs")
    require(np.issubdtype(ids.dtype, np.integer) and np.all(ids[1:] > ids[:-1]) and 990 not in ids and np.unique(ids).size == 919, "Native sorted unique node support differs")
    require(edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] > 0 and np.issubdtype(edges.dtype, np.integer) and edges.min() >= 0 and edges.max() < 919, "Native edges differ")
    require(np.isfinite(xyz).all() and np.isfinite(thickness).all(), "Nonfinite native inputs")
    stats, device = protocol["statistics"], torch.device("cuda")
    def tensor(value):
        return torch.as_tensor(np.array(value, dtype=np.float32, copy=True), device=device)
    normalized_xyz = (tensor(xyz) - tensor(stats["coordinate_min"])) / tensor(stats["coordinate_span"])
    normalized_thickness = (tensor(thickness).reshape(919, 1) - stats["thickness_mean"]) / stats["thickness_std"]
    def batch(globals_values):
        return {"xyz": normalized_xyz, "thickness": normalized_thickness, "globals": (tensor(globals_values) - tensor(stats["global_mean"])) / tensor(stats["global_std"]),
                "edge_index": torch.as_tensor(np.array(edges, dtype=np.int64, copy=True), device=device)}
    adapter = load_module(scratch / protocol["adapter_source"], "_frozen_model_adapters")
    specification = protocol["models"][args.model]
    require(canonical(adapter.model_config(args.model, 70, global_dim=2)) == canonical(specification["config"]), "Default constructor config differs from frozen protocol")
    checkpoint = torch.load(run / args.model / "train/checkpoint.pt", map_location="cpu", weights_only=True) if phase == "reload" else None
    if checkpoint is not None:
        require(checkpoint["model"] == args.model and checkpoint["protocol_sha256"] == args.protocol_sha256 and checkpoint["updates"] == 3 and canonical(checkpoint["config"]) == canonical(specification["config"]), "Checkpoint binding differs")
    model, config, source = adapter.build_model(args.model, 70, global_dim=2, **({"config": checkpoint["config"]} if checkpoint is not None else {}))
    model.to(device)
    require(sum(parameter.numel() for parameter in model.parameters()) == specification["expected_parameters"], "Parameter count differs")
    for key in ("module", "class_name", "source_sha256"):
        require(source[key] == specification["model_source"][key], "Installed model implementation differs")
    require(Path(source["source_file"]).resolve().is_relative_to(environment), "Model implementation escaped clean environment")
    module_origins(environment, frozen)
    validation_batch = batch(protocol["data"]["validation_globals"])
    adapter.validate_batch(validation_batch, require_edges=args.model == "meshgraphnet")
    result = {"model": args.model, "phase": phase, "protocol_sha256": args.protocol_sha256, "allocation_sha256": args.allocation_sha256,
              "parameters": specification["expected_parameters"], "source": source, "fullfit_claimed": False, "accuracy_claimed": False}
    if phase == "train":
        require(arrays[TARGET].shape == (919, 10, 7) and np.isfinite(arrays[TARGET]).all(), "Pilot target shape/values differ")
        target = ((tensor(arrays[TARGET]) - tensor(stats["field_mean"])) / tensor(stats["field_std"])).reshape(1, 919, 70)
        train_batch = batch(protocol["data"]["train_globals"])
        adapter.validate_batch(train_batch, require_edges=args.model == "meshgraphnet")
        optimizer, steps = torch.optim.Adam(model.parameters(), lr=0.001), []
        model.train()
        for step in range(1, 4):
            optimizer.zero_grad(set_to_none=True)
            prediction = adapter.forward_model(args.model, model, train_batch)
            require(tuple(prediction.shape) == (1, 919, 70) and bool(torch.isfinite(prediction).all()), "Invalid output shape/values")
            prediction.retain_grad()
            loss = torch.mean((prediction - target) ** 2)
            require(bool(torch.isfinite(loss)), "Nonfinite training loss")
            loss.backward()
            require(prediction.grad is not None and bool(torch.isfinite(prediction.grad).all()), "Nonfinite output gradients")
            head_grad = prediction.grad.abs().sum(dim=(0, 1)).reshape(10, 7)
            require(bool((head_grad > 0).all()), "Every time/channel head must contribute gradients")
            parameters = list(model.named_parameters())
            nonzero, unused = 0, []
            for name, parameter in parameters:
                if parameter.grad is None:
                    unused.append(name); continue
                require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite parameter gradient")
                nonzero += int(bool((parameter.grad != 0).any()))
            require(nonzero > 0, "All parameter gradients are zero")
            before = [parameter.detach().clone() for _, parameter in parameters]
            optimizer.step()
            require(all(bool(torch.isfinite(parameter).all()) for _, parameter in parameters), "Nonfinite updated parameter")
            changed = sum(int(bool((parameter.detach() != old).any())) for (_, parameter), old in zip(parameters, before))
            require(changed > 0, "Adam did not change actual parameter bytes")
            steps.append({"step": step, "loss": float(loss.detach()), "nonzero_gradient_parameters": nonzero, "unused_gradient_parameters": unused, "changed_parameters": changed,
                          "head_gradient_abs_sums": head_grad.detach().cpu().reshape(70).tolist(), "channel_gradient_abs_sums": head_grad.sum(dim=0).detach().cpu().tolist(), "available_gradients_finite": True})
        model.eval()
        with torch.no_grad():
            anchor = adapter.forward_model(args.model, model, validation_batch)
        require(tuple(anchor.shape) == (1, 919, 70) and bool(torch.isfinite(anchor).all()), "Invalid targetless anchor")
        checkpoint_path, anchor_path = phase_dir / "checkpoint.pt", phase_dir / "validation-anchor.pt"
        with checkpoint_path.open("xb") as stream:
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "config": config, "model": args.model, "updates": 3, "seed": seed, "protocol_sha256": args.protocol_sha256}, stream)
        with anchor_path.open("xb") as stream:
            torch.save({"normalized_prediction": anchor.detach().cpu().reshape(919, 10, 7), "case": "holdout_validation_01", "model": args.model, "protocol_sha256": args.protocol_sha256}, stream)
        result.update(status="passed_three_updates_and_targetless_anchor", updates=steps,
                      checkpoint={"path": checkpoint_path.relative_to(run).as_posix(), "bytes": checkpoint_path.stat().st_size, "sha256": hash_file(checkpoint_path)},
                      validation_anchor={"path": anchor_path.relative_to(run).as_posix(), "bytes": anchor_path.stat().st_size, "sha256": hash_file(anchor_path)})
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
        anchor = torch.load(run / args.model / "train/validation-anchor.pt", map_location="cpu", weights_only=True)
        require(anchor["case"] == "holdout_validation_01" and anchor["model"] == args.model and anchor["protocol_sha256"] == args.protocol_sha256 and tuple(anchor["normalized_prediction"].shape) == (919, 10, 7), "Anchor binding differs")
        with torch.no_grad():
            prediction = adapter.forward_model(args.model, model, validation_batch)
        require(tuple(prediction.shape) == (1, 919, 70) and bool(torch.isfinite(prediction).all()), "Invalid reload output")
        difference = float((prediction.detach().cpu().reshape(919, 10, 7) - anchor["normalized_prediction"]).abs().max())
        require(math.isfinite(difference) and difference <= 1e-5, "Fresh reload exceeded normalized tolerance")
        result.update(status="passed_fresh_targetless_reload", maximum_absolute_normalized_difference=difference, tolerance=1e-5)
    result.update(module_origins=module_origins(environment, frozen), observed_data_files=sorted(journal.observed), denied_access=journal.denied,
                  test_payload_files_opened=0, test_payload_hashes_computed=0, validation_target_files_opened=0,
                  pilot_target_files_opened=int(TARGET in journal.observed), output_shape=[1, 919, 70], reshaped_output=[919, 10, 7])
    require(journal.observed == allowed and not journal.denied, "Observed data access does not match phase whitelist")
    save_new(phase_dir / "result.json", result)
    return 0


def cleanup_owned(job, process):
    errors, remaining = [], None
    try: job.terminate()
    except BaseException as error: errors.append("Job termination: " + str(error))
    if process is not None:
        try:
            if process.poll() is None: process.terminate()
            process.wait(timeout=30)
        except BaseException as error:
            errors.append("Root reap: " + str(error))
            try: process.kill(); process.wait(timeout=10)
            except BaseException as failure: errors.append("Root kill/reap: " + str(failure))
    try:
        deadline = time.monotonic() + 10
        remaining = job.pids()
        while remaining and time.monotonic() < deadline:
            time.sleep(0.1); remaining = job.pids()
    except BaseException as error: errors.append("Job empty verification: " + str(error))
    if process is not None:
        try:
            handle = getattr(process, "_handle", None)
            if handle is not None: handle.Close()
        except BaseException as error: errors.append("Root handle close: " + str(error))
    return {"cleanup_errors": errors, "child_reaped": process is not None and process.returncode is not None,
            "remaining_job_pids": remaining, "descendants_reaped": remaining == []}


def one_phase(args, root, protocol, allocation, run, guard, model, phase, deadline):
    limits, directory = allocation["resource_limits"], run / model / phase
    plain(directory, run, exists=False); directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "tmp"; temporary.mkdir()
    first = guard.available_memory()
    require(first >= limits["preflight_min_available_bytes"] and shutil.disk_usage(root).free >= limits["disk_floor_bytes"], "Resource preflight failed; no retry")
    time.sleep(10)
    second = guard.available_memory()
    require(second >= limits["preflight_min_available_bytes"] and time.monotonic() < deadline, "Stable resource/wall preflight failed; no retry")
    command = [str(root / protocol["environment"]["relative_venv"] / "Scripts/python.exe"), "-I", "-B", "-X", "utf8", str(run / "scratch" / protocol["runtime_source"]),
               "--workspace", str(root), "--protocol", str(run / "protocol.json"), "--protocol-sha256", args.protocol_sha256,
               "--allocation", str(run / "allocation.json"), "--allocation-sha256", args.allocation_sha256, "--run-dir", str(run), "--model", model, "--child-phase", phase]
    if phase == "reload":
        train_path = run / model / "train/result.json"
        train, train_bytes = read_json(train_path)
        require(train["status"] == "passed_three_updates_and_targetless_anchor" and len(train["updates"]) == 3, "Successful new training receipt required")
        binding = {"model": model, "protocol_sha256": args.protocol_sha256, "allocation_sha256": args.allocation_sha256, "train_receipt_sha256": sha(train_bytes)}
        for key, filename in (("checkpoint", "checkpoint.pt"), ("validation_anchor", "validation-anchor.pt")):
            path = plain(run / model / "train" / filename, run)
            require(train[key]["path"] == path.relative_to(run).as_posix() and hash_file(path) == train[key]["sha256"] and path.stat().st_size == train[key]["bytes"], "New train artifact binding mismatch")
            binding[key] = train[key]
        save_new(directory / "artifact-binding.json", binding)
        command += ["--binding-sha256", hash_file(directory / "artifact-binding.json")]
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"} and not key.startswith("PIP_")}
    if environment.get("CUDA_VISIBLE_DEVICES") == "-1":
        environment.pop("CUDA_VISIBLE_DEVICES")
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")})
    environment.update(PYTHONHASHSEED="20261015", TEMP=str(temporary), TMP=str(temporary), TORCH_HOME=str(temporary / "torch"), XDG_CACHE_HOME=str(temporary / "cache"), WARP_CACHE_DIR=str(temporary / "warp"), CUDA_CACHE_PATH=str(temporary / "cuda"))
    record = {"model": model, "phase": phase, "status": "running", "preflight_available_bytes": [first, second], "CPU_threads": 1, "priority": "BelowNormal", "peak_family_RSS_bytes": 0, "automatic_retry": False}
    start = time.monotonic()
    process, failure, interrupted = None, None, [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    with (directory / "child.log").open("x", encoding="utf-8") as log, guard.ProcessJob() as job:
        try:
            # Delay Ctrl+C until the owned suspended child can be contained or reaped.
            signal.signal(signal.SIGINT, lambda *unused: interrupted.__setitem__(0, True))
            process = subprocess.Popen(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 4)
            kernel = job.kernel
            kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            require(bool(kernel.GetProcessAffinityMask(int(process._handle), ctypes.byref(mask), ctypes.byref(system_mask))), "Cannot inspect child CPU affinity")
            require(bool(kernel.SetProcessAffinityMask(int(process._handle), mask.value & -mask.value)), "Cannot bind child to one CPU")
            save_new(directory / "launch.json", {"pid": process.pid, "protocol_sha256": args.protocol_sha256, "allocation_sha256": args.allocation_sha256, "created_suspended": True})
            job.assign_and_resume(process)
            record.update(pid=process.pid, contained_before_resume=True)
            require(not interrupted[0], "Interrupted after owned child containment")
            print(json.dumps({"model": model, "phase": phase, "pid": process.pid, "contained_before_resume": True}), flush=True)
            while process.poll() is None:
                require(not interrupted[0], "Interrupted during functional phase")
                members = job.pids(); measurements = {pid: guard.rss(pid) for pid in members}
                still_current = set(job.pids())
                require(all(value is not None or pid not in still_current for pid, value in measurements.items()), "Current Job member RSS unavailable")
                require(members or process.poll() is not None, "Running root escaped Job membership")
                family = sum(value for value in measurements.values() if value is not None)
                record["peak_family_RSS_bytes"] = max(record["peak_family_RSS_bytes"], family)
                require(family <= limits["family_RSS_cap_bytes"] and guard.available_memory() >= limits["host_floor_bytes"] and shutil.disk_usage(root).free >= limits["disk_floor_bytes"] and time.monotonic() <= deadline, "Allocated resource guard stopped the phase; no retry")
                time.sleep(0.5)
            process.wait()
            require(not interrupted[0] and process.returncode == 0, "Child phase failed/interrupted; stop all six models")
            require(not job.pids(), "Child left descendants; stop all six models")
            result, _ = read_json(directory / "result.json")
            require(result["model"] == model and result["phase"] == phase and result["protocol_sha256"] == args.protocol_sha256, "Child result binding differs")
            record.update(status="passed", exit_code=0)
        except BaseException as error:
            failure = error
            record.update(status="failed_stop_all_no_retry", error=str(error), exit_code=process.returncode if process is not None else None)
        finally:
            record.update(cleanup_owned(job, process))
            if interrupted[0] or record["cleanup_errors"] or not record["child_reaped"] or not record["descendants_reaped"]:
                record["status"] = "failed_stop_all_no_retry"
                failure = failure or RuntimeError("Interrupt or owned-process cleanup failure")
            signal.signal(signal.SIGINT, previous_handler)
            record["wall_seconds"] = time.monotonic() - start
            record["RSS_observation"] = "0.5-second sampled family working-set lower bound; not an OS hard memory limit"
            save_new(directory / "supervisor.json", record)
    if failure is not None:
        raise failure


def parent(args):
    require(sys.flags.no_site and os.name == "nt", "Parent requires Windows Python -S")
    root, protocol, allocation, protocol_bytes, allocation_bytes = context(args)
    require(isinstance(args.run_name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_name), "A new bounded run name is required")
    base = plain(root / "work/expanded-clean-runtime", root)
    runs = base / "newfunctional-runs"; plain(runs, root, exists=False); runs.mkdir(exist_ok=True)
    run = runs / args.run_name
    require(not os.path.lexists(run), "Run already exists; no overwrite or automatic retry")
    run.mkdir(); scratch = run / "scratch"; scratch.mkdir()
    with (run / "protocol.json").open("xb") as stream: stream.write(protocol_bytes)
    with (run / "allocation.json").open("xb") as stream: stream.write(allocation_bytes)
    for row in protocol["source_files"]:
        source = relative_path(root, row["origin"])
        require(hash_file(source) == row["sha256"], "Restoration source SHA256 differs")
        destination = plain(scratch / safe_relative(row["scratch"]), scratch, exists=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream: stream.write(source.read_bytes())
    environment = relative_path(root, protocol["environment"]["relative_venv"])
    for row in protocol["core_contract"]["files"]:
        source = plain(environment / "Lib/site-packages" / safe_relative(row["path"]), environment)
        data = source.read_bytes()
        require(sha(data) == row["sha256"], "New-environment core snapshot mismatch")
        destination = plain(scratch / "core" / safe_relative(row["path"]), scratch, exists=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream: stream.write(data)
    source_check(protocol, scratch)
    guard_record = next(row for row in protocol["source_files"] if row["sha256"] == GUARD_SHA)
    guard = load_module(scratch / guard_record["scratch"], "_frozen_resource_guard")
    deadline = time.monotonic() + allocation["resource_limits"]["wall_seconds"]
    result = {"status": "failed_stop_all_no_retry", "protocol_sha256": args.protocol_sha256, "allocation_sha256": args.allocation_sha256,
              "actual_executed": True, "fullfit_claimed": False, "accuracy_claimed": False, "functional_checks": [], "automatic_retry": False,
              "core_source_evidence_snapshots": 40, "core_snapshot_import_path_added": False}
    try:
        for model in MODELS:
            for phase in ("train", "reload"):
                one_phase(args, root, protocol, allocation, run, guard, model, phase, deadline)
            result["functional_checks"].append({"model": model, "updates": 3, "fresh_targetless_reload": "passed", "fullfit_claimed": False})
        result["status"] = "passed_fixed_functional6_only"
    except BaseException as error:
        result["error"] = str(error)
        raise
    finally:
        save_new(run / "result.json", result)
    return 0


def selfcheck():
    require(not (FORBIDDEN_MODULES & set(sys.modules)), "Mock selfcheck must remain framework-free")
    assert canonical({"a": (1, 2)}) == canonical({"a": [1, 2]})
    assert safe_relative(TARGET).as_posix() == TARGET
    for value in ("../target.npy", "/absolute", "runs\\test.npy", "runs/a//b", "C:/target"):
        try: safe_relative(value)
        except ValueError: pass
        else: raise AssertionError("Unsafe path accepted")
    common = set(COMMON)
    assert TARGET not in common and not any("holdout_test" in name or "holdout_validation" in name for name in common | {TARGET})
    for prediction, expected in (([0.0, 1.0], [0.0, 1.0 + 1e-6]),):
        assert max(abs(left - right) for left, right in zip(prediction, expected)) <= 1e-5
    assert set(MODELS) == {"transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet"}
    class MockHandle:
        closed = False
        def Close(self): self.closed = True
    class MockProcess:
        returncode = None
        _handle = MockHandle()
        terminated = False
        def poll(self): return self.returncode
        def terminate(self): self.terminated = True; self.returncode = 1
        def wait(self, timeout): return self.returncode
        def kill(self): self.terminate()
    class MockJob:
        terminated = False
        def terminate(self): self.terminated = True
        def pids(self): return []
    process, job = MockProcess(), MockJob()
    cleanup = cleanup_owned(job, process)
    assert process.terminated and process._handle.closed and job.terminated and cleanup["child_reaped"] and cleanup["descendants_reaped"] and not cleanup["cleanup_errors"]
    journal = object.__new__(AccessJournal); journal.busy = False
    try: journal.hook("socket.sendto", ())
    except PermissionError: pass
    else: raise AssertionError("Network event accepted")
    print(json.dumps({"status": "selfcheck_passed", "mock_only": True, "framework_imports": False, "array_or_checkpoint_reads": False, "actual_core_snapshot_copies": 0, "processes_started": False, "network": False, "writes": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--protocol", type=Path); parser.add_argument("--protocol-sha256")
    parser.add_argument("--allocation", type=Path); parser.add_argument("--allocation-sha256")
    parser.add_argument("--run-name"); parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--child-phase", choices=("train", "reload")); parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--binding-sha256"); parser.add_argument("--run-approved", action="store_true"); parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        require(sys.flags.no_site, "Use Python -S for mock selfcheck"); selfcheck(); return 0
    require(all(getattr(args, key) is not None for key in ("workspace", "protocol", "protocol_sha256", "allocation", "allocation_sha256")), "External protocol/allocation paths and hashes are mandatory")
    require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (args.protocol_sha256, args.allocation_sha256)), "External SHA256 values required")
    if args.child_phase:
        require(args.model is not None and args.run_dir is not None and (args.child_phase != "reload" or args.binding_sha256 is not None), "Bound child phase arguments required")
        return child(args)
    require(args.run_approved, "Explicit --run-approved is required after the GPU allocation")
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
