"""Strict train-only cache reader. Never constructs the all-split benchmark Cache."""
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Cache path escapes its frozen root: {relative}")
    return path


def read_train_contract(manifest, normalization, batch_size, expected_manifest, expected_normalization):
    """Only metadata is opened here; held-out target names never become file reads."""
    manifest, normalization = Path(manifest), Path(normalization)
    assert batch_size in (1, 2, 5)
    manifest_sha = file_hash(manifest)
    assert manifest_sha == expected_manifest, "Frozen manifest changed"
    assert file_hash(normalization) == expected_normalization, "Frozen training normalization changed"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    ready_path = manifest.with_name("READY.json")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    assert ready["status"] == "ready" and ready["writer_exit_code"] == 0
    assert ready["manifest_sha256"] == manifest_sha and ready["expected_cases"] == len(meta["runs"])
    split = meta["split"]
    all_ids = [case for group in split.values() for case in group]
    assert len(all_ids) == len(set(all_ids)) and set(all_ids) == set(meta["runs"])
    cases = list(split["train"][:batch_size])
    assert len(cases) == batch_size
    assert not set(cases).intersection(split.get("validation", []) + split.get("test", []))
    selected_names = {meta["coordinates"], meta["edge_index"]}
    for case in cases:
        row = meta["runs"][case]
        selected_names.update((row["thickness"], row["target"]))
        if row.get("coordinates"):
            selected_names.add(row["coordinates"])
    forbidden = {meta["runs"][case]["target"]
                 for group in ("validation", "test") for case in split.get(group, [])}
    assert not selected_names.intersection(forbidden), "Selected file aliases a held-out label"
    return {"manifest": manifest, "normalization": normalization, "meta": meta,
            "stats": json.loads(normalization.read_text(encoding="utf-8")), "ready": ready,
            "ready_sha256": file_hash(ready_path), "cases": cases,
            "selected_names": sorted(selected_names), "forbidden_label_names": sorted(forbidden)}


class TrainProbeData:
    def __init__(self, contract):
        import numpy as np

        self.contract = contract
        self.meta, self.stats = contract["meta"], contract["stats"]
        self.root, self.cases = contract["manifest"].parent, contract["cases"]
        self.opened_files = []
        inventory = {row["name"]: row for row in contract["ready"]["files"]}
        for name in contract["selected_names"]:
            path = contained(self.root, name)
            record = inventory[name]
            assert path.stat().st_size == record["bytes"] and file_hash(path) == record["sha256"], name
            self.opened_files.append({"name": name, "bytes": record["bytes"], "sha256": record["sha256"]})

        def load(name):
            assert name in contract["selected_names"] and name not in contract["forbidden_label_names"]
            return np.load(contained(self.root, name), mmap_mode="r", allow_pickle=False)

        base_xyz, self.edges = load(self.meta["coordinates"]), load(self.meta["edge_index"])
        assert base_xyz.ndim == 2 and base_xyz.shape[1] == 3
        self.n = len(base_xyz)
        assert self.edges.shape[0] == 2 and np.issubdtype(self.edges.dtype, np.integer)
        assert self.edges.min() >= 0 and self.edges.max() < self.n
        mean, std = np.asarray(self.stats["field_mean"], np.float32), np.asarray(self.stats["field_std"], np.float32)
        low, span = np.asarray(self.stats["coordinate_min"]), np.asarray(self.stats["coordinate_span"])
        global_mean, global_std = np.asarray(self.stats["global_mean"]), np.asarray(self.stats["global_std"])
        assert np.all(std > 0) and np.all(span > 0) and np.all(global_std > 0)
        xyzs, scalars, globals_, targets = [], [], [], []
        for case in self.cases:
            row = self.meta["runs"][case]
            xyz = load(row["coordinates"]) if row.get("coordinates") else base_xyz
            assert np.array_equal(xyz, base_xyz), "Probe requires the declared fixed mesh"
            scalar, target = load(row["thickness"]), load(row["target"])
            assert scalar.shape == (self.n, 1) and target.dtype == np.float32
            assert target.shape == (self.n, len(self.meta["time_values"]), len(self.meta["field_names"]))
            assert np.isfinite(xyz).all() and np.isfinite(scalar).all() and np.isfinite(target).all()
            xyzs.append(((xyz-low)/span).astype(np.float32))
            scalars.append(((scalar-self.stats["thickness_mean"])/self.stats["thickness_std"]).astype(np.float32))
            globals_.append(((np.asarray(row["globals"])-global_mean)/global_std).astype(np.float32))
            targets.append(((target-mean)/std).reshape(self.n, -1))
        self.xyz, self.scalar = np.stack(xyzs), np.stack(scalars)
        self.globals, self.targets = np.stack(globals_), np.stack(targets)
        assert np.isfinite(self.globals).all() and np.isfinite(self.targets).all()
        assert self.xyz.min() >= 0 and self.xyz.max() <= 1
        self.out_dim = self.targets.shape[-1]

    def to_device(self, device):
        import torch

        batch = {"xyz": torch.as_tensor(self.xyz, device=device),
                 "scalar": torch.as_tensor(self.scalar, device=device),
                 "globals": torch.as_tensor(self.globals, device=device),
                 "edge_index": torch.tensor(self.edges, dtype=torch.long, device=device)}
        return batch, torch.as_tensor(self.targets, device=device)
