"""Small streaming cache interface and physical-field metrics (no Torch import)."""
import hashlib
import json
from pathlib import Path
import numpy as np


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_ready(manifest, expected_sha256=None):
    """Read-only verification of the producer's flat frozen-cache inventory."""
    manifest = Path(manifest)
    ready_path = manifest.with_name("READY.json")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    assert ready["status"] == "ready" and ready["writer_exit_code"] == 0
    current_hash = file_hash(manifest)
    assert ready["manifest_sha256"] == current_hash
    if expected_sha256:
        assert current_hash == expected_sha256, "Unexpected frozen manifest"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(meta["runs"]) == ready["expected_cases"]
    for row in ready["files"]:
        path = manifest.parent / row["name"]
        assert path.stat().st_size == row["bytes"] and file_hash(path) == row["sha256"], path
    return {"ready_sha256": file_hash(ready_path), "manifest_sha256": current_hash,
            "expected_cases": ready["expected_cases"], "verified_files": len(ready["files"])}


class Cache:
    def __init__(self, manifest):
        self.path = Path(manifest)
        self.root = self.path.parent
        self.meta = json.loads(self.path.read_text(encoding="utf-8"))
        self.xyz = np.load(self.root / self.meta["coordinates"], mmap_mode="r")
        self.edges = np.load(self.root / self.meta["edge_index"], mmap_mode="r")
        self.split = self.meta["split"]
        self.names = self.meta["field_names"]
        self.units = self.meta["field_units"]
        ids = sum(self.split.values(), [])
        assert len(ids) == len(set(ids)) and set(ids) == set(self.meta["runs"])
        assert self.xyz.ndim == 2 and self.xyz.shape[1] == 3 and np.isfinite(self.xyz).all()
        assert self.edges.shape[0] == 2 and self.edges.min() >= 0 and self.edges.max() < len(self.xyz)
        first = self.target(ids[0])
        self.shape = first.shape
        assert self.shape == (len(self.xyz), len(self.meta["time_values"]), len(self.names))
        self.contract = {"manifest_sha256": file_hash(self.path), "coordinate_sha256": array_hash(self.xyz), "edge_sha256": array_hash(self.edges),
                         "shape": list(self.shape), "fields": self.names, "units": self.units,
                         "times": self.meta["time_values"], "split": self.split,
                         "source_ready_sha256": self.meta.get("source_ready_sha256"),
                         "label_source": self.meta.get("label_source"),
                         "static_feature_name": self.meta.get("static_feature_name", "thickness"),
                         "case_inputs_and_labels": {
                             case: {"target_sha256": file_hash(self.root / self.meta["runs"][case]["target"]),
                                    "thickness_sha256": file_hash(self.root / self.meta["runs"][case]["thickness"]),
                                    "coordinates_sha256": file_hash(self.root / self.meta["runs"][case]["coordinates"]) if "coordinates" in self.meta["runs"][case] else array_hash(self.xyz),
                                    "globals": self.meta["runs"][case]["globals"],
                                    "actual_time_values": self.meta["runs"][case].get("actual_time_values")} for case in ids}}

    def target(self, case):
        value = np.load(self.root / self.meta["runs"][case]["target"], mmap_mode="r")
        assert value.dtype == np.float32 and np.isfinite(value).all(), case
        return value

    def inputs(self, case):
        entry = self.meta["runs"][case]
        thickness = np.load(self.root / entry["thickness"], mmap_mode="r")
        assert thickness.shape == (len(self.xyz), 1) and np.isfinite(thickness).all(), case
        return thickness, np.asarray(entry["globals"], dtype=np.float32)

    def coordinates(self, case):
        path = self.meta["runs"][case].get("coordinates")
        value = np.load(self.root / path, mmap_mode="r") if path else self.xyz
        assert value.shape == self.xyz.shape and np.isfinite(value).all(), case
        return value

    def statistics(self):
        count = 0
        mean = np.zeros(len(self.names), dtype=np.float64)
        field_m2 = np.zeros_like(mean)
        mean_field = np.zeros(self.shape, dtype=np.float64)
        thickness_mean = thickness_m2 = 0.
        thickness_count = 0
        global_rows = []
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for case in self.split["train"]:
            y = self.target(case)
            assert y.shape == self.shape, case
            local_count = y.shape[0] * y.shape[1]
            local_mean = np.mean(y, axis=(0,1), dtype=np.float64)
            local_m2 = np.var(y, axis=(0,1), dtype=np.float64)*local_count
            delta = local_mean-mean
            field_m2 += local_m2 + delta**2*count*local_count/(count+local_count)
            mean += delta*local_count/(count+local_count)
            count += local_count
            mean_field += y
            thickness, global_values = self.inputs(case)
            local_count = thickness.size
            local_mean = float(np.mean(thickness, dtype=np.float64))
            delta = local_mean-thickness_mean
            thickness_m2 += float(np.var(thickness, dtype=np.float64))*local_count + delta**2*thickness_count*local_count/(thickness_count+local_count)
            thickness_mean += delta*local_count/(thickness_count+local_count)
            thickness_count += local_count
            global_rows.append(global_values)
            xyz = self.coordinates(case)
            low = np.minimum(low, xyz.min(0))
            high = np.maximum(high, xyz.max(0))
        std = np.sqrt(np.maximum(field_m2/count, 0))
        constant = std == 0
        # Constant channels have no observed variation; their physical units are preserved.
        std[constant] = 1.
        groups = []
        if self.names[:3] == ["displacement_x", "displacement_y", "displacement_z"]:
            vector_scale = float(np.sqrt(np.sum(field_m2[:3]/count)/3)) or 1.
            std[:3] = vector_scale
            groups.append({"fields": self.names[:3], "scale": vector_scale,
                           "definition": "sqrt(trace(training displacement covariance)/3), shared by all three vector components"})
        tm = thickness_mean
        ts = np.sqrt(max(thickness_m2/thickness_count, 0)) or 1.
        gg = np.asarray(global_rows)
        gm, gs = gg.mean(0, dtype=np.float64), gg.std(0, dtype=np.float64)
        gs[gs < 1e-12] = 1.
        span = high-low
        span[span < 1e-12] = 1.
        margin = .01
        low, span = low-margin*span, span*(1+2*margin)
        return {"field_mean": mean.tolist(), "field_std": std.tolist(), "constant_training_fields": constant.tolist(),
                "normalization_groups": groups,
                "thickness_mean": float(tm), "thickness_std": float(ts), "global_mean": gm.tolist(), "global_std": gs.tolist(),
                "coordinate_min": low.tolist(), "coordinate_span": span.tolist(), "coordinate_margin_fraction": margin}, (mean_field/len(global_rows)).astype(np.float32)


def physical_metrics(prediction, truth, names, units, case_ids, time_values):
    """Arrays [case,node,time,field]; scalar channels and displacement vector separately."""
    p, t = np.asarray(prediction), np.asarray(truth)
    assert p.shape == t.shape and p.ndim == 4 and p.shape[-1] == len(names)
    assert np.isfinite(p).all() and np.isfinite(t).all()
    result = {}
    for c, (name, unit) in enumerate(zip(names, units)):
        err = p[..., c].astype(np.float64)-t[..., c]
        squared = err**2
        den = float(np.sum(t[..., c].astype(np.float64)**2))
        per_case = []
        for j, case in enumerate(case_ids):
            target_peak = float(np.max(np.abs(t[j, ..., c])))
            predicted_peak = float(np.max(np.abs(p[j, ..., c])))
            true_location = np.unravel_index(np.argmax(np.abs(t[j,...,c])), t[j,...,c].shape)
            predicted_location = np.unravel_index(np.argmax(np.abs(p[j,...,c])), p[j,...,c].shape)
            per_case.append({"case": case, "sse": float(squared[j].sum()),
                             "truth_sum_square": float(np.sum(t[j,...,c].astype(np.float64)**2)),
                             "count": int(squared[j].size), "truth_absolute_peak": target_peak,
                             "prediction_absolute_peak": predicted_peak, "absolute_peak_error": abs(predicted_peak-target_peak),
                             "truth_peak_node_index": int(true_location[0]), "truth_peak_time_index": int(true_location[1]),
                             "truth_peak_time": time_values[true_location[1]],
                             "prediction_at_truth_peak": float(p[j,true_location[0],true_location[1],c]),
                             "absolute_error_at_truth_peak": float(abs(err[j,true_location[0],true_location[1]])),
                             "predicted_peak_node_index": int(predicted_location[0]), "predicted_peak_time_index": int(predicted_location[1])})
        nonnegative_field = any(token in name.lower() for token in ("vm", "von_mises", "epsp", "plastic", "frobenius", "magnitude"))
        result[name] = {"unit": unit, "relative_l2": float(np.sqrt(squared.sum()/den)) if den else None,
                        "sum_squared_error": float(squared.sum()), "truth_sum_square": den,
                        "truth_l2_norm": float(np.sqrt(den)),
                        "relative_l2_definition": "sqrt(sum((prediction-truth)^2)/sum(truth^2)) over all evaluation cases, support points and times; undefined when denominator is zero",
                        "rmse": float(np.sqrt(squared.mean())), "absolute_error_p95": float(np.quantile(np.abs(err), .95)),
                        "absolute_error_max": float(np.abs(err).max()),
                        "truth_exact_nonzero_count": int(np.count_nonzero(t[...,c])), "count": int(t[...,c].size),
                        "truth_nonfinite_count": int(np.count_nonzero(~np.isfinite(t[...,c]))),
                        "prediction_nonfinite_count": int(np.count_nonzero(~np.isfinite(p[...,c]))),
                        "prediction_negative_count_for_nonnegative_field": int(np.count_nonzero(p[...,c] < 0)) if nonnegative_field else None,
                        "prediction_min": float(np.min(p[...,c])), "prediction_max": float(np.max(p[...,c])),
                        "per_timestep_rmse": np.sqrt(squared.mean((0, 1))).tolist(),
                        "per_case": per_case}
    if names[:3] == ["displacement_x", "displacement_y", "displacement_z"]:
        error = p[..., :3].astype(np.float64)-t[..., :3]
        squared = error**2
        norm = np.linalg.norm(error, axis=-1)
        centered_p = p[..., :3].astype(np.float64)
        centered_t = t[..., :3].astype(np.float64)
        centered_p -= centered_p.mean(1, keepdims=True)
        centered_t -= centered_t.mean(1, keepdims=True)
        displacement_den = float(np.sum(t[..., :3].astype(np.float64)**2))
        centered_den = float(np.linalg.norm(centered_t.astype(np.float64)))
        result["displacement_vector"] = {
            "unit": units[0], "relative_l2": float(np.sqrt(squared.sum()/displacement_den)) if displacement_den else None,
            "sum_squared_error": float(squared.sum()), "truth_sum_square": displacement_den,
            "component_rmse": float(np.sqrt(squared.mean())), "vector_rmse": float(np.sqrt(np.mean(np.sum(squared, -1)))),
            "vector_error_p95": float(np.quantile(norm, .95)), "vector_error_max": float(norm.max()),
            "centroid_removed_relative_l2": float(np.linalg.norm(centered_p.astype(np.float64)-centered_t)/centered_den) if centered_den else None}
    return result


def self_check():
    truth = np.ones((2, 3, 4, 1), dtype=np.float32)
    result = physical_metrics(truth*2, truth, ["stress"], ["GPa"], ["a", "b"], [1,2,3,4])["stress"]
    assert result["relative_l2"] == result["rmse"] == result["absolute_error_p95"] == 1.
    assert all(case["sse"] == 12 for case in result["per_case"])
    rigid = np.ones((1, 3, 2, 3), dtype=np.float32)
    result = physical_metrics(rigid, rigid, ["displacement_x", "displacement_y", "displacement_z"], ["mm"]*3, ["a"], [1,2])
    assert result["displacement_vector"]["centroid_removed_relative_l2"] is None


if __name__ == "__main__":
    self_check()
    print("physical metric self-check passed")
