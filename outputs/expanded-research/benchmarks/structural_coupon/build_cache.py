"""Freeze native element fields and explicit case splits for common benchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(work, summaries, design_path, out):
    design = json.loads(design_path.read_text(encoding="utf-8"))
    split = design["split"]
    case_ids = sum(split.values(), [])
    assert set(split) == {"train", "validation", "test"}
    assert len(case_ids) == len(set(case_ids)) and all(split.values())
    assert not out.exists(), "Use a fresh cache version; never overwrite frozen arrays/splits"
    out.mkdir(parents=True)
    (out / "runs").mkdir()
    manifest = {"schema_version": 1, "benchmark": "RD-E1101 LAW2 structural tensile pilot",
                "coordinates": "coordinates.npy", "coordinate_units": "mm", "edge_index": "edge_index.npy",
                "thickness_units": "mm", "time_values": list(range(1, 11)), "time_units": "ms",
                "time_definition": "Nominal requested animation times; initial zero frame excluded; actual per-case times and jitter retained below. No temporal interpolation.",
                "field_names": None, "field_units": None, "global_keys": ["yield_UTS_scale", "velocity_scale"],
                "global_definition": "Iflag1 engineering yield and engineering UTS scaled together; E/rho/nu/UTS strain fixed. Original velocity curve values scaled, shape unchanged.",
                "split": split, "split_scope": design["scope"], "runs": {},
                "support": "808 original shell element centers in original ELEMENT_ID order; real shared-edge adjacency; no nodal smoothing.",
                "basis": "Native co-rotational local shell axes; same fixed source mesh in every case, no global tensor transform.",
                "location": "Stress/total-strain/EPSP at shell thickness integration point 3 of 5. Separate native shell von Mises scalar has a distinct output definition.",
                "shear_convention": "Tensor epsilon_xy=engineering gamma_xy/2; converter padding zz/yz/zx excluded.",
                "strain_measure": "Native solver accumulated total strain from full nonlinear shell; not displacement-derived. No independent proof of exact equivalence to a named finite-strain measure.",
                "scope": "Public source example sensitivity exercise; no failure model, product data, or evidence of whole-appliance generalization.",
                "source_inputs_redistribution": "Original Altair input decks/ZIP are local only; no redistribution grant found. This cache is private work output pending publication review.",
                "design_sha256": digest(design_path)}
    frozen = {"design": design, "cases": {}}
    ref_xyz = ref_edges = None
    for case_id in case_ids:
        case = work / "runs" / case_id
        summary = json.loads((summaries / case_id / "pilot-summary.json").read_text(encoding="utf-8"))
        p = summary["parameters"]
        xyz = np.load(case / "coordinates.npy")
        edges = np.load(case / "edge_index.npy")
        target = np.load(case / "target_with_initial.npy")
        assert target.shape == (808, 11, 8) and target.dtype == np.float32 and np.isfinite(target).all()
        assert np.max(np.abs(target[:, 0])) < 1e-7
        actual = np.asarray(summary["actual_times_ms"])
        nominal = np.arange(11)
        jitter = float(np.max(np.abs(actual - nominal)))
        assert len(actual) == 11 and jitter < .001, (case_id, actual)
        if ref_xyz is None:
            ref_xyz, ref_edges = xyz, edges
            np.save(out / "coordinates.npy", xyz)
            np.save(out / "edge_index.npy", edges)
            np.save(out / "thickness.npy", np.full((808, 1), 2.5, dtype=np.float32))
            manifest["field_names"], manifest["field_units"] = summary["native_fields"], summary["units"]
        else:
            assert np.array_equal(xyz, ref_xyz) and np.array_equal(edges, ref_edges)
            assert summary["native_fields"] == manifest["field_names"] and summary["units"] == manifest["field_units"]
        dest = out / "runs" / (case_id + ".npy")
        np.save(dest, np.ascontiguousarray(target[:, 1:]))
        report = (case / "tensile_LAW2_0000.out").read_text(errors="replace")
        coefficients = {}
        for key, marker in [("A_GPa", "YIELD COEFFICIENT A"), ("B_GPa", "YIELD COEFFICIENT B"), ("n", "YIELD COEFFICIENT N")]:
            coefficients[key] = float(re.search(re.escape(marker) + r"[^=\r\n]*=\s*([0-9.Ee+\-]+)", report).group(1))
        manifest["runs"][case_id] = {"target": "runs/" + dest.name, "thickness": "thickness.npy",
                                     "globals": [p["strength_scale"], p["velocity_scale"]],
                                     "actual_time_values_ms": actual[1:].tolist(), "max_time_jitter_ms": jitter,
                                     "target_sha256": digest(dest), "native_npz_sha256": digest(case / "native_fem_fields.npz"),
                                     "prepared_sha256": p["prepared_sha256"], "solver_calculated_LAW2_coefficients": coefficients}
        frozen["cases"][case_id] = {"parameters": p, "execution": summary["execution"], "energy": summary["energy"],
                                     "force_diagnostic": summary["force_diagnostic"], "final_max": summary["final_max"],
                                     "final_std": summary["final_std"], "solver_calculated_LAW2_coefficients": coefficients,
                                     "max_time_jitter_ms": jitter, "labels_sha256": manifest["runs"][case_id]["target_sha256"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "frozen-label-metadata.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    shutil.copyfile(design_path, out / "design.json")
    print(json.dumps({"manifest": str(out / "manifest.json"), "cases": len(case_ids), "shape": [808, 10, 8],
                      "manifest_sha256": digest(out / "manifest.json"), "array_bytes": sum(p.stat().st_size for p in out.rglob("*.npy"))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.work.resolve(), args.summaries.resolve(), args.design.resolve(), args.out.resolve())
