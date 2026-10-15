"""Freeze verified original ANIM projections into the common benchmark cache.

Source extraction is owned by the FEM workflow. This script neither invents a
strain field nor mixes the new total-strain rerun with original published cases.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from benchmark_data import array_hash, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True, help="Atomic global READY.json written only after extraction process exit0")
    parser.add_argument("--curated-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.curated_cache / "manifest.json").read_text(encoding="utf-8"))
    cases = sum(old["split"].values(), [])
    ready_hash = file_hash(args.ready)
    ready = json.loads(args.ready.read_text(encoding="utf-8"))
    assert ready["status"] == "ready" and ready["writer_exit_code"] == 0 and ready["expected_cases"] == 131
    inventory = {row["run"]: row for row in ready["cases"]}
    assert len(inventory) == len(cases) == 131 and set(inventory) == set(cases)
    source_manifests = {}
    missing = []
    for case in cases:
        path = args.source / case / "manifest.json"
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if value.get("status") != "complete" or value.get("reader_backend") != "direct":
            missing.append(case)
        else:
            assert file_hash(path) == inventory[case]["manifest_sha256"], f"Source manifest changed after READY: {case}"
            for item in inventory[case]["files"]:
                frozen = args.source / case / item["name"]
                assert frozen.stat().st_size == item["bytes"] and file_hash(frozen) == item["sha256"], f"Frozen source file mismatch: {frozen}"
            source_manifests[case] = value
    if missing:
        raise SystemExit(f"Original-case cache incomplete ({len(cases)-len(missing)}/{len(cases)}); missing: {missing}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs").mkdir(exist_ok=True)
    (args.out / "source_manifests").mkdir(exist_ok=True)
    shutil.copyfile(args.ready, args.out / "source_READY.json")
    reference_case = cases[0]
    root = args.source / reference_case
    xyz = np.load(root / "coords.npy")
    node_ids = np.load(root / "node_id.npy")
    original_indices = np.load(args.curated_cache / old["source_point_indices"])
    curated_xyz = np.load(args.curated_cache / old["coordinates"])
    assert np.allclose(xyz, curated_xyz, atol=1e-4, rtol=0), "ANIM/curated coordinate correspondence failed"
    assert len(node_ids) == len(np.unique(node_ids)) == len(xyz)
    np.save(args.out / "coordinates.npy", xyz)
    np.save(args.out / "node_id.npy", node_ids)
    np.save(args.out / "source_point_indices.npy", original_indices)
    shutil.copyfile(args.curated_cache / old["edge_index"], args.out / "edge_index.npy")
    fields = source_manifests[reference_case]["nodal_target_channels"]
    units = source_manifests[reference_case]["nodal_target_units"]
    times = np.load(root / "nominal_time_ms.npy")
    assert len(fields) == 7 and np.array_equal(times, np.arange(0,101,10))
    assert fields == ["displacement_x", "displacement_y", "displacement_z", "inplane_vm_upper", "inplane_vm_lower", "EPSP_upper", "EPSP_lower"]
    runs = {}
    for i, case in enumerate(cases):
        current = args.source / case
        manifest_before = (current / "manifest.json").read_bytes()
        info = source_manifests[case]
        assert info["provenance"] == "original_public_OpenRadioss_ANIM", case
        assert info["nodal_target_channels"] == fields and info["nodal_target_units"] == units, case
        assert np.array_equal(np.load(current / "node_id.npy"), node_ids), case
        assert np.array_equal(np.load(current / "curated_node_indices.npy"), original_indices), case
        assert np.allclose(np.load(current / "coords.npy"), xyz, atol=1e-4, rtol=0), case
        assert np.array_equal(np.load(current / "nominal_time_ms.npy"), times), case
        y = np.load(current / "nodal_targets.npy", mmap_mode="r")
        assert y.shape == (len(xyz), len(times), 7) and y.dtype == np.float32 and np.isfinite(y).all(), case
        assert np.all(y[...,3:] >= 0), case
        target_path, thick_path = f"runs/{case}.npy", f"runs/{case}_thickness.npy"
        np.save(args.out / target_path, y[:,1:,:])
        thickness = np.load(current / "thickness.npy").reshape(len(xyz),1)
        assert np.isfinite(thickness).all() and np.all(thickness > 0), case
        np.save(args.out / thick_path, thickness.astype(np.float32))
        assert (current / "manifest.json").read_bytes() == manifest_before, f"Source changed while freezing {case}"
        (args.out / "source_manifests" / f"{case}.json").write_bytes(manifest_before)
        runs[case] = {"target": target_path, "thickness": thick_path, "globals": old["runs"][case]["globals"],
                      "actual_time_values": np.load(current / "time_ms.npy")[1:].tolist(),
                      "source_manifest_sha256": file_hash(current / "manifest.json")}
        if i == 0 or (i+1) % 20 == 0:
            print(f"Frozen {i+1}/{len(cases)} original ANIM cases", flush=True)
    assert file_hash(args.ready) == ready_hash, "Source READY changed while freezing benchmark cache"
    manifest = {**old, "label_source": "Original published OpenRadioss ANIM; binary float32 extraction and deterministic initial-area cell-to-node projection",
                "source_ready_sha256": ready_hash,
                "dataset_revision": ready.get("dataset_revision"),
                "static_feature_name": "nodal_thickness", "static_feature_unit": "mm",
                "coordinates": "coordinates.npy", "node_id": "node_id.npy", "edge_index": "edge_index.npy", "source_point_indices": "source_point_indices.npy",
                "coordinate_sha256": array_hash(xyz), "time_values": times[1:].tolist(),
                "field_names": fields, "field_units": units, "runs": runs,
                "field_definition": "Displacement is nodal solver output. VM upper/lower is the in-plane invariant sqrt(sxx^2-sxx*syy+syy^2+3*sxy^2) of each local shell tensor, projected by initial shell area. EPSP upper/lower is scalar solver output projected by the same rule. These are surface-specific nodal averaged fields, not raw-cell hotspots or through-thickness-averaged solver VONM.",
                "strain_note": "EPSP is equivalent plastic strain. Total strain was not exported in these original 131 cases and is absent here.",
                "time_note": "Common nominal frame times10..100ms define the one-shot outputs; actual solver save times are recorded separately for every case.",
                "projection_note": "Local tensor components are never directly averaged across shell bases. Nonnegative scalar invariants are projected; averaging may attenuate cell-level hotspots.",
                "graph_note": "Actual curated shell perimeter adjacency on node-ID/coordinate-verified ANIM correspondence; no geometric nearest-neighbor substitute."}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Frozen independent-case multi-field cache ready", flush=True)


if __name__ == "__main__":
    main()
