"""Stream public curated VTP displacement into an aligned, mmap-friendly cache.

This is preprocessing only. No Torch import and no synthetic stress/strain labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read(path):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32).copy()
    pd = poly.GetPointData()
    thickness = vtk_to_numpy(pd.GetArray("thickness")).astype(np.float32).copy()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    times = sorted(float(n.split("_t")[1]) for n in names if n.startswith("displacement_t"))
    target = np.stack([vtk_to_numpy(pd.GetArray(f"displacement_t{t:.3f}")) for t in times], 1).astype(np.float32)
    assert target.shape == (len(xyz), len(times), 3) and np.isfinite(target).all()
    cells = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int64).copy()
    offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray()).astype(np.int64).copy()
    return xyz, thickness, target, np.asarray(times), cells, offsets, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="Existing experiment metrics.json, split is reused exactly")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs").mkdir(exist_ok=True)
    paths = sorted((args.data / "TRAINING_DATA").glob("*.vtp")) + sorted((args.data / "VALIDATION_DATA").glob("*.vtp"))
    features = json.loads((args.data / "GLOBAL_FEATURES.json").read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))["split"]
    all_names = sum(split.values(), [])
    assert len(all_names) == len(set(all_names)) == len(paths) == 131
    xyz0, thick0, _, times0, cells, offsets, names = read(paths[0])
    selected = np.flatnonzero(thick0 > 0)
    inverse = np.full(len(xyz0), -1, dtype=np.int64)
    inverse[selected] = np.arange(len(selected))
    edges = set()
    retained_cells = 0
    for a, b in zip(offsets[:-1], offsets[1:]):
        points = cells[a:b]
        if not np.all(inverse[points] >= 0):
            continue
        retained_cells += 1
        mapped = inverse[points]
        for u, v in zip(mapped, np.roll(mapped, -1)):
            if u != v:
                edges.add((int(u), int(v)))
                edges.add((int(v), int(u)))
    edge_index = np.array(sorted(edges), dtype=np.int64).T
    degrees = np.bincount(edge_index[0], minlength=len(selected))
    assert edge_index.shape[0] == 2 and edge_index.max() < len(selected)
    np.save(args.out / "coordinates.npy", xyz0[selected])
    np.save(args.out / "source_point_indices.npy", selected)
    np.save(args.out / "edge_index.npy", edge_index)
    global_keys = ["thickness_scale", "velocity_x", "rwall_origin_y"]
    runs = {}
    for index, path in enumerate(paths):
        xyz, thick, y, times, current_cells, current_offsets, _ = read(path)
        assert np.array_equal(times[:len(times0)], times0), path
        assert xyz.shape == xyz0.shape and np.allclose(xyz[selected], xyz0[selected], atol=1e-4), path
        assert np.array_equal(current_cells, cells) and np.array_equal(current_offsets, offsets), path
        assert np.all(thick[selected] > 0), path
        target = y[selected, 1:len(times0), :]
        target_path = f"runs/{path.stem}.npy"
        thickness_path = f"runs/{path.stem}_thickness.npy"
        np.save(args.out / target_path, target)
        np.save(args.out / thickness_path, thick[selected, None])
        runs[path.stem] = {"target": target_path, "thickness": thickness_path,
                           "globals": [features[path.stem][k] for k in global_keys]}
        if index == 0 or (index + 1) % 20 == 0:
            print(f"Cached {index+1}/{len(paths)} runs", flush=True)
    manifest = {
        "schema": "bumper_multifield_cache_v1", "label_source": "Original published curated FEM VTP displacement only",
        "coordinates": "coordinates.npy", "source_point_indices": "source_point_indices.npy", "edge_index": "edge_index.npy",
        "coordinate_sha256": digest(xyz0[selected]), "edge_sha256": digest(edge_index),
        "time_values": times0[1:].tolist(), "time_unit": "ms", "coordinate_unit": "mm",
        "field_names": ["displacement_x", "displacement_y", "displacement_z"], "field_units": ["mm"]*3,
        "global_keys": global_keys, "split": split, "runs": runs,
        "points": len(selected), "original_points": len(xyz0), "polygons_original": len(offsets)-1,
        "polygons_retained": retained_cells, "directed_edges": edge_index.shape[1], "isolated_points": int(np.sum(degrees == 0)),
        "graph_note": "Perimeter edges of actual curated VTP polygons; retained positive-thickness points. Raw shell-only graph will be compared when ANIM labels are available.",
        "normalization": "None applied in cache; training computes all scales using train split only.",
        "license_note": "Source model declares Altair CC BY-NC4.0. Cache is local and is not redistributed.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ["points","polygons_original","polygons_retained","directed_edges","isolated_points"]}), flush=True)


if __name__ == "__main__":
    main()
