"""Bounded trajectory loader and FEM kinematic strain (no Torch/TensorFlow)."""
import itertools
import json
import pathlib

import numpy as np
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent


def mesh_edges(cells):
    pairs = np.concatenate([cells[:, [a, b]] for a, b in itertools.combinations(range(4), 2)])
    return np.unique(np.concatenate([pairs, pairs[:, ::-1]]), axis=0).T.astype(np.int64)


def edge_values(positions, edges):
    diff = positions[edges[0]] - positions[edges[1]]
    return np.concatenate([diff, np.linalg.norm(diff, axis=1, keepdims=True)], axis=1).astype(np.float32)


def world_edges(positions, mesh, radius=0.03):
    n = len(positions)
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    if len(pairs):
        existing = mesh[0] * n + mesh[1]
        pairs = pairs[~np.isin(pairs[:, 0] * n + pairs[:, 1], existing)]
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return pairs.T.astype(np.int64).reshape(2, -1)


def deformation_strain(reference, current, cells):
    """Per-tetra Green-Lagrange tensor; reference/current must share node ordering."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    ref_tet = reference[cells]
    now_tet = current[cells]
    dm = (ref_tet[:, 1:] - ref_tet[:, :1]).transpose(0, 2, 1)
    ds = (now_tet[:, 1:] - now_tet[:, :1]).transpose(0, 2, 1)
    det = np.linalg.det(dm)
    assert np.all(np.abs(det) > 1e-15), "Degenerate reference tetrahedron"
    f = ds @ np.linalg.inv(dm)
    strain = 0.5 * (f.transpose(0, 2, 1) @ f - np.eye(3))
    return strain[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]].astype(np.float32), np.abs(det / 6)


def self_check():
    ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tet = np.array([[0, 1, 2, 3]])
    translation, _ = deformation_strain(ref, ref + [4, 5, 6], tet)
    np.testing.assert_allclose(translation, 0, atol=1e-12)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rigid, _ = deformation_strain(ref, ref @ rotation.T, tet)
    np.testing.assert_allclose(rigid, 0, atol=1e-12)
    stretch, _ = deformation_strain(ref, ref @ np.diag([1.2, 1, 1]), tet)
    np.testing.assert_allclose(stretch, [[0.22, 0, 0, 0, 0, 0]], atol=1e-7)
    edges = mesh_edges(tet)
    assert edges.shape == (2, 12) and not np.any(edges[0] == edges[1])
    assert world_edges(ref, edges, radius=3).shape == (2, 0)


def load_trajectory(path):
    with np.load(path) as record:
        result = {k: record[k] for k in record.files}
    for name in ["cells", "mesh_pos", "node_type"]:
        result[name] = result[name][0]
    result["mesh_edges"] = mesh_edges(result["cells"])
    result["mesh_edge_values"] = edge_values(result["mesh_pos"], result["mesh_edges"])
    result["name"] = pathlib.Path(path).stem
    return result


def build_statistics(paths):
    """Equal-trajectory weighting follows the pinned NVIDIA example; train only."""
    means, seconds, edge_means, edge_seconds = [], [], [], []
    for path in paths:
        record = load_trajectory(path)
        y = np.concatenate([np.diff(record["world_pos"], axis=0), record["stress"][1:]], axis=-1)
        means.append(np.mean(y, axis=(0, 1), dtype=np.float64))
        seconds.append(np.mean(y.astype(np.float64) ** 2, axis=(0, 1)))
        edge = record["mesh_edge_values"]
        edge_means.append(np.mean(edge, axis=0, dtype=np.float64))
        edge_seconds.append(np.mean(edge.astype(np.float64) ** 2, axis=0))
    mean = np.mean(means, axis=0)
    edge_mean = np.mean(edge_means, axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds, axis=0) - mean ** 2, 1e-24))
    edge_std = np.sqrt(np.maximum(np.mean(edge_seconds, axis=0) - edge_mean ** 2, 1e-24))
    return {"target_mean": mean.tolist(), "target_std": std.tolist(),
            "edge_mean": edge_mean.tolist(), "edge_std": edge_std.tolist(),
            "fit_trajectories": [pathlib.Path(p).stem for p in paths],
            "weighting": "Equal trajectory weight; all 399 training increments; no validation/test statistics",
            "strain_reference": "Per-tetra Green-Lagrange derived from mesh_pos and world_pos; not original COMSOL strain output"}


if __name__ == "__main__":
    self_check()
    paths = sorted((ROOT / "data").glob("train_*.npz"))
    out = ROOT.parents[1] / "outputs/expanded-research/benchmarks/deforming_plate"
    statistics = build_statistics(paths)
    statistics["kinematic_self_checks"] = "translation, rigid rotation, 20% axial stretch, tetra mesh edges: passed"
    (out / "normalization.json").write_text(json.dumps(statistics, indent=2), encoding="utf8")
    inspection = []
    for path in sorted((ROOT / "data").glob("*.npz")):
        record = load_trajectory(path)
        edges = world_edges(record["world_pos"][0], record["mesh_edges"])
        strain, volume = deformation_strain(record["mesh_pos"], record["world_pos"][-1], record["cells"])
        inspection.append({"name": record["name"], "nodes": len(record["mesh_pos"]),
                           "mesh_directed_edges": record["mesh_edges"].shape[1], "world_directed_edges_t0": edges.shape[1],
                           "cells": len(record["cells"]), "strain_finite": bool(np.isfinite(strain).all()),
                           "max_abs_derived_green_lagrange_component_final": float(abs(strain).max()),
                           "node_types": {str(v): int((record["node_type"] == v).sum()) for v in np.unique(record["node_type"])}})
    (out / "data-inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf8")
    print(json.dumps(statistics, indent=2))
    print(json.dumps(inspection, indent=2))
