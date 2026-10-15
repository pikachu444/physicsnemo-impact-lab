"""Project rotationally invariant native cell quantities; never average local tensor components."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_mesh(path):
    nodes, elements, section = {}, {}, ""
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("/"):
            section = line
        elif section == "/NODE":
            nodes[int(line[:10])] = [float(line[10:30]), float(line[30:50]), float(line[50:70])]
        elif section.startswith(("/SHELL/", "/SH3N/")):
            count = 4 if section.startswith("/SHELL/") else 3
            values = [int(line[i*10:(i+1)*10]) for i in range(count+1)]
            elements[values[0]] = values[1:]
    assert len(nodes) == 920 and len(elements) == 808
    return nodes, elements


def build(cell_cache, starter, work, out):
    assert not out.exists(), "Use a fresh cache; never overwrite frozen evidence"
    cell_meta = json.loads((cell_cache / "manifest.json").read_text(encoding="utf-8"))
    nodes, elements = source_mesh(starter)
    element_ids = sorted(elements)
    node_ids = sorted({n for conn in elements.values() for n in conn})
    assert len(node_ids) == 919 and set(nodes)-set(node_ids) == {990}
    lookup = {node: i for i, node in enumerate(node_ids)}
    coordinates = np.asarray([nodes[node] for node in node_ids], dtype=np.float32)
    area = []
    projection = np.zeros((len(node_ids), len(element_ids)), dtype=np.float64)
    edge_set = set()
    for j, eid in enumerate(element_ids):
        conn = elements[eid]
        polygon = np.asarray([nodes[n][:2] for n in conn], dtype=np.float64)
        a = .5*abs(np.sum(polygon[:,0]*np.roll(polygon[:,1], -1)-polygon[:,1]*np.roll(polygon[:,0], -1)))
        assert a > 0
        area.append(a)
        for n in conn:
            projection[lookup[n], j] = a/len(conn)
        for first, second in zip(conn, conn[1:]+conn[:1]):
            edge_set.update([(lookup[first], lookup[second]), (lookup[second], lookup[first])])
    lumped_area = projection.sum(1)
    assert (lumped_area > 0).all()
    projection /= lumped_area[:, None]
    assert np.allclose(projection.sum(1), 1)
    area = np.asarray(area)
    out.mkdir(parents=True)
    (out / "runs").mkdir()
    np.save(out / "coordinates.npy", coordinates)
    np.save(out / "edge_index.npy", np.asarray(sorted(edge_set), dtype=np.int64).T)
    np.save(out / "node_ids.npy", np.asarray(node_ids, dtype=np.int64))
    np.save(out / "element_ids.npy", np.asarray(element_ids, dtype=np.int64))
    np.save(out / "lumped_nodal_area_mm2.npy", lumped_area)
    np.save(out / "cell_area_mm2.npy", area)
    np.save(out / "thickness.npy", np.full((919, 1), 2.5, dtype=np.float32))
    meta = {key: value for key, value in cell_meta.items() if key not in ("runs", "field_names", "field_units", "support", "basis", "location")}
    meta.update(benchmark="RD-E1101 native cell invariants projected to physical shell nodes",
                field_names=["displacement_x", "displacement_y", "displacement_z",
                             "von_mises_shell_scalar_projected", "equivalent_plastic_strain_projected",
                             "total_strain_principal_major_inplane_projected", "total_strain_principal_minor_inplane_projected"],
                field_units=["mm", "mm", "mm", "MPa", "1", "1", "1"], runs={},
                support="919 original nodes connected to physical shell elements; auxiliary RBODY master990 excluded. Real original shell-edge node adjacency.",
                basis="Displacement in global coordinates verified against physical-node current minus initial geometry. Four invariant scalars computed before projection; no average of local tensor components.",
                location="Channels0:3 raw physical-node ANIM displacement. Channels3:7 are projected cell invariants: native shell von Mises scalar, IP3 EPSP, eigenvalues of IP3 in-plane native total strain. Projected invariants are not raw solver nodal outputs.",
                projection={"definition": "w_ic = initial_area_c / number_of_vertices_c for incident cells; projected_i = sum_c(w_ic invariant_c)/sum_c(w_ic). Initial area and connectivity are fixed across cases.",
                            "conservation": "Lumped-nodal-area integral equals original cell-area integral before float32 rounding.",
                            "limitations": "Projection smooths element discontinuities and extrema. In-plane principal values are not asserted equal to 3D principal values. No displacement-derived strain is used.",
                            "native_cell_manifest_sha256": sha(cell_cache / "manifest.json"), "source_starter_sha256": sha(starter)},
                scalar_channels_derived_from_native_tensors=["total_strain_principal_major_inplane_projected", "total_strain_principal_minor_inplane_projected"])
    diagnostics = {}
    for case_id, entry in cell_meta["runs"].items():
        raw = np.load(cell_cache / entry["target"], mmap_mode="r")
        assert raw.shape == (808, 10, 8)
        e = raw[..., 3:6].astype(np.float64)
        mean = (e[...,0]+e[...,1])*.5
        radius = np.sqrt(((e[...,0]-e[...,1])*.5)**2 + e[...,2]**2)
        invariants = np.stack([raw[...,7], raw[...,6], mean+radius, mean-radius], -1)
        projected64 = np.einsum("ij,jtc->itc", projection, invariants, optimize=True)
        source_run = work / "runs" / case_id
        displacement_path = source_run / "nodal_displacement_with_initial.npy"
        displacement = np.load(displacement_path)
        assert displacement.shape == (919, 11, 3)
        assert np.array_equal(np.load(source_run / "physical_node_ids.npy"), node_ids)
        target = np.concatenate([displacement[:,1:], projected64.astype(np.float32)], axis=-1)
        assert target.shape == (919, 10, 7) and target.dtype == np.float32
        assert np.isfinite(target).all() and (target[...,3:5] >= 0).all()
        dest = out / "runs" / (case_id+".npy")
        np.save(dest, target)
        integral = np.einsum("i,itc->tc", lumped_area, projected64)
        original = np.einsum("i,itc->tc", area, invariants)
        residual = np.max(np.abs(integral-original), axis=0)
        assert np.allclose(integral, original, rtol=1e-12, atol=1e-8)
        copied = {key: value for key, value in entry.items() if key not in ("target", "target_sha256", "thickness")}
        copied.update(target="runs/"+dest.name, target_sha256=sha(dest), thickness="thickness.npy",
                      native_cell_target_sha256=entry["target_sha256"], raw_nodal_displacement_sha256=sha(displacement_path))
        meta["runs"][case_id] = copied
        diagnostics[case_id] = {"max_integral_abs_residual_before_float32_by_channel": residual.tolist(),
                                "cell_invariant_absolute_max": np.max(np.abs(invariants), axis=(0,1)).tolist(),
                                "projected_nodal_absolute_max": np.max(np.abs(target[...,3:]), axis=(0,1)).tolist(),
                                "raw_displacement_absolute_max_mm": np.max(np.abs(displacement[:,1:]), axis=(0,1)).tolist()}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "projection-diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    shutil.copyfile(cell_cache / "frozen-label-metadata.json", out / "frozen-label-metadata.json")
    print(json.dumps({"manifest": str(out / "manifest.json"), "manifest_sha256": sha(out / "manifest.json"),
                      "shape": [919, 10, 7], "cases": len(meta["runs"])}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-cache", type=Path, required=True)
    parser.add_argument("--starter", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.cell_cache.resolve(), args.starter.resolve(), args.work.resolve(), args.out.resolve())
