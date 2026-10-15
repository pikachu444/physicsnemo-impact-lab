"""Preserve raw ANIM global nodal displacement on source physical shell nodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

from build_nodal_cache import source_mesh


def extract(case):
    nodes, elements = source_mesh(case / "tensile_LAW2_0000.rad")
    physical_ids = np.asarray(sorted({n for conn in elements.values() for n in conn}), dtype=np.int64)
    assert len(physical_ids) == 919 and set(nodes)-set(physical_ids) == {990}
    values, times = [], []
    frames = sorted(case.glob("tensile_LAW2A[0-9][0-9][0-9].vtk"))
    assert len(frames) == 11
    for frame in frames:
        reader = vtkUnstructuredGridReader()
        reader.SetFileName(str(frame))
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.ReadAllFieldsOn()
        reader.Update()
        grid = reader.GetOutput()
        point_ids = vtk_to_numpy(grid.GetPointData().GetArray("NODE_ID"))
        rows = {int(n): i for i, n in enumerate(point_ids)}
        order = np.asarray([rows[int(n)] for n in physical_ids])
        displacement = vtk_to_numpy(grid.GetPointData().GetArray("Displacement"))[order].copy()
        assert displacement.shape == (919, 3) and np.isfinite(displacement).all()
        values.append(displacement)
        times.append(float(vtk_to_numpy(grid.GetFieldData().GetArray("TIME"))[0]))
    target = np.stack(values, 1).astype(np.float32)
    assert target.shape == (919, 11, 3) and np.max(np.abs(target[:,0])) < 1e-7
    dest = case / "nodal_displacement_with_initial.npy"
    if dest.exists():
        assert np.array_equal(np.load(dest), target), "Refusing to change existing displacement evidence"
    else:
        np.save(dest, target)
        np.save(case / "physical_node_ids.npy", physical_ids)
    metadata = {"source": "ANIM /VECT/DISP converted Displacement vector; no numerical displacement derivation",
                "basis": "global", "units": "mm", "shape": list(target.shape), "physical_node_count": 919,
                "excluded_auxiliary_master_node": 990, "actual_times_ms": times,
                "final_min_mm": target[:,-1].min(0).tolist(), "final_max_mm": target[:,-1].max(0).tolist(),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (case / "nodal-displacement-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"case": case.name, **metadata}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    extract(parser.parse_args().case.resolve())
