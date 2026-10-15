"""Extract native FEM stress/total-strain/EPSP; no displacement-derived strain labels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader

FIELD_NAMES = ["stress_xx_local", "stress_yy_local", "stress_xy_local", "total_strain_xx_local", "total_strain_yy_local", "total_strain_xy_local", "equivalent_plastic_strain", "von_mises_shell_scalar"]
FIELD_UNITS = ["MPa", "MPa", "MPa", "1", "1", "1", "1", "MPa"]
STRESS = "2DELEM_Stress_(layer__3)_"
STRAIN = "2DELEM_Strain_(layer__3)_"
EPSP = "2DELEM_Plast_Strn_Layer___3_"


def read_source(path):
    nodes, elements = {}, {}
    section = ""
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("/"):
            section = line
            continue
        if section == "/NODE":
            nodes[int(line[:10])] = [float(line[10:30]), float(line[30:50]), float(line[50:70])]
        elif section.startswith(("/SHELL/", "/SH3N/")):
            count = 4 if section.startswith("/SHELL/") else 3
            values = [int(line[i * 10:(i + 1) * 10]) for i in range(count + 1)]
            elements[values[0]] = values[1:]
    assert len(nodes) == 920 and len(elements) == 808, (len(nodes), len(elements))
    return nodes, elements


def read_vtk(path):
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllTensorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    return reader.GetOutput()


def extract(case, solver, out, plot=True):
    out.mkdir(parents=True, exist_ok=True)
    parameters = json.loads((case / "parameters.json").read_text())
    import re
    starter_report = (case / "tensile_LAW2_0000.out").read_text(errors="replace")
    derived_law2 = {}
    for key, marker in [("A_GPa", "YIELD COEFFICIENT A"), ("B_GPa", "YIELD COEFFICIENT B"), ("n", "YIELD COEFFICIENT N")]:
        found = re.search(re.escape(marker) + r"[^=\r\n]*=\s*([0-9.Ee+\-]+)", starter_report)
        assert found, f"Missing solver-calculated material coefficient: {marker}"
        derived_law2[key] = float(found.group(1))
    nodes, elements = read_source(case / "tensile_LAW2_0000.rad")
    physical_nodes = {n for conn in elements.values() for n in conn}
    ids = np.array(sorted(elements), dtype=np.int64)
    xyz = np.array([np.mean([nodes[n] for n in elements[e]], axis=0) for e in ids], dtype=np.float32)
    incidence = {}
    for i, eid in enumerate(ids):
        conn = elements[eid]
        for a, b in zip(conn, conn[1:] + conn[:1]):
            incidence.setdefault(tuple(sorted((a, b))), []).append(i)
    edges = sorted({(a, b) for attached in incidence.values() for a in attached for b in attached if a != b})
    edge_index = np.array(edges, dtype=np.int64).T
    assert edge_index.shape[0] == 2 and len(np.unique(edge_index)) == len(ids)
    frames = sorted(case.glob("tensile_LAW2A[0-9][0-9][0-9]"))
    assert len(frames) >= 2
    tensors, strains, plastic, von_mises_native, times, checks = [], [], [], [], [], []
    converter = solver / "exec/anim_to_vtk_win64.exe"
    for frame in frames:
        vtk = frame.with_suffix(".vtk")
        if not vtk.exists():
            with vtk.open("wb") as stdout, (case / (frame.name + ".converter-stderr.log")).open("wb") as stderr:
                result = subprocess.run([str(converter), frame.name], cwd=case, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode == 0, frame
        grid = read_vtk(vtk)
        cells, points = grid.GetCellData(), grid.GetPointData()
        eid = vtk_to_numpy(cells.GetArray("ELEMENT_ID"))
        part = vtk_to_numpy(cells.GetArray("PART_ID"))
        rows = {int(e): i for i, e in enumerate(eid) if int(e) in elements and int(part[i]) == 3}
        assert len(rows) == len(ids)
        order = np.array([rows[int(e)] for e in ids])
        nid = vtk_to_numpy(points.GetArray("NODE_ID"))
        for e, index in zip(ids, order):
            cell = grid.GetCell(int(index))
            actual = {int(nid[cell.GetPointId(j)]) for j in range(cell.GetNumberOfPoints())}
            assert actual == set(elements[e]), f"Element connectivity differs: {e}"
        stress = vtk_to_numpy(cells.GetArray(STRESS))[order].reshape(-1, 3, 3).copy()
        strain = vtk_to_numpy(cells.GetArray(STRAIN))[order].reshape(-1, 3, 3).copy()
        epsp = vtk_to_numpy(cells.GetArray(EPSP))[order].copy()
        assert np.allclose(stress, stress.transpose(0, 2, 1), atol=1e-8)
        assert np.allclose(strain, strain.transpose(0, 2, 1), atol=1e-8)
        # Converter-filled zz/yz/zx zeros are deliberately not physical training labels.
        s = stress[:, [0, 1, 0], [0, 1, 1]]
        e = strain[:, [0, 1, 0], [0, 1, 1]]
        vonm = np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2)
        direct_vm = vtk_to_numpy(cells.GetArray("2DELEM_Von_Mises"))[order]
        displacement = vtk_to_numpy(points.GetArray("Displacement"))
        current_xyz = vtk_to_numpy(grid.GetPoints().GetData())
        initial = np.array([nodes[int(n)] for n in nid])
        physical_mask = np.array([int(n) in physical_nodes for n in nid])
        geometry_error = float(np.max(np.abs(current_xyz[physical_mask] - initial[physical_mask] - displacement[physical_mask])))
        auxiliary_reference_offset = float(np.max(np.abs(current_xyz[~physical_mask] - initial[~physical_mask] - displacement[~physical_mask])))
        assert geometry_error < 0.002, "Coordinate/displacement mismatch beyond ASCII conversion precision"
        times.append(float(vtk_to_numpy(grid.GetFieldData().GetArray("TIME"))[0]))
        tensors.append(s)
        strains.append(e)
        plastic.append(epsp)
        von_mises_native.append(direct_vm.copy())
        checks.append({"time_ms": times[-1], "von_mises_crosscheck_max_abs_MPa": float(np.max(np.abs(vonm - direct_vm))) * 1000,
                       "coordinate_minus_displacement_max_abs_mm": geometry_error,
                       "excluded_auxiliary_node_reference_offset_max_abs_mm": auxiliary_reference_offset,
                       "vtk_cells": grid.GetNumberOfCells(), "kept_physical_shell_cells": len(ids),
                       "excluded_rigid_body_auxiliary_cells": grid.GetNumberOfCells() - len(ids)})
    native_s = np.stack(tensors, axis=1)
    native_e = np.stack(strains, axis=1)
    epsp = np.stack(plastic, axis=1)[..., None]
    native_vm = np.stack(von_mises_native, axis=1)[..., None]
    target = np.concatenate([native_s * 1000, native_e, epsp, native_vm * 1000], axis=-1).astype(np.float32)
    assert target.shape == (808, len(frames), 8) and np.isfinite(target).all() and (epsp >= 0).all()
    assert np.max(np.abs(target[:, 0])) < 1e-7 and np.max(target[..., 6]) > 0
    np.savez_compressed(case / "native_fem_fields.npz", element_ids=ids, coordinates_initial_mm=xyz, edge_index=edge_index,
                        actual_times_ms=np.array(times), stress_GPa=native_s, total_strain_tensor=native_e, equivalent_plastic_strain=epsp, von_mises_shell_GPa=native_vm)
    np.save(case / "target_with_initial.npy", target)
    np.save(case / "coordinates.npy", xyz)
    np.save(case / "edge_index.npy", edge_index)
    converter_th = solver / "exec/th_to_csv_win64.exe"
    csv_path = case / "tensile_LAW2T01.csv"
    if not csv_path.exists():
        with (case / "th-converter.stdout.log").open("wb") as stdout, (case / "th-converter.stderr.log").open("wb") as stderr:
            result = subprocess.run([str(converter_th), "tensile_LAW2T01"], cwd=case, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0
    with csv_path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header, history = rows[0], np.array(rows[1:], dtype=float)
    assert len(header) == 45 and "102" in header[24] and "616" in header[30] and "Rigid_Body" in header[36]
    # T01 RBODY stores impulse; its derivative is a diagnostic force, not a direct force label.
    time = history[:, 0]
    force = -np.gradient(history[:, 36], time, edge_order=2)
    extension = history[:, 30]
    gauge_length = float(np.linalg.norm(np.array(nodes[616]) - np.array(nodes[102])))
    assert abs(gauge_length - 80) < 1e-8
    engineering_strain = extension / gauge_length
    engineering_stress = force * 1000 / (12 * 2.5)
    curve = np.column_stack([time, extension, force, engineering_strain, engineering_stress, history[:, 1], history[:, 2], history[:, 12]])
    np.savetxt(out / "force_extension_diagnostic.csv", curve, delimiter=",", header="time_ms,gauge_extension_mm,reaction_from_impulse_kN,engineering_strain,engineering_stress_MPa,internal_energy_J,kinetic_energy_J,hourglass_energy_J", comments="")
    late = time >= 1
    summary = {"case": parameters["case"], "parameters": parameters, "native_fields": FIELD_NAMES, "units": FIELD_UNITS,
               "solver_calculated_LAW2_coefficients": derived_law2,
               "actual_times_ms": times, "support": "808 original physical shell elements, sorted by original ELEMENT_ID; no nodal stress smoothing",
               "physical_mesh_node_count": len(physical_nodes),
               "basis": "native element local co-rotational basis; no conversion to global coordinates",
               "shear_convention": "tensor epsilon_xy=engineering gamma_xy/2; no additional scaling in extraction",
               "strain_measure": "solver native accumulated total strain output for full nonlinear QEPH shell; not displacement-derived; exact finite-strain-measure equivalence not independently proved",
               "location": "tensor stress/total strain/EPSP:thickness integration point3 of5; vonMises:separate native shell scalar output, not asserted identical to IP3 tensor invariant", "converter_sha256": hashlib.sha256(converter.read_bytes()).hexdigest(),
               "checks": checks, "final_min": target[:, -1].min(0).tolist(), "final_max": target[:, -1].max(0).tolist(),
               "final_std": target[:, -1].std(0).tolist(), "final_positive_plastic_cell_fraction": float(np.mean(epsp[:, -1, 0] > 1e-6)),
               "force_diagnostic": {"source": "negative derivative of RBODY T01 impulse, no filter", "gauge_length_mm": gauge_length,
                                    "area_initial_mm2": 30, "final_force_kN": float(force[-1]), "final_engineering_strain": float(engineering_strain[-1]),
                                    "final_engineering_stress_MPa": float(engineering_stress[-1])},
               "energy": {"final_internal_J": float(history[-1, 1]), "final_kinetic_J": float(history[-1, 2]),
                          "max_kinetic_over_internal_after1ms": float(np.max(history[late, 2] / np.maximum(history[late, 1], 1e-20))),
                          "final_hourglass_over_internal": float(history[-1, 12] / history[-1, 1])},
               "execution": json.loads((case / "execution-metrics.json").read_text()),
               "labels_sha256": hashlib.sha256((case / "native_fem_fields.npz").read_bytes()).hexdigest()}
    (out / "pilot-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        polygons = [[nodes[n][:2] for n in elements[e]] for e in ids]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), constrained_layout=True)
        for ax, col, title in zip(axes.flat[:3], [7, 3, 6], ["Native shell von Mises scalar [MPa]", "Native total strain xx, varying local axes [1]", "Native equivalent plastic strain, IP3 [1]"]):
            collection = PolyCollection(polygons, array=target[:, -1, col], cmap="viridis", edgecolors="none")
            ax.add_collection(collection)
            ax.autoscale()
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("Initial X [mm]")
            ax.set_ylabel("Initial Y [mm]")
            fig.colorbar(collection, ax=ax, shrink=.7)
        axes[1, 1].plot(extension, force, linewidth=1)
        axes[1, 1].set(xlabel="Gauge extension [mm]", ylabel="Reaction from impulse derivative [kN]", title="Diagnostic force-extension; unfiltered")
        fig.suptitle(f"DP600 LAW2 pilot: native solver labels at actual t={times[-1]:.6g} ms; initial geometry")
        fig.savefig(out / "pilot_fields.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"case": parameters["case"], "target_shape": list(target.shape), "final_field_max": summary["final_max"], "summary": str(out / "pilot-summary.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    extract(args.case.resolve(), args.solver.resolve(), args.out.resolve(), not args.no_plot)
