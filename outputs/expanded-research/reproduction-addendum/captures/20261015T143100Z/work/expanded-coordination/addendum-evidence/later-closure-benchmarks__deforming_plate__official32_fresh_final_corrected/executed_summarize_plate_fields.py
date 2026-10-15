"""Inspect saved predictions only: native stress and coordinate-derived strain hotspots."""
import argparse
import hashlib
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.run / "field-diagnostics.json"
    if output.exists():
        raise FileExistsError(output)
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf8"))
    rows = []
    for result in metrics["results"]:
        row = {"trajectory": result["trajectory"], "status": result["status"]}
        if result["status"] != "completed":
            row["failure_context"] = result.get("failure_context")
            rows.append(row)
            continue
        path = args.run / result["field_file"]
        with path.open("rb") as stream:
            row["fields_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with np.load(path) as fields:
            free_ids = np.flatnonzero(fields["free_mask"])
            truth = fields["truth_stress"][:, free_ids, 0]
            pred = fields["prediction_stress"][:, free_ids, 0]
            t, j = np.unravel_index(np.argmax(truth), truth.shape)
            pt, pj = np.unravel_index(np.argmax(pred), pred.shape)
            row["stress_FEM_peak"] = {"source_loading_step": int(t + 1),
                "array_node_index_not_solver_id": int(free_ids[j]),
                "reference_xyz": fields["reference_positions"][free_ids[j]].tolist(),
                "FEM_native_scalar_stress": float(truth[t, j]),
                "AI_at_same_node_and_step": float(pred[t, j]),
                "absolute_error": float(abs(pred[t, j] - truth[t, j])),
                "relative_error": float(abs(pred[t, j] - truth[t, j]) / abs(truth[t, j])) if truth[t, j] else None,
                "units": "Unconfirmed dataset scalar-stress units; no MPa conversion"}
            row["stress_AI_peak"] = {"source_loading_step": int(pt + 1),
                "array_node_index_not_solver_id": int(free_ids[pj]),
                "AI_native_scalar_stress": float(pred[pt, pj]),
                "FEM_at_same_node_and_step": float(truth[pt, pj]),
                "reference_location_distance_from_FEM_peak": float(np.linalg.norm(fields["reference_positions"][free_ids[pj]] - fields["reference_positions"][free_ids[j]])),
                "distance_units": "Unconfirmed dataset coordinate units"}
            true_strain, pred_strain = fields["truth_derived_strain"], fields["predicted_derived_strain"]
            frame, cell, component = np.unravel_index(np.argmax(abs(true_strain)), true_strain.shape)
            exact, predicted = float(true_strain[frame, cell, component]), float(pred_strain[frame, cell, component])
            row["derived_GL_max_abs_reference_component"] = {
                "source_loading_step": int(fields["strain_frames"][frame] + 1),
                "array_tetrahedron_index_not_solver_id": int(fields["original_array_tetrahedron_indices_not_solver_ids"][cell]),
                "component": ["xx", "yy", "zz", "xy", "yz", "xz"][component],
                "FEM_positions_derived_GL": exact, "AI_positions_derived_GL_at_same_cell_step_component": predicted,
                "absolute_error": abs(predicted - exact),
                "relative_error": abs(predicted - exact) / abs(exact) if exact else None,
                "scope": "Only saved sampled strain frames; unweighted tetrahedra/tensor components; dimensionless; no native COMSOL strain labels"}
        row.update({"displacement_relative_l2": result["rollout_free_displacement"]["relative_l2"],
                    "stress_relative_l2": result["rollout_free_stress"]["relative_l2"],
                    "derived_GL_relative_l2": result["derived_GL_strain"]["relative_l2"],
                    "prediction_inverted_frames": sum(v > 0 for v in result["all_frame_predicted_inverted_tetrahedra"]),
                    "reference_inverted_frames": sum(v > 0 for v in result["all_frame_truth_inverted_tetrahedra"]),
                    "maximum_predicted_inverted_tetrahedra": max(result["all_frame_predicted_inverted_tetrahedra"]),
                    "negative_predicted_stress_values": result["predicted_negative_von_mises_values"],
                    "AR399_graph_forward_geometry_seconds": result["evaluation_seconds"],
                    "AR_graph_preparation_seconds": result["AR_graph_preparation_seconds"],
                    "AR_forward_output_seconds": result["AR_forward_and_output_transfer_seconds"],
                    "derived_strain_seconds": result["derived_strain_seconds"],
                    "maximum_AR_total_directed_edges": result["maximum_AR_total_directed_edges"]})
        rows.append(row)
    report = {"evaluation_role": metrics["evaluation_role"], "checkpoint_sha256": metrics["checkpoint_sha256"],
              "selected_epoch": metrics["best_epoch"], "results": rows,
              "interpretation": "Diagnostic re-reading of saved arrays; no new neural inference or model selection",
              "stress_units": "Native dataset units, unconfirmed; no assumed MPa conversion",
              "strain": "Green-Lagrange derived from source/AI nodal coordinates, not native COMSOL strain labels"}
    with output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
