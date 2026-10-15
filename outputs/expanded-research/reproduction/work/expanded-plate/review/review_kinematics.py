"""Independent, CPU-only checks of the plate strain and mesh helper contract."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "outputs/expanded-research/benchmarks/deforming_plate/review"
spec = importlib.util.spec_from_file_location("plate_data_under_review", ROOT / "plate_data.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

reference = np.array([[.3, -.7, .2], [1.4, -.2, .5], [.4, .8, .1], [-.1, .3, 1.9]])
tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
affine = np.array([[1.2, .25, -.1], [.07, .9, .2], [.05, -.15, 1.1]])
angle = .83
rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
expected_tensor = .5 * (affine.T @ affine - np.eye(3))
expected = expected_tensor[[0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]][None]
results = {}
for name, current in {
    "arbitrary_affine": reference @ affine.T + np.array([3.1, -.2, .7]),
    "superposed_rigid_rotation": reference @ affine.T @ rotation.T + np.array([1.2, .8, -.6]),
}.items():
    observed, volume = module.deformation_strain(reference, current, tet)
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-8)
    results[name] = {"max_abs_error": float(abs(observed - expected).max()),
                     "reference_volume": float(volume[0])}
permutation = np.array([2, 0, 3, 1])
inverse = np.argsort(permutation)
observed, _ = module.deformation_strain(reference[permutation],
                                       (reference @ affine.T)[permutation], inverse[tet])
np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-8)
results["consistent_node_permutation"] = {"max_abs_error": float(abs(observed - expected).max())}
try:
    module.deformation_strain(np.zeros((4, 3)), reference, tet)
except AssertionError:
    results["degenerate_reference_refused"] = True
else:
    raise AssertionError("Degenerate reference must not produce plausible strain")
module.self_check()
results["worker_self_check"] = "passed"

rows = []
for path in sorted((ROOT / "data").glob("*.npz")):
    with np.load(path) as data:
        x = data["mesh_pos"][0].astype(np.float64)
        cells = data["cells"][0]
        kinds = data["node_type"][0, :, 0]
        dm = np.stack([x[cells[:, i]] - x[cells[:, 0]] for i in (1, 2, 3)], axis=-1)
        det = np.linalg.det(dm)
        final = data["world_pos"][-1].astype(np.float64)
        ds = np.stack([final[cells[:, i]] - final[cells[:, 0]] for i in (1, 2, 3)], axis=-1)
        det_f = np.linalg.det(ds @ np.linalg.inv(dm))
        plate_mask = ~(kinds[cells] == 1).any(axis=-1)
        initial_strain, _ = module.deformation_strain(x, data["world_pos"][0], cells)
        rows.append({"file": path.name, "nodes": len(x), "tetrahedra": len(cells),
                     "all_reference_determinants_positive": bool((det > 0).all()),
                     "max_reference_condition": float(np.linalg.cond(dm).max()),
                     "initial_world_equals_mesh": bool(np.array_equal(data["world_pos"][0], data["mesh_pos"][0])),
                     "initial_strain_abs_max": float(abs(initial_strain).max()),
                     "plate_tetrahedra_excluding_type1": int(plate_mask.sum()),
                     "final_inverted_plate_tetrahedra": int((det_f[plate_mask] <= 0).sum()),
                     "final_plate_detF_min": float(det_f[plate_mask].min())})
report = {"plate_data_sha256": hashlib.sha256((ROOT / "plate_data.py").read_bytes()).hexdigest(),
          "kinematic_checks": results, "actual_data_geometry_checks": rows,
          "scope": "No Torch import, GPU, training, or claim of original COMSOL strain-label validation."}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "kinematics-review.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
