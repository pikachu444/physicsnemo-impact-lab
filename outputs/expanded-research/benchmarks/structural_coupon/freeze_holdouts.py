"""Create the authorized eight disjoint interior FEM validation/test conditions."""
import argparse
import hashlib
import json
from pathlib import Path

import scipy
from scipy.stats import qmc


def freeze(out):
    assert not out.exists(), "Do not overwrite a frozen design"
    unit = qmc.Sobol(d=2, scramble=True, seed=20261015).random_base2(m=3)
    assert ((unit > 0) & (unit < 1)).all()
    runs = []
    for i, point in enumerate(unit):
        group = "validation" if i < 4 else "test"
        index = i + 1 if i < 4 else i - 3
        runs.append({"case": f"holdout_{group}_{index:02d}", "split": group,
                     "unit_square_point": point.tolist(), "strength_scale": float(.9 + .2 * point[0]),
                     "velocity_scale": float(.75 + .5 * point[1])})
    anchors = ["pilot_base", "corner_s090_v075", "corner_s090_v125", "corner_s110_v075", "corner_s110_v125"]
    design = {"scope": "Initial train anchor envelope plus frozen disjoint interior validation/test FEM cases. These eight cases are not training data; test labels must not select N, hyperparameters or stopping epochs. Total future training DOE size remains undetermined.",
              "parameter_bounds": {"yield_UTS_scale": [.9, 1.1], "velocity_scale": [.75, 1.25]},
              "parameter_bounds_scope": "Parent-authorized algorithm sensitivity assumptions, not measured material scatter or product tolerance.",
              "generator": {"method": "scipy.stats.qmc.Sobol", "dimension": 2, "scramble": True, "seed": 20261015,
                            "power": 3, "scipy_version": scipy.__version__,
                            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "allocation": "First four points validation, last four test; all are distinct interior FEM conditions. A future training sequence requires a different seed and duplicate checks."},
              "split": {"train": anchors, "validation": [r["case"] for r in runs if r["split"] == "validation"],
                        "test": [r["case"] for r in runs if r["split"] == "test"]},
              "holdout_runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(json.dumps({"design": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "runs": runs}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    freeze(parser.parse_args().out.resolve())
