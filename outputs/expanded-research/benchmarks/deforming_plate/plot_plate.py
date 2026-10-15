"""Render actual held-out plate predictions; no synthetic reference results."""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--trajectory")
    args = parser.parse_args()
    metrics = json.loads((args.run / "metrics.json").read_text())
    trajectory = args.trajectory or metrics["results"][min(1, len(metrics["results"]) - 1)]["trajectory"]
    with np.load(args.run / f"{trajectory}_evaluation.npz") as f:
        data = {k: f[k] for k in f.files}
    frame = min(359, len(data["truth_positions"]) - 1)
    free = data["free_mask"]
    reference = data["reference_positions"]
    strain_index = int(np.argmin(abs(data["strain_frames"] - frame)))
    strain_frame = int(data["strain_frames"][strain_index])
    cells = data["cells"]
    xy = reference[free, :2]
    cell_xy = reference[cells].mean(axis=1)[:, :2]
    true_disp = np.linalg.norm(data["truth_positions"][frame, free] - reference[free], axis=-1)
    pred_disp = np.linalg.norm(data["prediction_positions"][frame, free] - reference[free], axis=-1)
    true_stress, pred_stress = data["truth_stress"][frame, free, 0], data["prediction_stress"][frame, free, 0]
    true_strain = data["truth_derived_strain"][strain_index, :, 2]
    pred_strain = data["predicted_derived_strain"][strain_index, :, 2]
    rows = [(xy, true_disp, pred_disp, f"Displacement magnitude, step {frame + 1}\n[dataset coordinate units]"),
            (xy, true_stress, pred_stress, f"Nodal von Mises stress, step {frame + 1}\n[dataset stress units]"),
            (cell_xy, true_strain, pred_strain, f"Derived Green-Lagrange Ezz, step {strain_frame + 1}\n[dimensionless; NOT COMSOL strain output]")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    for row, (coords, exact, predicted, label) in enumerate(rows):
        vmin = float(min(exact.min(), predicted.min()))
        vmax = float(max(exact.max(), predicted.max()))
        if vmax == vmin:
            vmax = vmin + 1e-12
        for col, values in enumerate([exact, predicted, abs(predicted - exact)]):
            ax = axes[row, col]
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=8, cmap="inferno" if col < 2 else "magma",
                            vmin=vmin if col < 2 else 0, vmax=vmax if col < 2 else max(float(values.max()), 1e-12))
            ax.set_aspect("equal")
            ax.set_xlabel("Reference x [dataset units]")
            ax.set_ylabel("Reference y [dataset units]")
            ax.set_title(["FEM reference", "PhysicsNeMo prediction", "Absolute error"][col] + "\n" + label, fontsize=10)
            fig.colorbar(sc, ax=ax, shrink=0.7)
    role = metrics.get("evaluation_role", "development after pilot inspection")
    fig.suptitle(f"Official source trajectory {trajectory}, role: {role}\nFree plate nodes / non-actuator tetrahedra; autoregressive; actuator and clamp prescribed", fontsize=13)
    fig.savefig(args.run / "fields_comparison.png", dpi=160)
    plt.close(fig)
    history_path = args.run / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.semilogy([r["epoch"] for r in history], [r["train_mse"] for r in history], label="Training (with input noise)")
        ax.semilogy([r["epoch"] for r in history], [r["validation_mse"] for r in history], label="Teacher-forced validation (clean)")
        ax.axvline(metrics["best_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
        ax.set(xlabel="Epoch", ylabel="Normalized 4-channel MSE", title="Plate learning curve (bounded training)")
        ax.legend()
        fig.savefig(args.run / "learning_curve.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
