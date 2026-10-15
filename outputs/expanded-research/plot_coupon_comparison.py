"""Plot frozen same-support coupon metrics; no FEM/ML arrays or models loaded."""
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    root = Path(__file__).resolve().parent
    model_root = root / "benchmarks/structural_coupon/models"
    names = [("transolver", "Transolver"), ("flare", "FLARE"),
             ("geotransolver", "GeoTransolver"), ("geoflare", "GeoFLARE (GALE_FA)"),
             ("meshgraphnet", "MeshGraphNet"), ("figconvunet", "FIGConvUNet")]
    neural = []
    for key, label in names:
        path = model_root / key / "metrics.json"
        data = read(path)
        assert data["status"] == "actual_supervised_training_and_heldout_evaluation_completed", path
        neural.append((label, data["metrics"][key], data))
    pod = read(root / "benchmarks/rom/coupon13/metrics.json")
    direct = read(root / "benchmarks/rom/coupon13-direct-rbf/metrics.json")
    contracts = [item[2]["contract"] for item in neural] + [pod["contract"], direct["contract"]]
    for contract in contracts[1:]:
        assert contract["manifest_sha256"] == contracts[0]["manifest_sha256"]
        assert contract["shape"] == contracts[0]["shape"]
        assert contract["split"] == contracts[0]["split"]
        assert contract["fields"] == contracts[0]["fields"]
    methods = [("Training mean", neural[0][2]["metrics"]["training_mean_field"], "#777777"),
               ("POD + ridge", pod["test_metrics"]["pod_ridge"], "#2c7fb8"),
               ("POD + RBF", pod["test_metrics"]["pod_rbf"], "#2c7fb8"),
               ("Direct RBF", direct["metrics"]["rbf_direct_field"], "#2c7fb8")]
    methods.extend((label, metric, "#d9732c") for label, metric, _ in neural)
    fields = [("Displacement vector L2 [%]", "displacement_vector", "relative_l2", 100),
              ("Projected VM stress RMSE [MPa]", "von_mises_shell_scalar_projected", "rmse", 1),
              ("Projected EPSP RMSE [1]", "equivalent_plastic_strain_projected", "rmse", 1),
              ("Total strain principal major RMSE [1]", "total_strain_principal_major_inplane_projected", "rmse", 1),
              ("Total strain principal minor RMSE [1]", "total_strain_principal_minor_inplane_projected", "rmse", 1)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    labels = [item[0] for item in methods]
    colors = [item[2] for item in methods]
    for ax, (title, field, key, factor) in zip(axes.flat, fields):
        values = [item[1][field][key] * factor for item in methods]
        assert all(value > 0 for value in values)
        ax.barh(labels, values, color=colors)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", which="major", alpha=.25)
        ax.tick_params(axis="y", labelsize=9)
    ax = axes.flat[-1]
    fractions = []
    for _, metrics, _ in methods:
        field = metrics["von_mises_shell_scalar_projected"]
        fractions.append(100 * field["prediction_negative_count_for_nonnegative_field"] / field["count"])
    ax.barh(labels, fractions, color=colors)
    ax.invert_yaxis()
    ax.set_title("Raw negative VM predictions [% of point-times]", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=.25)
    fig.suptitle("OpenCourant coupon: 5 training / 4 validation / 4 held-out simulations\n"
                 "919 nodes, 10 stored times; initial model configurations and recorded budgets", fontsize=14)
    fig.savefig(root / "coupon-algorithm-comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"methods": len(methods), "same_support_split_fields_verified": True,
                      "output": "coupon-algorithm-comparison.png"}))


if __name__ == "__main__":
    main()
