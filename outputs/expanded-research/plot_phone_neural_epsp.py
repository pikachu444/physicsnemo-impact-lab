"""Both frozen phone tests in a FEM-defined EPSP envelope; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import psutil
    if psutil.virtual_memory().available < 1.5 * 1024**3:
        raise SystemExit("DEFER: available host RAM below1.5GiB; no arrays or plots imported")
    report = json.loads((args.run / "test_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["test_performance_passes"] == 1
    assert digest(args.cache) == report["frozen_data_verification"]["manifest_sha256"]
    meta = json.loads(args.cache.read_text(encoding="utf-8"))
    assert report["cases"] == meta["split"]["test"]
    assert report["field_names"] == meta["field_names"]
    assert not args.output.exists(), "Use a new output directory; preserve prior evidence"
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xyz = np.load(args.cache.parent / meta["coordinates"], mmap_mode="r")
    truth_path, pred_path = (args.run / "heldout" / name for name in ("truth.npy", "predictions.npy"))
    truth, prediction = (np.load(path, mmap_mode="r") for path in (truth_path, pred_path))
    assert list(truth.shape) == report["shape"] and prediction.shape == truth.shape
    channel = report["field_names"].index("epsp")
    positive = np.zeros(xyz.shape[0], dtype=bool)
    for case in range(len(report["cases"])):
        positive |= np.any(truth[case, :, :, channel] > 0, axis=1)
    assert positive.any(), "No true positive EPSP support; do not invent a hotspot"
    low = xyz[positive, :2].min(axis=0) - 1.0
    high = xyz[positive, :2].max(axis=0) + 1.0
    keep = np.all((xyz[:, :2] >= low) & (xyz[:, :2] <= high), axis=1)
    fields = [(truth[i, keep, -1, channel], prediction[i, keep, -1, channel])
              for i in range(len(report["cases"]))]
    lower = min(float(v.min()) for pair in fields for v in pair)
    upper = max(float(v.max()) for pair in fields for v in pair)
    error_max = max(float(np.abs(p-t).max()) for t, p in fields)
    fig, axes = plt.subplots(len(fields), 3, figsize=(12, 4*len(fields)),
                             layout="constrained", squeeze=False)
    for row, ((t, p), case) in enumerate(zip(fields, report["cases"])):
        for col, values in enumerate((t, p, np.abs(p-t))):
            order = np.argsort(np.abs(values), kind="stable")
            ax = axes[row, col]
            image = ax.scatter(xyz[keep, 0][order], xyz[keep, 1][order], c=values[order],
                s=12, cmap="magma" if col == 2 else "viridis",
                vmin=0 if col == 2 else lower,
                vmax=max(error_max, 1e-30) if col == 2 else upper)
            ax.set_aspect("equal")
            ax.set_title(("FEM EPSP", "Raw " + report["architecture"], "Absolute error")[col])
            ax.set_xlabel("Reference X [mm]")
            if col == 0:
                ax.set_ylabel(case + "\nReference Y [mm]")
            if col in (1, 2):
                fig.colorbar(image, ax=axes[row, :2] if col == 1 else ax, label="[1]")
    fig.suptitle("Both fixed test cases; final nominal20us, no clipping\n"
                 "ROI: union of positive FEM EPSP over all stored times +1mm display margin")
    args.output.mkdir(parents=True)
    output = args.output / "epsp_both_tests_truth_support_zoom.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    evidence = {"status": "saved_prediction_postprocessing_completed", "model_calls": 0,
        "scope": "All fixed test cases; FEM-defined display envelope, no error-based selection, refitting or clipping.",
        "cases": report["cases"], "time": report["time_values"][-1], "time_unit": report["time_unit"],
        "roi_reference_xy_min_mm": low.tolist(), "roi_reference_xy_max_mm": high.tolist(),
        "roi_node_count": int(keep.sum()), "total_nodes": int(xyz.shape[0]),
        "final_time_true_positive_counts": [int(np.count_nonzero(truth[i, :, -1, channel] > 0)) for i in range(len(fields))],
        "overlap_policy": "Reference XY projection; coincident points drawn in increasing absolute value order. Not a deformed3D surface.",
        "shared_truth_prediction_range": [lower, upper], "absolute_error_range": [0, error_max],
        "source_sha256": digest(Path(__file__)), "cache_sha256": digest(args.cache),
        "test_report_sha256": digest(args.run / "test_report.json"),
        "truth_sha256": digest(truth_path), "prediction_sha256": digest(pred_path),
        "image_sha256": digest(output), "rss_bytes_at_end": psutil.Process().memory_info().rss}
    (args.output / "postprocessing.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "roi_nodes": int(keep.sum()),
                      "true_final_positive_counts": evidence["final_time_true_positive_counts"]}))


if __name__ == "__main__":
    main()
