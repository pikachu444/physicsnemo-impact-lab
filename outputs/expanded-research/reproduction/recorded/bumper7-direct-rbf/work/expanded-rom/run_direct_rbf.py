"""Streamed direct RBF field regression on the common FEM benchmark cache.

This fixed-mesh baseline maps global design parameters to the whole FEM field.
It is kernel ridge regression, not a physical solver or a PhysicsNeMo model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

from benchmark_data import Cache, physical_metrics


GAMMAS = (0.1, 0.3, 1.0, 3.0)
ALPHAS = (1e-4, 0.01, 0.1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def close_map(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def read_block(path, begin, end):
    # Releasing each mapping keeps previously touched pages out of process RSS.
    array = np.load(path, mmap_mode="r")
    block = np.array(array[:, begin:end], dtype=np.float64, copy=True)
    close_map(array)
    return block


def rbf(a, b, gamma):
    distance = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * distance)


def normalized_globals(cache, cases, stats):
    rows = np.asarray([cache.meta["runs"][case]["globals"] for case in cases], dtype=np.float64)
    return (rows - np.asarray(stats["global_mean"])) / np.asarray(stats["global_std"])


def write_training_matrix(cache, cases, destination, mean_path, scale, chunk):
    dimension = int(np.prod(cache.shape))
    matrix = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                       shape=(len(cases), dimension))
    close_map(matrix)
    mean = np.load(mean_path, mmap_mode="r")
    mean_flat = mean.reshape(-1)
    for row, case in enumerate(cases):
        target = cache.target(case)
        assert target.shape == cache.shape, case
        flat = target.reshape(-1)
        matrix = np.load(destination, mmap_mode="r+")
        for begin in range(0, dimension, chunk):
            end = min(begin + chunk, dimension)
            channel_scale = scale[np.arange(begin, end) % len(scale)]
            matrix[row, begin:end] = (flat[begin:end] - mean_flat[begin:end]) / channel_scale
        matrix.flush()
        close_map(matrix)
        close_map(target)
    close_map(mean)


def field_inner_products(train_path, validation_path, dimension, chunk):
    train = np.load(train_path, mmap_mode="r")
    validation = np.load(validation_path, mmap_mode="r")
    gram = np.zeros((len(train), len(train)), dtype=np.float64)
    cross = np.zeros((len(validation), len(train)), dtype=np.float64)
    energy = np.zeros(len(validation), dtype=np.float64)
    close_map(train)
    close_map(validation)
    for begin in range(0, dimension, chunk):
        end = min(begin + chunk, dimension)
        train_block = read_block(train_path, begin, end)
        validation_block = read_block(validation_path, begin, end)
        gram += train_block @ train_block.T
        cross += validation_block @ train_block.T
        energy += np.einsum("ij,ij->i", validation_block, validation_block)
    return gram / dimension, cross / dimension, energy / dimension


def validation_mse(weights, gram, cross, energy):
    # This is ||weights @ Ytrain - Yvalidation||^2 / D without field reconstruction.
    per_case = np.einsum("ij,jk,ik->i", weights, gram, weights)
    per_case -= 2 * np.einsum("ij,ij->i", weights, cross)
    per_case += energy
    assert np.min(per_case) > -1e-8, per_case
    return float(np.maximum(per_case, 0).mean())


def predict_test(cache, output, train_path, mean_path, scale, xtrain, kernel, gamma,
                 stats, chunk, reference=None):
    cases = cache.split["test"]
    shape = (len(cases), *cache.shape)
    dimension = int(np.prod(cache.shape))
    prediction = np.lib.format.open_memmap(output / "predictions.npy", mode="w+", dtype=np.float32, shape=shape)
    truth = np.lib.format.open_memmap(output / "truth.npy", mode="w+", dtype=np.float32, shape=shape)
    close_map(prediction)
    close_map(truth)
    weights = np.linalg.solve(kernel, rbf(normalized_globals(cache, cases, stats), xtrain, gamma).T).T
    mean = np.load(mean_path, mmap_mode="r")
    mean_flat = mean.reshape(-1)
    maximum_difference = 0.
    prediction_seconds = 0.
    for row, case in enumerate(cases):
        tick = time.perf_counter()
        prediction = np.load(output / "predictions.npy", mmap_mode="r+")
        truth = np.load(output / "truth.npy", mmap_mode="r+")
        target = cache.target(case)
        truth[row] = target
        flat = prediction[row].reshape(-1)
        expected = np.load(reference, mmap_mode="r") if reference else None
        if expected is not None:
            assert expected.shape == shape, "Reference predictions have another field/support shape"
        for begin in range(0, dimension, chunk):
            end = min(begin + chunk, dimension)
            block = read_block(train_path, begin, end)
            field = weights[row] @ block
            field *= scale[np.arange(begin, end) % len(scale)]
            field += mean_flat[begin:end]
            flat[begin:end] = field
            if expected is not None:
                difference = np.abs(flat[begin:end] - expected[row].reshape(-1)[begin:end])
                maximum_difference = max(maximum_difference, float(difference.max()))
        prediction.flush()
        truth.flush()
        prediction_seconds += time.perf_counter() - tick
        close_map(prediction)
        close_map(truth)
        close_map(target)
        if expected is not None:
            close_map(expected)
    close_map(mean)
    if reference:
        assert maximum_difference == 0., maximum_difference
    prediction = np.load(output / "predictions.npy", mmap_mode="r")
    truth = np.load(output / "truth.npy", mmap_mode="r")
    metrics = physical_metrics(prediction, truth, cache.names, cache.units, cases, cache.meta["time_values"])
    close_map(prediction)
    close_map(truth)
    return metrics, prediction_seconds, maximum_difference


def self_check():
    rng = np.random.default_rng(17)
    train_globals = rng.normal(size=(5, 3))
    val_globals = rng.normal(size=(2, 3))
    train = rng.normal(size=(5, 24)).astype(np.float32)
    validation = rng.normal(size=(2, 24)).astype(np.float32)
    with tempfile.TemporaryDirectory(prefix="rbf-selfcheck-", dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        np.save(root / "train.npy", train)
        np.save(root / "validation.npy", validation)
        gram, cross, energy = field_inner_products(root / "train.npy", root / "validation.npy", 24, 7)
        kernel = rbf(train_globals, train_globals, .3) + np.eye(5) * .01
        weights = np.linalg.solve(kernel, rbf(val_globals, train_globals, .3).T).T
        explicit = np.mean((weights @ train.astype(np.float64) - validation) ** 2)
        assert np.isclose(validation_mse(weights, gram, cross, energy), explicit, rtol=1e-12)
        fitted = np.linalg.solve(kernel, train)
        assert np.allclose(weights @ train, rbf(val_globals, train_globals, .3) @ fitted)
        # Selection consumes validation inner products; no test array participates.
        assert gram.shape == (5, 5) and cross.shape == (2, 5)
        assert np.array_equal(read_block(root / "train.npy", 4, 11), train[:, 4:11])
        np.save(root / "xyz.npy", rng.normal(size=(3, 3)).astype(np.float32))
        np.save(root / "edges.npy", np.asarray([[0, 1], [1, 2]], dtype=np.int64))
        runs = {}
        for i in range(8):
            case = f"case{i}"
            np.save(root / f"{case}.npy", rng.normal(size=(3, 2, 1)).astype(np.float32))
            np.save(root / f"{case}_thickness.npy", np.ones((3, 1), dtype=np.float32) * (i + 1))
            runs[case] = {"target": f"{case}.npy", "thickness": f"{case}_thickness.npy", "globals": [i, i*i]}
        meta = {"coordinates": "xyz.npy", "edge_index": "edges.npy", "field_names": ["stress"],
                "field_units": ["GPa"], "time_values": [1., 2.], "runs": runs,
                "split": {"train": [f"case{i}" for i in range(5)], "validation": ["case5", "case6"], "test": ["case7"]}}
        write_json(root / "manifest.json", meta)
        cache = Cache(root / "manifest.json")
        stats, mean = cache.statistics()
        np.save(root / "mean.npy", mean)
        scale = np.asarray(stats["field_std"], dtype=np.float32)
        write_training_matrix(cache, cache.split["train"], root / "span.npy", root / "mean.npy", scale, 4)
        globals_ = normalized_globals(cache, cache.split["train"], stats)
        kernel = rbf(globals_, globals_, .3) + np.eye(5) * .01
        (root / "first").mkdir()
        (root / "fresh").mkdir()
        predict_test(cache, root / "first", root / "span.npy", root / "mean.npy", scale,
                     globals_, kernel, .3, stats, 4)
        _, _, difference = predict_test(cache, root / "fresh", root / "span.npy", root / "mean.npy", scale,
                                        globals_, kernel, .3, stats, 4, root / "first/predictions.npy")
        assert difference == 0.
        for case in cache.split["validation"] + cache.split["test"]:
            np.save(root / runs[case]["target"], np.full((3, 2, 1), 1e6, dtype=np.float32))
            cache.meta["runs"][case]["globals"] = [1e6, 1e6]
        stats_after, mean_after = cache.statistics()
        assert stats_after == stats and np.array_equal(mean_after, mean), "Nontraining cases affected fitted statistics"
        assert np.array_equal(globals_, normalized_globals(cache, cache.split["train"], stats_after))
        close_map(cache.xyz)
        close_map(cache.edges)
    print("RBF streamed reconstruction and validation-selection self-check passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, help="Prior predictions.npy for exact fresh-process inference comparison")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--normalization",type=Path)
    parser.add_argument("--mean-field",type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    assert args.cache and args.out and args.chunk > 0, "--cache and --out are required"
    assert args.mode == "train" or args.checkpoint, "infer mode requires --checkpoint"
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint.json", "predictions.npy", "training_centered_scaled.npy"):
        assert not (args.out / name).exists(), f"Choose a fresh output directory: {args.out / name}"
    import psutil
    process = psutil.Process()
    started = time.perf_counter()
    cache = Cache(args.cache)
    initial_manifest_sha256 = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if args.expected_manifest_sha256:
        assert initial_manifest_sha256 == args.expected_manifest_sha256
    if args.ready_file:
        ready = json.loads(args.ready_file.read_text(encoding="utf-8"))
        assert ready["status"] == "ready" and ready["writer_exit_code"] == 0
        assert ready["manifest_sha256"] == initial_manifest_sha256
        for entry in ready["files"]:
            artifact = args.ready_file.parent / entry["name"]
            assert artifact.stat().st_size == entry["bytes"]
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
        shutil.copyfile(args.ready_file,args.out/"dataset_READY_evidence.json")
    assert cache.split["train"] and cache.split["validation"] and cache.split["test"]
    dimension = int(np.prod(cache.shape))
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "source").mkdir(exist_ok=True)
    shutil.copyfile(__file__, args.out / "executed_run_baselines.py")
    shutil.copyfile(Path(__file__).with_name("benchmark_data.py"), args.out / "executed_benchmark_data.py")
    shutil.copyfile(__file__, args.out / "source/run_baselines.py")
    shutil.copyfile(Path(__file__).with_name("benchmark_data.py"), args.out / "source/benchmark_data.py")
    candidates = []
    sampled_rss = process.memory_info().rss
    selection_seconds = 0.

    if args.mode == "train":
        normal_path, shared_mean = args.normalization or cache.root/"train-normalization.json", args.mean_field or cache.root/"training_mean_field.npy"
        if normal_path.is_file() and shared_mean.is_file():
            stats = json.loads(normal_path.read_text(encoding="utf-8"))
            mean = np.load(shared_mean,mmap_mode="r")
            if args.normalization:
                verified_stats,verified_mean=cache.statistics()
                assert stats==verified_stats and np.array_equal(mean,verified_mean), "Frozen train statistics differ"
                del verified_mean
                shutil.copyfile(args.normalization,args.out/"normalization.json")
        else:
            stats, mean = cache.statistics()
        mean_path = args.out / "training_mean_field.npy"
        np.save(mean_path, mean)
        del mean
        scale = np.asarray(stats["field_std"], dtype=np.float32)
        assert np.isfinite(scale).all() and (scale > 0).all()
        train_path = args.out / "training_centered_scaled.npy"
        validation_path = args.out / "validation_centered_scaled.npy"
        write_training_matrix(cache, cache.split["train"], train_path, mean_path, scale, args.chunk)
        write_training_matrix(cache, cache.split["validation"], validation_path, mean_path, scale, args.chunk)
        xtrain = normalized_globals(cache, cache.split["train"], stats)
        xvalidation = normalized_globals(cache, cache.split["validation"], stats)
        selection_start = time.perf_counter()
        gram, cross, energy = field_inner_products(train_path, validation_path, dimension, args.chunk)
        best_score = float("inf")
        for gamma in GAMMAS:
            unregularized = rbf(xtrain, xtrain, gamma)
            validation_kernel = rbf(xvalidation, xtrain, gamma)
            for alpha in ALPHAS:
                kernel = unregularized + alpha * np.eye(len(xtrain))
                weights = np.linalg.solve(kernel, validation_kernel.T).T
                score = validation_mse(weights, gram, cross, energy)
                row = {"gamma": gamma, "alpha": alpha, "validation_normalized_mse": score}
                candidates.append(row)
                if score < best_score:
                    best_score, best = score, row.copy()
                print(json.dumps(row), flush=True)
        selection_seconds = time.perf_counter() - selection_start
        kernel = rbf(xtrain, xtrain, best["gamma"]) + best["alpha"] * np.eye(len(xtrain))
        np.savez_compressed(args.out / "kernel_checkpoint.npz", global_train_normalized=xtrain,
                            kernel_regularized=kernel)
        payload = {"architecture": "rbf_direct_field", "contract": cache.contract,
                   "normalization": stats, "model_config": {"gamma": best["gamma"], "alpha": best["alpha"]},
                   "validation_normalized_mse": best_score, "training_cases": cache.split["train"],
                   "training_matrix": train_path.name, "training_mean_field": mean_path.name,
                   "kernel_checkpoint": "kernel_checkpoint.npz", "source_sha256": source_hash,
                   "checkpoint_note": "Small JSON/NPZ parameters require the adjacent centered training-field matrix and mean-field NPY sidecars for inference."}
        write_json(args.out / "checkpoint.json", payload)
        write_json(args.out / "selection.json", {"candidates": candidates, "selected": best,
                   "selection_split": "validation", "test_used_for_selection": False,
                   "field_inner_product_dimension": dimension})
        del gram, cross, energy
    else:
        payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        assert payload["contract"] == cache.contract, "Cache mesh/field/time/split contract changed"
        assert payload["training_cases"] == cache.split["train"]
        stats = payload["normalization"]
        scale = np.asarray(stats["field_std"], dtype=np.float32)
        train_path = args.checkpoint.parent / payload["training_matrix"]
        mean_path = args.checkpoint.parent / payload["training_mean_field"]
        with np.load(args.checkpoint.parent / payload["kernel_checkpoint"]) as small:
            xtrain = small["global_train_normalized"]
            kernel = small["kernel_regularized"]
    sampled_rss = max(sampled_rss, process.memory_info().rss)
    metrics, inference_seconds, difference = predict_test(
        cache, args.out, train_path, mean_path, scale, xtrain, kernel,
        payload["model_config"]["gamma"], stats, args.chunk, args.reference)
    report = {"status": "rbf_direct_field_training_and_heldout_evaluation_completed" if args.mode == "train" else "fresh_rbf_checkpoint_inference_completed",
              "architecture": "rbf_direct_field", "mode": args.mode, "contract": cache.contract,
              "model_config": payload["model_config"], "normalization": stats,
              "validation_normalized_mse": payload["validation_normalized_mse"],
              "selection_candidate_count": len(candidates) if args.mode == "train" else None,
              "selection_seconds": selection_seconds, "metrics": {"rbf_direct_field": metrics},
              "test_inference_with_field_io_seconds": inference_seconds,
              "inference_timing_note": "Includes streamed training-field reconstruction, target-cache read and prediction/truth file writes; excludes metric calculation and initial kernel-weight solve.",
              "normalization_note": "Cache.statistics uses training cases only. Field residuals are centered on the training spatial mean field and scaled by the common per-channel training std; global parameters use training mean/std.",
              "selection_note": "Twelve RBF gamma/alpha combinations; validation equal-weight normalized MSE across nodes, time and channels. Test cases are read for final evaluation only.",
              "prediction_scope": f"Fixed node-aligned mesh and existing {len(stats['global_mean'])}-global-parameter regime; no novel topology or appliance generalization is tested.",
              "kernel_definition": "exp(-gamma * squared Euclidean distance in train-normalized globals), unit amplitude, ridge alpha, no polynomial term",
              "out_of_core_note": "Float32 disk training matrix; float64 inner products accumulated in coordinate chunks. No full dual-coefficient field matrix or candidate prediction ensemble is held in RAM.",
              "chunk_coordinates": args.chunk, "source_sha256": source_hash,
              "peak_sampled_rss_bytes": max(sampled_rss, process.memory_info().rss),
              "memory_note": "RSS sampled at setup/selection/evaluation boundaries; lower bound on process peak.",
              "total_seconds": time.perf_counter() - started}
    if args.reference:
        report["fresh_checkpoint_prediction_max_absolute_difference"] = difference
    assert hashlib.sha256(args.cache.read_bytes()).hexdigest() == initial_manifest_sha256
    report["peak_process_rss_bytes"] = int(getattr(process.memory_info(),"peak_wset",process.memory_info().rss))
    report["dataset_freeze"] = {"manifest_sha256":initial_manifest_sha256,"ready_sha256":hashlib.sha256(args.ready_file.read_bytes()).hexdigest() if args.ready_file else None}
    report["shared_normalization_sha256"] = hashlib.sha256(args.normalization.read_bytes()).hexdigest() if args.normalization else None
    report["normalization_note"] = "Shared frozen train-only field scales: displacement XYZ share sqrt(trace covariance/3); scalar fields use individual training std. Training spatial mean field removed; equal channel/node/time MSE, no labels clipped."
    report["prediction_scope"] = "Registered mesh, globals listed in common cache; initial13 coupon cases split5train/4validation/4test. No unseen topology or whole-appliance generalization tested."
    from threadpoolctl import threadpool_info
    report["threadpools"] = threadpool_info()
    assert all(item.get("num_threads",1)==1 for item in report["threadpools"])
    write_json(args.out / "metrics.json", report)
    print(json.dumps({"status": report["status"], "selected": report["model_config"],
                      "seconds": report["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
