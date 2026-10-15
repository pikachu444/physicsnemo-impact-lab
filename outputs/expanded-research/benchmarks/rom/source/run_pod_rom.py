"""Train-only snapshot POD with validation-selected RBF/ridge, streamed on CPU."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import psutil

SCRIPT = Path(__file__).resolve()
HELPER = next(path for path in [SCRIPT.parent/"benchmark_data.py", SCRIPT.parent/"executed_benchmark_data.py",
                               SCRIPT.parents[1]/"expanded-ml"/"benchmark_data.py"] if path.is_file())


def load_helper(path):
    spec=importlib.util.spec_from_file_location("rom_data_helper",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA_HELPER = load_helper(HELPER)
Cache, physical_metrics = DATA_HELPER.Cache, DATA_HELPER.physical_metrics


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_block(path, start, end, dtype=np.float64):
    # Releasing each mmap view keeps large snapshot files outside process RSS.
    value = np.load(path, mmap_mode="r")
    result = np.array(value.reshape(-1)[start:end], dtype=dtype, copy=True)
    del value
    return result


def snapshot_block(path, start, end):
    value = np.load(path, mmap_mode="r")
    result = np.array(value[:, start:end], dtype=np.float64, copy=True)
    del value
    return result


def poly_features(x, degree):
    parts = [np.ones((len(x), 1)), x]
    if degree == 2:
        parts += [(x[:, i] * x[:, j])[:, None]
                  for i, j in itertools.combinations_with_replacement(range(x.shape[1]), 2)]
    return np.concatenate(parts, axis=1)


def kernel(x, y, gamma):
    distance = np.maximum(np.sum(x*x, 1)[:, None] + np.sum(y*y, 1)[None] - 2*x@y.T, 0.)
    return np.exp(-gamma*distance)


def regress(train_x, query_x, coefficients, config):
    alpha = config["alpha"]
    if config["method"] == "pod_rbf":
        matrix = kernel(train_x, train_x, config["gamma"])
        weights = np.linalg.solve(matrix + alpha*np.eye(len(matrix)), coefficients)
        return kernel(query_x, train_x, config["gamma"])@weights, weights
    tx, qx = poly_features(train_x, config["degree"]), poly_features(query_x, config["degree"])
    penalty = np.eye(tx.shape[1])
    penalty[0, 0] = 0.
    weights = np.linalg.solve(tx.T@tx + alpha*penalty, tx.T@coefficients)
    return qx@weights, weights


def objective(coefficients, true_projection, total_norm, dimensions):
    sse = total_norm - 2*np.sum(coefficients*true_projection) + np.sum(coefficients**2)
    assert sse > -1e-7*max(1., total_norm), sse
    return max(float(sse), 0.) / dimensions


def self_check():
    rng = np.random.default_rng(20261015)
    z = rng.normal(size=(9, 31))
    z -= z.mean(0)
    gram = z@z.T
    eig, u = np.linalg.eigh(gram)
    order = np.argsort(eig)[::-1][:6]
    eig, u = eig[order], u[:, order]
    s = np.sqrt(eig)
    v = (u/s).T@z
    assert np.allclose(v@v.T, np.eye(6))
    y = rng.normal(size=(3, 31))
    c = rng.normal(size=(3, 6))
    b = y@z.T@(u/s)
    assert np.allclose(objective(c, b, np.sum(y*y), y.size), np.mean((c@v-y)**2))
    assert np.allclose((c/s)@u.T@z, c@v)
    x = rng.normal(size=(9, 2))
    assert regress(x, x[:3], z[:, :4], {"method":"pod_rbf", "gamma":.3, "alpha":.01})[0].shape == (3,4)
    assert regress(x, x[:3], z[:, :4], {"method":"pod_ridge", "degree":2, "alpha":.01})[0].shape == (3,4)


def memory(process):
    state = process.memory_info()
    return int(getattr(state, "peak_wset", state.rss))


def verify_ready(path, cache):
    ready=json.loads(path.read_text(encoding="utf-8"))
    assert ready["status"]=="ready" and ready["writer_exit_code"]==0, "Dataset writer is not successfully finished"
    assert ready["expected_cases"]==len(cache.meta["runs"]), "READY case count differs"
    if "cases" in ready:
        assert len(ready["cases"])==len(cache.meta["runs"])
        assert {case["run"] for case in ready["cases"]}==set(cache.meta["runs"]), "READY case identities differ"
    checked=0
    if ready.get("files"):
        assert sha(cache.path)==ready["manifest_sha256"], "Top-level READY manifest hash differs"
        groups=[(path.parent,ready["files"])]
    else:
        groups=[]
        for case in ready["cases"]:
            directory=path.parent/case["run"]
            assert sha(directory/"manifest.json")==case["manifest_sha256"], case["run"]
            groups.append((directory,case["files"]))
    for directory,files in groups:
        for item in files:
            artifact=directory/item["name"]
            assert artifact.stat().st_size==item["bytes"] and sha(artifact)==item["sha256"],str(artifact)
            checked+=1
    return {"case_count":len(cache.meta["runs"]),"artifact_hashes_verified":checked,"ready_sha256":sha(path),"dataset_revision":ready.get("dataset_revision")}


def normalized_cross(cache, ids, snapshot_path, mean_path, std, chunk):
    dims = int(np.prod(cache.shape))
    cross = np.zeros((len(ids), len(cache.split["train"])))
    total_norm = 0.
    for start in range(0, dims, chunk):
        end = min(start+chunk, dims)
        mu = read_block(mean_path, start, end)
        scale = std[np.arange(start, end) % len(std)]
        train = snapshot_block(snapshot_path, start, end)
        query = np.stack([(read_block(cache.root/cache.meta["runs"][case]["target"], start, end)-mu)/scale for case in ids])
        cross += query@train.T
        total_norm += float(np.sum(query*query))
    return cross, total_norm


def reconstruct(cache, ids, sample_weights, snapshot_path, mean_path, std, out, chunk):
    dims = int(np.prod(cache.shape))
    target = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(ids), *cache.shape))
    del target
    for start in range(0, dims, chunk):
        end = min(start+chunk, dims)
        train = snapshot_block(snapshot_path, start, end)
        scale = std[np.arange(start, end) % len(std)]
        physical = (sample_weights@train)*scale + read_block(mean_path, start, end)
        target = np.load(out, mmap_mode="r+")
        target.reshape(len(ids), -1)[:, start:end] = physical.astype(np.float32)
        target.flush()
        del target


def prediction_from_checkpoint(cache, case_ids, directory, out_path, chunk):
    meta = json.loads((directory/"checkpoint.json").read_text(encoding="utf-8"))
    assert meta["contract"] == cache.contract, "Mesh/fields/units/times/split contract differs"
    assert meta["cache_manifest_sha256"] == sha(cache.path), "Cache manifest changed"
    assert sha(meta["snapshots"]) == meta["snapshot_sha256"], "Snapshot values changed"
    if "mean_field_sha256" in meta:
        assert sha(meta["mean_field"]) == meta["mean_field_sha256"], "Training mean field changed"
    weights = np.load(directory/"checkpoint.npz")
    queries = np.stack([cache.inputs(case)[1] for case in case_ids])
    qx = ((queries-meta["global_mean"])/meta["global_std"])[:, meta["active_global_indices"]]
    train_x = weights["train_x"]
    config = meta["selected"]
    if config["method"] == "pod_rbf":
        c = kernel(qx, train_x, config["gamma"])@weights["regression_weights"]
    else:
        c = poly_features(qx, config["degree"])@weights["regression_weights"]
    sample_weights = (c/weights["singular_values"])@weights["left_vectors"].T
    reconstruct(cache, case_ids, sample_weights, Path(meta["snapshots"]), Path(meta["mean_field"]),
                np.asarray(meta["field_std"]), out_path, chunk)


def plot_results(out, cache, names, units, predictions, truth, metrics, energy, candidates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1,len(energy)+1), energy)
    axes[0].set(xlabel="POD rank", ylabel="Cumulative centered snapshot variance", ylim=(0,1.01))
    for method in ["pod_rbf", "pod_ridge"]:
        points = {}
        for row in candidates:
            if row["method"] == method:
                points[row["rank"]] = min(points.get(row["rank"], float("inf")), row["validation_normalized_mse"])
        axes[1].plot(list(points), list(points.values()), "o-", label=method)
    axes[1].set(xlabel="POD rank", ylabel="Best validation normalized MSE", yscale="log")
    axes[1].legend()
    fig.tight_layout();fig.savefig(out/"rank_selection.png", dpi=150);plt.close(fig)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, max(4,2*len(names))), squeeze=False)
    for j,(name,unit) in enumerate(zip(names,units)):
        for method in predictions:
            axes[j,0].plot(cache.meta["time_values"], metrics[method][name]["per_timestep_rmse"], label=method)
        axes[j,0].set(ylabel=f"{name}\nRMSE [{unit}]")
    axes[0,0].legend();axes[-1,0].set_xlabel("Time ["+cache.meta.get("time_unit","unspecified")+"]")
    fig.tight_layout();fig.savefig(out/"field_rmse.png", dpi=150);plt.close(fig)
    chosen="pod_rbf"
    field = next((i for i,n in enumerate(names) if "vm" in n.lower()), 0)
    a,b = truth[0,:,-1,field], predictions[chosen][0,:,-1,field]
    lo,hi=float(min(a.min(),b.min())),float(max(a.max(),b.max()))
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    for ax,values,title in zip(axes,[a,b],["FEM reference",chosen]):
        im=ax.scatter(cache.xyz[:,0],cache.xyz[:,1],c=values,s=3,vmin=lo,vmax=hi)
        ax.set(title=title,xlabel="Reference X [mm]",ylabel="Reference Y [mm]")
        ax.set_aspect("equal")
    fig.colorbar(im,ax=axes.ravel().tolist(),label=f"{names[field]} [{units[field]}]")
    fig.suptitle("Reference-coordinate field map; "+cache.split["test"][0]+", final stored time")
    fig.savefig(out/"field_comparison.png",dpi=150,bbox_inches="tight");plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--work",type=Path)
    parser.add_argument("--chunk-features",type=int,default=4096)
    parser.add_argument("--reuse",type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--ready-file",type=Path)
    parser.add_argument("--normalization",type=Path)
    parser.add_argument("--mean-field",type=Path)
    parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    self_check()
    if args.self_check:
        print("POD Gram/reconstruction/selection objective and regressor self-checks passed",flush=True)
        return
    assert args.cache and args.out and args.chunk_features>0
    if not args.reuse:
        assert args.work, "Training needs a separate snapshot work directory"
        for existing in [args.out/"metrics.json",args.out/"pod_rbf/checkpoint.json",args.work/"normalized_centered_training_snapshots.npy"]:
            assert not existing.exists(), "Choose fresh output/work directories; preserving previous evidence: "+str(existing)
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();process=psutil.Process()
    frozen_helper = args.reuse.parent/"executed_benchmark_data.py" if args.reuse else HELPER
    if not frozen_helper.is_file():frozen_helper=HELPER
    cache=load_helper(frozen_helper).Cache(args.cache)
    initial_manifest_sha = sha(cache.path)
    if args.expected_manifest_sha256:
        assert initial_manifest_sha == args.expected_manifest_sha256, "Manifest is not the frozen expected revision"
    if args.ready_file:
        assert args.ready_file.is_file(), "Dataset READY evidence missing"
        freeze_verification = verify_ready(args.ready_file,cache)
        write_json(args.out/"dataset_freeze_verification.json",freeze_verification)
        shutil.copyfile(args.ready_file,args.out/"dataset_READY_evidence.json")
    if args.reuse:
        prediction_from_checkpoint(cache,cache.split["test"],args.reuse,args.out/"fresh_predictions.npy",args.chunk_features)
        original=np.load(args.reuse/"predictions.npy",mmap_mode="r")
        fresh=np.load(args.out/"fresh_predictions.npy",mmap_mode="r")
        difference=float(np.max(np.abs(original-fresh)))
        assert difference==0.,difference
        write_json(args.out/"inference_report.json",{"status":"fresh_checkpoint_reload_exact_match","maximum_absolute_difference":difference,"case_ids":cache.split["test"],"elapsed_seconds":time.perf_counter()-started,"peak_process_rss_bytes":memory(process),"gpu_used":False,"data_helper_sha256":sha(frozen_helper)})
        print("Fresh checkpoint inference matches saved predictions exactly",flush=True)
        return
    assert args.work
    args.work.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,args.out/"executed_run_pod_rom.py")
    shutil.copyfile(HELPER,args.out/"executed_benchmark_data.py")
    executed_source = {Path(__file__).name:sha(args.out/"executed_run_pod_rom.py"), HELPER.name:sha(args.out/"executed_benchmark_data.py")}
    normal_path=args.normalization or cache.root/"train-normalization.json"
    mean_path=args.mean_field or cache.root/"training_mean_field.npy"
    if normal_path.exists() and mean_path.exists():
        stats=json.loads(normal_path.read_text(encoding="utf-8"))
        if args.normalization:
            verified_stats,verified_mean=cache.statistics()
            assert stats==verified_stats, "Frozen statistics differ from train-only recomputation"
            assert np.array_equal(verified_mean,np.load(mean_path,mmap_mode="r")), "Frozen mean field differs from training mean"
            del verified_mean
        shutil.copyfile(mean_path,args.work/"training_mean_field.npy")
    else:
        stats,mean=cache.statistics()
        np.save(args.work/"training_mean_field.npy",mean)
        del mean
    mean_path=args.work/"training_mean_field.npy"
    write_json(args.out/"normalization.json",stats)
    field_std=np.asarray(stats["field_std"])
    dims=int(np.prod(cache.shape));train_ids=cache.split["train"];val_ids=cache.split["val"] if "val" in cache.split else cache.split["validation"]
    n=len(train_ids)
    snapshot_path=args.work/"normalized_centered_training_snapshots.npy"
    value=np.lib.format.open_memmap(snapshot_path,mode="w+",dtype=np.float32,shape=(n,dims));del value
    mean=np.load(mean_path,mmap_mode="r")
    for i,case in enumerate(train_ids):
        y=cache.target(case)
        centered=((y.astype(np.float64)-mean)/field_std).astype(np.float32)
        value=np.load(snapshot_path,mmap_mode="r+");value[i]=centered.reshape(-1);value.flush()
        del value,y,centered
        if (i+1)%20==0 or i==0:print(f"Prepared train-only POD snapshot {i+1}/{n}",flush=True)
    del mean
    snapshot_seconds=time.perf_counter()-started
    tick=time.perf_counter();gram=np.zeros((n,n))
    for start in range(0,dims,args.chunk_features):
        end=min(start+args.chunk_features,dims)
        z=snapshot_block(snapshot_path,start,end)
        gram+=z@z.T
    eig,u=np.linalg.eigh(gram);order=np.argsort(eig)[::-1];eig,u=eig[order],u[:,order]
    assert eig[0]>1e-12, "No case-to-case training response variation; inspect FEM labels before fitting a parametric POD"
    positive=eig>max(eig[0]*1e-12,1e-12);eig,u=eig[positive],u[:,positive]
    s=np.sqrt(eig);energy=np.cumsum(eig)/eig.sum()
    assert np.allclose(u.T@u,np.eye(len(eig)),atol=1e-8)
    assert np.isclose(eig.sum(),np.trace(gram),rtol=1e-8)
    thresholds=[.90,.95,.99,.999,.9999]
    ranks=sorted(set(min(int(np.searchsorted(energy,t))+1,len(eig)) for t in thresholds))
    coefficients=u*s
    gram_seconds=time.perf_counter()-tick
    print(json.dumps({"positive_modes":len(eig),"candidate_ranks":ranks,"gram_seconds":gram_seconds,"peak_rss":memory(process)}),flush=True)
    tick=time.perf_counter()
    cross,valnorm=normalized_cross(cache,val_ids,snapshot_path,mean_path,field_std,args.chunk_features)
    true_projection=cross@(u/s)
    raw_x=np.stack([cache.inputs(case)[1] for case in train_ids])
    active=np.flatnonzero(raw_x.std(0)>1e-12)
    gm,gs=np.asarray(stats["global_mean"]),np.asarray(stats["global_std"])
    train_x=((raw_x-gm)/gs)[:,active]
    val_x=((np.stack([cache.inputs(case)[1] for case in val_ids])-gm)/gs)[:,active]
    candidates=[]
    for rank in ranks:
        configs=[{"method":"pod_rbf","gamma":g,"alpha":a} for g in [.1,.3,1.,3.] for a in [1e-4,.01,.1]]
        configs += [{"method":"pod_ridge","degree":d,"alpha":a} for d in [1,2] for a in [1e-4,.01,.1]]
        for cfg in configs:
            prediction,_=regress(train_x,val_x,coefficients[:,:rank],cfg)
            score=objective(prediction,true_projection[:,:rank],valnorm,len(val_ids)*dims)
            candidates.append({**cfg,"rank":rank,"training_energy_retained":float(energy[rank-1]),"validation_normalized_mse":score,"validation_projection_lower_bound_mse":float(max(valnorm-np.sum(true_projection[:,:rank]**2),0)/(len(val_ids)*dims))})
    selected={method:min((r for r in candidates if r["method"]==method),key=lambda r:r["validation_normalized_mse"]) for method in ["pod_rbf","pod_ridge"]}
    selection_seconds=time.perf_counter()-tick
    write_json(args.out/"validation_candidates.json",candidates)
    ids=cache.split["test"]
    truth=np.lib.format.open_memmap(args.work/"test_truth.npy",mode="w+",dtype=np.float32,shape=(len(ids),*cache.shape))
    for i,case in enumerate(ids):truth[i]=cache.target(case)
    truth.flush();del truth
    truth=np.load(args.work/"test_truth.npy",mmap_mode="r")
    tx=((np.stack([cache.inputs(case)[1] for case in ids])-gm)/gs)[:,active]
    predictions={};metrics={};inference={}
    for method,cfg in selected.items():
        destination=args.out/method;destination.mkdir(exist_ok=True)
        rank=cfg["rank"]
        tick=time.perf_counter();c,weights=regress(train_x,tx,coefficients[:,:rank],cfg)
        sample_weights=(c/s[:rank])@u[:,:rank].T
        path=destination/"predictions.npy"
        reconstruct(cache,ids,sample_weights,snapshot_path,mean_path,field_std,path,args.chunk_features)
        inference[method]=time.perf_counter()-tick
        predictions[method]=np.load(path,mmap_mode="r")
        metrics[method]=physical_metrics(predictions[method],truth,cache.names,cache.units,ids,cache.meta["time_values"])
        metrics[method]["normalized_mse"]=float(np.mean(((predictions[method].astype(np.float64)-truth)/field_std)**2))
        np.savez(destination/"checkpoint.npz",left_vectors=u[:,:rank],singular_values=s[:rank],regression_weights=weights,train_x=train_x,all_eigenvalues=eig)
        write_json(destination/"checkpoint.json",{"selected":cfg,"contract":cache.contract,"cache_manifest_sha256":sha(cache.path),"snapshot_sha256":sha(snapshot_path),"snapshots":str(snapshot_path.resolve()),"mean_field":str(mean_path.resolve()),"mean_field_sha256":sha(mean_path),"field_std":field_std.tolist(),"global_mean":gm.tolist(),"global_std":gs.tolist(),"active_global_indices":active.tolist(),"note":"Dual POD basis reconstructs from local training snapshots; checkpoint depends on this local cache. Basis and source FEM-derived arrays are not cleared for redistribution."})
        np.savez_compressed(destination/"heldout_predictions.npz",prediction=predictions[method],truth=truth,case_ids=np.asarray(ids),field_names=np.asarray(cache.names),field_units=np.asarray(cache.units),time_values=np.asarray(cache.meta["time_values"]))
    plot_results(args.out,cache,cache.names,cache.units,predictions,truth,metrics,energy,candidates)
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
        assert all(p.get("num_threads",1)==1 for p in pools),pools
    except ImportError:
        pools="threadpoolctl unavailable; thread environment set to1 before NumPy import"
    report={"status":"actual_POD_fit_validation_selection_and_heldout_evaluation_completed","data_manifest":str(cache.path.resolve()),"cache_manifest_sha256":sha(cache.path),"source_sha256":executed_source,"snapshot_sha256":sha(snapshot_path),"seed":20261015,"determinism":"linear algebra deterministic; seed only used by assert self-check, no randomized POD","gpu_used":False,"python":platform.python_version(),"numpy":np.__version__,"threadpools":pools,"contract":cache.contract,"normalization":stats,"snapshot_dimensions":[n,dims],"positive_modes":len(eig),"candidate_energy_thresholds":thresholds,"candidate_ranks":ranks,"cumulative_training_energy":energy.tolist(),"pod_energy_definition":"Fraction of train-only case-to-case snapshot variance after per-channel standardization and subtraction of the training mean field; not physical strain/kinetic energy or fraction of uncentered raw signal.","selected_by_validation_only":selected,"test_metrics":metrics,"timing_seconds":{"snapshot_preparation":snapshot_seconds,"gram_eigendecomposition":gram_seconds,"validation_projection_and_selection":selection_seconds,"heldout_prediction_including_streamed_reconstruction":inference,"total_including_io_plots":time.perf_counter()-started},"peak_process_rss_bytes":memory(process),"host_available_memory_bytes_at_end":psutil.virtual_memory().available,"input_information":"Per-run global parameters only, normalized with train statistics; constant inputs dropped. Fixed geometry/reference coordinates are encoded by POD basis. No heldout response is a predictor input.","loss":"Mean squared error of physical field residual divided by train-only channel std, equally weighted per node/time/channel. Spatial temporal mean field is also learned on train split only.","scope":"Exploratory public benchmark; same fixed geometry, heldout GPU-model split cases previously exist in public FEM results; no blindtest or appliance generalization claim.","local_data_warning":"Prediction NPZ/snapshots/mean field contain FEM-derived data; local evidence, excluded from public redistribution pending license handling."}
    assert sha(cache.path) == initial_manifest_sha, "Cache manifest changed during POD fit/evaluation"
    report["loss"] = "Mean squared residual divided by the shared train-only field scales. XYZ use a common rotation-invariant vector scale when specified by normalization metadata; scalar fields retain their train scales. Equal node/time/channel weights; mean field uses train cases only. No response values replaced or clipped."
    report["active_input_keys"] = [cache.meta.get("global_keys",list(range(raw_x.shape[1])))[i] for i in active]
    report["training_global_min"] = raw_x.min(0).tolist()
    report["training_global_max"] = raw_x.max(0).tolist()
    report["dataset_freeze"] = {"expected_manifest_sha256":args.expected_manifest_sha256,"initial_and_final_manifest_sha256":initial_manifest_sha,"ready_evidence_sha256":sha(args.ready_file) if args.ready_file else None}
    report["shared_normalization_sha256"] = sha(args.normalization) if args.normalization else None
    report["scope"] = "Exploratory structural FEM benchmark with provenance and case split in the frozen cache. No test-driven tuning, independent blind certification, new topology or appliance generalization claimed."
    write_json(args.out/"metrics.json",report)
    print(json.dumps({"selected":selected,"test_normalized_mse":{k:v["normalized_mse"] for k,v in metrics.items()},"seconds":report["timing_seconds"],"peak_rss":report["peak_process_rss_bytes"]}),flush=True)


if __name__=="__main__":main()
