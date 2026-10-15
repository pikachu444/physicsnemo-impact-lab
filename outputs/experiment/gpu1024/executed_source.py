"""Reproducible reduced CPU PhysicsNeMo trial on public Radioss bumper FEM.

This is a new lightweight driver, not a claim to reproduce the official recipe.
The official validation runs remain untouched until the final evaluation.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/CURATED_DATA_VTP")
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/experiment")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--nodes", type=int, default=1024)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
execution_source = Path(__file__).read_bytes()
execution_source_sha256 = hashlib.sha256(execution_source).hexdigest()
(args.out/"executed_source.py").write_bytes(execution_source)
SEED = 20261015
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
process = psutil.Process()
device = torch.device(args.device)
if device.type == "cuda":
    assert torch.cuda.is_available(), "CUDA was requested but is unavailable"
    torch.cuda.reset_peak_memory_stats()
peak_rss = process.memory_info().rss
start = time.perf_counter()

def read_vtp(path):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    point_data = poly.GetPointData()
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32).copy()
    thickness = vtk_to_numpy(point_data.GetArray("thickness")).astype(np.float32).copy()
    fields = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    times = sorted(float(name.split("_t")[1]) for name in fields if name.startswith("displacement_t"))
    disp = np.stack([vtk_to_numpy(point_data.GetArray(f"displacement_t{t:.3f}")) for t in times], axis=1).astype(np.float32)
    assert disp.shape == (len(xyz), len(times), 3) and np.isfinite(disp).all()
    return xyz, thickness, disp, times, fields

paths = sorted((args.data / "TRAINING_DATA").glob("*.vtp"))
test_paths = sorted((args.data / "VALIDATION_DATA").glob("*.vtp"))
assert len(paths) > 20 and len(test_paths) > 1
xyz0, thickness0, disp0, times, fields = read_vtp(paths[0])
eligible = np.flatnonzero(thickness0 > 0)
indices = eligible[np.linspace(0, len(eligible) - 1, min(args.nodes, len(eligible)), dtype=int)]
assert len(indices) == len(np.unique(indices))
features = json.loads((args.data / "GLOBAL_FEATURES.json").read_text(encoding="utf-8"))
global_keys = ["thickness_scale", "velocity_x", "rwall_origin_y"]

def load(paths_to_read):
    xx, yy, gg, coordinates = [], [], [], []
    for path in paths_to_read:
        xyz, thickness, disp, current_times, current_fields = read_vtp(path)
        assert current_times[:len(times)] == times and xyz.shape == xyz0.shape
        assert np.allclose(xyz[indices], xyz0[indices], atol=1e-4), f"Unaligned deformable mesh: {path}"
        disp = disp[:, :len(times)]
        assert (thickness[indices] > 0).all()
        g = np.array([features[path.stem][key] for key in global_keys], dtype=np.float32)
        x = np.column_stack((thickness[indices], np.broadcast_to(g, (len(indices), 3))))
        xx.append(x)
        yy.append(disp[indices, 1:, :].reshape(len(indices), -1))
        gg.append(g)
        coordinates.append(xyz[indices])
    return np.stack(xx), np.stack(yy), np.stack(gg), np.stack(coordinates)

x_all, y_all, g_all, xyz_all = load(paths)
x_test, y_test, g_test, xyz_test = load(test_paths)
order = np.random.default_rng(SEED).permutation(len(paths))
val_ids, train_ids = order[:12], order[12:]
x_train, y_train, g_train = x_all[train_ids], y_all[train_ids], g_all[train_ids]
x_val, y_val, g_val = x_all[val_ids], y_all[val_ids], g_all[val_ids]
xyz_train, xyz_val = xyz_all[train_ids], xyz_all[val_ids]
stats = {
    "x_mean": x_train.mean((0, 1)), "x_std": np.maximum(x_train.std((0, 1)), 1e-6),
    "y_mean": y_train.mean((0, 1)), "y_std": np.maximum(y_train.std((0, 1)), 1.0),
    "xyz_mean": xyz_train.mean((0, 1)), "xyz_std": np.maximum(xyz_train.std((0, 1)), 1.0),
}
def tensor(array):
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device)
def normalize(x, y, xyz):
    return (tensor((x-stats["x_mean"])/stats["x_std"]),
            tensor((y-stats["y_mean"])/stats["y_std"]),
            tensor((xyz-stats["xyz_mean"])/stats["xyz_std"]))
xt, yt, pt = normalize(x_train, y_train, xyz_train)
xv, yv, pv = normalize(x_val, y_val, xyz_val)
xe, ye, pe = normalize(x_test, y_test, xyz_test)
print(f"FEM data loaded: train={len(train_ids)} val={len(val_ids)} test={len(test_paths)} nodes={len(indices)}/{len(xyz0)} times={times}", flush=True)
load_seconds = time.perf_counter() - start

from physicsnemo.models.transolver import Transolver
model_config = dict(functional_dim=4, out_dim=y_train.shape[-1], embedding_dim=3,
                    n_layers=2, n_hidden=48, n_head=4, slice_num=8,
                    use_te=False, mlp_ratio=2)
model = Transolver(**model_config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
print(f"PhysicsNeMo model instantiated: {sum(p.numel() for p in model.parameters())} parameters; device={device}", flush=True)
best_val = float("inf")
history = []
training_start = time.perf_counter()
def checkpoint_payload(state):
    return {"model_state_dict":state,"model_config":model_config,
            "normalization":{k:torch.from_numpy(v) for k,v in stats.items()},
            "node_indices":torch.from_numpy(indices),"times":times,
            "best_epoch":best_epoch,"best_validation_normalized_mse":best_val}

for epoch in range(args.epochs):
    epoch_start = time.perf_counter()
    model.train()
    total = 0.0
    for item in torch.randperm(len(xt)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(xt[item:item+1], pt[item:item+1])
        loss = torch.mean((pred - yt[item:item+1]) ** 2)
        assert torch.isfinite(loss), "Nonfinite training loss"
        loss.backward()
        optimizer.step()
        total += loss.detach().item()
    scheduler.step()
    model.eval()
    with torch.inference_mode():
        val_loss = np.mean([torch.mean((model(xv[i:i+1], pv[i:i+1])-yv[i:i+1])**2).item() for i in range(len(xv))])
    if val_loss < best_val:
        best_val = float(val_loss)
        best_epoch = epoch + 1
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(checkpoint_payload(best_state), args.out/"transolver_best_checkpoint.pt")
    row = {"epoch": epoch+1, "training_normalized_mse": total/len(xt), "validation_normalized_mse": val_loss,
           "seconds": time.perf_counter()-epoch_start}
    history.append(row)
    with (args.out/"training_history.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    peak_rss = max(peak_rss, process.memory_info().rss)
    if epoch == 0 or (epoch+1) % 5 == 0:
        print(json.dumps(row), flush=True)
training_seconds = time.perf_counter()-training_start
model.load_state_dict(best_state)
model.eval()
with torch.inference_mode():
    for _ in range(3):
        model(xe[0:1], pe[0:1])
    if device.type == "cuda":torch.cuda.synchronize()
    inference_start = time.perf_counter()
    predictions = torch.cat([model(xe[i:i+1], pe[i:i+1]) for i in range(len(xe))])
    if device.type == "cuda":torch.cuda.synchronize()
    inference_seconds = time.perf_counter()-inference_start
    pred_test = predictions.cpu().numpy()*stats["y_std"]+stats["y_mean"]
mean_test = np.broadcast_to(y_train.mean(0), y_test.shape)

# Kernel regression is a separate non-PhysicsNeMo field surrogate comparator.
from sklearn.kernel_ridge import KernelRidge
gm, gs = g_train.mean(0), np.maximum(g_train.std(0), 1e-6)
gtr, gva, gte = (g_train-gm)/gs, (g_val-gm)/gs, (g_test-gm)/gs
kernel_start = time.perf_counter()
best_kernel_val = float("inf")
for gamma in (0.1, 0.3, 1.0, 3.0):
    for alpha in (1e-4, 1e-2, 0.1):
        kernel = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
        kernel.fit(gtr, (y_train-y_train.mean(0)).reshape(len(y_train), -1))
        vp = kernel.predict(gva).reshape(y_val.shape)+y_train.mean(0)
        score = float(np.mean((vp-y_val)**2))
        if score < best_kernel_val:
            best_kernel_val, best_kernel = score, kernel
            best_kernel_config = {"gamma":gamma,"alpha":alpha}
kernel_pred = best_kernel.predict(gte).reshape(y_test.shape)+y_train.mean(0)
kernel_seconds = time.perf_counter()-kernel_start

def metrics(prediction, target):
    p = prediction.reshape(len(prediction), len(indices), -1, 3)
    t = target.reshape(p.shape)
    squared = (p-t).astype(np.float64)**2
    den = np.sum(t.astype(np.float64)**2)
    return {"displacement_relative_l2": float(np.sqrt(squared.sum()/den)),
            "displacement_rmse_native_units": float(np.sqrt(squared.mean())),
            "per_timestep_relative_l2": np.sqrt(squared.sum((0,1,3))/np.sum(t.astype(np.float64)**2,(0,1,3))).tolist(),
            "per_timestep_rmse_native_units": np.sqrt(squared.mean((0,1,3))).tolist(),
            "per_run_relative_l2": np.sqrt(squared.sum((1,2,3))/np.sum(t.astype(np.float64)**2,(1,2,3))).tolist()}
results = {"physicsnemo_transolver":metrics(pred_test,y_test), "zero_displacement":metrics(np.zeros_like(y_test),y_test),
           "training_mean_field":metrics(mean_test,y_test), "rbf_kernel_field":metrics(kernel_pred,y_test)}
report = {
    "status":f"reduced_{device.type}_public_fem_training_and_heldout_evaluation_completed",
    "not_claimed":["official full crash recipe reproduction", "FEM solver rerun", "real appliance drop prediction", "stress/plasticity/fracture prediction"],
    "seed":SEED, "physicsnemo_version":importlib.metadata.version("nvidia-physicsnemo"),
    "execution_source_sha256":execution_source_sha256,
    "torch_version":torch.__version__, "python_version":platform.python_version(), "device":str(device), "threads":4,
    "model_config":model_config,"parameters":sum(p.numel() for p in model.parameters()),
    "epochs":args.epochs,"best_epoch":best_epoch,"best_validation_normalized_mse":best_val,
    "full_nodes":len(xyz0),"deformable_nodes":len(eligible),"sampled_nodes":len(indices),
    "time_values_as_stored":times,"unit_note":"VTP has no embedded units. Public Run100 OpenRadioss starter /BEGIN explicitly declares kg, mm, ms; these source model units are used for this dataset.",
    "units":{"displacement":"mm","coordinates":"mm","time":"ms","velocity":"mm/ms"},
    "license_note":"Public source model headers declare Altair CC BY-NC 4.0. Download source from its publisher; no source FEM data redistribution in this code package.",
    "common_time_window":"0..100 as stored; training Run194 and Run203 also contain t=110, excluded consistently to use the common time window.",
    "point_data_fields":fields,"global_feature_keys":global_keys,
    "split":{"train":[paths[i].stem for i in train_ids],"validation":[paths[i].stem for i in val_ids],"test":[p.stem for p in test_paths]},
    "normalization":"Training runs only; internal validation chooses checkpoint/hyperparameters; official VALIDATION_DATA used as final heldout test.",
    "load_seconds":load_seconds,"training_seconds":training_seconds,"test_inference_seconds":inference_seconds,
    "test_inference_seconds_per_run":inference_seconds/len(xe),"peak_sampled_rss_bytes":peak_rss,
    "memory_note":"RSS sampled at initialization and epoch boundaries; maximum is a lower bound on the true process peak.",
    "inference_timing_note":"Forward calls only after 3 warmup calls, excluding data loading, output transfer and disk I/O; CUDA synchronized if used.",
    "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
    "kernel_scope":"Fixed, verified node-aligned mesh; global design parameters to sampled displacement field. Does not test new topology or arbitrary geometry.",
    "kernel_selection_and_fit_seconds":kernel_seconds,"kernel_selected":best_kernel_config,
    "metrics":results,
}
(args.out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
with (args.out/"training_history.csv").open("w",newline="",encoding="utf-8") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
torch.save({**checkpoint_payload(best_state),"split":report["split"]}, args.out/f"transolver_{device.type}_checkpoint.pt")
np.savez_compressed(args.out/"heldout_predictions.npz",node_indices=indices,coords=xyz_test,truth=y_test,transolver=pred_test,mean=mean_test,kernel=kernel_pred,time=times[1:])
np.savez_compressed(args.out/"kernel_checkpoint.npz",X_fit=best_kernel.X_fit_,dual_coef=best_kernel.dual_coef_,g_mean=gm,g_std=gs,y_mean=y_train.mean(0),**best_kernel_config)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot([r["epoch"] for r in history],[r["training_normalized_mse"] for r in history],label="train")
ax[0].plot([r["epoch"] for r in history],[r["validation_normalized_mse"] for r in history],label="internal validation")
ax[0].set(xlabel="Epoch",ylabel="Normalized MSE",yscale="log");ax[0].legend()
for name,value in results.items():ax[1].plot(times[1:],value["per_timestep_relative_l2"],label=name)
ax[1].set(xlabel="Time (ms)",ylabel="Heldout displacement relative L2");ax[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(args.out/"learning_and_error.png",dpi=160);plt.close(fig)
truth3=y_test.reshape(len(y_test),len(indices),-1,3)[0,:,-1]
pred3=pred_test.reshape(len(y_test),len(indices),-1,3)[0,:,-1]
fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
values=[np.linalg.norm(truth3,axis=-1),np.linalg.norm(pred3,axis=-1),np.linalg.norm(pred3-truth3,axis=-1)]
for ax,title,color in zip(axes,["FEM displacement magnitude","Transolver prediction","Vector error magnitude"],values):
    scatter=ax.scatter(xyz_test[0,:,1],xyz_test[0,:,0],c=color,s=7,cmap="viridis",vmin=0,vmax=max(values[0].max(),values[1].max()) if title!="Vector error magnitude" else values[2].max())
    ax.set(title=title,xlabel="Reference Y (mm)",ylabel="Reference X (mm)");fig.colorbar(scatter,ax=ax,label="mm")
fig.suptitle(f"Heldout {test_paths[0].stem}, source t={times[-1]}; {len(indices)} sampled deformable nodes")
fig.tight_layout();fig.savefig(args.out/"fem_vs_prediction.png",dpi=160);plt.close(fig)
print(json.dumps({"best_epoch":best_epoch,"training_seconds":training_seconds,"metrics":results},indent=2),flush=True)
