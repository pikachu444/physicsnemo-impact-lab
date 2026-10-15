"""Probe actual training-step budget and full-mesh inference after sampled training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
import torch
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--data",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
source=Path(__file__).read_bytes()
(args.out/"executed_probe.py").write_bytes(source)
assert torch.cuda.is_available()
torch.set_num_threads(4)
checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
stats={k:v.numpy() for k,v in checkpoint["normalization"].items()}
globals_data=json.loads((args.data/"GLOBAL_FEATURES.json").read_text(encoding="utf-8"))
keys=["thickness_scale","velocity_x","rwall_origin_y"]

def read(name,split):
    reader=vtkXMLPolyDataReader();reader.SetFileName(str(args.data/split/(name+".vtp")));reader.Update()
    poly=reader.GetOutput();pd=poly.GetPointData()
    xyz=vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32).copy()
    thickness=vtk_to_numpy(pd.GetArray("thickness")).astype(np.float32).copy()
    target=np.stack([vtk_to_numpy(pd.GetArray(f"displacement_t{t:.3f}")) for t in checkpoint["times"][1:]],axis=1).astype(np.float32)
    return xyz,thickness,target

xyz0,thickness0,target0=read("Run100","TRAINING_DATA")
indices=np.flatnonzero(thickness0>0)

def tensors(name,xyz,thickness,target,ids):
    g=np.array([globals_data[name][k] for k in keys],dtype=np.float32)
    x=np.column_stack((thickness[ids],np.broadcast_to(g,(len(ids),3))))
    y=target[ids].reshape(len(ids),-1)
    x=torch.from_numpy((x-stats["x_mean"])/stats["x_std"]).unsqueeze(0).cuda()
    p=torch.from_numpy((xyz[ids]-stats["xyz_mean"])/stats["xyz_std"]).unsqueeze(0).cuda()
    y=torch.from_numpy((y-stats["y_mean"])/stats["y_std"]).unsqueeze(0).cuda()
    return x,p,y

from physicsnemo.models.transolver import Transolver
model=Transolver(**checkpoint["model_config"]).cuda()
model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
roundtrip_run=checkpoint["split"]["test"][0]
rx,rt,ry=read(roundtrip_run,"VALIDATION_DATA")
roundtrip_ids=checkpoint["node_indices"].numpy()
xx,pp,yy=tensors(roundtrip_run,rx,rt,ry,roundtrip_ids)
with torch.inference_mode():
    reloaded_prediction=model(xx,pp).squeeze(0).cpu().numpy()*stats["y_std"]+stats["y_mean"]
with np.load(args.checkpoint.with_name("heldout_predictions.npz")) as saved:
    expected=saved["transolver"][0]
    assert expected.shape==reloaded_prediction.shape
    max_difference=float(np.max(np.abs(reloaded_prediction-expected)))
    assert np.allclose(reloaded_prediction,expected,rtol=1e-5,atol=1e-4),max_difference
roundtrip={"passed":True,"run":roundtrip_run,"max_absolute_difference_mm":max_difference,"rtol":1e-5,"atol_mm":1e-4,
           "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
(args.out/"checkpoint_roundtrip.json").write_text(json.dumps(roundtrip,indent=2),encoding="utf-8")
print("Checkpoint roundtrip "+json.dumps(roundtrip),flush=True)
del xx,pp,yy,reloaded_prediction
probes=[]
for n in [1024,4096,len(indices)]:
    ids=indices[np.linspace(0,len(indices)-1,n,dtype=int)]
    x,p,y=tensors("Run100",xyz0,thickness0,target0,ids)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    def step():
        optimizer.zero_grad(set_to_none=True)
        prediction=model(x,p)
        loss=((prediction-y)**2).mean();assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        return loss.detach().item()
    step();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    durations=[]
    for _ in range(3):
        torch.cuda.synchronize();started=time.perf_counter();loss=step();torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
    result={"nodes":n,"mean_train_step_seconds":float(np.mean(durations)),"train_step_seconds":durations,
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"sampled_process_rss_bytes":psutil.Process().memory_info().rss,"last_loss":loss}
    probes.append(result);print(json.dumps(result),flush=True)
    del optimizer,x,p,y
    torch.cuda.empty_cache()

model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
mean=np.zeros((len(indices),len(checkpoint["times"])-1,3),dtype=np.float64)
for name in checkpoint["split"]["train"]:
    xyz,thickness,target=read(name,"TRAINING_DATA")
    assert np.allclose(xyz[indices],xyz0[indices],atol=1e-4)
    mean+=target[indices]
mean/=len(checkpoint["split"]["train"])
sse=den=mean_sse=0.0
per_run=[]
for name in checkpoint["split"]["test"]:
    xyz,thickness,target=read(name,"VALIDATION_DATA")
    assert np.allclose(xyz[indices],xyz0[indices],atol=1e-4)
    x,p,y=tensors(name,xyz,thickness,target,indices)
    with torch.inference_mode():
        model(x,p);torch.cuda.synchronize();started=time.perf_counter();prediction=model(x,p);torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        prediction=prediction.squeeze(0).cpu().numpy()*stats["y_std"]+stats["y_mean"]
    prediction=prediction.reshape(target[indices].shape)
    truth=target[indices].astype(np.float64)
    run_sse=float(np.sum((prediction-truth)**2));run_den=float(np.sum(truth**2))
    sse+=run_sse;den+=run_den;mean_sse+=float(np.sum((mean-truth)**2))
    per_run.append({"run":name,"displacement_relative_l2":float(np.sqrt(run_sse/run_den)),"inference_seconds":seconds})
    del x,p,y,prediction
report={"status":"completed","device":torch.cuda.get_device_name(),"training_nodes":len(checkpoint["node_indices"]),"inference_nodes":len(indices),
        "execution_source_sha256":hashlib.sha256(source).hexdigest(),"checkpoint_roundtrip":roundtrip,
        "claim":"Training-step memory/time feasibility and full-node inference using a model trained at sampled nodes. Full-node training was not performed.",
        "units":{"length":"mm","time":"ms"},"training_step_probes":probes,"heldout_per_run":per_run,
        "full_mesh_displacement_relative_l2":float(np.sqrt(sse/den)),"full_mesh_displacement_rmse_mm":float(np.sqrt(sse/(len(per_run)*len(indices)*(len(checkpoint["times"])-1)*3))),
        "full_mesh_training_mean_field_relative_l2":float(np.sqrt(mean_sse/den)),"zero_displacement_relative_l2":1.0,
        "inference_timing_note":"One warmup per run; synchronized forward call only; excludes data loading, transfers and disk I/O.",
        "memory_note":"CUDA peak allocation covers real FP32 forward/backward/Adam steps after a warmup. Host RSS is sampled, not a true peak."}
(args.out/"full_mesh_probe.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
