"""Measure a larger Transolver on a real 1024-node FEM training sample."""
import argparse
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
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
assert torch.cuda.is_available();torch.set_num_threads(4);torch.manual_seed(20261015)
checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
stats={k:v.numpy() for k,v in checkpoint["normalization"].items()}
ids=checkpoint["node_indices"].numpy();assert len(ids)==1024
r=vtkXMLPolyDataReader();r.SetFileName(str(args.data/"TRAINING_DATA/Run100.vtp"));r.Update();p=r.GetOutput();pd=p.GetPointData()
xyz=vtk_to_numpy(p.GetPoints().GetData()).astype(np.float32)[ids]
thickness=vtk_to_numpy(pd.GetArray("thickness")).astype(np.float32)[ids]
target=np.stack([vtk_to_numpy(pd.GetArray(f"displacement_t{t:.3f}"))[ids] for t in checkpoint["times"][1:]],axis=1).astype(np.float32).reshape(len(ids),-1)
gd=json.loads((args.data/"GLOBAL_FEATURES.json").read_text())["Run100"]
g=np.array([gd[k] for k in ["thickness_scale","velocity_x","rwall_origin_y"]],dtype=np.float32)
features=np.column_stack((thickness,np.broadcast_to(g,(len(ids),3))))
x=torch.from_numpy((features-stats["x_mean"])/stats["x_std"]).unsqueeze(0).cuda()
pos=torch.from_numpy((xyz-stats["xyz_mean"])/stats["xyz_std"]).unsqueeze(0).cuda()
y=torch.from_numpy((target-stats["y_mean"])/stats["y_std"]).unsqueeze(0).cuda()
from physicsnemo.models.transolver import Transolver
config={**checkpoint["model_config"],"n_layers":4,"n_hidden":96,"slice_num":16,"n_head":4}
model=Transolver(**config).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
def step():
    optimizer.zero_grad(set_to_none=True);prediction=model(x,pos);loss=((prediction-y)**2).mean()
    assert torch.isfinite(loss);loss.backward();optimizer.step()
for _ in range(3):step()
torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();durations=[]
for _ in range(10):
    torch.cuda.synchronize();started=time.perf_counter();step();torch.cuda.synchronize();durations.append(time.perf_counter()-started)
mean=float(np.mean(durations));memory=torch.cuda.max_memory_allocated()
report={"status":"completed","nodes":len(ids),"model_config":config,"parameters":sum(p.numel() for p in model.parameters()),
        "step_seconds":durations,"mean_step_seconds":mean,"peak_cuda_allocated_bytes":memory,"sampled_rss_bytes":psutil.Process().memory_info().rss,
        "estimated_60epoch_112run_training_seconds_with_10percent_overhead":mean*60*112*1.1,
        "execution_gate_passed":memory<2_000_000_000 and mean*60*112*1.1<300,
        "note":"Real FP32 forward/backward/Adam step after warmup; estimate is not measured full training runtime. GPU allocation excludes CUDA context and dataset cache."}
(args.out/"capacity_probe.json").write_text(json.dumps(report,indent=2));(args.out/"executed_probe.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(report,indent=2),flush=True)
