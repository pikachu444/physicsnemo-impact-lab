"""Inference-only reuse of a saved PhysicsNeMo checkpoint on a matching FEM mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--vtp",type=Path,required=True)
parser.add_argument("--globals",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
parser.add_argument("--reference-predictions",type=Path)
parser.add_argument("--mesh-contract",type=Path)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
device=torch.device(args.device)
assert device.type!="cuda" or torch.cuda.is_available()
torch.set_num_threads(4)
contract_path=args.mesh_contract or args.checkpoint.with_name("mesh_contract.json")
contract=json.loads(contract_path.read_text(encoding="utf-8"))
assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==contract["checkpoint_sha256"],"Checkpoint/mesh contract mismatch"
checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
stats={k:v.numpy() for k,v in checkpoint["normalization"].items()}
ids=checkpoint["node_indices"].numpy()
assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and ids.min()>=0 and len(np.unique(ids))==len(ids)
assert len(ids)==contract["selected_node_count"]
assert hashlib.sha256(np.ascontiguousarray(ids,dtype="<i8").tobytes()).hexdigest()==contract["node_indices_int64_le_sha256"]
reader=vtkXMLPolyDataReader();reader.SetFileName(str(args.vtp));reader.Update()
poly=reader.GetOutput();pd=poly.GetPointData()
xyz=vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32).copy()
thickness=vtk_to_numpy(pd.GetArray("thickness")).astype(np.float32).copy()
assert ids.max()<len(xyz) and (thickness[ids]>0).all()
assert len(xyz)==contract["point_count"]
assert hashlib.sha256(np.ascontiguousarray(xyz[ids],dtype="<f4").tobytes()).hexdigest()==contract["reference_coordinates_float32_le_sha256"],"Input mesh geometry or node order differs from the trained reference"
globals_data=json.loads(args.globals.read_text(encoding="utf-8"))[args.vtp.stem]
g=np.array([globals_data[key] for key in ["thickness_scale","velocity_x","rwall_origin_y"]],dtype=np.float32)
assert np.isfinite(g).all() and np.isfinite(xyz).all() and np.isfinite(thickness).all()
outside=[key for key,value,lo,hi in zip(contract["global_feature_keys"],g,contract["training_global_min"],contract["training_global_max"]) if value<lo-1e-5 or value>hi+1e-5]
x=np.column_stack((thickness[ids],np.broadcast_to(g,(len(ids),3))))
x=torch.from_numpy((x-stats["x_mean"])/stats["x_std"]).unsqueeze(0).to(device)
p=torch.from_numpy((xyz[ids]-stats["xyz_mean"])/stats["xyz_std"]).unsqueeze(0).to(device)
from physicsnemo.models.transolver import Transolver
model=Transolver(**checkpoint["model_config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
with torch.inference_mode():
    for _ in range(3):model(x,p)
    if device.type=="cuda":torch.cuda.synchronize()
    started=time.perf_counter();prediction=model(x,p)
    if device.type=="cuda":torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    prediction=prediction.squeeze(0).cpu().numpy()*stats["y_std"]+stats["y_mean"]
report={"status":"inference_completed","run":args.vtp.stem,"device":str(device),"nodes":len(ids),
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "forward_seconds_excluding_loading_and_transfers":seconds,"units":{"length":"mm","time":"ms"},
        "scope":"Matching fixed node ordering and source physical regime only; this helper performs no fitting."}
report["within_training_global_bounds"]=not outside
report["outside_training_global_features"]=outside
fields=dict(node_indices=ids,coordinates_mm=xyz[ids],displacement_mm=prediction.reshape(len(ids),-1,3),time_ms=checkpoint["times"][1:])
names=[f"displacement_t{t:.3f}" for t in checkpoint["times"][1:]]
if all(pd.HasArray(name) for name in names):
    truth=np.stack([vtk_to_numpy(pd.GetArray(name))[ids] for name in names],axis=1).astype(np.float32)
    error=prediction.reshape(truth.shape).astype(np.float64)-truth
    report["displacement_relative_l2"]=float(np.linalg.norm(error)/np.linalg.norm(truth.astype(np.float64)))
    report["displacement_component_rmse_mm"]=float(np.sqrt(np.mean(error**2)))
    fields["reference_displacement_mm"]=truth
if args.reference_predictions:
    assert "split" in checkpoint,"Roundtrip comparison requires the completed transolver_cuda_checkpoint.pt, not the intermediate best checkpoint."
    with np.load(args.reference_predictions) as saved:
        index=checkpoint["split"]["test"].index(args.vtp.stem)
        expected=saved["transolver"][index]
        assert np.array_equal(ids,saved["node_indices"])
        assert np.array_equal(xyz[ids],saved["coords"][index])
        difference=float(np.max(np.abs(prediction-expected)))
        assert np.allclose(prediction,expected,rtol=1e-5,atol=1e-4),difference
    report["checkpoint_roundtrip"]={"passed":True,"max_absolute_difference_mm":difference,"rtol":1e-5,"atol_mm":1e-4}
np.savez_compressed(args.out/"inference_result.npz",**fields)
(args.out/"inference_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
(args.out/"executed_inference.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(report,indent=2),flush=True)
