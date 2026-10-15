"""Predict saved FEM-response fields from geometry/parameters with no FEM labels.

NPZ input: coordinates[N,3], static_feature[N,1], globals[G], edge_index[2,E].
The fixed-topology research checkpoint rejects a different mesh or node order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault("PHYSICSNEMO_FORCE_TE","False")
os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
import numpy as np
import torch
from benchmark_data import array_hash
from model_adapters import build_model, forward_model, validate_batch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/"prediction.npy").exists(),"Choose a new inference output directory"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    stats,contract=payload["normalization"],payload["contract"]
    with np.load(args.inputs) as inputs:
        assert set(inputs.files)=={"coordinates","static_feature","globals","edge_index"}, "Predictor inputs contain unexpected fields; FEM target labels are not accepted"
        xyz=np.asarray(inputs["coordinates"],dtype=np.float32)
        static=np.asarray(inputs["static_feature"],dtype=np.float32)
        globals_=np.asarray(inputs["globals"],dtype=np.float32)
        edges=np.asarray(inputs["edge_index"],dtype=np.int64)
    assert array_hash(xyz)==contract["coordinate_sha256"],"Checkpoint is restricted to the same geometry/node order"
    assert array_hash(edges)==contract["edge_sha256"],"Checkpoint graph differs"
    assert static.shape==(contract["shape"][0],1)
    expected_globals=len(stats["global_mean"])
    assert globals_.shape==(expected_globals,)
    training_globals=np.array([contract["case_inputs_and_labels"][c]["globals"] for c in contract["split"]["train"]])
    outside=(globals_<training_globals.min(0)-1e-6)|(globals_>training_globals.max(0)+1e-6)
    device=torch.device(args.device)
    def tensor(value,dtype=torch.float32):
        return torch.tensor(value,dtype=dtype,device=device)
    batch={"xyz":tensor((xyz-stats["coordinate_min"])/np.asarray(stats["coordinate_span"])),
           "thickness":tensor((static-stats["thickness_mean"])/stats["thickness_std"]),
           "globals":tensor((globals_-np.asarray(stats["global_mean"]))/np.asarray(stats["global_std"])),
           "edge_index":tensor(edges,torch.long)}
    validate_batch(batch,require_edges=payload["architecture"]=="meshgraphnet")
    model,_,_=build_model(payload["architecture"],contract["shape"][1]*contract["shape"][2],config=payload["model_config"],global_dim=expected_globals)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    with torch.inference_mode():
        for _ in range(3):
            forward_model(payload["architecture"],model,batch)
        if device.type=="cuda":torch.cuda.synchronize()
        tick=time.perf_counter()
        normalized=forward_model(payload["architecture"],model,batch)
        if device.type=="cuda":torch.cuda.synchronize()
        seconds=time.perf_counter()-tick
        result=normalized[0].cpu().numpy().reshape(contract["shape"])*np.asarray(stats["field_std"],dtype=np.float32)+np.asarray(stats["field_mean"],dtype=np.float32)
    assert np.isfinite(result).all()
    np.save(args.out/"prediction.npy",result)
    report={"status":"fresh_checkpoint_prediction_without_FEM_target_inputs_completed","architecture":payload["architecture"],
            "inputs":str(args.inputs.resolve()),"input_sha256":hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),"shape":list(result.shape),
            "field_names":contract["fields"],"field_units":contract["units"],"time_values":contract["times"],
            "parameters_outside_training_envelope":outside.tolist(),"scope":"Prediction only; accuracy for a new parameter condition requires separate FEM or experimental reference. Fixed mesh and node order are enforced.",
            "warm_forward_seconds_excluding_transfers":seconds,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/"prediction_metadata.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    main()
