"""Bind a completed checkpoint to its fixed reference mesh using hashes only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--reference-vtp",type=Path,required=True)
parser.add_argument("--globals",type=Path,required=True)
args=parser.parse_args()
directory=args.checkpoint.parent
report=json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
with np.load(directory/"heldout_predictions.npz") as arrays:
    ids=arrays["node_indices"]
    coordinates=arrays["coords"][0]
reader=vtkXMLPolyDataReader();reader.SetFileName(str(args.reference_vtp));reader.Update()
poly=reader.GetOutput()
xyz=vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float32)
assert np.array_equal(xyz[ids],coordinates),"Exact reference mesh mismatch"
globals_data=json.loads(args.globals.read_text(encoding="utf-8"))
keys=report["global_feature_keys"]
g=np.array([[globals_data[name][key] for key in keys] for name in report["split"]["train"]])
contract={"checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
          "point_count":len(xyz),"selected_node_count":len(ids),
          "node_indices_int64_le_sha256":hashlib.sha256(np.ascontiguousarray(ids,dtype="<i8").tobytes()).hexdigest(),
          "reference_coordinates_float32_le_sha256":hashlib.sha256(np.ascontiguousarray(coordinates,dtype="<f4").tobytes()).hexdigest(),
          "reference_vtp_sha256":hashlib.sha256(args.reference_vtp.read_bytes()).hexdigest(),
          "global_feature_keys":keys,"training_global_min":g.min(0).tolist(),"training_global_max":g.max(0).tolist(),
          "geometry_rule":"Exact fixed reference coordinates and node order required; no new topology/geometry claim.",
          "training_source_sha256":report["execution_source_sha256"]}
(directory/"mesh_contract.json").write_text(json.dumps(contract,indent=2),encoding="utf-8")
print(json.dumps(contract,indent=2))
