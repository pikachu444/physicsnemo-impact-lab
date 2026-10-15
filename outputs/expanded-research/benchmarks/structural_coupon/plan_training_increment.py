"""Prepare eight independent interior training conditions; no solver execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc


def plan(design_path, out):
    assert not out.exists(), "Do not overwrite a prepared proposal"
    frozen=json.loads(design_path.read_text(encoding="utf-8"))
    anchors=np.array([[1,1],[.9,.75],[.9,1.25],[1.1,.75],[1.1,1.25]])
    existing=np.concatenate([anchors,np.asarray([[r["strength_scale"],r["velocity_scale"]] for r in frozen["holdout_runs"]])])
    points=qmc.Sobol(d=2,scramble=True,seed=20261016).random_base2(m=3)
    scaled=np.array([.9,.75])+points*np.array([.2,.5])
    distances=np.max(np.abs(scaled[:,None,:]-existing[None,:,:]),axis=-1)
    pair=np.max(np.abs(scaled[:,None,:]-scaled[None,:,:]),axis=-1)+np.eye(8)
    assert distances.min()>1e-10 and pair.min()>1e-10
    runs=[{"case":f"proposed_train_interior_{i+1:02d}","split":"train","unit_square_point":point.tolist(),
           "strength_scale":float(value[0]),"velocity_scale":float(value[1])}
          for i,(point,value) in enumerate(zip(points,scaled))]
    proposal={"status":"proposed_not_executed","solver_runs_performed":0,
              "scope":"Only an eight-point proposal; parent must decide generation after initial fair validation learning curves. No present cache/split/label file is modified.",
              "current_training_cases":5,"proposed_training_cases_after_increment":13,
              "fixed_validation_cases":4,"fixed_test_cases":4,"current_total_fem_conditions":13,
              "proposed_total_fem_conditions_after_increment":21,
              "frozen_holdout_design_sha256":hashlib.sha256(design_path.read_bytes()).hexdigest(),
              "parameter_bounds":{"yield_UTS_scale":[.9,1.1],"velocity_scale":[.75,1.25]},
              "parameter_bounds_scope":"Algorithm sensitivity assumptions; same fixed geometry and first10ms source interval, not material scatter/product tolerance.",
              "generator":{"method":"scipy.stats.qmc.Sobol","scramble":True,"dimension":2,"seed":20261016,"power":3,
                           "scipy_version":scipy.__version__,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "duplicate_check":{"metric":"maximum absolute difference between parameter coordinates","tolerance":1e-10,
                                  "minimum_distance_to_existing13":float(distances.min()),"minimum_pair_distance_among_new8":float(pair.min())},
              "execution_policy":"No FEM commands are issued by this script. Preserve existing val/test points. If authorized, run serial one-thread cases and rebuild a new immutable cache/version, with train-only normalization shared by every algorithm. Test must not select training N or settings.",
              "runs":runs}
    out.write_text(json.dumps(proposal,indent=2),encoding="utf-8")
    print(json.dumps({"proposal":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"status":proposal["status"],"solver_runs":0},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    plan(a.design.resolve(),a.out.resolve())
