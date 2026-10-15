"""Frozen phone EPSP maps in a truth-support envelope, no model selection."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("--cache",type=Path,required=True)
parser.add_argument("--pod",type=Path,required=True)
parser.add_argument("--direct",type=Path,required=True)
args=parser.parse_args()
meta=json.loads(args.cache.read_text(encoding="utf-8"))
xyz=np.load(args.cache.parent/meta["coordinates"],mmap_mode="r")
ids=meta["split"]["test"]
c=next(i for i,name in enumerate(meta["field_names"]) if name.lower()=="epsp")
truth=[np.load(args.cache.parent/meta["runs"][case]["target"],mmap_mode="r")[:,-1,c] for case in ids]
active=np.any(np.stack([value!=0 for value in truth]),axis=0)
assert active.any(),"No final-time EPSP truth support; do not invent a plastic hotspot"
low=xyz[active,:2].min(0)-1.;high=xyz[active,:2].max(0)+1.
keep=np.all((xyz[:,:2]>=low)&(xyz[:,:2]<=high),axis=1)
directories=[args.pod/"pod_rbf",args.pod/"pod_ridge",args.direct]
ranges=[min(float(value.min()) for value in truth),max(float(value.max()) for value in truth)]
for directory in directories:
    p=np.load(directory/"predictions.npy",mmap_mode="r")
    ranges=[min(ranges[0],float(p[:,:,-1,c].min())),max(ranges[1],float(p[:,:,-1,c].max()))]
horizontal=1 if np.ptp(xyz[:,1])>np.ptp(xyz[:,0]) else 0
vertical=1-horizontal
for directory in directories:
    p=np.load(directory/"predictions.npy",mmap_mode="r")
    fig,axes=plt.subplots(len(ids),3,figsize=(12,4*len(ids)),layout="constrained",squeeze=False)
    error_max=max(float(np.abs(p[i,keep,-1,c]-t[keep]).max()) for i,t in enumerate(truth))
    for i,(case,t) in enumerate(zip(ids,truth)):
        values=[t[keep],p[i,keep,-1,c],np.abs(p[i,keep,-1,c]-t[keep])]
        for j,(value,title) in enumerate(zip(values,["FEM EPSP",directory.name,"Absolute error"])):
            order=np.argsort(np.abs(value),kind="stable")
            im=axes[i,j].scatter(xyz[keep,horizontal][order],xyz[keep,vertical][order],c=value[order],s=12,
                cmap="magma" if j==2 else "viridis",vmin=0 if j==2 else ranges[0],vmax=max(error_max,1e-30) if j==2 else ranges[1])
            axes[i,j].set_aspect("equal");axes[i,j].set_title(title)
            axes[i,j].set_xlabel("Reference "+"XY"[horizontal]+" [mm]")
            if j==0:axes[i,j].set_ylabel(case+"\nReference "+"XY"[vertical]+" [mm]")
            if j in [1,2]:fig.colorbar(im,ax=axes[i,:2] if j==1 else axes[i,2],label="[1]")
    fig.suptitle("EPSP: fixed envelope of all heldout final-time nonzero FEM nodes (+1 mm display margin)\nReference XY; overlapping points drawn in increasing |value| order")
    fig.savefig(directory/"epsp_truth_support_zoom.png",dpi=160);plt.close(fig)
    report={"scope":"Frozen postprocessing only; all fixed test cases included, no threshold/error-based test selection, refitting or clipping.","case_ids":ids,"time":meta["time_values"][-1],"time_unit":meta["time_unit"],
        "roi_definition":"Exact nonzero final-time FEM EPSP support union over all fixed test cases; reference XY bounding rectangle plus 1 mm display-only margin. Margin is not an accuracy or success threshold.",
        "roi_reference_xy_min_mm":low.tolist(),"roi_reference_xy_max_mm":high.tolist(),"roi_node_count":int(keep.sum()),"final_time_truth_nonzero_counts":[int(np.count_nonzero(t)) for t in truth],
        "overlap_policy":"At coincident reference XY, higher absolute scalar values are drawn last to expose localized fields; this is not a 3D deformed-surface view.","shared_truth_prediction_color_range":ranges,"absolute_error_color_range":[0.,error_max],"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/"epsp_truth_support_zoom.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(str(directory/"epsp_truth_support_zoom.png"),flush=True)
