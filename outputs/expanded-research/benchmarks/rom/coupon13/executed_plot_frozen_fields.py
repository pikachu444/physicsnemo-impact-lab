"""Visualize already frozen predictions; no training or model selection."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--pod",type=Path,required=True)
    parser.add_argument("--direct",type=Path)
    args=parser.parse_args()
    meta=json.loads(args.cache.read_text(encoding="utf-8"))
    case=meta["split"]["test"][0]
    xyz=np.load(args.cache.parent/meta["coordinates"],mmap_mode="r")
    truth=np.load(args.cache.parent/meta["runs"][case]["target"],mmap_mode="r")
    fields=meta["field_names"];units=meta["field_units"]
    chosen=[j for j,n in enumerate(fields) if any(token in n.lower() for token in ["von_mises","vm_","plastic","epsp","principal"])]
    assert chosen, "No stress/strain fields in this frozen cache"
    directories=[args.pod/"pod_rbf",args.pod/"pod_ridge"]+([args.direct] if args.direct else [])
    for directory in directories:
        predicted=np.load(directory/"predictions.npy",mmap_mode="r")
        assert predicted.shape[1:]==truth.shape
        fig,axes=plt.subplots(len(chosen),3,figsize=(13,1.6*len(chosen)+.7),layout="constrained",squeeze=False)
        description={}
        for row,j in enumerate(chosen):
            reference=np.asarray(truth[:,-1,j]);prediction=np.asarray(predicted[0,:,-1,j])
            error=np.abs(prediction-reference)
            lo,hi=float(min(reference.min(),prediction.min())),float(max(reference.max(),prediction.max()))
            if hi==lo:hi=lo+1e-12
            emax=float(error.max())
            for col,(value,title) in enumerate(zip([reference,prediction,error],["FEM projected nodal scalar",directory.name,"Absolute error"])):
                im=axes[row,col].scatter(xyz[:,0],xyz[:,1],c=value,s=3 if len(xyz)>3000 else 6,
                    cmap="magma" if col==2 else "viridis",vmin=0 if col==2 else lo,vmax=max(emax,1e-12) if col==2 else hi)
                axes[row,col].set_aspect("equal")
                axes[row,col].set_title(title,fontsize=10)
                if row==len(chosen)-1:axes[row,col].set_xlabel("Reference X ["+meta.get("coordinate_unit","mm")+"]")
                if col==0:axes[row,col].set_ylabel("Reference Y ["+meta.get("coordinate_unit","mm")+"]")
                if col==1:fig.colorbar(im,ax=axes[row,:2],shrink=.85,label=fields[j]+" ["+units[j]+"]")
                if col==2:fig.colorbar(im,ax=axes[row,2],shrink=.85,label="Absolute error ["+units[j]+"]")
            description[fields[j]]={"shared_fem_prediction_color_range":[lo,hi],"absolute_error_color_range":[0.,emax],"unit":units[j]}
        fig.suptitle(f"{case}, stored time {meta['time_values'][-1]} {meta.get('time_unit','')} | Reference geometry; native/derived scalars projected to nodes",fontsize=11)
        fig.savefig(directory/"stress_strain_comparison.png",dpi=160)
        plt.close(fig)
        report={"scope":"Frozen heldout predictions only. No refitting, rank changes, test selection or response clipping.","case":case,"time":meta["time_values"][-1],"geometry":"Reference XY, nodal projected scalars. Hotspot/peak is a nodal projection result, not original element/integration-point maximum.","figure_fields":description,"physical_diagnostics":{}}
        for j,name in enumerate(fields):
            p=predicted[...,j]
            nonnegative=any(t in name.lower() for t in ["vm","von_mises","plastic","epsp"])
            report["physical_diagnostics"][name]={"unit":units[j],"total_values":int(p.size),"nonfinite_count":int(np.count_nonzero(~np.isfinite(p))),"negative_count_if_nonnegative_field":int(np.count_nonzero(p<0)) if nonnegative else None,"minimum":float(p.min()),"maximum":float(p.max()),"unclipped":True}
        (directory/"frozen_field_diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        del predicted
        print(str(directory/"stress_strain_comparison.png"),flush=True)


if __name__=="__main__":main()
