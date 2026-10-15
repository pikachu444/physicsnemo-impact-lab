"""Visualize already frozen predictions; no training or model selection."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse
import hashlib
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
    parser.add_argument("--time-unit")
    args=parser.parse_args()
    meta=json.loads(args.cache.read_text(encoding="utf-8"))
    case=meta["split"]["test"][0]
    xyz=np.load(args.cache.parent/meta["coordinates"],mmap_mode="r")
    truth=np.load(args.cache.parent/meta["runs"][case]["target"],mmap_mode="r")
    fields=meta["field_names"];units=meta["field_units"]
    time_unit=args.time_unit or meta.get("time_unit",meta.get("time_units","unspecified"))
    chosen=[j for j,n in enumerate(fields) if any(token in n.lower() for token in ["von_mises","vm_","plastic","epsp","principal"])]
    assert chosen, "No stress/strain fields in this frozen cache"
    directories=[args.pod/"pod_rbf",args.pod/"pod_ridge"]+([args.direct] if args.direct else [])
    horizontal=1 if np.ptp(xyz[:,1])>np.ptp(xyz[:,0]) else 0
    vertical=1-horizontal
    axis_names=["X","Y"]
    ranges={j:[float(truth[:,-1,j].min()),float(truth[:,-1,j].max())] for j in chosen}
    for directory in directories:
        values=np.load(directory/"predictions.npy",mmap_mode="r")
        for j in chosen:
            ranges[j][0]=min(ranges[j][0],float(values[0,:,-1,j].min()))
            ranges[j][1]=max(ranges[j][1],float(values[0,:,-1,j].max()))
        del values
    for directory in directories:
        predicted=np.load(directory/"predictions.npy",mmap_mode="r")
        assert predicted.shape[1:]==truth.shape
        fig,axes=plt.subplots(len(chosen),3,figsize=(13,1.6*len(chosen)+.7),layout="constrained",squeeze=False)
        description={}
        for row,j in enumerate(chosen):
            factor=1000. if units[j]=="GPa" else 1.
            display_unit="MPa" if units[j]=="GPa" else "1" if units[j] in ["dimensionless","1"] else units[j]
            reference=np.asarray(truth[:,-1,j])*factor;prediction=np.asarray(predicted[0,:,-1,j])*factor
            error=np.abs(prediction-reference)
            lo,hi=[value*factor for value in ranges[j]]
            if hi==lo:hi=lo+1e-12
            emax=float(error.max())
            name=fields[j].lower()
            layer=" upper" if "upper" in name else " lower" if "lower" in name else ""
            label=("Native shell VM -> nodes" if "von_mises" in name else "In-plane VM"+layer+" -> nodes" if "vm_" in name else "Total in-plane strain: major" if "principal_major" in name else "Total in-plane strain: minor" if "principal_minor" in name else "Equivalent plastic strain"+layer)
            for col,(value,title) in enumerate(zip([reference,prediction,error],[label+"\nFEM",directory.name,"Absolute error"])):
                im=axes[row,col].scatter(xyz[:,horizontal],xyz[:,vertical],c=value,s=3 if len(xyz)>3000 else 6,
                    cmap="magma" if col==2 else "viridis",vmin=0 if col==2 else lo,vmax=max(emax,1e-12) if col==2 else hi)
                axes[row,col].set_aspect("equal")
                axes[row,col].set_title(title,fontsize=10)
                if row==len(chosen)-1:axes[row,col].set_xlabel("Reference "+axis_names[horizontal]+" ["+meta.get("coordinate_unit","mm")+"]")
                if col==0:axes[row,col].set_ylabel("Reference "+axis_names[vertical]+" ["+meta.get("coordinate_unit","mm")+"]")
                if col==1:fig.colorbar(im,ax=axes[row,:2],shrink=.85,label="["+display_unit+"]")
                if col==2:fig.colorbar(im,ax=axes[row,2],shrink=.85,label="Abs. error ["+display_unit+"]")
            description[fields[j]]={"shared_fem_prediction_color_range":[lo,hi],"absolute_error_color_range":[0.,emax],"display_unit":display_unit,"source_unit":units[j],"display_multiplier":factor}
        fig.suptitle(f"{case} | t={meta['time_values'][-1]} {time_unit} | Reference XY; projected nodal scalars",fontsize=11)
        fig.savefig(directory/"stress_strain_comparison.png",dpi=160)
        plt.close(fig)
        report={"scope":"Frozen heldout predictions only. No refitting, rank changes, test selection or response clipping.","case":case,"time":meta["time_values"][-1],"time_unit":time_unit,"postprocess_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"geometry":"Reference XY, nodal projected scalars. Hotspot/peak is a nodal projection result, not original element/integration-point maximum.","figure_fields":description,"physical_diagnostics":{}}
        report["horizontal_reference_axis"]=axis_names[horizontal]
        report["vertical_reference_axis"]=axis_names[vertical]
        for j,name in enumerate(fields):
            p=predicted[...,j]
            nonnegative=any(t in name.lower() for t in ["vm","von_mises","plastic","epsp"])
            count=int(np.count_nonzero(p<0)) if nonnegative else None
            report["physical_diagnostics"][name]={"unit":units[j],"total_values":int(p.size),"nonfinite_count":int(np.count_nonzero(~np.isfinite(p))),"negative_count_if_nonnegative_field":count,"negative_fraction_if_nonnegative_field":count/p.size if nonnegative else None,"minimum":float(p.min()),"maximum":float(p.max()),"unclipped":True}
        (directory/"frozen_field_diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        del predicted
        print(str(directory/"stress_strain_comparison.png"),flush=True)


if __name__=="__main__":main()
