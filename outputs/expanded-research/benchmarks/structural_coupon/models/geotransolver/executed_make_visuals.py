"""Scientific comparison figures from saved, physical-unit inference arrays."""
import argparse
import json
from pathlib import Path
from textwrap import fill
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from importlib.util import module_from_spec, spec_from_file_location

# Preserve the exact CPU cache reader beside this postprocessing script.
_spec = spec_from_file_location("postprocess_benchmark_data", Path(__file__).with_name("executed_postprocess_benchmark_data.py"))
_reader = module_from_spec(_spec)
_spec.loader.exec_module(_reader)
Cache = _reader.Cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--time", type=int, default=-1)
    args = parser.parse_args()
    cache = Cache(args.cache)
    coordinate_unit = cache.meta.get("coordinate_unit",cache.meta.get("coordinate_units"))
    time_unit = cache.meta.get("time_unit",cache.meta.get("time_units"))
    prediction = np.load(args.result / "predictions.npy", mmap_mode="r")
    truth = np.load(args.result / "truth.npy", mmap_mode="r")
    assert prediction.shape == truth.shape == (len(cache.split["test"]), *cache.shape)
    metadata = json.loads((args.result / "metrics.json").read_text(encoding="utf-8"))
    case, time = args.case, args.time % cache.shape[1]
    axes_ids = np.argsort(np.ptp(cache.xyz, axis=0))[-2:][::-1]
    a, b = axes_ids
    xyz = cache.coordinates(cache.split["test"][case])
    plt.rcParams.update({"font.size": 9})
    def display_name(name):
        if name.startswith("displacement_"):return "Displacement "+name[-1]
        if "von_mises" in name or "vm" in name:return "Projected von Mises stress"
        if "plastic" in name.lower() or "epsp" in name.lower():return "Projected equivalent plastic strain"
        if "major" in name:return "Projected total strain, in-plane major"
        if "minor" in name:return "Projected total strain, in-plane minor"
        return name.replace("_"," ")
    for first in range(0, len(cache.names), 4):
        selected = list(range(first, min(first+4, len(cache.names))))
        fig, axes = plt.subplots(len(selected), 3, figsize=(12, 1.6*len(selected)+1), squeeze=False, constrained_layout=True)
        for row, c in enumerate(selected):
            field, unit = cache.names[c], cache.units[c]
            reference = truth[case,:,time,c]
            predicted = prediction[case,:,time,c]
            error = np.abs(predicted-reference)
            norm = Normalize(vmin=min(float(reference.min()),float(predicted.min())), vmax=max(float(reference.max()),float(predicted.max())))
            cmap = "coolwarm" if norm.vmin < 0 else "viridis"
            images = []
            for ax, value, title in zip(axes[row], [reference,predicted,error], ["FEM reference",metadata.get("architecture","AI"),"Absolute error"]):
                image = ax.scatter(xyz[:,a],xyz[:,b],c=value,s=3,cmap=cmap if title!="Absolute error" else "magma",norm=norm if title!="Absolute error" else Normalize(0,max(float(error.max()),1e-15)),rasterized=True)
                images.append(image)
                row_title = fill(display_name(field), width=32)+"\n"+title if title=="FEM reference" else title
                ax.set(title=row_title, xlabel=f"{'xyz'[a]} ({coordinate_unit})", ylabel=f"{'xyz'[b]} ({coordinate_unit})", aspect="equal")
            fig.colorbar(images[0],ax=axes[row,:2],label=unit,shrink=.8)
            fig.colorbar(images[2],ax=axes[row,2],label=unit,shrink=.8)
        fig.suptitle(f"{cache.split['test'][case]} | nominal time={cache.meta['time_values'][time]} {time_unit} | N={len(xyz):,}\nShared FEM/prediction color scale; original support coordinates")
        fig.savefig(args.result/f"field_comparison_{first//4+1}.png",dpi=150)
        plt.close(fig)
    for c,(field,unit) in enumerate(zip(cache.names,cache.units)):
        reference,predicted=truth[case,:,time,c],prediction[case,:,time,c]
        error=np.abs(predicted-reference)
        norm=Normalize(min(float(reference.min()),float(predicted.min())),max(float(reference.max()),float(predicted.max())))
        fig,axes=plt.subplots(1,3,figsize=(12,2.6),constrained_layout=True)
        images=[]
        for ax,value,title in zip(axes,[reference,predicted,error],["FEM reference",metadata.get("architecture","AI"),"Absolute error"]):
            image=ax.scatter(xyz[:,a],xyz[:,b],c=value,s=4,cmap=("coolwarm" if norm.vmin<0 else "viridis") if title!="Absolute error" else "magma",norm=norm if title!="Absolute error" else Normalize(0,max(float(error.max()),1e-15)),rasterized=True)
            images.append(image)
            ax.set(title=title,xlabel=f"{'xyz'[a]} ({coordinate_unit})",ylabel=f"{'xyz'[b]} ({coordinate_unit})",aspect="equal")
        fig.colorbar(images[0],ax=axes[:2],orientation="horizontal",label=f"{display_name(field)} ({unit})",shrink=.8,pad=.12)
        fig.colorbar(images[2],ax=axes[2],orientation="horizontal",label=f"Absolute error ({unit})",shrink=.8,pad=.12)
        fig.suptitle(f"{display_name(field)} | {cache.split['test'][case]} | nominal {cache.meta['time_values'][time]} {time_unit}\nSame reference support and FEM/AI color scale; raw predictions without clipping")
        fig.savefig(args.result/f"field_{c:02d}_{field}.png",dpi=160)
        plt.close(fig)
    if cache.names[:3] == ["displacement_x","displacement_y","displacement_z"]:
        reference = truth[case,:,time,:3]
        predicted = prediction[case,:,time,:3]
        magnitudes = [np.linalg.norm(reference,axis=-1),np.linalg.norm(predicted,axis=-1)]
        norm = Normalize(0,max(v.max() for v in magnitudes))
        fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for ax,u,color,title in zip(axes,[reference,predicted],magnitudes,["FEM","AI prediction"]):
            deformed = xyz+u
            ax.scatter(xyz[:,a],xyz[:,b],c="0.85",s=1,rasterized=True)
            image=ax.scatter(deformed[:,a],deformed[:,b],c=color,s=3,cmap="viridis",norm=norm,rasterized=True)
            ax.set(title=title,xlabel=f"{'xyz'[a]} (mm)",ylabel=f"{'xyz'[b]} (mm)",aspect="equal")
        fig.colorbar(image,ax=axes,label="Displacement magnitude (mm)",shrink=.8)
        fig.suptitle(f"{cache.split['test'][case]} | {cache.meta['time_values'][time]} ms | true deformation scale 1x | N={len(xyz):,}\nGray: initial configuration")
        fig.savefig(args.result/"deformation_1x.png",dpi=170)
        plt.close(fig)
    history=args.result/"training_history.csv"
    if history.exists():
        values=np.genfromtxt(history,delimiter=",",names=True)
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for field,title in [("training_normalized_mse","train"),("validation_normalized_mse","internal validation")]:
            ax.plot(np.atleast_1d(values["epoch"]),np.atleast_1d(values[field]),label=title)
        ax.set(xlabel="Epoch",ylabel="Mean normalized squared error",yscale="log",title=metadata.get("architecture","Model"))
        ax.legend();fig.savefig(args.result/"learning_curve.png",dpi=160);plt.close(fig)


if __name__ == "__main__":
    main()
