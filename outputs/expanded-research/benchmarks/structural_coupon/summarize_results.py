"""Summarize measured FEM throughput and distinct physical label responses."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(cell_cache, nodal_cache, work, summaries, out):
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((nodal_cache/"manifest.json").read_text(encoding="utf-8"))
    native = json.loads((cell_cache/"manifest.json").read_text(encoding="utf-8"))
    rows, details = [], {}
    for group, ids in meta["split"].items():
        for case_id in ids:
            source = json.loads((summaries/case_id/"pilot-summary.json").read_text(encoding="utf-8"))
            entry = meta["runs"][case_id]
            target = np.load(nodal_cache/entry["target"], mmap_mode="r")
            stages = {value["stage"]:value for value in source["execution"]}
            coefficients = entry["solver_calculated_LAW2_coefficients"]
            row = {"case":case_id,"split":group,"yield_UTS_scale":entry["globals"][0],"velocity_scale":entry["globals"][1],
                   "starter_seconds":stages["starter"]["wall_seconds"],"engine_seconds":stages["engine"]["wall_seconds"],
                   "starter_sampled_peak_rss_bytes":stages["starter"]["sampled_peak_rss_bytes"],
                   "engine_sampled_peak_rss_bytes":stages["engine"]["sampled_peak_rss_bytes"],
                   "native_cell_final_VM_max_MPa":source["final_max"][7],"native_cell_final_EPSP_max":source["final_max"][6],
                   "nodal_final_u_x_max_mm":float(target[:,-1,0].max()),"nodal_final_VM_projected_max_MPa":float(target[:,-1,3].max()),
                   "nodal_final_EPSP_projected_max":float(target[:,-1,4].max()),
                   "nodal_final_strain_principal_major_projected_max":float(target[:,-1,5].max()),
                   "max_KE_over_IE_after1ms":source["energy"]["max_kinetic_over_internal_after1ms"],
                   "final_hourglass_over_IE":source["energy"]["final_hourglass_over_internal"],
                   "LAW2_A_GPa":coefficients["A_GPa"],"LAW2_B_GPa":coefficients["B_GPa"],"LAW2_n":coefficients["n"],
                   "max_time_jitter_ms":entry["max_time_jitter_ms"]}
            rows.append(row)
            details[case_id] = {"physical_checks":source["checks"],"force_diagnostic":source["force_diagnostic"],
                                "energy":source["energy"],"prepared_input_hashes":entry["prepared_sha256"],
                                "canonical_cell_target_sha256":native["runs"][case_id]["target_sha256"],
                                "primary_nodal_target_sha256":entry["target_sha256"]}
    with (out/"dataset-summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary={"cases":len(rows),"split_counts":{group:len(ids) for group,ids in meta["split"].items()},
             "all_solver_exit_codes_zero":all(v["exit_code"]==0 and v["terminated_reason"] is None for ids in meta["split"].values() for c in ids for v in json.loads((work/"runs"/c/"execution-metrics.json").read_text())),
             "total_solver_wall_seconds":sum(r["starter_seconds"]+r["engine_seconds"] for r in rows),
             "engine_wall_seconds_range":[min(r["engine_seconds"] for r in rows),max(r["engine_seconds"] for r in rows)],
             "maximum_sampled_starter_rss_bytes":max(r["starter_sampled_peak_rss_bytes"] for r in rows),
             "maximum_sampled_engine_rss_bytes":max(r["engine_sampled_peak_rss_bytes"] for r in rows),
             "maximum_KE_over_IE_after1ms":max(r["max_KE_over_IE_after1ms"] for r in rows),
             "maximum_frame_jitter_ms":max(r["max_time_jitter_ms"] for r in rows),
             "memory_measurement":"Per solver process RSS sampled every .05s; lower bound on instantaneous maximum; not total system/ML-process memory.",
             "timing_measurement":"Actual local sequential one-thread stages, with other research processes sharing host; not isolated performance guarantees.",
             "native_cell_shape":[808,10,8],"primary_nodal_shape":[919,10,7],"cache_array_bytes":sum(p.stat().st_size for p in nodal_cache.rglob("*.npy")),
             "fem_private_work_bytes":sum(p.stat().st_size for p in work.rglob("*") if p.is_file()),
             "cost":"No paid resource created; existing local computer used. Electricity and hardware ownership cost not measured.",
             "canonical_cell_manifest_sha256":hashlib.sha256((cell_cache/"manifest.json").read_bytes()).hexdigest(),
             "primary_nodal_manifest_sha256":hashlib.sha256((nodal_cache/"manifest.json").read_bytes()).hexdigest(),
             "details":details}
    (out/"dataset-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,field,title in zip(axes,["native_cell_final_VM_max_MPa","native_cell_final_EPSP_max","nodal_final_strain_principal_major_projected_max"],
                              ["Native cell von Mises max [MPa]","Native cell EPSP max [1]","Projected nodal in-plane total strain major max [1]"]):
        scatter=ax.scatter([r["yield_UTS_scale"] for r in rows],[r["velocity_scale"] for r in rows],c=[r[field] for r in rows],cmap="viridis",s=90,edgecolors="black")
        for r in rows:
            ax.annotate({"train":"T","validation":"V","test":"H"}[r["split"]],(r["yield_UTS_scale"],r["velocity_scale"]),ha="center",va="center",color="white",fontsize=8)
        ax.set(xlabel="Engineering yield/UTS scale",ylabel="Velocity scale",title=title)
        fig.colorbar(scatter,ax=ax,shrink=.7)
    fig.suptitle("Structural FEM sensitivity dataset; T=train anchor, V=validation, H=held-out test; actual ~10 ms")
    fig.savefig(out/"parameter-response.png",dpi=150)
    plt.close(fig)
    print(json.dumps({key:value for key,value in summary.items() if key!="details"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ["cell-cache","nodal-cache","work","summaries","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    summarize(a.cell_cache.resolve(),a.nodal_cache.resolve(),a.work.resolve(),a.summaries.resolve(),a.out.resolve())
