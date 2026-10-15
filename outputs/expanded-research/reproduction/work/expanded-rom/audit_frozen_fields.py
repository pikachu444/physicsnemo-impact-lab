"""Author-side recomputation and raw physical-support diagnostics, no refitting."""
import os
for key in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]:os.environ[key]="1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def epsp_history(values):
    source=np.asarray(values)
    delta=np.diff(source.astype(np.float64),axis=1)
    decrease=-delta[delta<0]
    spacing=np.abs(np.spacing(np.maximum(np.abs(source[:,:-1]),np.abs(source[:,1:])))).astype(np.float64)
    ratios=decrease/spacing[delta<0]
    qs=[.5,.95,.99,1.]
    return {"stored_dtype":str(source.dtype),"difference_definition":"np.diff(time) in float64 of the stored float32 endpoints; exact negative signs retained without tolerance or clipping",
        "interval_count":int(delta.size),"negative_increment_count":int(decrease.size),
        "minimum_delta":float(delta.min()),"maximum_delta":float(delta.max()),
        "decrease_magnitude_quantile_probabilities":qs,
        "decrease_magnitude_quantiles":np.quantile(decrease,qs).tolist() if decrease.size else None,
        "decrease_to_max_endpoint_float32_ulp_quantiles":np.quantile(ratios,qs).tolist() if ratios.size else None,
        "float32_endpoint_spacing_min":float(spacing.min()),"float32_endpoint_spacing_max":float(spacing.max()),
        "interpretation":"A negative increment is observed history inconsistency, not an automatic error verdict. Endpoint ULP ratios describe floating-point resolution; no success threshold was chosen. This diagnostic applies to accumulated EPSP, not total strain Frobenius or signed principal strains."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--pod",type=Path,required=True)
    parser.add_argument("--direct",type=Path,required=True)
    parser.add_argument("--raw-cache",type=Path)
    args=parser.parse_args()
    meta=json.loads(args.cache.read_text(encoding="utf-8"))
    pod=json.loads((args.pod/"metrics.json").read_text(encoding="utf-8"))
    direct=json.loads((args.direct/"metrics.json").read_text(encoding="utf-8"))
    assert pod["contract"]==direct["contract"]
    assert pod["normalization"]==direct["normalization"]
    assert pod["shared_normalization_sha256"]==direct["shared_normalization_sha256"]
    ids=meta["split"]["test"]
    variants=[("pod_rbf",args.pod/"pod_rbf",pod["test_metrics"]["pod_rbf"]),
              ("pod_ridge",args.pod/"pod_ridge",pod["test_metrics"]["pod_ridge"]),
              ("direct_rbf",args.direct,direct["metrics"]["rbf_direct_field"])]
    reports={}
    for method,directory,reported in variants:
        predicted=np.load(directory/"predictions.npy",mmap_mode="r")
        stats={name:{"sse":0.,"truth_square":0.,"count":0,"nonzero_sse":0.,"nonzero_square":0.,"nonzero_count":0,"zero_sse":0.,"zero_count":0,"negative_count":0,"negative_on_nonzero":0,"prediction_min":float("inf"),"prediction_max":float("-inf"),"nonfinite_count":0,"peaks":[],"epsp_history":[]} for name in meta["field_names"]}
        for case_index,case in enumerate(ids):
            truth=np.load(args.cache.parent/meta["runs"][case]["target"],mmap_mode="r")
            for c,name in enumerate(meta["field_names"]):
                p=np.asarray(predicted[case_index,...,c],dtype=np.float64)
                t=np.asarray(truth[...,c],dtype=np.float64)
                e=p-t;s=e*e;support=t!=0.;r=stats[name]
                r["sse"]+=float(s.sum());r["truth_square"]+=float(np.sum(t*t));r["count"]+=t.size
                r["nonzero_sse"]+=float(s[support].sum());r["nonzero_square"]+=float(np.sum(t[support]**2));r["nonzero_count"]+=int(support.sum())
                r["zero_sse"]+=float(s[~support].sum());r["zero_count"]+=int((~support).sum())
                r["negative_count"]+=int(np.count_nonzero(p<0));r["negative_on_nonzero"]+=int(np.count_nonzero((p<0)&support))
                r["prediction_min"]=min(r["prediction_min"],float(p.min()));r["prediction_max"]=max(r["prediction_max"],float(p.max()))
                r["nonfinite_count"]+=int(np.count_nonzero(~np.isfinite(p)))
                if "epsp" in name.lower() or "plastic" in name.lower():
                    r["epsp_history"].append({"case":case,"prediction":epsp_history(predicted[case_index,...,c]),"truth":epsp_history(truth[...,c])})
                index=np.unravel_index(np.argmax(np.abs(t)),t.shape)
                r["peaks"].append({"case":case,"truth_peak_node_index":int(index[0]),"truth_peak_time_index":int(index[1]),"truth_peak_time":meta["time_values"][index[1]],"truth_value":float(t[index]),"prediction_at_truth_peak":float(p[index]),"absolute_error_at_truth_peak":float(abs(e[index])),"magnitude_underprediction_at_truth_peak":float(max(abs(t[index])-abs(p[index]),0.))})
            del truth
        fields={}
        for c,name in enumerate(meta["field_names"]):
            r=stats[name]
            l2=float(np.sqrt(r["sse"]/r["truth_square"])) if r["truth_square"] else None
            rmse=float(np.sqrt(r["sse"]/r["count"]))
            assert l2 is None or np.isclose(l2,reported[name]["relative_l2"],rtol=1e-12)
            assert np.isclose(rmse,reported[name]["rmse"],rtol=1e-12,atol=1e-30)
            assert r["nonfinite_count"]==0
            nonnegative=any(token in name.lower() for token in ["vm","von_mises","plastic","epsp","frobenius"])
            fields[name]={"unit":meta["field_units"][c],"recomputed_relative_l2":l2,"recomputed_rmse":rmse,
                "rmse_in_MPa_if_stress_GPa":rmse*1000 if meta["field_units"][c]=="GPa" else None,
                "total_count":r["count"],"truth_exact_nonzero_support_count":r["nonzero_count"],"truth_zero_support_count":r["zero_count"],
                "rmse_on_nonzero_truth_support":float(np.sqrt(r["nonzero_sse"]/r["nonzero_count"])) if r["nonzero_count"] else None,
                "relative_l2_on_nonzero_truth_support":float(np.sqrt(r["nonzero_sse"]/r["nonzero_square"])) if r["nonzero_square"] else None,
                "rmse_on_zero_truth_support":float(np.sqrt(r["zero_sse"]/r["zero_count"])) if r["zero_count"] else None,
                "negative_count_if_nonnegative_field":r["negative_count"] if nonnegative else None,
                "negative_fraction_if_nonnegative_field":r["negative_count"]/r["count"] if nonnegative else None,
                "negative_on_nonzero_truth_support_if_nonnegative_field":r["negative_on_nonzero"] if nonnegative else None,
                "prediction_min":r["prediction_min"],"prediction_max":r["prediction_max"],"nonfinite_count":r["nonfinite_count"],"peaks":r["peaks"]}
            if r["epsp_history"]:fields[name]["accumulated_epsp_history"]=r["epsp_history"]
        reports[method]=fields
    result={"status":"all_separate_formula_metric_assertions_passed","scope":"Author-side diagnostic; no model or hyperparameter changes, no clipping or target replacements. Independent review is a separate gate.","cache_manifest_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest(),"shared_normalization_sha256":pod["shared_normalization_sha256"],"same_contract_split_fields_and_normalization":True,"support_definition":"Exact nonzero nodal truth, without a newly selected numerical threshold. Source derivation/projection is defined in the frozen cache. Nodal peaks do not establish original element or integration-point extrema.","source_field_definition":meta.get("field_definition"),"source_strain_note":meta.get("strain_note"),"physical_diagnostics":reports}
    if args.raw_cache:
        status=[];policies=set();layers=set()
        for case in sum(meta["split"].values(),[]):
            raw=json.loads((args.raw_cache/case/"manifest.json").read_text(encoding="utf-8"))
            policies.add(raw.get("shell_status_policy","unspecified"));layers.add(raw.get("upper_lower_semantics","unspecified"))
            counts=[f["shell_status_zero_count"] for f in raw["frame_checks"]]
            status.append({"case":case,"status_zero_count_max":max(counts),"status_zero_count_sum_over_frames":sum(counts)})
        result["source_shell_status"]={"policy":sorted(policies),"upper_lower_semantics":sorted(layers),"cases_with_status_zero":[r for r in status if r["status_zero_count_max"]],"maximum_status_zero_count_per_frame":max(r["status_zero_count_max"] for r in status),"cases_checked":len(status)}
    (args.pod/"frozen_support_peak_diagnostics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":result["status"],"methods":list(reports),"case_count":len(ids),"field_count":len(meta["field_names"])}),flush=True)


if __name__=="__main__":main()
