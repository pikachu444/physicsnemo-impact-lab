"""Posthoc physical support diagnostics; no fitting, clipping or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--result",type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads(args.cache.read_text(encoding="utf-8"))
    pred=np.load(args.result/"predictions.npy",mmap_mode="r")
    truth=np.load(args.result/"truth.npy",mmap_mode="r")
    assert pred.shape==truth.shape and pred.shape[-1]==len(meta["field_names"])
    fields={}
    for c,(name,unit) in enumerate(zip(meta["field_names"],meta["field_units"])):
        p,t=pred[...,c].astype(np.float64),truth[...,c].astype(np.float64)
        e=p-t
        row={"unit":unit,"truth_absolute_peak":float(np.abs(t).max()),"prediction_absolute_peak":float(np.abs(p).max()),
             "component_rmse":float(np.sqrt(np.mean(e**2))),"absolute_error_max":float(np.abs(e).max()),
             "truth_exact_nonzero_count":int(np.count_nonzero(t)),"prediction_exact_nonzero_count":int(np.count_nonzero(p)),
             "prediction_nonfinite_count":int(np.count_nonzero(~np.isfinite(p))),"count":int(t.size)}
        if "plastic" in name.lower() or "epsp" in name.lower():
            mask=t>0
            pmask=p>0
            tp=int(np.count_nonzero(mask&pmask));fp=int(np.count_nonzero(~mask&pmask));fn=int(np.count_nonzero(mask&~pmask))
            row.update({"support_definition":"Reference EPSP >0 exactly; prediction >0 exactly. No clipping/threshold tuning. This is a nodal averaged projected support, not a raw-cell damage criterion.",
                        "reference_positive_support_count":int(mask.sum()),"reference_zero_support_count":int((~mask).sum()),
                        "rmse_on_reference_positive_support":float(np.sqrt(np.mean(e[mask]**2))) if mask.any() else None,
                        "rmse_on_reference_zero_support":float(np.sqrt(np.mean(e[~mask]**2))) if (~mask).any() else None,
                        "prediction_absolute_peak_on_reference_zero_support":float(np.abs(p[~mask]).max()) if (~mask).any() else None,
                        "prediction_negative_count":int(np.count_nonzero(p<0)),"true_positive":tp,"false_positive":fp,"false_negative":fn,
                        "positive_support_precision":tp/(tp+fp) if tp+fp else None,"positive_support_recall":tp/(tp+fn) if tp+fn else None,
                        "positive_support_iou":tp/(tp+fp+fn) if tp+fp+fn else None})
            row["per_case_support_errors"]=[]
            dp,dt=np.diff(p,axis=2),np.diff(t,axis=2)
            def negative_increment_summary(value):
                decreases=-value[value<0]
                return {"minimum_delta":float(value.min()),"maximum_delta":float(value.max()),
                    "negative_count":int(decreases.size),"zero_count":int(np.count_nonzero(value==0)),"count":int(value.size),
                    "decrease_magnitude_quantiles":{str(q):float(np.quantile(decreases,q)) for q in [.5,.9,.95,.99,1.]} if decreases.size else None}
            row["history_diagnostics"]={"definition":"Exact negative time increments at fixed projected support; raw cumulative EPSP is expected not to decrease. Report reference rounding/decrease separately; no correction is applied.",
                "prediction_negative_time_increment_count":int(np.count_nonzero(dp<0)),
                "reference_negative_time_increment_count":int(np.count_nonzero(dt<0)),
                "prediction_maximum_time_decrease":float(max(0.,-dp.min())),
                "reference_maximum_time_decrease":float(max(0.,-dt.min())),
                "prediction_raw_delta_distribution":negative_increment_summary(dp),
                "reference_raw_delta_distribution":negative_increment_summary(dt),
                "stored_prediction_dtype":str(pred.dtype),"stored_reference_dtype":str(truth.dtype),
                "float32_machine_epsilon":float(np.finfo(np.float32).eps),
                "interpretation":"Strict negative counts include roundoff-sized changes. Quantile magnitudes and the same truth statistics are reported without an invented pass/fail threshold."}
            for i,case in enumerate(meta["split"]["test"]):
                positive=mask[i]
                row["per_case_support_errors"].append({"case":case,"reference_positive_count":int(positive.sum()),
                    "reference_positive_rmse":float(np.sqrt(np.mean(e[i][positive]**2))) if positive.any() else None,
                    "reference_zero_rmse":float(np.sqrt(np.mean(e[i][~positive]**2))) if (~positive).any() else None,
                    "reference_peak":float(t[i].max()),"prediction_absolute_peak":float(np.abs(p[i]).max())})
        fields[name]=row
    report={"scope":"Posthoc diagnostics of saved raw predictions against the same heldout FEM references; no training or model selection",
            "fields":fields,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    major=[i for i,n in enumerate(meta["field_names"]) if "principal_major" in n]
    minor=[i for i,n in enumerate(meta["field_names"]) if "principal_minor" in n]
    if len(major)==len(minor)==1:
        difference=pred[...,minor[0]].astype(np.float64)-pred[...,major[0]]
        reference_difference=truth[...,minor[0]].astype(np.float64)-truth[...,major[0]]
        report["principal_order_diagnostics"]={"prediction_minor_greater_than_major_count":int(np.count_nonzero(difference>0)),
            "reference_minor_greater_than_major_count":int(np.count_nonzero(reference_difference>0)),
            "prediction_maximum_order_violation":float(max(0.,difference.max())),
            "definition":"A projected in-plane major principal strain must be >= its paired minor principal strain. No reordering is applied."}
    if meta["field_names"][:3]==["displacement_x","displacement_y","displacement_z"]:
        vector_norm=float(np.linalg.norm(truth[...,:3].astype(np.float64)))
        tiny=[]
        for name in meta["field_names"][:3]:
            c=meta["field_names"].index(name)
            fields[name]["reference_component_norm_to_vector_norm_ratio"]=float(np.linalg.norm(truth[...,c].astype(np.float64))/vector_norm) if vector_norm else None
            if vector_norm and fields[name]["reference_component_norm_to_vector_norm_ratio"]<1e-10:tiny.append(name)
        report["near_zero_reference_components"]=tiny
        report["near_zero_diagnostic_definition"]="Component reference L2 norm / vector reference L2 norm <1e-10; reporting diagnostic only, no field removal or model selection."
        if tiny:report["near_zero_axis_note"]="For these near-zero reference components, emphasize absolute error and predicted magnitude in physical units. Relative error divided by a near-zero reference is not a useful standalone accuracy claim."
    (args.result/"field_diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    shutil.copyfile(__file__,args.result/"executed_field_diagnostics.py")
    print(json.dumps({"status":"raw_field_diagnostics_completed","fields":list(fields)}),flush=True)


if __name__=="__main__":
    main()
