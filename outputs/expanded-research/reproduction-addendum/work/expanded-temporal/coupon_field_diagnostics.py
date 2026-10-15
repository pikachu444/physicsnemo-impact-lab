"""Posthoc diagnostics of immutable saved predictions; NumPy only."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies/coupon13'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def increment_summary(values):
    difference=np.diff(values.astype(np.float64),axis=2)
    decreases=-difference[difference<0]
    return {'minimum_delta':float(difference.min()),'maximum_delta':float(difference.max()),
            'strict_negative_increment_count':int(decreases.size),'exact_zero_increment_count':int(np.count_nonzero(difference==0)),
            'increment_count':int(difference.size),
            'decrease_magnitude_quantiles':{str(q):float(np.quantile(decreases,q)) for q in [.5,.9,.95,.99,1.]} if decreases.size else None,
            'maximum_time_decrease':float(decreases.max()) if decreases.size else 0.}
for strategy in ['time_conditional','ar_rollout_training','one_step_teacher_forcing']:
    path=OWN/'checkpoints'/strategy/'test-predictions.npz'
    archive=np.load(path);pred,truth=archive['prediction'],archive['truth']
    before=sha(path)
    report={'scope':'Posthoc diagnostic only; no training, clipping, threshold tuning, or model selection',
            'prediction_archive_sha256':before,'source_sha256':sha(Path(__file__)),
            'case_node_time_field_shape':list(pred.shape),'epsp_predicted':pred.shape[-1]>3,
            'dtype':str(pred.dtype),'float32_machine_epsilon':float(np.finfo(np.float32).eps)}
    if pred.shape[-1]>3:
        i=4;p,t=pred[...,i].astype(np.float64),truth[...,i].astype(np.float64)
        mask=t>0;pmask=p>0;err=p-t
        tp=int(np.count_nonzero(mask&pmask));fp=int(np.count_nonzero(~mask&pmask));fn=int(np.count_nonzero(mask&~pmask))
        report['equivalent_plastic_strain_projected']={
            'history_definition':'Strict fixed-node differences along saved future1..10ms axis;9 increments, no inferred t0 prediction. Truth and prediction receive identical diagnostics.',
            'prediction_history':increment_summary(p),'reference_history':increment_summary(t),
            'interpretation':'Cumulative raw EPSP should not decrease; projected reference rounding/decrease is reported separately. Strict negatives include roundoff, so compare magnitude distributions; no pass/fail tolerance was invented.',
            'stored_prediction_dtype':str(pred.dtype),'stored_reference_dtype':str(truth.dtype),
            'reference_absolute_peak':float(np.max(np.abs(t))),'prediction_absolute_peak':float(np.max(np.abs(p))),
            'reference_peak_times_float32_epsilon':float(np.max(np.abs(t))*np.finfo(np.float32).eps),
            'reference_positive_count':int(mask.sum()),'reference_zero_count':int((~mask).sum()),
            'prediction_negative_count':int(np.count_nonzero(p<0)),
            'reference_positive_support_rmse':float(np.sqrt(np.mean(err[mask]**2))) if mask.any() else None,
            'reference_zero_support_rmse':float(np.sqrt(np.mean(err[~mask]**2))) if (~mask).any() else None,
            'support_definition':'Reference>0 and prediction>0 exactly, no learned threshold; projected nodal EPSP, not a cell failure criterion',
            'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'positive_support_precision':tp/(tp+fp) if tp+fp else None,'positive_support_recall':tp/(tp+fn) if tp+fn else None,
            'positive_support_iou':tp/(tp+fp+fn) if tp+fp+fn else None}
        ordering=predict_order=pred[...,6].astype(np.float64)-pred[...,5]
        reference_order=truth[...,6].astype(np.float64)-truth[...,5]
        report['principal_order_diagnostics']={'prediction_minor_greater_than_major_count':int(np.count_nonzero(ordering>0)),
            'reference_minor_greater_than_major_count':int(np.count_nonzero(reference_order>0)),
            'prediction_maximum_violation':float(max(0.,ordering.max())),'no_reordering_applied':True}
    else:report['reason_not_applicable']='Original AR/TF wrappers predict displacement3 only; no EPSP channel exists.'
    assert sha(path)==before,'Immutable prediction archive changed'
    (OUT/strategy/'field_diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'strategy':strategy,'epsp_predicted':report['epsp_predicted'],
                     'history':report.get('equivalent_plastic_strain_projected',{}).get('prediction_history')}))
