import pathlib,json,csv
p=pathlib.Path('outputs/expanded-research/benchmarks/deforming_plate')
a=json.loads((p/'pilot8_validation/metrics.json').read_text());b=json.loads((p/'noise0_pilot8/metrics.json').read_text());t=json.loads((p/'pilot8/metrics.json').read_text())
r={}
for label,d,v in [('noise_0.003',t,a['pooled_metrics']),('noise_0',b,b['validation_pooled_metrics'])]:
 r[label]={'training_seconds':d['train_seconds'],'model_parameters':119284,'optimizer_updates':1920,'architecture':'HybridMeshGraphNet width48 processor3 output4','validation':v,'validation_final_inverted_tetrahedra':{i['trajectory']:i['predicted_inverted_tetrahedra_per_sampled_frame'][-1] for i in (a['results'] if label=='noise_0.003' else b['validation_results'])},'development':d['pooled_metrics'],'development_final_inverted_tetrahedra':{i['trajectory']:i['predicted_inverted_tetrahedra_per_sampled_frame'][-1] for i in d['results']}}
r['interpretation']='Noise and target normalization/loss weighting change together. This is a training-configuration sensitivity experiment, not an isolated noise-only causal ablation. Neither setting yields adequate displacement/stress/strain accuracy. Original test0/1 are development after inspection; final test2..7 remains unused.'
(p/'noise-sensitivity-comparison.json').write_text(json.dumps(r,indent=2),encoding='utf8')
print(json.dumps({k:{n:v[n] for n in ['validation_final_inverted_tetrahedra','development_final_inverted_tetrahedra']} for k,v in r.items() if isinstance(v,dict)},indent=2))
