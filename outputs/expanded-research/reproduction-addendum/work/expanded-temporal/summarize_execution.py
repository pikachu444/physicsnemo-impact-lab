from pathlib import Path
import datetime, hashlib, json
ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records={name:json.loads((OUT/'coupon13'/name/'result.json').read_text()) for name in
         ['time_conditional','ar_rollout_training','one_step_teacher_forcing']}
probes={device:json.loads((OUT/f'official-wrapper-probe-{device}.json').read_text()) for device in ['cpu','cuda']}
for device,p in probes.items():
    assert len(p['rows'])==10
    assert sum(r['status']=='forward_backward_update_pass' for r in p['rows'])==6
    assert all(r.get('inference_target_value_independence_max_abs')==0 for r in p['rows'] if r['status']=='forward_backward_update_pass')
    assert all(r['status']=='runtime_failure' for r in p['rows'] if r['requested_target_channels']==7 and r['strategy']!='time_conditional')
for name,r in records.items():
    fresh=json.loads((OUT/'coupon13'/name/'fresh-process-reload.json').read_text())
    assert fresh['prediction_max_abs_difference']==0 and fresh['truth_max_abs_difference']==0
    assert r['driver_sha256']==sha(OWN/'run_temporal.py')
common_source=json.loads((ROOT/'outputs/expanded-research/benchmarks/bumper/probes/geotransolver/probe.json').read_text())['model_source']
assert sha(Path(common_source['source_file']))==common_source['source_sha256']
summary={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'actual_trained_models':3,'actual_training_conditions_per_model':5,'validation_conditions':4,'heldout_test_conditions':4,
 'model_implementation':common_source,'source_wrapper_classes_unchanged':True,
 'cpu_probe_combinations':10,'cuda_probe_combinations':10,'per_device_successes':6,'per_device_expected_dimension_failures':4,
 'cpu_to_cuda_fallback_for_training':False,'all_three_training_devices':'cuda',
 'all_three_fresh_process_predictions_max_abs_difference':0,
 'total_training_including_validation_seconds':sum(r['training_including_validation_seconds'] for r in records.values()),
 'accuracy_assessment':'TimeconditionalC7 beats train-mean displacement/EPSP/strain baselines but loses stress and weak-axis/planarity checks. AR/TF3 free rollout loses train-mean baseline. No appliance accuracy validation.',
 'new_domain_status':'Bumper/whole-phone source/data adapter preparation only; no extra GPU allocation or domain training claimed here',
 'models':{name:{k:r[k] for k in ['channels','epochs','updates','best_epoch','training_including_validation_seconds','four_case_inference_seconds','parameter_count','rss_observed_peak_bytes','cuda_allocated_peak_bytes','field_names','field_units','metrics']} for name,r in records.items()},
 'publications':{'raw_FEM_decks_included':False,'raw_FEM_target_NPY_or_NPZ_included':False,'environment_binaries_included':False,
                 'local_checkpoints':'work/expanded-temporal/checkpoints/<strategy>/best.pt',
                 'local_reference_and_predictions':'work/expanded-temporal/checkpoints/<strategy>/test-predictions.npz'},
 'figures':['coupon13-temporal-learning-and-errors.png','coupon13-timeconditional-stress-strain-fields.png'],
 'probe_note':'drop_test wrapper checked on coupon data is a wrapper API test, not phone-domain training'}
(OUT/'execution-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
paths=[OWN/n for n in ['fetch_sources.py','source_audit.py','run_temporal.py','make_visuals.py','summarize_execution.py','normalization.json','domain_source_audit.py','run_domain_temporal.py']]
paths+=list((OWN/'sources').rglob('*.py'))+list((OWN/'sources').rglob('*.yaml'))
paths+=[OWN/n for n in ['probe-cpu.log','probe-cpu-exit.json','probe-cpu-initial-api-failure.log','probe-train-gpu.log','probe-train-gpu-exit.json','reload-gpu.log','reload-gpu-exit.json']]
manifest={'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths],
 'layout':'Keep source scripts at work/expanded-temporal relative to the research workspace root; all producer caches retain the paths stated in the scripts. Source audit reconstructs boundary_features.npy from the licensed, locally obtained original source deck.',
 'execution_order':['fetch_sources.py','source_audit.py','run_temporal.py --allocated --mode probe --device cpu',
                    'run_temporal.py --allocated --mode probe-train-all --device cuda --epochs 30',
                    'run_temporal.py --allocated --mode reload-all --device cuda','make_visuals.py','summarize_execution.py'],
 'warning':'--allocated requires an explicit exclusive GPU/memory slot from the orchestrator; it does not mean a paid resource is authorized.'}
(OUT/'reproduction-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({'trained_models':3,'all_checks_passed':True,'driver_matches_all_training_reports':True,'reproduction_files':len(paths),'training_seconds':summary['total_training_including_validation_seconds']}))
