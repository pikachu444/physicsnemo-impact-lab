"""Independent stdlib-only plate provenance/JSON audit; never opens array payloads."""
import ctypes
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/expanded-research/benchmarks/deforming_plate'
TRAIN = BASE / 'official32_valpilot'
FINAL = BASE / 'official32_fresh_final_corrected'
OUT = BASE / 'review'
PREFIX = 'official32-final-metadata-audit'
started = time.perf_counter()
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
handle = kernel.GetCurrentProcess()
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
affinity_ok = bool(kernel.SetProcessAffinityMask(handle, 1))
priority_ok = bool(kernel.SetPriorityClass(handle, 0x4000))
checks, failures, inventory = [], [], {}

def check(name, condition, detail=None):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
    if not condition:
        failures.append(name)

def sha(path):
    path = pathlib.Path(path)
    assert path.suffix.lower() not in {'.npz', '.npy'}, 'Array file access forbidden'
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    value = h.hexdigest()
    inventory[str(path.relative_to(ROOT))] = value
    return value

def read(path):
    path = pathlib.Path(path)
    assert path.suffix == '.json'
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def close(a, b):
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-12)

OUT.mkdir(parents=True, exist_ok=True)
snapshot = OUT / (PREFIX + '.executed.py')
with snapshot.open('xb') as stream:
    stream.write(pathlib.Path(__file__).read_bytes())
source_hash = sha(snapshot)
protocol = read(BASE / 'official32-validation-protocol.json')
lock = read(BASE / 'official32-completion-source-lock-v2.json')
corr = read(BASE / 'official32-final-correction-lock-v1.json')
config = read(TRAIN / 'config.json')
norm = read(TRAIN / 'normalization.json')
ta = read(TRAIN / 'training-attempt.json')
tm = read(TRAIN / 'metrics.json')
history = read(TRAIN / 'history.json')
selection = read(TRAIN / 'selection_history.json')
gate = read(FINAL / 'pre-evaluation-lock.json')
fa = read(FINAL / 'final-attempt.json')
fm = read(FINAL / 'metrics.json')
contract = read(BASE / 'final-split-contract.json')
release = read(FINAL / 'gpu-release.json')
old = read(BASE / 'official32_fresh_final/final-attempt.json')
expected_train = [f'train_{i:04d}' for i in range(32)]
fresh = [f'test_{i:04d}' for i in range(2, 8)]
valid = [f'valid_{i:04d}' for i in range(6)]
lock_hash = sha(BASE / 'official32-completion-source-lock-v2.json')
protocol_hash = sha(BASE / 'official32-validation-protocol.json')
correction_hash = sha(BASE / 'official32-final-correction-lock-v1.json')
check('v2/protocol pins', lock_hash == '5992ae6deed73d0b6fc4476de35db6b8483d070400597cf9f0a350de5d7af640' and protocol_hash == lock['protocol_sha256'] == ta['protocol_sha256'] == 'a71497614937762d2ca7434d8b5c6e2f91d3fecb4c750ab9efd97192e776bf33')
check('training and final source lock receipts', sha(TRAIN/'preexecution_source_lock.json') == lock_hash == ta['source_lock_sha256'] == gate['source_lock_sha256'])
check('training protocol receipt', sha(TRAIN/'pretraining_protocol.json') == protocol_hash)
for name, value in lock['source_sha256'].items():
    check('source9 '+name, sha(ROOT/'work/expanded-plate'/name) == value == sha(FINAL/('executed_'+name)))
check('correction driver and lock pins', sha(ROOT/'work/expanded-plate/run_frozen_final_corrected.py') == corr['corrected_driver_sha256'] == sha(FINAL/'executed_run_frozen_final_corrected.py') and correction_hash == fa['correction_lock_sha256'] == gate['correction_lock_sha256'] == sha(FINAL/'preexecution_correction_lock.json'))
cp_hash = sha(TRAIN/'checkpoint.pt')
check('selected checkpoint c29 hash', cp_hash == 'c29e4ecd258553622ecee13fc054a58854950e667b3a838610ab52f799339429' == ta['checkpoint_pt_sha256'] == tm['checkpoint_sha256'] == gate['checkpoint_sha256'] == fa['checkpoint_sha256'] == fm['checkpoint_sha256'] == corr['selected_checkpoint_sha256'])
for filename, receipt in [('config.json','config_json_sha256'),('normalization.json','normalization_json_sha256')]:
    check(filename+' stable receipt', sha(TRAIN/filename) == ta[receipt] == gate[filename.replace('.json','_sha256')])
check('training/final metrics receipts', sha(TRAIN/'metrics.json') == ta['metrics_sha256'] and sha(FINAL/'metrics.json') == fa['metrics_sha256'])
check('split/train-only normalization', norm['fit_trajectories'] == expected_train == config['splits']['train'] == contract['training'] == protocol['training'] and config['splits']['valid'] == valid == protocol['validation'] and config['splits']['test'] == [] and contract['development'] == ['test_0000','test_0001'] and contract['fresh_final_test'] == fresh == protocol['fresh_final_test'] == gate['fresh_final_test'])
check('split contract pin', sha(BASE/'final-split-contract.json') == gate['split_contract_sha256'])
check('train recipe actual config', config['parameters'] == 3605636 and (config['hidden'],config['processor'],config['epochs'],config['frames_per_trajectory'],config['train_count']) == (128,15,5,8,32) and config['validation_only'] and not config['test_npz_opened_or_hashed'] and not ta['test_npz_opened_or_hashed'])
check('all frozen required options', all(config[k] == (protocol[k] if k in {'selection_epochs','selection_trajectories'} else v) for k,v in protocol['required_options'].items()))
check('five completed epoch records', [r['epoch'] for r in history] == [1,2,3,4,5])
progress = {}
with (BASE/'official32_valpilot-completion.log').open(encoding='utf-8',errors='replace') as stream:
    for line in stream:
        if line.startswith('{') and 'training_progress' in line:
            row=json.loads(line)
            if row.get('stage')=='training_progress' and row.get('completed_trajectories')==32:
                progress[row['epoch']]=row['updates']
check('1280 updates from final progress each epoch', progress == {i:256 for i in range(1,6)}, progress)
check('per-update LR corroborates1280', close(history[-1]['learning_rate_after_updates'], 1e-4*0.9999917**1280))
ranked=[]
for rec in selection:
    epoch=rec['epoch']; rows=[]
    for case in ['valid_0000','valid_0005']:
        rows.append(read(TRAIN/f'selection_epoch_{epoch:03d}__{case}_metrics.json'))
    complete=all(r['status']=='completed' and r['steps']==399 and all(len(r[k])==399 for k in ['all_frame_predicted_inverted_tetrahedra','all_frame_truth_inverted_tetrahedra']) for r in rows)
    invalid=any(any(v>0 for v in r['all_frame_predicted_inverted_tetrahedra']) for r in rows)
    worst=max(r[f]['relative_l2'] for r in rows for f in ['rollout_free_displacement','rollout_free_stress'])
    key=[0 if complete else 1, int(invalid),worst]
    check(f'selection epoch{epoch} independently recomputed key', rec['selection_key']==key)
    check(f'candidate epoch{epoch} file pin', sha(TRAIN/rec['checkpoint_file'])==rec['checkpoint_sha256'])
    ranked.append((tuple(key)+(epoch,),epoch))
check('selection frozen epochs and chosen5', [r['epoch'] for r in selection]==[1,3,5] and min(ranked)[1] == tm['best_epoch'] == gate['selected_epoch'] == fa['selected_epoch'] == fm['best_epoch'] == 5 and tm['selected_rollout_ranking']==gate['selected_validation_rank'])
check('training validation completed/exact6', ta['status']=='completed_actual_validation_experiment' and [r['trajectory'] for r in tm['results']]==valid and tm['evaluation_completed_all_cases'] and tm['evaluation_split']=='valid' and not tm['test_npz_opened_or_hashed'])
check('permanent marker matches gate', sha(TRAIN/'fresh-final-access.json') == sha(FINAL/'pre-evaluation-lock.json') == fa['pre_evaluation_lock_sha256'])
driver=(FINAL/'executed_run_frozen_final_corrected.py').read_text(encoding='utf-8')
check('marker ordering source contract', driver.index('marker.open("x"') < driver.index('fresh_test_content_access_started"] = True') < driver.index('sha(path) == gate["fresh_expected_npz_sha256"]') < driver.index('import psutil') < driver.index('results = trainer.evaluate'))
check('new PID completed final exact6', fa['process_id'] != ta['process_id'] and fa['status']=='completed_fixed_checkpoint_fresh_test' and fm['evaluation_completed_all_cases'] and fm['evaluation_failed_case_count']==0 and [r['trajectory'] for r in fm['results']]==fresh and fm['evaluation_split']=='test' and fa['prediction_attempt_count_per_case']==1)
check('failed pregate preserved/data0', old['status']=='failed' and old['fresh_test_content_access_started'] is False and sha(BASE/'official32_fresh_final/final-attempt.json') == corr['preserved_failed_attempt_sha256'] and corr['test_prediction_count_before_correction']==0 and not (BASE/'official32_fresh_final/pre-evaluation-lock.json').exists() and (BASE/'official32_fresh_final/failure-traceback.txt').is_file())
check('exit0/release recorded not repolled', release['unified_session_reaped_exit_code']==0 and release['owned_process_id']==fa['process_id'] and release['owned_process_present'] is False and release['compute_apps_raw']==[] and release['additional_neural_inference_after_completion']==0 and release['additional_training_after_completion']==0)
pooled={}; case_summary=[]
for name in fresh:
    r=read(FINAL/(name+'_metrics.json'))
    embedded=next(v for v in fm['results'] if v['trajectory']==name)
    check(name+' percase equals aggregate JSON', r==embedded)
    check(name+' completed399 noTF all geometry/graphs', r['status']=='completed' and r['steps']==399 and r['teacher_forcing_computed'] is False and all(len(r[k])==399 for k in ['all_frame_predicted_inverted_tetrahedra','all_frame_truth_inverted_tetrahedra','all_frame_predicted_minimum_detF','all_frame_truth_minimum_detF','all_frame_graph_sizes']))
    check(name+' GL21/native-unit caveat', len(r['derived_GL_strain_frames'])==21 and r['derived_GL_strain_frames'][-1]==399 and 'not an original COMSOL' in r['strain_label_caveat'] and r['stress_units']=='unconfirmed dataset stress units')
    check(name+' saved field exists metadata only', (FINAL/r['field_file']).is_file())
    for field, value in r.items():
        if isinstance(value,dict) and 'relative_l2' in value:
            s=pooled.setdefault(field,{'sse':0.0,'truth_sum_square':0.0,'values':0,'max':0.0})
            s['sse']+=value['component_rmse']**2*value['values'];s['truth_sum_square']+=value['truth_l2']**2;s['values']+=value['values'];s['max']=max(s['max'],value['max_abs_component_error'])
    case_summary.append({'case':name,'u_relative_l2':r['rollout_free_displacement']['relative_l2'],'stress_relative_l2':r['rollout_free_stress']['relative_l2'],'derived_GL_relative_l2':r['derived_GL_strain']['relative_l2'],'predicted_inverted_frames':sum(v>0 for v in r['all_frame_predicted_inverted_tetrahedra']),'reference_inverted_frames':sum(v>0 for v in r['all_frame_truth_inverted_tetrahedra']),'negative_stress_count':r['predicted_negative_von_mises_values'],'AR_wall_seconds':r['evaluation_seconds']})
for name,s in pooled.items():
    actual=fm['pooled_metrics'][name]
    s['relative_l2']=math.sqrt(s['sse']/s['truth_sum_square']) if s['truth_sum_square'] else None
    s['component_rmse']=math.sqrt(s['sse']/s['values'])
    check('pooled JSON reconstruction '+name, actual['values']==s['values'] and close(actual['component_rmse'],s['component_rmse']) and close(actual['relative_l2'],s['relative_l2']) and actual['max_abs_component_error']==s['max'])
check('accuracy not prequalified', not fm['accuracy_qualified'] and not tm['accuracy_qualified'] and not fm['original_paper_recipe_reproduced'])
report={'status':'PASS_SCOPED_METADATA_AND_JSON_AGGREGATE_AUDIT' if not failures else 'FAIL_METADATA_AUDIT','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor_source_sha256':source_hash,'executed_source':str(snapshot.relative_to(ROOT)),'CPU_affinity_single_core_applied':affinity_ok,'below_normal_priority_applied':priority_ok,'framework_imports':False,'array_payload_files_opened_or_hashed':0,'checkpoint_deserialized':False,'new_training_or_inference':0,'maximum_hash_chunk_bytes':1024*1024,'wall_seconds':time.perf_counter()-started,'checks':checks,'failures':failures,'source_and_receipt_hashes':inventory,'completed_updates_log_corroborated':1280 if progress=={i:256 for i in range(1,6)} else None,'selected_epoch':fm['best_epoch'],'fresh_cases':case_summary,'pooled_reconstructed_from_rounded_case_metrics':pooled,'resources_recorded_not_remeasured':{'training_wall_including_imports_setup_io':ta['elapsed_wall_seconds_including_imports_setup_and_io'],'training_metrics_timer':tm['train_seconds'],'training_timer_scope':tm['train_timing_scope'],'training_memory':ta['memory'],'training_allocator':ta['cuda'],'final_wall_including_imports_gates_io':fa['elapsed_wall_seconds_including_imports_gates_io'],'final_memory':fa['memory'],'final_allocated_bytes':fm['cuda_peak_allocated_bytes'],'final_reserved_bytes':fm['cuda_peak_reserved_bytes'],'device_usage_observations_present':(FINAL/'external-resource-observations.jsonl').is_file()},'limits':['JSON arithmetic checks reuse producer per-case metrics; not independent raw/FEM/prediction-array recomputation.','Source and checkpoint byte hashes verified; checkpoint tensors not loaded.','Fresh array hashes in gate are recorded claims, not reverified in this audit.','Marker/access order is source plus matching receipts, not OS file-open tracing.','One attempt and no extra inference supported by exclusive marker/source/receipts; no global syscall proof.','New-PID strict checkpoint load is supported; no same-validation numerical reload equality check performed.','Allocator allocated/reserved differ from total device usage; host RSS differs from commit. External device observation samples were not numerically re-audited.','Derived GL covers21 sampled frames; not native COMSOL strain.','Runtime completed but errors/inversions do not qualify engineering accuracy; no visual review.']}
with (OUT/(PREFIX+'.json')).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
lines=['# Official-size plate final metadata audit','',report['status'],'','This is independent stdlib metadata/provenance and JSON aggregate consistency only. It is not raw-array verification, a visual audit or an accuracy qualification.','',f"Checks: {len(checks)}, failures: {len(failures)}. CPU1 applied: {affinity_ok}; no framework/array/NN access.",f"Selected epoch: {fm['best_epoch']}; fresh test2..7 six completed AR399 attempts; no teacher forcing.",f"Checkpoint: `{cp_hash}`.",'','| Case | u relative L2 | stress relative L2 | derived GL relative L2 | predicted/reference inverted frames |','|---|---:|---:|---:|---:|']
for r in case_summary:lines.append(f"| {r['case']} | {r['u_relative_l2']:.9g} | {r['stress_relative_l2']:.9g} | {r['derived_GL_relative_l2']:.9g} | {r['predicted_inverted_frames']}/{r['reference_inverted_frames']} |")
lines+=['','Relative L2 values above are fractions, not percentages. Native stress units remain unconfirmed; derived GL is dimensionless.',f"Training wrapper wall {ta['elapsed_wall_seconds_including_imports_setup_and_io']:.3f}s; final cold wall {fa['elapsed_wall_seconds_including_imports_gates_io']:.3f}s. Training metrics timer includes validation/selection/checkpoint I/O and is not optimizer-only. Sampled RSS, Windows peak working set, commit, tensor allocation and allocator reserve remain separate fields.",'','Limitations:']+['- '+x for x in report['limits']]
if failures:lines+=['','Failed checks:']+['- '+x for x in failures]
with (OUT/(PREFIX+'.md')).open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failures':failures,'wall_seconds':report['wall_seconds'],'selected_epoch':fm['best_epoch'],'fresh_cases':len(case_summary)}))
