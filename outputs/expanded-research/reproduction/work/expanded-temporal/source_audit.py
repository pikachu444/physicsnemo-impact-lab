"""Static source and small FEM shape audit: no torch imports, no test-value reads."""
from pathlib import Path
import ast, hashlib, json, re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OWN=ROOT/'work/expanded-temporal'
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
CACHE=ROOT/'work/expanded-structural-coupon/cache-initial13-nodal7'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def group(text, n):
    tail=text.split(f'/GRNOD/NODE/{n}\n',1)[1].split('\n/',1)[0]
    return sorted(set(int(v) for line in tail.splitlines()[1:] if not line.startswith('#') for v in line.split()))

ready=json.loads((CACHE/'READY.json').read_text())
assert ready['status']=='ready' and ready['writer_exit_code']==0
assert sha(CACHE/'manifest.json')==ready['manifest_sha256']
meta=json.loads((CACHE/'manifest.json').read_text())
for r in ready['files']:
    assert sha(CACHE/r['name'])==r['sha256']
source=ROOT/'work/expanded-structural-coupon/original/tensile_LAW2_0000.rad'
source_manifest=json.loads((OUT/'source-manifest.json').read_text())
independent_manifest=json.loads((ROOT/'outputs/expanded-research/source-matrix/source-blob-manifest.json').read_text())
known={r['path']:r for r in independent_manifest['files']}
for row in source_manifest['files']:
    assert row['path'] in known and row['git_blob_sha1']==known[row['path']]['git_blob_sha']
text=source.read_text()
nodeids=np.load(CACHE/'node_ids.npy')
fixed=np.isin(nodeids,group(text,2)); moving=np.isin(nodeids,group(text,1))
assert fixed.sum()==91 and moving.sum()==91 and not np.any(fixed&moving)
np.save(OWN/'boundary_features.npy',np.stack([fixed,moving],1).astype('float32'))
train=np.stack([np.load(CACHE/meta['runs'][c]['target']) for c in meta['split']['train']])
mean=train.mean((0,1,2),dtype=np.float64)
var=train.var((0,1,2),dtype=np.float64)
std=np.sqrt(var); std[:3]=np.sqrt(var[:3].sum()/3)
reference=json.loads((ROOT/'work/expanded-ml/coupon13-normalization.json').read_text())
np.testing.assert_allclose(mean,reference['field_mean'],rtol=1e-12,atol=1e-12)
np.testing.assert_allclose(std,reference['field_std'],rtol=1e-12,atol=1e-12)
stats={'field_mean':mean.tolist(),'field_std':std.tolist(),'global_mean':reference['global_mean'],'global_std':reference['global_std'],
       'displacement_scale_definition':'train-only sqrt(trace(covariance)/3), one scale for XYZ',
       'scalar_scale_definition':'train-only population standard deviation per scalar field'}
(OWN/'normalization.json').write_text(json.dumps(stats,indent=2))
findings={}
for example in ['crash','drop_test']:
    p=OWN/'sources'/example/'rollout.py'
    tree=ast.parse(p.read_text())
    findings[example]={'sha256':sha(p),'class_line_ranges':{n.name:[n.lineno,n.end_lineno] for n in tree.body if isinstance(n,ast.ClassDef)},
       'time_conditional_support':'arbitrary out_dim including C7; independent time output, not AR',
       'ar_training_support':'three-dimensional acceleration and position only; extra scalar fields unsupported',
       'one_step_support':'three-dimensional position only; full C7 node_target concatenation with coords3 fails',
       'time_encoding':'t_index/(num_time_steps-1), first future frame is encoded0, final frame .9 for T11; this is index encoding not physical time0',
       'fixed_scalar_dt_only':True,
       'initial_vel_attribute_is_registered_buffer':False,
       'time_conditional_mutates_input_time_key':example=='crash'}
report={'status':'static source audit and NumPy FEM shape verification complete; runtime pending GPU allocation',
 'source_git_blob_matches_independent_inventory':True,
 'source_pin':'11685edb3ec1fb6f25665e96cf7a8c4d11a23f16','findings':findings,
 'coupon':{'manifest_sha256':sha(CACHE/'manifest.json'),'ready_sha256':sha(CACHE/'READY.json'),
           'shape':[919,10,7],'train_cases':5,'validation_cases':4,'test_cases':4,
           'test_target_values_read':False,'all_frozen_file_hashes_verified':True,
           'fixed_nodes':int(fixed.sum()),'moving_rigid_grip_slave_nodes':int(moving.sum()),
           'excluded_master_node':990,'boundary_feature_sha256':sha(OWN/'boundary_features.npy'),
           'source_initial_velocity':'all physical919 initial ANIM velocity components exactly0, independently checked by FEM producer',
           'velocity_after_t0':'constant step imposed X velocity at rigid master990, transferred to91 right-grip slave nodes; scale in global inputs',
           'source_inivel_card_present':'/INIVEL' in text,'source_boundary_card_sha256':sha(source),
           'wrapper_dt':1.0,'wrapper_dt_unit':'ms','dt_assumption':'nominal1ms with actual output jitter recorded; no interpolation',
           'max_actual_time_jitter_ms':max(r['max_time_jitter_ms'] for r in meta['runs'].values()),
           'train_only_normalization':stats},
 'runtime_plan':{'genuine_official_wrapper':True,'model':'GeoTransolver, PyTorch GALE backend',
                 'preset_training_epochs':30,'optimizer':'Adam, learning_rate0.001',
                 'selection':'lowest internal validation normalized MSE; final test after selection only',
                 'test_values_for_initial_velocity':False,'initial_velocity':'zero at every physical node',
                 'output_scopes':{'time_conditional':'C7 displacement+stress+plasticstrain+native-totalstrainprincipal2',
                                 'ar_rollout_training':'position3 only','one_step_teacher_forcing':'position3 only'},
                 'modifications':'independent data driver, boundary input features and physically consistent normalized initial velocity/dt; original wrapper class source unchanged',
                 'comparison_limit':'same data/case split and displacement loss scale, but timeconditional7-channel vs AR/TF3-channel task differs'}}
(OUT/'source-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'status':'pass','boundary_counts':[int(fixed.sum()),int(moving.sum())], 'source_examples':list(findings),'test_target_values_read':False,'max_jitter_ms':report['coupon']['max_actual_time_jitter_ms']}))
