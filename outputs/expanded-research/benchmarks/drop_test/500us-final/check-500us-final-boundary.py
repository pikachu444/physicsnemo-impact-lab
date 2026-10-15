"""Compare a zero-duration final-checkpoint reload at identical saved time."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import importlib.util
import json
from pathlib import Path
from resource_gate import guard as resource_guard
def guard(stage):
    result=resource_guard(stage,minimum_available=768*1024**2)
    result['configured_minimum_available_bytes']=768*1024**2
    return result
guard('before_boundary_numpy')
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/expanded-research/benchmarks/drop_test'
spec=importlib.util.spec_from_file_location('anim',BASE/'read-solid-anim.py')
anim=importlib.util.module_from_spec(spec);spec.loader.exec_module(anim)
old_path=ROOT/json.loads((BASE/'500us-final/summary.json').read_text())['last_frame']['file']
new_paths=sorted((ROOT/'work/expanded-drop/500us-final-boundary-run').glob('Cell_Phone_DropA[0-9][0-9][0-9]'))
assert new_paths
old=anim.read_solid_anim(old_path,geometry=True,nodal_stress=True,all_valid_tensors=True,nodal_vectors=True)
new=anim.read_solid_anim(new_paths[0],geometry=True,nodal_stress=True,all_valid_tensors=True,nodal_vectors=True)
checks={name:bool(np.array_equal(old[name],new[name])) for name in ['coords','node_ids','connectivity_8','solid_active_flags']}
for group in ['scalar_fields','node_scalar_fields','tensor_fields_6_xx_yy_zz_xy_yz_xz','node_vector_fields']:
    assert old[group].keys()==new[group].keys()
    checks.update({f'{group}/{name}':bool(np.array_equal(value,new[group][name])) for name,value in old[group].items()})
result={'source_frame':str(old_path.relative_to(ROOT)),'reloaded_frame':str(new_paths[0].relative_to(ROOT)),
        'source_actual_time_s':old['time_s'],'reloaded_actual_time_s':new['time_s'],
        'same_actual_time_s':bool(old['time_s']==new['time_s']),
        'all_native_fields_bitwise_equal':all(checks.values()),'field_checks':checks,
        'scope':'Final0.5ms checkpoint reload only; does not replace the unequal-time guard-stop boundary disclosure.',
        'resource':guard('boundary_comparison_complete')}
(BASE/'500us-final/final-checkpoint-boundary-bitwise.json').write_text(json.dumps(result,indent=2))
assert result['same_actual_time_s'] and result['all_native_fields_bitwise_equal'],result
print(json.dumps(result,indent=2))
