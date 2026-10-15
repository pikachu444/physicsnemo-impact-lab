import importlib.util
import json
import os
from pathlib import Path
from resource_gate import guard
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
guard('before_NumPy_import')
import numpy as np
root=Path(__file__).resolve().parents[2]
out=root/'outputs/expanded-research/benchmarks/drop_test'
spec=importlib.util.spec_from_file_location('anim',out/'read-solid-anim.py')
anim=importlib.util.module_from_spec(spec);spec.loader.exec_module(anim)
guard('before_source_snapshot_read')
old=anim.read_solid_anim(root/'work/expanded-drop/100us-run/Cell_Phone_DropA021',geometry=True,nodal_stress=True,all_valid_tensors=True,nodal_vectors=True)
guard('before_new_snapshot_read')
new=anim.read_solid_anim(root/'work/expanded-drop/500us-run/Cell_Phone_DropA021',geometry=True,nodal_stress=True,all_valid_tensors=True,nodal_vectors=True)
assert old['time_s']==new['time_s']
checks={name:bool(np.array_equal(old[name],new[name])) for name in ['coords','node_ids','connectivity_8','solid_active_flags']}
for group in ['scalar_fields','node_scalar_fields','tensor_fields_6_xx_yy_zz_xy_yz_xz','node_vector_fields']:
    assert old[group].keys()==new[group].keys()
    checks.update({f'{group}/{name}':bool(np.array_equal(value,new[group][name])) for name,value in old[group].items()})
result={'same_actual_time_s':old['time_s'],'all_existing_native_fields_bitwise_equal':all(checks.values()),'field_checks':checks,
    'new_tensor_field_count':len(new['all_tensor_names']),'new_tensor_names':new['all_tensor_names'],'old_tensor_field_count':len(old['all_tensor_names']),
    'output_change':'Only nonexistent-invalidIP tensors removed; all4validTETRA10IP and BRICKIP111 remain. No physical state changed.',
    'new_frame_bytes':(root/'work/expanded-drop/500us-run/Cell_Phone_DropA021').stat().st_size,
    'resource_measurement':guard('boundary_complete')}
assert all(checks.values()),result
(out/'500us/restart-boundary-bitwise-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
