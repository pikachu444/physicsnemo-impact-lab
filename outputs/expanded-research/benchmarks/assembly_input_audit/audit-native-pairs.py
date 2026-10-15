"""Check uninterpolated FEM-derived C6 raw-frame caches at coincident pairs.

NumPy mmap only. The native-frame cache follows cache-phone-case.py's GPS
stress/volume-projected strain definitions; it is distinct from raw ANIM tensors.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]
started=time.perf_counter();audit=json.loads((OUT/'assembly-input-audit.json').read_text())
rows=list(csv.DictReader((OUT/'coordinate-pairs.csv').open()))
pairs=np.array([[int(x['index_a']),int(x['index_b'])] for x in rows if x['bitwise_equal_source_float64']=='True'])
assert pairs.shape==(132,2)
maximum=np.zeros(6);examples=[None]*6;nonzero=np.zeros((132,6),bool);sources=[]
for case in audit['case_target_evidence']:
    name=case['run'];metadata=ROOT/'outputs/expanded-research/benchmarks/drop_test/doe/canonical/runs'/name/'metadata.json'
    meta=json.loads(metadata.read_text());path=ROOT/'work/expanded-drop'/f'{name}-raw-target.npy'
    before=path.stat();target=np.load(path,mmap_mode='r')
    assert target.shape==(len(meta['raw_frame_records']),76065,6)
    left=target[:,pairs[:,0],:].astype(np.float64);right=target[:,pairs[:,1],:].astype(np.float64)
    diff=abs(left-right);nonzero|=np.any(diff!=0,axis=0)
    for c in range(6):
        value=float(diff[:,:,c].max())
        if value>maximum[c]:
            t,q=np.unravel_index(int(np.argmax(diff[:,:,c])),diff[:,:,c].shape)
            maximum[c]=value
            examples[c]={'run':name,'raw_frame':meta['raw_frame_records'][t],
              'node_indices':pairs[q].tolist(),'node_ids':None,
              'field':audit['fields'][c],'unit':audit['field_units'][c],
              'values':[float(left[t,q,c]),float(right[t,q,c])],'abs_difference':value}
    del target
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    sources.append({'run':name,'bytes':after.st_size,'sha256':digest,'raw_frames':len(meta['raw_frame_records']),
      'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()})
ids=np.load(ROOT/'outputs/expanded-research/benchmarks/drop_test/doe/canonical/shared/node_ids_full.npy')
for example in examples:
    if example is not None:example['node_ids']=ids[example['node_indices']].tolist()
record={'status':'uninterpolated_FEM_derived_pair_check_completed','pairs':132,'cases':9,
  'scope':'Native actual ANIM times including t0 and retained terminal snapshot. C6 raw-frame cache uses the existing verified GPS/strain-projection definitions; these are not raw tensor components.',
  'field_names':audit['fields'],'field_units':audit['field_units'],'max_abs_difference_per_channel':maximum.tolist(),
  'pairs_nonzero_any_native_case_time_per_channel':nonzero.sum(axis=0).tolist(),'max_examples_per_channel':examples,
  'raw_cache_source_hashes':sources,'GPU_used':False,'cpu_threads_environment':1,
  'elapsed_seconds':time.perf_counter()-started,'peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None)}
(OUT/'native-pair-audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('status','max_abs_difference_per_channel','elapsed_seconds','peak_working_set_bytes')}))
