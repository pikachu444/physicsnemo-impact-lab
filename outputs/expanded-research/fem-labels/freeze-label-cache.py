"""Verify every label artifact and atomically commit the 131-case READY gate.

Run only after extract-ready-caches.py has actually returned exit code zero.
No training process may rely on individual case completion before this gate.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,default=ROOT/'work/expanded-fem-labels/cache')
    p.add_argument('--split-metrics',type=Path,default=ROOT/'outputs/experiment/fulltraining/metrics.json')
    p.add_argument('--writer-exit-code',type=int,required=True)
    p.add_argument('--selected-manifest',type=Path,default=ROOT/'work/expanded-fem-labels/bulk-inventory/selected-anim-manifest.json')
    p.add_argument('--report-dir',type=Path,default=OUT)
    a=p.parse_args();start=time.perf_counter()
    if a.writer_exit_code!=0:raise ValueError('Require actual successful writer exit code')
    ready=a.cache_root/'READY.json'
    if ready.exists():raise ValueError('Existing frozen READY must be preserved')
    split=json.loads(a.split_metrics.read_text())['split']
    expected=sorted(split['train']+split['validation']+split['test'])
    assert len(expected)==131 and len(set(expected))==131
    assert [len(split[k]) for k in ('train','validation','test')]==[112,12,7]
    extractor_sha=sha(OUT/'extract-anim-fields.py');direct_sha=sha(OUT/'direct-anim-reader.py')
    pinned=json.loads(a.selected_manifest.read_text())
    revision='78f1302fba4d75926c6fdd22d01d5133efc14742'
    assert pinned['revision']==revision and pinned['selected_run_count']==131 and pinned['selected_file_count']==1441
    required_shapes={'coords.npy':(13675,3),'node_id.npy':(13675,),'curated_node_indices.npy':(13675,),
      'thickness.npy':(13675,),'nodal_targets.npy':(13675,11,7),'time_ms.npy':(11,),'nominal_time_ms.npy':(11,),
      'shell_element_id.npy':(13626,),'shell_part_id.npy':(13626,),'shell_node_id.npy':(13626,4),
      'shell_initial_area_mm2.npy':(13626,),'shell_stress_local_inplane.npy':(11,13626,7,3),
      'shell_epsp.npy':(11,13626,7),'shell_animation_status.npy':(11,13626)}
    channels=['displacement_x','displacement_y','displacement_z','inplane_vm_upper','inplane_vm_lower','EPSP_upper','EPSP_lower']
    units=['mm']*3+['GPa']*2+['dimensionless']*2
    cases=[];totals=0;maxdisp=0.;maxtime=0.;statuszero=0;peak=0
    initial_ids=initial_xyz=None
    target_min=np.full(7,np.inf);target_max=np.full(7,-np.inf)
    for run in expected:
        folder=a.cache_root/run;mp=folder/'manifest.json';manifest_raw=mp.read_bytes();m=json.loads(manifest_raw)
        assert m['status']=='complete' and m['run']==run and m['schema_version']==2
        assert m['reader_backend']=='direct' and m['extractor_sha256']==extractor_sha and m['direct_reader_sha256']==direct_sha
        assert m['provenance']=='original_public_OpenRadioss_ANIM'
        assert m['dataset_revision']==revision and m['nodal_target_channels']==channels and m['nodal_target_units']==units
        expected_sources={Path(x['path']).name:(x['bytes'],x['lfs_sha256']) for x in pinned['files'] if x['run']==run}
        observed_sources={x['name']:(x['bytes'],x['sha256']) for x in m['source_files']}
        assert len(expected_sources)==11 and observed_sources==expected_sources,'Pinned LFS source mismatch '+run
        assert m['nodal_target_shape']==[13675,11,7] and m['real_shell_elements']==13626
        assert m['isolated_positive_thickness_nodes']==0
        assert m['projection_constant_test_max_abs_error']<1e-12
        files=m['output_files'];assert {x['name'] for x in files}==set(required_shapes)
        for item in files:
            f=folder/item['name'];assert f.is_file() and f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],str(f)
            arr=np.load(f,mmap_mode='r')
            assert arr.shape==required_shapes[item['name']],item['name']
            for i in range(0,len(arr),1024):assert np.isfinite(arr[i:i+1024]).all(),f.name
            totals+=item['bytes']
            del arr
        ids=np.load(folder/'node_id.npy');xyz=np.load(folder/'coords.npy')
        if initial_ids is None:initial_ids=ids;initial_xyz=xyz
        else:assert np.array_equal(ids,initial_ids) and np.array_equal(xyz,initial_xyz)
        times=np.load(folder/'time_ms.npy');nominal=np.load(folder/'nominal_time_ms.npy')
        assert times.shape==(11,) and times[0]==0 and np.all(np.diff(times)>0)
        assert np.array_equal(nominal,np.arange(11)*10)
        maxtime=max(maxtime,float(abs(times-nominal).max()))
        assert np.max(abs(times-nominal))<=.01,'Unexpected nominal-frame correspondence '+run
        target=np.load(folder/'nodal_targets.npy',mmap_mode='r');assert target.shape==(13675,11,7) and target.dtype==np.float32
        assert np.max(abs(target[:,0]))==0
        target_min=np.minimum(target_min,target.min(axis=(0,1)));target_max=np.maximum(target_max,target.max(axis=(0,1)))
        status=np.load(folder/'shell_animation_status.npy',mmap_mode='r');assert status.shape==(11,13626) and np.isin(status,[0,1]).all()
        statuszero+=int(np.sum(status==0));del target,status
        maxdisp=max(maxdisp,max(x['source_anim_vs_curated_displacement_component_rmse_mm'] for x in m['frame_checks']))
        peak=max(peak,m.get('python_peak_working_set_bytes') or 0)
        assert mp.read_bytes()==manifest_raw,'Concurrent writer modified '+run
        cases.append({'run':run,'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),'files':files})
    value={'status':'ready','expected_cases':131,'writer_exit_code':a.writer_exit_code,
      'dataset_revision':'78f1302fba4d75926c6fdd22d01d5133efc14742','split':split,
      'extractor_sha256':extractor_sha,'direct_reader_sha256':direct_sha,'cases':cases,
      'selected_source_manifest_sha256':sha(a.selected_manifest),'selected_source_bytes':pinned['selected_bytes'],
      'nodal_target_shape_per_case':[13675,11,7],
      'channels':['displacement_x','displacement_y','displacement_z','inplane_vm_upper','inplane_vm_lower','EPSP_upper','EPSP_lower'],
      'units':['mm']*3+['GPa']*2+['dimensionless']*2,
      'validation':{'artifact_bytes':totals,'all_artifact_sha256_verified':True,'all_arrays_finite':True,'same_reference_node_ids_coordinates_across_cases':True,
        'max_source_anim_vs_curated_displacement_component_rmse_mm':maxdisp,'max_actual_vs_nominal_time_delta_ms':maxtime,
        'status_zero_shell_frame_entries':statuszero,'target_min':target_min.tolist(),'target_max':target_max.tolist(),
        'max_reported_extractor_working_set_bytes':peak,'freeze_verification_seconds':time.perf_counter()-start},
      'frozen_utc':datetime.now(timezone.utc).isoformat(),
      'reader_contract':'Treat cache immutable after READY. Verify READY hash, case manifest hashes and required NPY hashes before training. Preserve nominal-frame matching and per-case actual times.'}
    temp=ready.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(ready)
    summary={k:v for k,v in value.items() if k not in ('cases','split')};summary['split_counts']={k:len(v) for k,v in split.items()};summary['READY_sha256']=sha(ready)
    a.report_dir.mkdir(parents=True,exist_ok=True)
    (a.report_dir/'dataset-build-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':'ready','cases':len(cases),'READY_sha256':sha(ready),'seconds':time.perf_counter()-start}))
if __name__=='__main__':main()
