"""One guarded NumPy-only audit of six frozen plate predictions against source NPZ.

Run with CPython -S, one explicitly supplied NumPy package parent, no evaluator
imports. Sources and arrays are immutable. Temporary NPY members are mmap-read.
"""
import argparse
import contextlib
import ctypes
import datetime
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/expanded-research/benchmarks/deforming_plate'
FINAL = BASE / 'official32_fresh_final_corrected'
TRAIN = BASE / 'official32_valpilot'
PREFIX = 'official32-streaming-numeric-audit-v1'
REVIEW = BASE / 'review'
MiB = 1024**2
started = time.perf_counter()
np = None
checks, failures, hashes = [], [], {}
context = {'stage': 'stdlib_setup'}
resource = {'rss_peak_bytes': 0, 'os_peak_working_set_bytes': 0,
            'host_available_min_bytes': None, 'guard_observations': 0}
report = {'status': 'PREPARING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'cases': [], 'completed_case_count': 0, 'checks': checks, 'failures': failures,
          'resource_limits': {'preflight_host_available_bytes': 640*MiB, 'host_available_floor_bytes': 512*MiB,
                              'RSS_peak_cap_bytes': 128*MiB, 'wall_cap_seconds': 600},
          'maximum_io_chunk_bytes': MiB, 'cell_chunk': 128, 'NN_inferences': 0,
          'framework_imports': 0, 'source_files_modified': 0}

class GuardStop(Exception):
    pass

class MemoryStatus(ctypes.Structure):
    _fields_ = [('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
                ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]

class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in
                ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]

kernel = ctypes.WinDLL('kernel32',use_last_error=True)
psapi = ctypes.WinDLL('psapi',use_last_error=True)
kernel.GetCurrentProcess.restype = ctypes.c_void_p
handle = kernel.GetCurrentProcess()
kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
kernel.SetPriorityClass.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
report['CPU_affinity_single_core_applied'] = bool(kernel.SetProcessAffinityMask(handle,1))
report['below_normal_priority_applied'] = bool(kernel.SetPriorityClass(handle,0x4000))

def guard(preflight=False):
    m=MemoryStatus();m.dwLength=ctypes.sizeof(m)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('GlobalMemoryStatusEx failed')
    p=ProcessMemory();p.cb=ctypes.sizeof(p)
    if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(p),p.cb):raise OSError('GetProcessMemoryInfo failed')
    available=int(m.ullAvailPhys)
    resource['guard_observations']+=1
    resource['rss_peak_bytes']=max(resource['rss_peak_bytes'],int(p.WorkingSetSize))
    resource['os_peak_working_set_bytes']=max(resource['os_peak_working_set_bytes'],int(p.PeakWorkingSetSize))
    resource['host_available_min_bytes']=available if resource['host_available_min_bytes'] is None else min(resource['host_available_min_bytes'],available)
    if preflight:report['preflight']={'available_bytes':available,'rss_bytes':int(p.WorkingSetSize),
                                    'threshold_bytes':640*MiB,'passed':available>=640*MiB}
    reason = ('preflight RAM below640MiB' if preflight and available<640*MiB else
              'host RAM below512MiB' if available<512*MiB else
              'OS peak RSS above128MiB' if p.PeakWorkingSetSize>128*MiB else
              'audit wall above600s' if time.perf_counter()-started>600 else None)
    if reason:raise GuardStop(reason)

def check(name, passed, details=None):
    r={'name':name,'passed':bool(passed)}
    if details is not None:r['details']=details
    checks.append(r)
    if not passed:failures.append(r)

def read(path):
    guard()
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        while b:=stream.read(MiB):
            h.update(b);guard()
    v=h.hexdigest();hashes[str(pathlib.Path(path).relative_to(ROOT))]=v
    return v

def json_close(name, got, expected, rtol=3e-6, atol=1e-9):
    ok=(got is None and expected is None) or (got is not None and expected is not None and math.isclose(got,expected,rel_tol=rtol,abs_tol=atol))
    check(name,ok,{'independent':got,'recorded':expected,'rtol':rtol,'atol':atol})

def extract_selected(archive, folder, names):
    """Never extract arbitrary archive paths; only exact whitelisted .npy names."""
    folder.mkdir(parents=True,exist_ok=False)
    result={}
    with zipfile.ZipFile(archive) as z:
        listing=z.namelist()
        for name in names:
            guard();member=name+'.npy'
            assert listing.count(member)==1 and '/' not in member and '\\' not in member
            dest=folder/member
            assert dest.resolve().parent==folder.resolve()
            with z.open(member) as src,dest.open('xb') as dst:
                while b:=src.read(65536):dst.write(b);guard()
            result[name]=dest
    return result

@contextlib.contextmanager
def mapped(path):
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    assert isinstance(a,np.memmap) and not a.dtype.hasobject
    try:yield a
    finally:
        a._mmap.close()

def edges(position,cells):
    t=np.asarray(position,dtype=np.float64)[cells]
    return (t[:,1:]-t[:,:1]).transpose(0,2,1)

def determinant(a):
    # Explicit scalar cofactor identity; independent of the production np.linalg.det.
    return (a[:,0,0]*(a[:,1,1]*a[:,2,2]-a[:,1,2]*a[:,2,1])
            -a[:,0,1]*(a[:,1,0]*a[:,2,2]-a[:,1,2]*a[:,2,0])
            +a[:,0,2]*(a[:,1,0]*a[:,2,1]-a[:,1,1]*a[:,2,0]))

def inverse(a):
    # Rows of inverse are cross-products of the three column vectors / det.
    c0,c1,c2=a[:,:,0],a[:,:,1],a[:,:,2]
    d=determinant(a)
    assert np.all(np.abs(d)>1e-15)
    return np.stack([np.cross(c1,c2),np.cross(c2,c0),np.cross(c0,c1)],axis=1)/d[:,None,None]

def green_lagrange(ds,inv_dm):
    f=np.einsum('eik,ekj->eij',ds,inv_dm,optimize=False)
    c=np.einsum('eki,ekj->eij',f,f,optimize=False)
    e=(c-np.eye(3))*0.5
    return e[:,[0,1,2,0,1,0],[0,1,2,1,2,2]].astype(np.float32)

def synthetic():
    ref=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
    cells=np.array([[0,1,2,3]])
    dm=edges(ref,cells);iv=inverse(dm)
    transforms=[('identity',ref,np.zeros(6),1),('translation',ref+[4,5,6],np.zeros(6),1),
                ('rigid_rotation',ref@np.array([[0,-1,0],[1,0,0],[0,0,1]]).T,np.zeros(6),1),
                ('known_stretch',ref@np.diag([1.2,1,1]),np.array([.22,0,0,0,0,0]),1),
                ('negative_det_reflection',ref@np.diag([-1,1,1]),np.zeros(6),-1)]
    for name,current,expected,sign in transforms:
        ds=edges(current,cells);g=green_lagrange(ds,iv)
        check('synthetic '+name,bool(np.allclose(g[0],expected,rtol=1e-6,atol=1e-7)) and int(np.sign(determinant(ds)[0]/determinant(dm)[0]))==sign)

def new_stats():return {'sse':0.,'truth_sum_square':0.,'values':0,'max_abs_error':0.}

def add_stats(s,p,t):
    pp=np.asarray(p,dtype=np.float64);tt=np.asarray(t,dtype=np.float64)
    if not (np.isfinite(pp).all() and np.isfinite(tt).all()):raise ValueError('Nonfinite numerical field')
    d=pp-tt
    s['sse']+=float(np.sum(d*d,dtype=np.float64));s['truth_sum_square']+=float(np.sum(tt*tt,dtype=np.float64))
    s['values']+=int(tt.size);s['max_abs_error']=max(s['max_abs_error'],float(np.max(np.abs(d))))

def finish_stats(s):
    return {**s,'relative_l2':math.sqrt(s['sse']/s['truth_sum_square']) if s['truth_sum_square'] else None,
            'component_rmse':math.sqrt(s['sse']/s['values']),'truth_l2':math.sqrt(s['truth_sum_square'])}

def compare_stats(prefix,s,target):
    check(prefix+' count',s['values']==target['values'])
    for k in ['relative_l2','component_rmse','truth_l2']:
        if k in target:json_close(prefix+' '+k,s[k],target[k])
    json_close(prefix+' max',s['max_abs_error'],target['max_abs_component_error'])

def maxdiff(a,b):
    return float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))

def case_audit(case,folder,gate,closure,norm,aggregate):
    context.update(stage='hash_source_and_saved_prediction',case=case,frame=None,cell_start=None)
    source=ROOT/'work/expanded-plate/data'/(case+'.npz')
    saved=FINAL/(case+'_evaluation.npz')
    check(case+' source NPZ hash',sha(source)==gate['fresh_expected_npz_sha256'][case])
    check(case+' saved NPZ hash',sha(saved)==closure['saved_prediction_NPZ_sha256'][saved.name])
    src=extract_selected(source,folder/'source',['world_pos','stress','mesh_pos','cells','node_type'])
    pred=extract_selected(saved,folder/'saved',['prediction_positions','truth_positions','prediction_stress','truth_stress',
           'reference_positions','free_mask','cells','predicted_derived_strain','truth_derived_strain','strain_frames',
           'original_array_tetrahedron_indices_not_solver_ids'])
    target=read(FINAL/(case+'_metrics.json'))
    with contextlib.ExitStack() as stack:
        ss={n:stack.enter_context(mapped(p)) for n,p in src.items()}
        pp={n:stack.enter_context(mapped(p)) for n,p in pred.items()}
        source_cells=np.array(ss['cells'][0],dtype=np.int64)
        mesh=np.array(ss['mesh_pos'][0],dtype=np.float64)
        types=np.array(ss['node_type'][0,:,0])
        mask=types==0
        cell_mask=np.all(types[source_cells]!=1,axis=1)
        cell_ids=np.flatnonzero(cell_mask);cells=source_cells[cell_mask]
        ref=np.array(ss['world_pos'][0],copy=True)
        n=len(ref);e=len(cells)
        assert ss['world_pos'].shape==(400,n,3) and ss['stress'].shape==(400,n,1)
        assert pp['prediction_positions'].shape==(399,n,3) and pp['prediction_stress'].shape==(399,n,1)
        assert pp['truth_positions'].shape==(399,n,3) and pp['truth_stress'].shape==(399,n,1)
        check(case+' source mesh/cells finite and supported',np.isfinite(mesh).all() and cells.min()>=0 and cells.max()<n)
        check(case+' exact node mask/cell mapping/reference',np.array_equal(mask,pp['free_mask']) and np.array_equal(cells,pp['cells']) and np.array_equal(cell_ids,pp['original_array_tetrahedron_indices_not_solver_ids']) and np.array_equal(ref,pp['reference_positions']))
        strain_frames=list(range(0,399,20))+[398]
        check(case+' GL21 alignment',np.array_equal(pp['strain_frames'],strain_frames) and target['derived_GL_strain_frames']==[f+1 for f in strain_frames] and pp['predicted_derived_strain'].shape==(21,e,6) and pp['truth_derived_strain'].shape==(21,e,6))
        accum={f:new_stats() for f in ['rollout_free_displacement','rollout_free_stress','derived_GL_strain',
                'zero_increment_baseline_free_displacement','zero_stress_baseline','training_mean_constant_stress_baseline']}
        pred_inv=[];true_inv=[];pred_det=[];true_det=[]
        raw_truth_delta=boundary_delta=glp_delta=glt_delta=0.
        glp_close=glt_close=True;negative=truth_negative=0
        min_stress=math.inf;max_truth_stress=-math.inf
        for frame in range(399):
            context.update(stage='frame_numeric',frame=frame+1,cell_start=None);guard()
            ppos=np.array(pp['prediction_positions'][frame],copy=True)
            tpos=np.array(ss['world_pos'][frame+1],copy=True)
            pstress=np.array(pp['prediction_stress'][frame],copy=True)
            tstress=np.array(ss['stress'][frame+1],copy=True)
            assert all(np.isfinite(x).all() for x in [ppos,tpos,pstress,tstress])
            raw_truth_delta=max(raw_truth_delta,maxdiff(pp['truth_positions'][frame],tpos),maxdiff(pp['truth_stress'][frame],tstress))
            if (~mask).any():boundary_delta=max(boundary_delta,maxdiff(ppos[~mask],tpos[~mask]))
            # Preserve source float32 subtraction before metric float64 accumulation.
            pu=ppos[mask]-ref[mask];tu=tpos[mask]-ref[mask]
            add_stats(accum['rollout_free_displacement'],pu,tu)
            add_stats(accum['rollout_free_stress'],pstress[mask],tstress[mask])
            add_stats(accum['zero_increment_baseline_free_displacement'],np.zeros_like(tu),tu)
            add_stats(accum['zero_stress_baseline'],np.zeros_like(tstress[mask]),tstress[mask])
            add_stats(accum['training_mean_constant_stress_baseline'],np.full_like(tstress[mask],norm['target_mean'][3]),tstress[mask])
            negative+=int(np.count_nonzero(pstress[mask]<0));truth_negative+=int(np.count_nonzero(tstress[mask]<0))
            min_stress=min(min_stress,float(pstress[mask].min()));max_truth_stress=max(max_truth_stress,float(tstress[mask].max()))
            ni=nj=0;mi=mj=math.inf
            sample_index=strain_frames.index(frame) if frame in strain_frames else None
            for begin in range(0,e,128):
                context['cell_start']=begin;guard()
                chunk=cells[begin:begin+128]
                dm=edges(mesh,chunk);dp=edges(ppos,chunk);dt=edges(tpos,chunk)
                dr=determinant(dm);assert np.all(np.abs(dr)>1e-15)
                pd=determinant(dp)/dr;td=determinant(dt)/dr
                assert np.isfinite(pd).all() and np.isfinite(td).all()
                ni+=int(np.count_nonzero(pd<=0));nj+=int(np.count_nonzero(td<=0));mi=min(mi,float(pd.min()));mj=min(mj,float(td.min()))
                if sample_index is not None:
                    iv=inverse(dm);gp=green_lagrange(dp,iv);gt=green_lagrange(dt,iv)
                    sp=np.array(pp['predicted_derived_strain'][sample_index,begin:begin+128],copy=True)
                    st=np.array(pp['truth_derived_strain'][sample_index,begin:begin+128],copy=True)
                    glp_delta=max(glp_delta,maxdiff(gp,sp));glt_delta=max(glt_delta,maxdiff(gt,st))
                    glp_close=glp_close and bool(np.allclose(gp,sp,rtol=3e-6,atol=1e-8));glt_close=glt_close and bool(np.allclose(gt,st,rtol=3e-6,atol=1e-8))
                    add_stats(accum['derived_GL_strain'],gp,gt)
            pred_inv.append(ni);true_inv.append(nj);pred_det.append(mi);true_det.append(mj)
            del ppos,tpos,pstress,tstress,pu,tu,dm,dp,dt,dr,pd,td
        check(case+' all399 saved truth vs source exact',raw_truth_delta==0,raw_truth_delta)
        check(case+' prescribed boundary all399 exact',boundary_delta==0,boundary_delta)
        check(case+' negative scalar stress counts',negative==target['predicted_negative_von_mises_values'] and truth_negative==target['truth_negative_von_mises_values'])
        json_close(case+' minimum predicted stress',min_stress,target['predicted_von_mises_minimum'])
        json_close(case+' maximum reference stress',max_truth_stress,target['truth_von_mises_maximum'])
        check(case+' GL tensor independent cofactor formula',glp_close and glt_close,{'prediction_max_abs_difference':glp_delta,'source_max_abs_difference':glt_delta,'rtol':3e-6,'atol':1e-8})
        check(case+' inversion counts pred/reference all399',pred_inv==target['all_frame_predicted_inverted_tetrahedra'] and true_inv==target['all_frame_truth_inverted_tetrahedra'])
        check(case+' minimum detF pred/reference all399',bool(np.allclose(pred_det,target['all_frame_predicted_minimum_detF'],rtol=3e-7,atol=1e-10)) and bool(np.allclose(true_det,target['all_frame_truth_minimum_detF'],rtol=3e-7,atol=1e-10)))
        finished={k:finish_stats(v) for k,v in accum.items()}
        for k,v in finished.items():
            compare_stats(case+' '+k,v,target[k])
            out=aggregate.setdefault(k,new_stats());out['sse']+=v['sse'];out['truth_sum_square']+=v['truth_sum_square'];out['values']+=v['values'];out['max_abs_error']=max(out['max_abs_error'],v['max_abs_error'])
        result={'case':case,'nodes':n,'free_nodes':int(mask.sum()),'audited_tetrahedra':e,'frames':399,'GL_frames':21,
                'metrics':finished,'predicted_inverted_frames':sum(v>0 for v in pred_inv),'reference_inverted_frames':sum(v>0 for v in true_inv),
                'predicted_negative_stress':negative,'reference_negative_stress':truth_negative,
                'saved_truth_vs_source_max_abs_difference':raw_truth_delta,'prescribed_boundary_max_abs_difference':boundary_delta,
                'GL_prediction_max_abs_difference':glp_delta,'GL_source_max_abs_difference':glt_delta}
    # Every mmap has closed; retain only compact numerical summaries between cases.
    for path in list(src.values())+list(pred.values()):
        assert path.resolve().is_relative_to(folder.resolve());path.unlink()
    return result

def main():
    global np
    args=argparse.ArgumentParser();args.add_argument('--numpy-site',required=True);a=args.parse_args()
    for suffix in ['.json','.md','.executed.py']:
        if (REVIEW/(PREFIX+suffix)).exists():raise FileExistsError('Preserve earlier attempt; no automatic retry')
    source=pathlib.Path(__file__).read_bytes()
    with (REVIEW/(PREFIX+'.executed.py')).open('xb') as stream:stream.write(source)
    report['auditor_source_sha256']=hashlib.sha256(source).hexdigest()
    report['python_flags']={'no_site':sys.flags.no_site,'isolated':sys.flags.isolated}
    try:
        assert sys.flags.no_site,'Python -S required'
        context['stage']='stdlib_preflight';guard(preflight=True)
        report['status']='RUNNING'
        site=pathlib.Path(a.numpy_site).resolve()
        assert (site/'numpy/__init__.py').is_file()
        sys.path.insert(0,str(site))
        for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
        import numpy as imported_numpy
        np=imported_numpy
        report['numpy_version']=np.__version__;report['numpy_file']=str(pathlib.Path(np.__file__).resolve())
        guard();assert not any(n in sys.modules for n in ['torch','physicsnemo','scipy','matplotlib'])
        synthetic();guard()
        gate=read(FINAL/'pre-evaluation-lock.json');closure=read(FINAL/'closure-provenance.json');norm=read(TRAIN/'normalization.json');metrics=read(FINAL/'metrics.json')
        lock=read(BASE/'official32-completion-source-lock-v2.json')
        check('source lock and checkpoint frozen',sha(BASE/'official32-completion-source-lock-v2.json')==gate['source_lock_sha256'] and sha(TRAIN/'checkpoint.pt')==gate['checkpoint_sha256'])
        for name,value in lock['source_sha256'].items():check('source '+name,sha(FINAL/('executed_'+name))==value)
        cases=[f'test_{i:04d}' for i in range(2,8)]
        assert gate['fresh_final_test']==cases
        temp=pathlib.Path(tempfile.mkdtemp(prefix='independent-stream-v1-',dir=ROOT/'work/expanded-plate'))
        report['temporary_root']=str(temp.relative_to(ROOT))
        pooled={}
        for case in cases:
            folder=temp/case;folder.mkdir(exist_ok=False)
            result=case_audit(case,folder,gate,closure,norm,pooled)
            report['cases'].append(result);report['completed_case_count']+=1;guard()
            print(json.dumps({'completed_case':case,'cases':report['completed_case_count'],'seconds':time.perf_counter()-started,'rss_peak_bytes':resource['os_peak_working_set_bytes']}),flush=True)
        report['pooled_metrics']={k:finish_stats(s) for k,s in pooled.items()}
        for k,s in report['pooled_metrics'].items():compare_stats('pooled '+k,s,metrics['pooled_metrics'][k])
        report['status']='PASS_SCOPED_SOURCE_AND_SAVED_FIELD_NUMERIC_AUDIT' if not failures else 'FAIL_NUMERIC_DIFFERENCES'
    except GuardStop as e:
        report.update(status='PARTIAL_RESOURCE_GUARD_STOP',stop_reason=str(e),stop_context=dict(context))
    except BaseException as e:
        report.update(status='FAILED_AUDIT_ATTEMPT',exception_type=type(e).__name__,exception=str(e),stop_context=dict(context))
    finally:
        report.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
                      resources=resource,source_and_array_hashes=hashes,
                      forbidden_modules_loaded=[n for n in ['torch','physicsnemo','scipy','matplotlib'] if n in sys.modules],
                      accuracy_qualified=False,
                      limitations=['Original source means downloaded DeepMind trajectory NPZ, not a newly solved COMSOL model or reconversion of TFRecord.',
                       'Array/node/tetrahedron indices are not solver IDs. Boundary nodes follow prescribed next source positions; only free nodes are predicted.',
                       'Stress is source nodal scalar von Mises in unconfirmed dataset units; no MPa conversion.',
                       'GL strain is independently derived from source/predicted coordinates on21 sampled frames, not native COMSOL strain.',
                       'Inversion is tetra detF<=0 relative to mesh_pos; predicted and source counts are separate. It is not a rupture/failure law.',
                       'No new neural inference, framework loading, source editing, plot or visual review. Numeric correspondence does not qualify model accuracy.',
                       'Streaming accumulation and independent cofactor matrix algebra may differ by floating-point rounding; declared metric and tensor tolerances are recorded.'])
        with (REVIEW/(PREFIX+'.json')).open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
        lines=['# Official-size plate independent streaming numerical audit','',report['status'],'',
               f"Completed cases: {report['completed_case_count']}/6; checks {len(checks)}; mismatches {len(failures)}.",
               f"Wall {report['wall_seconds']:.3f}s; OS peak working set {resource['os_peak_working_set_bytes']/MiB:.2f}MiB; minimum host available {resource['host_available_min_bytes']/MiB:.2f}MiB.",
               '','Source and saved prediction NPY members were accessed through guarded read-only mmap, one case and frame at a time. No neural inference or producer evaluator import.','',
               '|Case|u relative L2|source scalar stress relative L2|derived GL relative L2|predicted/reference inverted frames|','|---|---:|---:|---:|---:|']
        for c in report['cases']:
            m=c['metrics'];lines.append(f"|{c['case']}|{m['rollout_free_displacement']['relative_l2']:.9g}|{m['rollout_free_stress']['relative_l2']:.9g}|{m['derived_GL_strain']['relative_l2']:.9g}|{c['predicted_inverted_frames']}/{c['reference_inverted_frames']}|")
        lines+=['','Relative L2 values are fractions. This is numerical correspondence, not model accuracy acceptance.']
        if 'stop_reason' in report:lines+=['',report['stop_reason'],json.dumps(report['stop_context'])]
        if 'exception' in report:lines+=['',report['exception_type']+': '+report['exception'],json.dumps(report['stop_context'])]
        if failures:lines+=['','Mismatches:']+['- '+x['name'] for x in failures]
        lines+=['','Limits:']+['- '+s for s in report['limitations']]
        with (REVIEW/(PREFIX+'.md')).open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
        print(json.dumps({'status':report['status'],'completed_cases':report['completed_case_count'],'checks':len(checks),'mismatches':len(failures),'wall_seconds':report['wall_seconds'],'resources':resource,'stop_reason':report.get('stop_reason'),'exception':report.get('exception')}),flush=True)

if __name__=='__main__':main()
