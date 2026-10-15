"""One fixed first-fresh/final-frame plot; no model or producer evaluator import."""
import contextlib
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import sys
import sysconfig
import tempfile
import time
import zipfile

ROOT=pathlib.Path.cwd().resolve()
BASE=ROOT/'outputs/expanded-research/benchmarks/deforming_plate'
FINAL=BASE/'official32_fresh_final_corrected'
OUT=BASE/'review'
NAME='official32-first-fresh-field-comparison'
assert all(path.is_file() for path in (ROOT/'work/expanded-plate/data/test_0002.npz', FINAL/'test_0002_evaluation.npz', OUT/'official32-streaming-numeric-audit-v1.json')), 'Invoke from the research workspace with the original source data and saved predictions present.'
MiB=1024**2
start=time.perf_counter()
resources={'guard_observations':0,'rss_peak_bytes':0,'os_peak_working_set_bytes':0,'host_available_min_bytes':None}
report={'status':'PREPARING','layout_revision':3,'total_render_attempts_authorized':3,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case':'test_0002','source_loading_step':399,'selection':'First fresh split ID and last source frame, fixed by parent before seeing this plot; no error-based case/frame selection.',
        'new_neural_inference':0,'source_data_or_metrics_modified':0,'CPU_install_overlap':{
          'parent_advised_separate_CPU_install_may_start':True,'actual_overlap_verified':False,
          'qualification':'No process enumeration. This plot may overlap a separately authorized CPU-only installation; its actual interval is not established by this plotting receipt.'},
        'resource_limits':{'preflight_available_bytes':1024*MiB,'host_available_floor_bytes':512*MiB,'peak_RSS_cap_bytes':250*MiB,'wall_cap_seconds':180}}
stage='stdlib_setup'

class Stop(Exception):pass
class Mem(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
class Proc(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','commit','peak_commit']]
k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True)
k.GetCurrentProcess.restype=ctypes.c_void_p;handle=k.GetCurrentProcess()
k.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
p.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Proc),ctypes.c_ulong]
report['CPU_affinity_single_core_applied']=bool(k.SetProcessAffinityMask(handle,1))
report['below_normal_priority_applied']=bool(k.SetPriorityClass(handle,0x4000))
def guard(preflight=False):
    m=Mem();m.length=ctypes.sizeof(m);assert k.GlobalMemoryStatusEx(ctypes.byref(m))
    q=Proc();q.cb=ctypes.sizeof(q);assert p.GetProcessMemoryInfo(handle,ctypes.byref(q),q.cb)
    resources['guard_observations']+=1;resources['rss_peak_bytes']=max(resources['rss_peak_bytes'],q.ws);resources['os_peak_working_set_bytes']=max(resources['os_peak_working_set_bytes'],q.peak_ws)
    resources['host_available_min_bytes']=m.available if resources['host_available_min_bytes'] is None else min(resources['host_available_min_bytes'],m.available)
    if preflight:report['preflight']={'available_bytes':m.available,'rss_bytes':q.ws,'passed':m.available>=1024*MiB}
    if preflight and m.available<1024*MiB:raise Stop('Preflight available RAM below1GiB; no NumPy/Matplotlib/data load/render started')
    if m.available<512*MiB:raise Stop('Host available below512MiB')
    if q.peak_ws>250*MiB:raise Stop('OS peak working set above250MiB')
    if time.perf_counter()-start>180:raise Stop('Wall above180s')
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        while b:=f.read(MiB):h.update(b);guard()
    return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def extract(archive,folder,names):
    folder.mkdir()
    result={}
    with zipfile.ZipFile(archive) as z:
        listing=z.namelist()
        for name in names:
            member=name+'.npy';assert listing.count(member)==1 and '/' not in member and '\\' not in member
            dest=folder/member;assert dest.resolve().parent==folder.resolve()
            with z.open(member) as src,dest.open('xb') as dst:
                while b:=src.read(65536):dst.write(b);guard()
            result[name]=dest
    return result
@contextlib.contextmanager
def mm(path):
    a=np.load(path,mmap_mode='r',allow_pickle=False)
    try:yield a
    finally:a._mmap.close()
def edges(pos,cells):
    a=np.asarray(pos,dtype=np.float64)[cells]
    return (a[:,1:]-a[:,:1]).transpose(0,2,1)
def exx(reference,current,cells):
    a=edges(reference,cells);b=edges(current,cells)
    c0,c1,c2=a[:,:,0],a[:,:,1],a[:,:,2]
    det=np.einsum('ei,ei->e',c0,np.cross(c1,c2));assert np.all(abs(det)>1e-15)
    inv=np.stack([np.cross(c1,c2),np.cross(c2,c0),np.cross(c0,c1)],axis=1)/det[:,None,None]
    f0=np.einsum('eik,ek->ei',b,inv[:,:,0],optimize=False)
    return (.5*(np.einsum('ei,ei->e',f0,f0)-1)).astype(np.float32)
def range_of(a):return [float(a.min()),float(a.max())]

v1=OUT/'revisions/v1'
prior=read(v1/(NAME+'.json'))
assert hashlib.sha256((v1/(NAME+'.png')).read_bytes()).hexdigest()=='b383078b5a15af6ed58fb013503123748aa2200c6cf6de989ad27b604eb486a9'
assert hashlib.sha256((v1/(NAME+'.plot.py')).read_bytes()).hexdigest()=='5b57b61dee0ce501971b5c1d313fb1782196a85d8dda504fb48132ec6904d716'
assert hashlib.sha256((v1/(NAME+'.json')).read_bytes()).hexdigest()=='b53bfd3435e930d7a754ea3a79f29ac831ea3a84e1ea1f1450d38bb739f7758f'
v2=OUT/'revisions/v2'
prior_v2=read(v2/(NAME+'.json'))
assert hashlib.sha256((v2/(NAME+'.png')).read_bytes()).hexdigest()=='8631439b5b91e94fc4add638cd5f8d36fbbab3911b1387b1b35d45356fef4882'
assert hashlib.sha256((v2/(NAME+'.plot.py')).read_bytes()).hexdigest()=='051af7dc1aa4d76031625265066417a53d58f91c9c8056b0df7e0c92857778f8'
assert hashlib.sha256((v2/(NAME+'.json')).read_bytes()).hexdigest()=='d368d1ea16ef07f676e910224358a2741cb72f32fdc656045a26f523bdb37a2e'
assert read(OUT/(NAME+'.json')).get('layout_revision',1)==2,'Only one separately authorized v3 save-margin correction'
report['preserved_v1']={'PNG_sha256':prior['PNG_sha256'],'plot_source_sha256':prior['plot_source_sha256'],
                      'metadata_sha256':hashlib.sha256((v1/(NAME+'.json')).read_bytes()).hexdigest(),'folder':str(v1.relative_to(ROOT))}
report['preserved_v2']={'PNG_sha256':prior_v2['PNG_sha256'],'plot_source_sha256':prior_v2['plot_source_sha256'],
                      'metadata_sha256':hashlib.sha256((v2/(NAME+'.json')).read_bytes()).hexdigest(),'folder':str(v2.relative_to(ROOT))}
report['v3_layout_only_change']='Tight save bounding box with 0.15-inch padding; axes positions, labels, ranges, data, and masks unchanged from v2.'
fig=None
try:
    assert sys.flags.no_site,'Use CPython -S'
    report['plot_source_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    stage='stdlib_preflight';guard(preflight=True)
    temp=pathlib.Path(tempfile.mkdtemp(prefix='plate-first-fresh-plot-',dir=ROOT/'work/expanded-plate'))
    report['temporary_root']=str(temp.relative_to(ROOT))
    os.environ['MPLCONFIGDIR']=str(temp/'mplcache')
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
    sys.path[:0]=[sysconfig.get_path('purelib',scheme=sysconfig.get_preferred_scheme('user')),sysconfig.get_path('purelib')]
    import numpy as np
    stage='hash_and_data';guard()
    audit=read(OUT/'official32-streaming-numeric-audit-v1.json');assert audit['status']=='PASS_SCOPED_SOURCE_AND_SAVED_FIELD_NUMERIC_AUDIT'
    source=ROOT/'work/expanded-plate/data/test_0002.npz';saved=FINAL/'test_0002_evaluation.npz'
    source_hash=sha(source);saved_hash=sha(saved)
    assert source_hash==audit['source_and_array_hashes'][str(source.relative_to(ROOT))]
    assert saved_hash==audit['source_and_array_hashes'][str(saved.relative_to(ROOT))]
    report['source_NPZ_sha256']=source_hash;report['saved_prediction_NPZ_sha256']=saved_hash
    report['upstream_numeric_audit_sha256']=sha(OUT/'official32-streaming-numeric-audit-v1.json')
    source_files=extract(source,temp/'source',['world_pos','stress','mesh_pos','cells','node_type'])
    saved_files=extract(saved,temp/'saved',['prediction_positions','prediction_stress'])
    with contextlib.ExitStack() as stack:
        a={n:stack.enter_context(mm(path)) for n,path in source_files.items()}
        b={n:stack.enter_context(mm(path)) for n,path in saved_files.items()}
        reference=np.array(a['world_pos'][0],copy=True);truth=np.array(a['world_pos'][399],copy=True)
        pred=np.array(b['prediction_positions'][398],copy=True)
        free=np.array(a['node_type'][0,:,0]==0);all_cells=np.array(a['cells'][0],dtype=np.int64)
        cellmask=np.all(np.array(a['node_type'][0,:,0])[all_cells]!=1,axis=1)
        cells=all_cells[cellmask];mesh=np.array(a['mesh_pos'][0],dtype=np.float64)
        node_xy=reference[free,:2].astype(float);cell_xy=reference[cells].mean(axis=1)[:,:2].astype(float)
        u_source=np.sqrt(np.sum((truth[free].astype(float)-reference[free])**2,axis=1))
        u_pred=np.sqrt(np.sum((pred[free].astype(float)-reference[free])**2,axis=1))
        s_source=np.array(a['stress'][399,free,0],copy=True);s_pred=np.array(b['prediction_stress'][398,free,0],copy=True)
        g_source=exx(mesh,truth,cells);g_pred=exx(mesh,pred,cells)
    for path in list(source_files.values())+list(saved_files.values()):
        assert path.resolve().is_relative_to(temp.resolve());path.unlink()
    fields=[('Displacement magnitude |u|','unconfirmed coordinate units',node_xy,u_source,u_pred,'free nodes'),
            ('Native nodal von Mises stress','unconfirmed stress units',node_xy,s_source,s_pred,'free nodes'),
            ('Coordinate-derived GL E_xx','dimensionless; NOT native COMSOL strain',cell_xy,g_source,g_pred,'plate tetrahedra excluding actuator cells')]
    report['fields']=[]
    for name,units,xy,t,q,support in fields:
        assert np.isfinite(t).all() and np.isfinite(q).all()
        error=q.astype(float)-t
        low=min(float(t.min()),float(q.min()));high=max(float(t.max()),float(q.max()));bound=float(abs(error).max())
        report['fields'].append({'field':name,'units':units,'support':support,'support_count':len(t),'reference_range':range_of(t),'AI_range':range_of(q),'signed_error_range':range_of(error),
                                 'shared_reference_AI_color_range':[low,high],'error_symmetric_color_range':[-bound,bound],
                                 'XY_reference_bounds':[xy.min(axis=0).tolist(),xy.max(axis=0).tolist()],'color_normalization':'linear, full actual range; no clipping or data subsampling'})
    assert report['fields']==prior['fields'],'Layout-only correction must preserve every field range/support/XY bound'
    assert report['fields']==prior_v2['fields'],'Save-margin correction must preserve v2 field metadata exactly'
    assert source_hash==prior['source_NPZ_sha256'] and saved_hash==prior['saved_prediction_NPZ_sha256']
    report['v1_data_equivalence']={'field_ranges_supports_XY_and_normalization_exactly_equal':True,'source_and_saved_NPZ_hashes_equal':True,
                                   'case_frame_checkpoint_unchanged':True,'data_math_unchanged':True}
    report['v2_data_equivalence']={'field_ranges_supports_XY_and_normalization_exactly_equal':True,
                                  'source_and_saved_NPZ_hashes_equal':source_hash==prior_v2['source_NPZ_sha256'] and saved_hash==prior_v2['saved_prediction_NPZ_sha256'],
                                  'case_frame_checkpoint_unchanged':True,'data_math_and_axes_positions_unchanged':True}
    stage='matplotlib_import';guard()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    guard();report['numpy_version']=np.__version__;report['matplotlib_version']=matplotlib.__version__
    stage='render_3x3';fig,axes=plt.subplots(3,3,figsize=(9,12),dpi=140)
    fig.subplots_adjust(left=.085,right=.965,bottom=.15,top=.88,wspace=.30,hspace=.48)
    for row,(name,units,xy,t,q,support) in enumerate(fields):
        metadata=report['fields'][row];lo,hi=metadata['shared_reference_AI_color_range'];bound=metadata['error_symmetric_color_range'][1]
        if lo==hi:hi=lo+1
        if bound==0:bound=1
        for col,values in enumerate([t,q,q.astype(float)-t]):
            guard();ax=axes[row,col]
            norm=Normalize(vmin=lo,vmax=hi) if col<2 else Normalize(vmin=-bound,vmax=bound)
            artist=ax.scatter(xy[:,0],xy[:,1],c=values,s=5 if row<2 else 3,cmap='viridis' if col<2 else 'coolwarm',norm=norm,rasterized=True,linewidths=0)
            ax.set_aspect('equal',adjustable='box');ax.tick_params(labelsize=7)
            ax.set_xlabel('reference X',fontsize=8);ax.set_ylabel('reference Y' if col==0 else '',fontsize=8)
            ax.set_title(['Source FEM reference','Fixed AI prediction','Signed error: AI - source'][col] if row==0 else '',fontsize=9)
            cb=fig.colorbar(artist,ax=ax,pad=.025,fraction=.05)
            cb.locator=MaxNLocator(nbins=5);cb.formatter=FuncFormatter(lambda v,pos:format(v,'.3g'));cb.update_ticks();cb.ax.tick_params(labelsize=7)
        row_top=max(ax.get_position().y1 for ax in axes[row])
        fig.text(.085,row_top+.042,name+' — '+units,ha='left',va='bottom',fontsize=9)
    fig.suptitle('DeformingPlate: first fresh case test_0002 | source loading step 399\nFixed checkpoint epoch 5 | initial-reference XY projection',fontsize=11,y=.975)
    footer=('Each row shares the full source/AI color range; signed error uses a separate symmetric scale.\n'
            'Displacement/stress: free nodes. GL E_xx: plate tetrahedra, actuator cells excluded; initial cell-centre XY.\n'
            'GL is derived from coordinates, not a native COMSOL strain output. This single E_xx view differs from the six-component GL norm metric.\n'
            'Displacement-magnitude error differs from displacement-vector L2. Coordinate/stress units remain unconfirmed; no MPa conversion.\n'
            'First fresh case and final step fixed before rendering; no error-based selection, clipping, or model inference.')
    fig.text(.5,.025,footer,ha='center',va='bottom',fontsize=7,linespacing=1.45)
    guard();fig.savefig(OUT/(NAME+'.png'),dpi=140,facecolor='white',bbox_inches='tight',pad_inches=.15);plt.close(fig);fig=None;guard()
    report['PNG_sha256']=sha(OUT/(NAME+'.png'))
    with (OUT/(NAME+'.png')).open('rb') as stream:png_header=stream.read(24)
    assert png_header[:8]==b'\x89PNG\r\n\x1a\n' and png_header[12:16]==b'IHDR'
    report['PNG_pixels']=[int.from_bytes(png_header[16:20],'big'),int.from_bytes(png_header[20:24],'big')]
    report['status']='COMPLETED_SINGLE_FIXED_FIELD_FIGURE'
except Stop as exc:report.update(status='PARTIAL_RESOURCE_GUARD_STOP',stop_reason=str(exc),stop_stage=stage)
except BaseException as exc:report.update(status='FAILED_PLOT_ATTEMPT',exception_type=type(exc).__name__,exception=str(exc),stop_stage=stage)
finally:
    if fig is not None:
        try:plt.close(fig)
        except Exception:pass
    report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();report['wall_seconds']=time.perf_counter()-start;report['resources']=resources
    report['forbidden_modules_loaded']=[n for n in ['torch','physicsnemo','scipy','plate_evaluation','train_plate'] if n in sys.modules]
    report['visual_QA']='PNG must be opened after this single render; no automatic rerender'
    with (OUT/(NAME+'.json')).open('w',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'wall_seconds':report['wall_seconds'],'resources':resources,'PNG_sha256':report.get('PNG_sha256'),'stop_reason':report.get('stop_reason'),'exception':report.get('exception')}),flush=True)
