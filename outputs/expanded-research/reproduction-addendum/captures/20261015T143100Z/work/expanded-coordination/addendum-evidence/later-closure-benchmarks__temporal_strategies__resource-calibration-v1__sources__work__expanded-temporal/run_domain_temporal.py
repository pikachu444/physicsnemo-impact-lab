"""Streaming actual PhysicsNeMo temporal experiments for bumper / full phone.

--inspect uses NumPy only. Framework work requires an explicit --allocated flag.
Original pinned crash/drop wrapper classes remain unmodified. The data driver
uses one common geometric length and separately applies the shared field loss
normalization, so mixed physical units are never added to each other.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.metadata, inspect, json, os, sys, time

ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
sys.path.insert(0,str(ROOT/'work/expanded-ml'))
from benchmark_data import Cache, physical_metrics, verify_ready
import numpy as np
import run_temporal as original

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

class Domain:
    def __init__(self,name,manifest,normalization):
        self.name=name
        self.ready=verify_ready(manifest)
        self.cache=Cache(manifest)
        self.norm=json.loads(Path(normalization).read_text())
        self.normalization_sha256=sha(normalization)
        self.mean=np.array(self.norm['field_mean'],np.float32)
        self.std=np.array(self.norm['field_std'],np.float32)
        assert np.array_equal(self.std[:3],np.repeat(self.std[0],3))
        self.n,self.t,self.c=self.cache.shape
        assert self.t==10 and self.c==(7 if name=='bumper' else 6)
        for case in sum(self.cache.split.values(),[]):
            assert np.array_equal(self.cache.coordinates(case),self.cache.xyz), 'This driver requires a fixed physical reference geometry'
        self.units=self.cache.units;self.names=self.cache.names
        assert self.names[:3]==['displacement_x','displacement_y','displacement_z']
        self.origin=np.mean(self.cache.xyz,axis=0,dtype=np.float64).astype(np.float32)
        self.length=float(np.ptp(self.cache.xyz,axis=0).max())
        assert self.length>0
        self.coords=(np.asarray(self.cache.xyz)-self.origin)/self.length
        self.dt=10. if name=='bumper' else 2e-6
        self.velocity=np.asarray([-5.,0.,0.] if name=='bumper' else [-3200.900476,-3511.241739,-2618.285239],np.float32)
        np.testing.assert_allclose(self.cache.meta['time_values'],np.arange(1,11)*self.dt,rtol=1e-6)
        self.kinematics=self._stream_kinematic_stats()
        self.current=None

    def _stream_kinematic_stats(self):
        moments={name:[0,np.zeros(3),np.zeros(3)] for name in ['velocity','acceleration']}
        for case in self.cache.split['train']:
            u=self.cache.target(case)[...,:3]
            zero=np.zeros((self.n,1,3),np.float32)
            series=np.concatenate([zero,u],axis=1)/self.length
            vel=np.diff(series,axis=1)/self.dt
            acc=np.diff(vel,axis=1)/self.dt
            for key,value in [('velocity',vel),('acceleration',acc)]:
                count,mean,m2=moments[key]
                local_n=value.shape[0]*value.shape[1]
                local_mean=value.mean((0,1),dtype=np.float64)
                local_m2=value.var((0,1),dtype=np.float64)*local_n
                delta=local_mean-mean
                m2+=local_m2+delta**2*count*local_n/(count+local_n)
                mean+=delta*local_n/(count+local_n)
                moments[key]=[count+local_n,mean,m2]
        result={}
        for prefix,key in [('vel','velocity'),('acc','acceleration')]:
            count,mean,m2=moments[key]
            scale=float(np.sqrt(np.sum(m2/count)/3)) or 1.
            result[f'norm_{prefix}_mean']=mean.tolist()
            result[f'norm_{prefix}_std']=[scale]*3
        return result

    def contract(self):
        return {'domain':self.name,'cache_ready':self.ready,'shape':list(self.cache.shape),'split':self.cache.split,
                'names':self.names,'units':self.units,'time_values':self.cache.meta['time_values'],
                'coordinate_length_mm':self.length,'coordinate_origin_mm':self.origin.tolist(),
                'float32_coordinate_ulp_scale_mm':self.length*float(np.finfo(np.float32).eps),
                'position_loss_scale':(self.length/self.std[:3]).tolist(),
                'initial_velocity':self.velocity.tolist(),'velocity_unit':'mm/ms' if self.name=='bumper' else 'mm/s',
                'dt':self.dt,'time_unit':'ms' if self.name=='bumper' else 's',
                'time_assumption':('nominal scalar dt10ms; actual animation jitter retained; frame correspondence without interpolation' if self.name=='bumper' else 'constant dt2us on labels linearly interpolated in physical units within actual saved-frame brackets; no extrapolation; original12frame provenance remains separate'),
                'time_input_encoding':'original wrapper future-frame index t/10: encoded0 corresponds to first future dt, encoded0.9 to final10dt; this is not physical t=0..1',
                'initial_velocity_basis':'new-domains-source-and-resource-plan.json; source INIVEL vector and explicit group support verified, never future target differences',
                'initial_velocity_scope_limit':'all131 RAD decks not independently inspected; Run224 plus family velocity metadata' if self.name=='bumper' else 't0 velocity array not stored in ANIM; source card is the evidence',
                'static_feature_name':self.cache.meta.get('static_feature_name','shell_thickness'),
                'boundary_mask_policy':'coupon fixed/moving masks are not reused; no explicit tied/contact boundary mask is encoded',
                'kinematic_stats':self.kinematics,'kinematic_statistics_scope':'train-only source-independent a2..a10; first acceleration excluded as in original datapipe',
                'normalization':self.norm,'normalization_sha256':self.normalization_sha256,
                'memory_policy':'stream train statistics and mmap labels; at most one current case target held on GPU',
                'scope':self.cache.meta.get('scope',self.cache.meta.get('prediction_scope','published bumper conditions only'))}

    def attach(self,device):
        self.device=device
        self.coords_t=torch.from_numpy(self.coords).to(device)
        self.stats_t={'node':{k:torch.tensor(v,dtype=torch.float32,device=device).reshape(1,3) for k,v in self.kinematics.items()}}
        self.loss_scale=torch.tensor(self.length/self.std[:3],device=device)

    def sample(self,dp,case,channels,time_index=None,poison=False):
        key=(case,channels)
        if self.current is None or self.current[0]!=key:
            y=self.cache.target(case)
            normalized=np.empty((self.n,self.t,channels),np.float32)
            normalized[...,:3]=self.coords[:,None,:]+y[...,:3]/self.length
            if channels>3:normalized[...,3:]=(y[...,3:]-self.mean[3:])/self.std[3:]
            static,global_values=self.cache.inputs(case)
            static=(np.asarray(static)-self.norm['thickness_mean'])/self.norm['thickness_std']
            gv=(global_values-np.asarray(self.norm['global_mean']))/np.asarray(self.norm['global_std'])
            inputs={'coords':self.coords_t,'features':torch.from_numpy(static.astype(np.float32)).to(self.device)}
            globals_t={k:torch.tensor(float(v),device=self.device) for k,v in zip(self.cache.meta['global_keys'],gv)}
            self.current=(key,torch.from_numpy(normalized).to(self.device),inputs,globals_t)
        _,target,inputs,globals_t=self.current
        inputs=dict(inputs)
        if poison:target=torch.zeros_like(target)
        if time_index is not None:
            inputs['time']=torch.tensor(time_index/self.t,device=self.device)
            target=target[:,time_index]
        return dp.SimSample(node_features=inputs,node_target=target,global_features=globals_t)

    def loss(self,pred,target):
        error=pred-target
        position_error=error[...,:3]*self.loss_scale
        if error.shape[-1]>3:error=torch.cat([position_error,error[...,3:]],dim=-1)
        else:error=position_error
        return torch.mean(error**2)

    def physical(self,pred):
        value=pred.detach().cpu().numpy()
        out=np.empty_like(value)
        out[...,:3]=(value[...,:3]-self.coords[:,None,:])*self.length
        if value.shape[-1]>3:out[...,3:]=value[...,3:]*self.std[3:]+self.mean[3:]
        return out

def build(data,strategy,device):
    dp,wr=original.official('crash' if data.name=='bumper' else 'drop_test')
    channels=data.c if strategy=='time_conditional' else 3
    config=dict(n_layers=2,n_hidden=48,n_head=4,mlp_ratio=2,slice_num=8,dropout=0.,use_te=False,time_input=False,
                functional_dim=5 if strategy=='time_conditional' else 4,out_dim=channels,geometry_dim=3,
                global_dim=len(data.cache.meta['global_keys']),include_local_features=False,structured_shape=None,
                plus=False,concrete_dropout=False,state_mixing_mode='weighted',attention_type='GALE',num_time_steps=11)
    if strategy!='time_conditional':config.update(dt=data.dt,initial_vel=torch.tensor(data.velocity/data.length,device=device).reshape(1,3))
    model=getattr(wr,original.STRATEGIES[strategy])(**config).to(device)
    serial={**config}
    if 'initial_vel' in serial:serial['initial_vel']=serial['initial_vel'].cpu().tolist()
    core=Path(inspect.getfile(wr.GeoTransolver))
    source={'wrapper_file':str(OWN/'sources'/('crash' if data.name=='bumper' else 'drop_test')/'rollout.py'),
            'core_class_file':str(core),'core_class_sha256':sha(core),'physicsnemo_version':importlib.metadata.version('nvidia-physicsnemo'),
            'driver_sha256':sha(Path(__file__))}
    source['wrapper_sha256']=sha(source['wrapper_file'])
    return model,dp,channels,serial,source

def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()

def evaluate(data,model,dp,channels,cases,keep=False,teacher=False,poison=False):
    model.train(teacher); losses=[]; predictions=[]; truths=[]
    with torch.no_grad():
        for case in cases:
            sample=data.sample(dp,case,channels,poison=poison)
            pred=model(sample,data.stats_t)
            assert torch.isfinite(pred).all() and pred.shape==(data.n,data.t,channels)
            losses.append(float(data.loss(pred,sample.node_target)))
            if keep:
                predictions.append(data.physical(pred))
                truths.append(data.cache.target(case)[...,:channels])
    return float(np.mean(losses)),np.stack(predictions) if keep else None,np.stack(truths) if keep else None

def run_probe(data,strategy,device):
    model,dp,c,config,source=build(data,strategy,device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    case=data.cache.split['train'][0]
    sample=data.sample(dp,case,c,time_index=4 if strategy=='time_conditional' else None)
    torch.cuda.reset_peak_memory_stats();times=[];losses=[]
    before={k:p.detach().clone() for k,p in model.named_parameters()}
    for i in range(3):
        model.train();opt.zero_grad(set_to_none=True);sync(device);start=time.perf_counter()
        pred=model(sample,data.stats_t);assert pred.shape==sample.node_target.shape
        loss=data.loss(pred,sample.node_target);assert torch.isfinite(loss)
        loss.backward();opt.step();sync(device)
        times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
    _,a,_=evaluate(data,model,dp,c,[case],keep=True)
    _,b,_=evaluate(data,model,dp,c,[case],keep=True,poison=True)
    assert np.array_equal(a,b)
    report={'status':'actual_FEM_forward_backward_optimizer_probe_only','domain':data.name,'strategy':strategy,
            'channels':c,'full_nodes':data.n,'steps':3,'seconds':times,'losses':losses,
            'changed_parameters':sum(not torch.equal(before[k],p.detach()) for k,p in model.named_parameters()),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rss_observed_bytes':psutil.Process().memory_info().rss,
            'inference_target_value_independence_max_abs':float(np.max(np.abs(a-b))),
            'contract':data.contract(),'constructor':config,'source':source,'torch':torch.__version__,
            'same_dataset_30_epoch_estimate_seconds_excluding_validation':float(np.median(times[1:]))*len(data.cache.split['train'])*30*(10 if strategy=='time_conditional' else 1)}
    write(OUT/data.name/strategy/'probe.json',report)
    print(json.dumps({k:report[k] for k in ['domain','strategy','channels','full_nodes','seconds','cuda_peak_allocated_bytes','rss_observed_bytes','same_dataset_30_epoch_estimate_seconds_excluding_validation']}),flush=True)

def train(data,strategy,epochs,device,pilot=False):
    model,dp,c,config,source=build(data,strategy,device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    target_dir=OUT/data.name/strategy/('pilot' if pilot else 'fit')
    checkpoint_dir=OWN/'domain-checkpoints'/data.name/strategy/('pilot' if pilot else 'fit')
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    history=[];best=float('inf');best_epoch=None;updates=0;rss=psutil.Process().memory_info().rss
    torch.cuda.reset_peak_memory_stats();sync(device);start=time.perf_counter()
    for epoch in range(1,epochs+1):
        cases=list(data.cache.split['train']);np.random.shuffle(cases);losses=[];epoch_start=time.perf_counter()
        model.train()
        for case in cases:
            indexes=list(range(10)) if strategy=='time_conditional' else [None]
            if strategy=='time_conditional':np.random.shuffle(indexes)
            for index in indexes:
                sample=data.sample(dp,case,c,time_index=index)
                opt.zero_grad(set_to_none=True);pred=model(sample,data.stats_t)
                assert pred.shape==sample.node_target.shape
                loss=data.loss(pred,sample.node_target);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                updates+=1;losses.append(float(loss.detach()))
            rss=max(rss,psutil.Process().memory_info().rss)
        val,_,_=evaluate(data,model,dp,c,data.cache.split['validation'])
        if val<best:
            best=val;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'contract':data.contract(),'constructor':config,'source':source,'best_epoch':epoch},checkpoint_dir/'best.pt')
        row={'epoch':epoch,'updates':updates,'training_mse':float(np.mean(losses)),'validation_rollout_mse':val,'epoch_seconds':time.perf_counter()-epoch_start}
        history.append(row);write(target_dir/'history.json',history);print(json.dumps({'domain':data.name,'strategy':strategy,**row}),flush=True)
    sync(device);training_seconds=time.perf_counter()-start
    report={'status':'actual_one_epoch_training_and_validation_only' if pilot else 'actual_training_and_heldout_inference',
            'domain':data.name,'strategy':strategy,'channels':c,'epochs':epochs,'updates':updates,'best_epoch':best_epoch,
            'best_validation_rollout_mse':best,'training_including_validation_seconds':training_seconds,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rss_observed_peak_bytes':rss,
            'contract':data.contract(),'constructor':config,'source':source,'torch':torch.__version__,
            'stress_strain_predicted':c>3,'test_used_for_selection':False,'pilot_test_values_read':False if pilot else None}
    if not pilot:
        checkpoint=torch.load(checkpoint_dir/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['state_dict']);sync(device);eval_start=time.perf_counter()
        test,p,t=evaluate(data,model,dp,c,data.cache.split['test'],keep=True)
        sync(device);report['heldout_inference_seconds']=time.perf_counter()-eval_start
        report['test_rollout_mse']=test
        report['metrics']=physical_metrics(p,t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
        mean=np.zeros(data.cache.shape[:2]+(c,),np.float64)
        for case in data.cache.split['train']:mean+=data.cache.target(case)[...,:c]/len(data.cache.split['train'])
        report['train_mean_baseline']=physical_metrics(np.broadcast_to(mean,t.shape),t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
        report['zero_baseline']=physical_metrics(np.zeros_like(t),t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
        np.savez_compressed(checkpoint_dir/'test-predictions.npz',prediction=p,truth=t)
        report['prediction_sha256']=sha(checkpoint_dir/'test-predictions.npz')
        if strategy=='one_step_teacher_forcing':
            _,tf,_=evaluate(data,model,dp,c,data.cache.split['test'],keep=True,teacher=True)
            report['teacher_forced_diagnostic_uses_FEM_history']=physical_metrics(tf,t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
            np.savez_compressed(checkpoint_dir/'teacher-forcing-predictions.npz',prediction=tf,truth=t)
    assert sha(data.cache.path)==data.ready['manifest_sha256'],'Frozen source manifest changed'
    report['checkpoint_sha256']=sha(checkpoint_dir/'best.pt')
    write(target_dir/'result.json',report)
    print(json.dumps({k:report[k] for k in ['domain','strategy','channels','epochs','updates','best_epoch','training_including_validation_seconds']}),flush=True)

def reload(data,strategy,device):
    directory=OWN/'domain-checkpoints'/data.name/strategy/'fit'
    checkpoint=torch.load(directory/'best.pt',map_location=device,weights_only=False)
    assert checkpoint['contract']==data.contract(),'Source/normalization/initial condition contract changed'
    model,dp,c,config,source=build(data,strategy,device)
    assert checkpoint['constructor']==config,'Model constructor changed'
    for key in ['wrapper_sha256','core_class_sha256','physicsnemo_version','driver_sha256']:
        assert checkpoint['source'][key]==source[key],f'Implementation source changed: {key}'
    model.load_state_dict(checkpoint['state_dict'])
    _,p,t=evaluate(data,model,dp,c,data.cache.split['test'],keep=True)
    old=np.load(directory/'test-predictions.npz')
    assert np.array_equal(old['truth'],t),'Reference values changed'
    normalized_difference=(old['prediction']-p)/data.std[:c]
    assert np.isfinite(normalized_difference).all()
    assert float(np.max(np.abs(normalized_difference)))<=1e-5,'Fresh inference changed beyond1e-5 train-standard-deviation units'
    report={'fresh_process':True,'domain':data.name,'strategy':strategy,'checkpoint_sha256':sha(directory/'best.pt'),
            'prediction_max_abs_difference':float(np.max(np.abs(old['prediction']-p))),
            'per_field_max_abs_difference':np.max(np.abs(old['prediction']-p),axis=(0,1,2)).tolist(),
            'truth_max_abs_difference':float(np.max(np.abs(old['truth']-t))),
            'max_abs_difference_in_training_scale_units':float(np.max(np.abs(normalized_difference))),
            'source_and_constructor_gate_passed':True,'reproduction_tolerance_in_training_scale_units':1e-5}
    write(OUT/data.name/strategy/'fit'/'fresh-process-reload.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=['bumper','phone_full'],required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--normalization',required=True)
    ap.add_argument('--strategy',choices=list(original.STRATEGIES),default='time_conditional')
    ap.add_argument('--mode',choices=['inspect','probe','pilot','train','reload'],required=True)
    ap.add_argument('--epochs',type=int,default=1);ap.add_argument('--allocated',action='store_true')
    args=ap.parse_args()
    if args.mode!='inspect' and not args.allocated:ap.error('Explicit root GPU/memory allocation required: --allocated')
    os.environ['PHYSICSNEMO_FORCE_TE']='False';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    data=Domain(args.domain,Path(args.manifest),Path(args.normalization))
    if args.mode=='inspect':
        write(OUT/args.domain/'prepared-data-contract.json',data.contract())
        print(json.dumps({'status':'numpy_contract_ready_no_framework_import','domain':args.domain,'shape':list(data.cache.shape)}));sys.exit(0)
    import torch,psutil
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(20261015);np.random.seed(20261015)
    device=torch.device('cuda');data.attach(device)
    if args.mode=='probe':run_probe(data,args.strategy,device)
    elif args.mode in ['train','pilot']:train(data,args.strategy,1 if args.mode=='pilot' else args.epochs,device,pilot=args.mode=='pilot')
    else:reload(data,args.strategy,device)
