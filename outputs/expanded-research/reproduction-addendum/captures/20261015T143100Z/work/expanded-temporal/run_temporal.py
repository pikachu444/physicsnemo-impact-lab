"""Genuine pinned PhysicsNeMo temporal wrappers on generated structural FEM data.

No framework imports occur until --allocated is explicitly supplied. Each run is
an independent process under the root orchestrator's exclusive GPU allocation.
The original rollout.py classes are loaded unchanged, without model substitutes.
"""
from __future__ import annotations
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, os, sys, time, traceback

ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
CACHE=ROOT/'work/expanded-structural-coupon/cache-initial13-nodal7'
STRATEGIES={'time_conditional':'GeoTransolverTimeConditional',
            'ar_rollout_training':'GeoTransolverAutoregressiveRolloutTraining',
            'one_step_teacher_forcing':'GeoTransolverOneStepRollout'}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod
def official(example):
    # rollout imports a sibling named datapipe. Bind that exact pinned sibling;
    # no fake SimSample/model/shim or monkey patch is used.
    directory=OWN/'sources'/example
    datapipe=load_module('datapipe',directory/'datapipe.py')
    wrapper=load_module(f'pinned_{example}_rollout',directory/'rollout.py')
    return datapipe,wrapper

def build(wrapper,strategy,channels,device):
    config=dict(n_layers=2,n_hidden=48,n_head=4,mlp_ratio=2,slice_num=8,
                dropout=0.,use_te=False,time_input=False,functional_dim=6 if strategy=='time_conditional' else 5,
                out_dim=channels,geometry_dim=3,global_dim=2,include_local_features=False,
                structured_shape=None,plus=False,concrete_dropout=False,
                state_mixing_mode='weighted',attention_type='GALE',num_time_steps=11)
    if strategy!='time_conditional':
        config.update(dt=1.,initial_vel=torch.zeros(1,3,device=device))
    model=getattr(wrapper,STRATEGIES[strategy])(**config).to(device)
    serial=dict(config)
    if 'initial_vel' in serial: serial['initial_vel']=serial['initial_vel'].cpu().tolist()
    return model,serial

class Coupon:
    def __init__(self,device):
        self.device=device
        self.meta=json.loads((CACHE/'manifest.json').read_text())
        ready=json.loads((CACHE/'READY.json').read_text())
        assert ready['status']=='ready' and ready['writer_exit_code']==0
        assert ready['manifest_sha256']==sha(CACHE/'manifest.json')
        for row in ready['files']:
            assert sha(CACHE/row['name'])==row['sha256'],row['name']
        self.manifest_sha=sha(CACHE/'manifest.json')
        self.stats=json.loads((OWN/'normalization.json').read_text())
        self.mean=np.asarray(self.stats['field_mean'],dtype=np.float32)
        self.std=np.asarray(self.stats['field_std'],dtype=np.float32)
        xyz=np.load(CACHE/'coordinates.npy').astype(np.float32)
        # Constant translation of coordinates uses component means; position
        # differences and every displacement loss use the shared vector scale.
        self.origin=xyz.mean(0,dtype=np.float64).astype(np.float32)+self.mean[:3]
        self.xyz=(xyz-self.origin)/self.std[:3]
        self.xyz_t=torch.from_numpy(self.xyz).to(device)
        self.boundary=torch.from_numpy(np.load(OWN/'boundary_features.npy')).to(device)
        train_u=np.stack([self.truth(c)[...,:3] for c in self.meta['split']['train']])
        zero=np.zeros(train_u.shape[:2]+(1,3),dtype=np.float32)
        u=np.concatenate([zero,train_u],axis=2)/self.std[:3]
        vel=np.diff(u,axis=2) # nominal dt=1ms, initial velocity zero
        acc=np.diff(vel,axis=2)
        vm=vel.mean((0,1,2),dtype=np.float64).astype(np.float32)
        am=acc.mean((0,1,2),dtype=np.float64).astype(np.float32)
        vs=float(np.sqrt(vel.var((0,1,2),dtype=np.float64).sum()/3)) or 1.
        ass=float(np.sqrt(acc.var((0,1,2),dtype=np.float64).sum()/3)) or 1.
        self.kinematic_stats={'node':{k:torch.tensor(v,dtype=torch.float32,device=device).reshape(1,3) for k,v in {
            'norm_vel_mean':vm,'norm_vel_std':[vs]*3,'norm_acc_mean':am,'norm_acc_std':[ass]*3}.items()}}
        self.kinematic_serial={k:v.cpu().tolist() for k,v in self.kinematic_stats['node'].items()}
        self.trained_cache={}
    def truth(self,case): return np.load(CACHE/self.meta['runs'][case]['target'])
    def sample(self,datapipe,case,channels,time_index=None,poison_target=False):
        # Cache train/val tensors only; final test read is deferred to evaluate.
        key=(case,channels)
        if key not in self.trained_cache:
            y=self.truth(case).astype(np.float32)
            target=np.empty_like(y[...,:channels])
            target[...,:3]=self.xyz[:,None,:]+y[...,:3]/self.std[:3]
            if channels>3:
                target[...,3:]=(y[...,3:]-self.mean[3:])/self.std[3:]
            self.trained_cache[key]=torch.from_numpy(target).to(self.device)
        target=self.trained_cache[key]
        if poison_target: target=torch.zeros_like(target)
        nf={'coords':self.xyz_t,'features':self.boundary}
        if time_index is not None:
            nf['time']=torch.tensor(time_index/10.,device=self.device)
            target=target[:,time_index]
        globals_value=np.asarray(self.meta['runs'][case]['globals'],np.float32)
        globals_value=(globals_value-np.asarray(self.stats['global_mean']))/np.asarray(self.stats['global_std'])
        g={key:torch.tensor(float(value),dtype=torch.float32,device=self.device) for key,value in zip(self.meta['global_keys'],globals_value)}
        return datapipe.SimSample(node_features=nf,node_target=target,global_features=g)
    def physical(self,output,channels):
        p=output.detach().cpu().numpy().astype(np.float32)
        out=np.empty_like(p)
        out[...,:3]=(p[...,:3]-self.xyz[:,None,:])*self.std[:3]
        if channels>3: out[...,3:]=p[...,3:]*self.std[3:]+self.mean[3:]
        return out

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize()

def probe(device,output):
    dataset=Coupon(device)
    rows=[]
    for example in ['crash','drop_test']:
        dp,wr=official(example)
        for strategy,channels in [('time_conditional',7),('ar_rollout_training',3),('one_step_teacher_forcing',3),
                                  ('ar_rollout_training',7),('one_step_teacher_forcing',7)]:
            row={'example':example,'strategy':strategy,'requested_target_channels':channels,'device':str(device)}
            model=None
            try:
                torch.manual_seed(20261015)
                # AR7 requests an actual seven-output head; this must not be
                # advertised as seven-channel support merely because AR3 runs.
                model,cfg=build(wr,strategy,channels,device)
                sample=dataset.sample(dp,dataset.meta['split']['train'][0],channels,
                                      time_index=4 if strategy=='time_conditional' else None)
                before={k:v.detach().clone() for k,v in model.named_parameters()}
                optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                synchronize(device); start=time.perf_counter()
                model.train(); pred=model(sample,dataset.kinematic_stats)
                loss=torch.mean((pred-sample.node_target)**2)
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); synchronize(device)
                row.update(status='forward_backward_update_pass',training_shape=list(pred.shape),
                           loss=float(loss.detach()),seconds=time.perf_counter()-start,
                           parameter_count=sum(p.numel() for p in model.parameters()),
                           changed_parameters=sum(not torch.equal(before[k],p.detach()) for k,p in model.named_parameters()),config=cfg)
                sample=dataset.sample(dp,dataset.meta['split']['train'][0],channels)
                assert 'time' not in sample.node_features
                model.eval()
                with torch.no_grad():
                    a=model(sample,dataset.kinematic_stats)
                    poisoned=dataset.sample(dp,dataset.meta['split']['train'][0],channels,poison_target=True)
                    b=model(poisoned,dataset.kinematic_stats)
                row.update(inference_shape=list(a.shape),inference_target_value_independence_max_abs=float((a-b).abs().max()),
                           input_time_key_mutated='time' in sample.node_features)
            except Exception as e:
                row.update(status='runtime_failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['traceback','config']}),flush=True)
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
    write(output,{'data_manifest_sha256':dataset.manifest_sha,'source_manifest_sha256':sha(OUT/'source-manifest.json'),
                  'torch':torch.__version__,'device':str(device),'probe_only_not_accuracy_validation':True,'rows':rows})

def evaluate(model,dp,dataset,cases,channels,teacher_forced=False):
    model.train(teacher_forced)
    outputs=[]; truth=[]; loss=0.
    with torch.no_grad():
        for case in cases:
            sample=dataset.sample(dp,case,channels)
            pred=model(sample,dataset.kinematic_stats)
            assert pred.shape==sample.node_target.shape and torch.isfinite(pred).all()
            loss+=float(torch.mean((pred-sample.node_target)**2))
            outputs.append(dataset.physical(pred,channels))
            truth.append(dataset.truth(case)[...,:channels])
    return np.stack(outputs),np.stack(truth),loss/len(cases)

def train(strategy,epochs,device):
    output=OUT/'coupon13'/strategy
    output.mkdir(parents=True,exist_ok=True)
    weights=OWN/'checkpoints'/strategy
    weights.mkdir(parents=True,exist_ok=True)
    dataset=Coupon(device); dp,wr=official('crash')
    channels=7 if strategy=='time_conditional' else 3
    torch.manual_seed(20261015); np.random.seed(20261015)
    model,config=build(wr,strategy,channels,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    history=[]; best=float('inf'); best_epoch=None; updates=0
    peak_rss=0; process=psutil.Process()
    synchronize(device); start=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        pairs=[(case,t) for case in dataset.meta['split']['train'] for t in (range(10) if strategy=='time_conditional' else [None])]
        np.random.shuffle(pairs)
        for case,t in pairs:
            sample=dataset.sample(dp,case,channels,time_index=t)
            optimizer.zero_grad(set_to_none=True)
            pred=model(sample,dataset.kinematic_stats)
            assert pred.shape==sample.node_target.shape
            loss=torch.mean((pred-sample.node_target)**2)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); updates+=1; losses.append(float(loss.detach()))
            peak_rss=max(peak_rss,process.memory_info().rss)
        _,_,validation=evaluate(model,dp,dataset,dataset.meta['split']['validation'],channels)
        row={'epoch':epoch,'training_mse':float(np.mean(losses)),'validation_rollout_mse':validation,'updates':updates}
        history.append(row)
        if validation<best:
            best=validation; best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'config':config,'strategy':strategy,'channels':channels,
                        'normalization':dataset.stats,'kinematic_stats':dataset.kinematic_serial,
                        'data_manifest_sha256':dataset.manifest_sha,'best_epoch':epoch},weights/'best.pt')
        write(output/'history.json',history)
        print(json.dumps(row),flush=True)
    synchronize(device); train_seconds=time.perf_counter()-start
    checkpoint=torch.load(weights/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    synchronize(device); eval_start=time.perf_counter()
    prediction,truth,testloss=evaluate(model,dp,dataset,dataset.meta['split']['test'],channels)
    synchronize(device); eval_seconds=time.perf_counter()-eval_start
    sys.path.insert(0,str(ROOT/'work/expanded-ml'))
    from benchmark_data import physical_metrics
    metrics=physical_metrics(prediction,truth,dataset.meta['field_names'][:channels],dataset.meta['field_units'][:channels],
                             dataset.meta['split']['test'],dataset.meta['time_values'])
    zero_metrics=physical_metrics(np.zeros_like(truth),truth,dataset.meta['field_names'][:channels],dataset.meta['field_units'][:channels],
                                 dataset.meta['split']['test'],dataset.meta['time_values'])
    mean_field=np.mean(np.stack([dataset.truth(c)[...,:channels] for c in dataset.meta['split']['train']]),axis=0)
    mean_metrics=physical_metrics(np.broadcast_to(mean_field,truth.shape),truth,dataset.meta['field_names'][:channels],dataset.meta['field_units'][:channels],
                                 dataset.meta['split']['test'],dataset.meta['time_values'])
    predictions_path=weights/'test-predictions.npz'
    np.savez_compressed(predictions_path,prediction=prediction,truth=truth,case_ids=dataset.meta['split']['test'],
                        field_names=dataset.meta['field_names'][:channels],field_units=dataset.meta['field_units'][:channels],
                        time_values_ms=dataset.meta['time_values'])
    teacher=None
    if strategy=='one_step_teacher_forcing':
        teacher_p,_,teacher_loss=evaluate(model,dp,dataset,dataset.meta['split']['test'],channels,teacher_forced=True)
        teacher=physical_metrics(teacher_p,truth,dataset.meta['field_names'][:channels],dataset.meta['field_units'][:channels],
                                dataset.meta['split']['test'],dataset.meta['time_values'])
        np.savez_compressed(weights/'teacher-forcing-test-predictions.npz',prediction=teacher_p,truth=truth)
        teacher={'diagnostic_only_uses_ground_truth_history':True,'normalized_mse':teacher_loss,'metrics':teacher}
    report={'strategy':strategy,'channels':channels,'trained':True,'full_mesh_nodes':919,'epochs':epochs,'updates':updates,
            'best_epoch':best_epoch,'best_validation_rollout_mse':best,'test_rollout_mse':testloss,
            'training_including_validation_seconds':train_seconds,'four_case_inference_seconds':eval_seconds,
            'parameter_count':sum(p.numel() for p in model.parameters()),'rss_observed_peak_bytes':peak_rss,
            'cuda_allocated_peak_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'torch':torch.__version__,'device':str(device),'data_manifest_sha256':dataset.manifest_sha,
            'source_rollout_sha256':sha(OWN/'sources/crash/rollout.py'),'driver_sha256':sha(Path(__file__)),
            'normalization':dataset.stats,'kinematic_stats':dataset.kinematic_serial,'constructor':config,
            'acceleration_statistics_scope':'train-only second differences a2..a10 from eleven positions; first acceleration a1 from zero-velocity backstep is not included, matching original statistics scope',
            'field_names':dataset.meta['field_names'][:channels],'field_units':dataset.meta['field_units'][:channels],
            'metrics':metrics,'zero_displacement_or_scalar_baseline':zero_metrics,'train_mean_field_baseline':mean_metrics,
            'teacher_forcing_diagnostic':teacher,'prediction_sha256':sha(predictions_path),'checkpoint_sha256':sha(weights/'best.pt'),
            'fem_units':'mm,ms,MPa,dimensionless; original source kg-mm-ms stressGPa converted by FEM producer',
            'time_assumption':'original wrapper constant nominal1ms; reference actual animation jitter <=0.0002ms preserved in source manifest; no interpolation',
            'time_encoding':'TC first future frame1ms encoded0; final10ms encoded.9, matching original data/wrapper index encoding',
            'initial_velocity':'exact0 at t0 all physical nodes; known rightgrip step boundary represented by source-derived mask and velocity_scale global',
            'scope':'five training FEM conditions plus four internal validation/four test interior conditions; small parameter interpolation experiment, not appliance validation',
            'stress_strain_prediction':channels==7,
            'official_class_unchanged':True,'research_driver_changes':['source-derived fixed/moving boundary masks','train-only common displacement/velocity/acceleration vector scales','dt1ms and zero initial velocity adapted to this FEM example','Adam small model instead of official six-layer1000epoch default'],
            'api_compatibility':'pinned main YAML activation_checkpointing kwargs are unsupported by installed PhysicsNeMo2.2.2 GeoTransolver; omitted, while original temporal wrapper step checkpointing remains intact',
            'test_selection_policy':'test target values first used after all epochs and validation-selected checkpoint loading',
            'accuracy_pass_threshold':'none predefined; report physical errors and compare baselines, do not infer success from finite loss'}
    write(output/'result.json',report)
    print(json.dumps({k:report[k] for k in ['strategy','channels','epochs','updates','best_epoch','training_including_validation_seconds','four_case_inference_seconds']}),flush=True)

def reload_check(strategy,device):
    weights=OWN/'checkpoints'/strategy
    ckpt=torch.load(weights/'best.pt',map_location=device,weights_only=False)
    dataset=Coupon(device); dp,wr=official('crash')
    assert dataset.manifest_sha==ckpt['data_manifest_sha256']
    assert dataset.stats==ckpt['normalization'], 'Training normalization changed'
    assert dataset.kinematic_serial==ckpt['kinematic_stats'], 'Training kinematic normalization changed'
    model,_=build(wr,strategy,ckpt['channels'],device)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    pred,truth,_=evaluate(model,dp,dataset,dataset.meta['split']['test'],ckpt['channels'])
    prior=np.load(weights/'test-predictions.npz')
    result={'fresh_process':True,'strategy':strategy,'checkpoint_sha256':sha(weights/'best.pt'),
            'prediction_max_abs_difference':float(np.max(np.abs(prior['prediction']-pred))),
            'truth_max_abs_difference':float(np.max(np.abs(prior['truth']-truth))),
            'prediction_reproducible_at_atol_1e-4':bool(np.allclose(prior['prediction'],pred,rtol=1e-6,atol=1e-4)),
            'physical_units':'mixed fields: per-field maximum differences below',
            'per_field_max_abs_difference':np.max(np.abs(prior['prediction']-pred),axis=(0,1,2)).tolist()}
    write(OUT/'coupon13'/strategy/'fresh-process-reload.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--allocated',action='store_true',help='root explicitly allocated framework/GPU memory slot')
    ap.add_argument('--mode',choices=['probe','train','reload','probe-train-all','reload-all'],required=True)
    ap.add_argument('--strategy',choices=list(STRATEGIES),default='time_conditional')
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    if not args.allocated: ap.error('Framework import requires root allocation: --allocated')
    os.environ['PHYSICSNEMO_FORCE_TE']='False'
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np
    import torch
    import psutil
    torch.set_num_threads(1)
    device=torch.device(args.device)
    if device.type=='cuda': assert torch.cuda.is_available()
    try:
        if args.mode=='probe-train-all':
            import gc
            probe(device,OUT/f'official-wrapper-probe-{args.device}.json')
            for strategy in STRATEGIES:
                train(strategy,args.epochs,device)
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
        elif args.mode=='reload-all':
            import gc
            for strategy in STRATEGIES:
                reload_check(strategy,device)
                gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
        elif args.mode=='probe': probe(device,OUT/f'official-wrapper-probe-{args.device}.json')
        elif args.mode=='train': train(args.strategy,args.epochs,device)
        else: reload_check(args.strategy,device)
    except Exception:
        write(OUT/f'failure-{args.mode}-{args.strategy}-{args.device}.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'traceback':traceback.format_exc()})
        raise
