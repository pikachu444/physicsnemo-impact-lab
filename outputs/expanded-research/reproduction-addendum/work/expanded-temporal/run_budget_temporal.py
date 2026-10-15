"""Compute-budget profile; prepared and original execution sources stay frozen.

TC samples one future frame per case in a balanced, seeded10-epoch cycle.
AR/TF retain the original full C3 rollout. Every validation uses all frames.
No test prediction exists in probe/pilot/budget/reload; final is an explicit,
single evaluation after root selects the frozen checkpoint using validation.
"""
from pathlib import Path
import argparse,gc,json,os,time
import numpy as np
import run_domain_temporal as prepared

ROOT,OWN,OUT=prepared.ROOT,prepared.OWN,prepared.OUT
EXPECTED_PREPARED_SHA='955f25cf30a173c578eaa620cce6a0fae1e3d89bf072b0e21c5177e3854f3013'
PROFILE='budget_v1_cyclic_time'

def phases(cases,seed=20261015):
    offsets=np.resize(np.arange(10),len(cases));np.random.default_rng(seed).shuffle(offsets)
    mapping={case:int(value) for case,value in zip(cases,offsets)}
    for case in cases:assert sorted((epoch+mapping[case])%10 for epoch in range(10))==list(range(10))
    counts=np.bincount(offsets,minlength=10);assert counts.max()-counts.min()<=1
    return mapping

def contract(data):
    result=data.contract()
    result['budget_profile']=PROFILE
    result['prepared_source_sha256']=prepared.sha(Path(prepared.__file__))
    result['budget_source_sha256']=prepared.sha(Path(__file__))
    if data.name=='phone_full':
        provenance=ROOT/'outputs/expanded-research/benchmarks/drop_test/common-cache-provenance-clarification.json'
        result['provenance_clarification']={'path':str(provenance.relative_to(ROOT)),'sha256':prepared.sha(provenance),
            'meaning':'Nine newly executed local OpenCourant FEM runs of the public phone deck; not a ready-made public result dataset. Fields interpolated within actual saved-frame brackets, no extrapolation. Source volume-length feature is not shell thickness.'}
    return result

def paths(data,strategy,stage='fit'):
    out=OUT/data.name/strategy/PROFILE/stage
    private=OWN/'domain-checkpoints'/data.name/strategy/PROFILE/stage
    out.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    return out,private

def fit(data,strategy,budget,pilot=False):
    out,private=paths(data,strategy,'pilot' if pilot else 'fit')
    phase=phases(data.cache.split['train'])
    model,dp,c,config,source=prepared.build(data,strategy,device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(20261015)
    exposures={case:[0]*10 for case in data.cache.split['train']}
    history=[];train_seconds=0.;val_seconds=0.;best=float('inf');best_epoch=None;updates=0;epoch=0;rss=0
    torch.cuda.reset_peak_memory_stats();prepared.sync(device);wallstart=time.perf_counter()
    while train_seconds<budget:
        epoch+=1;cases=list(data.cache.split['train']);rng.shuffle(cases);losses=[];epoch_complete=True;before=train_seconds
        model.train()
        for case in cases:
            if train_seconds>=budget and not pilot:epoch_complete=False;break
            index=(epoch-1+phase[case])%10 if strategy=='time_conditional' else None
            prepared.sync(device);stepstart=time.perf_counter()
            sample=data.sample(dp,case,c,time_index=index)
            opt.zero_grad(set_to_none=True);pred=model(sample,data.stats_t)
            assert pred.shape==sample.node_target.shape
            loss=data.loss(pred,sample.node_target);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();prepared.sync(device)
            train_seconds+=time.perf_counter()-stepstart;updates+=1;losses.append(float(loss.detach()))
            if index is None:exposures[case]=[value+1 for value in exposures[case]]
            else:exposures[case][index]+=1
            rss=max(rss,psutil.Process().memory_info().rss)
        prepared.sync(device);vstart=time.perf_counter()
        val,_,_=prepared.evaluate(data,model,dp,c,data.cache.split['validation'])
        prepared.sync(device);vtime=time.perf_counter()-vstart;val_seconds+=vtime
        if val<best:
            best=val;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'contract':contract(data),'constructor':config,'source':source,
                        'best_epoch':epoch,'profile':PROFILE,'phase_offsets':phase,'training_budget_seconds':budget},private/'best.pt')
        row={'epoch':epoch,'epoch_complete':epoch_complete,'training_mse':float(np.mean(losses)),
             'validation_all_time_mse':val,'updates':updates,'cumulative_train_seconds':train_seconds,
             'epoch_train_seconds':train_seconds-before,'epoch_validation_seconds':vtime}
        history.append(row);prepared.write(out/('pilot-history.json' if pilot else 'history.json'),history)
        print(json.dumps({'domain':data.name,'strategy':strategy,**row}),flush=True)
        if pilot:break
    # Reproduce checkpoint on validation before touching the held-out test.
    checkpoint=torch.load(private/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    _,p,t=prepared.evaluate(data,model,dp,c,data.cache.split['validation'],keep=True)
    np.savez_compressed(private/'validation-predictions.npz',prediction=p,truth=t)
    metric_args=(data.names[:c],data.units[:c],data.cache.split['validation'],data.cache.meta['time_values'])
    validation_metrics=prepared.physical_metrics(p,t,*metric_args)
    mean=np.zeros(data.cache.shape[:2]+(c,),np.float64)
    for case in data.cache.split['train']:mean+=data.cache.target(case)[...,:c]
    mean/=len(data.cache.split['train'])
    mean_metrics=prepared.physical_metrics(np.broadcast_to(mean,t.shape),t,*metric_args)
    zero_metrics=prepared.physical_metrics(np.zeros_like(t),t,*metric_args)
    ballistic=np.broadcast_to(data.velocity[None,None,None,:]*np.asarray(data.cache.meta['time_values'])[None,None,:,None],t[...,:3].shape)
    ballistic_metrics=prepared.physical_metrics(ballistic,t[...,:3],data.names[:3],data.units[:3],data.cache.split['validation'],data.cache.meta['time_values'])
    report={'status':'one_epoch_resource_pilot_validation_only' if pilot else 'budget_fit_validation_only_test_not_evaluated',
            'domain':data.name,'strategy':strategy,'channels':c,'profile':PROFILE,'epochs_started':epoch,'updates':updates,
            'intended_optimizer_training_budget_seconds':budget,'actual_optimizer_training_seconds':train_seconds,
            'all_time_validation_seconds':val_seconds,'total_wall_seconds':time.perf_counter()-wallstart,
            'budget_definition':'Includes per-update input transfer, forward/backward, gradient clip, optimizer and synchronization; excludes validation and one-off loading/import/setup. At most one optimizer update overshoot.',
            'pilot_budget_policy':'A resource pilot completes exactly one whole epoch even if it exceeds the nominal comparison budget; pilot artifacts are separate and never overwritten by the budget fit.',
            'best_validation_all_time_mse':best,'best_epoch':best_epoch,'latest_validation_all_time_mse':history[-1]['validation_all_time_mse'],
            'best_validation_physical_metrics':validation_metrics,'validation_train_mean_baseline':mean_metrics,
            'validation_zero_baseline':zero_metrics,'validation_ballistic_displacement_only_baseline':ballistic_metrics,
            'ballistic_scope':'Known source initial velocity multiplied by target time, no contact/deformation update and no stress or strain prediction; included because rigid translation can dominate global displacement.',
            'minimum_at_last_epoch_or_budget_boundary':best_epoch==epoch,'convergence_claimed':False,
            'test_predicted':False,'test_used_for_selection':False,'constructor':config,'source':source,'contract':contract(data),
            'time_sampling':{'definition':'TC one frame per case per sampled epoch, fixed seeded balanced offsets plus(epoch-1) modulo10; AR/TF full C3 trajectories unchanged',
                             'seed':20261015,'phase_offsets':phase,'per_case_time_label_exposures':exposures,
                             'sampled_epoch_interpretation':'Ten completed TC sampled epochs cover the same case-time pairs as one full-time epoch, once each. Weights change between SGD steps; this is not an identical optimization trajectory.',
                             'validation':'Every validation case, every stored future time; no temporal subsampling.'},
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rss_observed_peak_bytes':rss,
            'checkpoint_sha256':prepared.sha(private/'best.pt'),'validation_prediction_sha256':prepared.sha(private/'validation-predictions.npz'),
            'task_limit':'TC predicts all C7/C6 scalar fields. Original AR/TF output C3 displacement only; no stress/strain support is claimed.'}
    prepared.write(out/('pilot-result.json' if pilot else 'result.json'),report)
    print(json.dumps({k:report[k] for k in ['domain','strategy','channels','updates','actual_optimizer_training_seconds','all_time_validation_seconds','best_epoch','test_predicted']}),flush=True)

def load_selected(data,strategy):
    out,private=paths(data,strategy)
    ckpt=torch.load(private/'best.pt',map_location=device,weights_only=False)
    assert ckpt['contract']==contract(data),'Data/profile/source contract changed'
    model,dp,c,config,source=prepared.build(data,strategy,device)
    assert config==ckpt['constructor']
    for key in ['wrapper_sha256','core_class_sha256','physicsnemo_version','driver_sha256']:assert source[key]==ckpt['source'][key]
    model.load_state_dict(ckpt['state_dict'])
    return out,private,model,dp,c,ckpt

def reload_validation(data,strategy):
    out,private,model,dp,c,ckpt=load_selected(data,strategy)
    _,p,t=prepared.evaluate(data,model,dp,c,data.cache.split['validation'],keep=True)
    original=np.load(private/'validation-predictions.npz');assert np.array_equal(original['truth'],t)
    delta=np.abs((original['prediction']-p)/data.std[:c]);assert np.isfinite(delta).all() and delta.max()<=1e-5
    prepared.write(out/'fresh-process-validation-reload.json',{'domain':data.name,'strategy':strategy,'test_predicted':False,
        'fresh_process':True,'checkpoint_sha256':prepared.sha(private/'best.pt'),'source_constructor_contract_gate_passed':True,
        'prediction_max_abs_difference':float(np.abs(original['prediction']-p).max()),'max_abs_difference_in_training_scale_units':float(delta.max()),
        'per_field_physical_max_abs_difference':np.max(np.abs(original['prediction']-p),axis=(0,1,2)).tolist()})

def final_test(data,strategy):
    out,private,model,dp,c,ckpt=load_selected(data,strategy)
    assert not (out/'final-test.json').exists() and not (private/'test-predictions.npz').exists(),'Final evaluation already exists; never retune on test'
    prepared.sync(device);start=time.perf_counter();loss,p,t=prepared.evaluate(data,model,dp,c,data.cache.split['test'],keep=True);prepared.sync(device)
    inference_seconds=time.perf_counter()-start
    metrics=prepared.physical_metrics(p,t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
    np.savez_compressed(private/'test-predictions.npz',prediction=p,truth=t)
    prepared.write(out/'final-test.json',{'status':'single_final_heldout_evaluation_after_validation_checkpoint_selection',
        'domain':data.name,'strategy':strategy,'profile':PROFILE,'selected_epoch':ckpt['best_epoch'],'normalized_mse':loss,
        'inference_seconds':inference_seconds,'metrics':metrics,'contract':contract(data),
        'checkpoint_sha256':prepared.sha(private/'best.pt'),'prediction_sha256':prepared.sha(private/'test-predictions.npz'),
        'test_used_for_selection':False,'no_retuning_after_this_evaluation':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--domain',choices=['bumper','phone_full'],required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--normalization',required=True)
    ap.add_argument('--strategy',choices=list(prepared.original.STRATEGIES),default='time_conditional')
    ap.add_argument('--mode',choices=['inspect','probe','pilot','budget','reload','final'],required=True)
    ap.add_argument('--train-budget-seconds',type=float,default=180.);ap.add_argument('--allocated',action='store_true')
    args=ap.parse_args()
    assert prepared.sha(Path(prepared.__file__))==EXPECTED_PREPARED_SHA,'Frozen prepared source changed'
    assert args.train_budget_seconds>0
    if args.mode!='inspect' and not args.allocated:ap.error('Root GPU/memory allocation required')
    os.environ['PHYSICSNEMO_FORCE_TE']='False';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    data=prepared.Domain(args.domain,Path(args.manifest),Path(args.normalization))
    if args.mode=='inspect':
        prepared.write(OUT/args.domain/PROFILE/'prepared-contract.json',{'contract':contract(data),'phase_offsets':phases(data.cache.split['train']),
            'no_framework_import':True,'test_predicted':False});print(json.dumps({'domain':data.name,'status':'numpy_inspect_pass'}));raise SystemExit(0)
    import torch,psutil
    prepared.torch=torch;prepared.psutil=psutil
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(20261015)
    device=torch.device('cuda');data.attach(device)
    if args.mode=='probe':prepared.run_probe(data,args.strategy,device)
    elif args.mode in ['pilot','budget']:fit(data,args.strategy,args.train_budget_seconds,pilot=args.mode=='pilot')
    elif args.mode=='reload':reload_validation(data,args.strategy)
    else:final_test(data,args.strategy)
