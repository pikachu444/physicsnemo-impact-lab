"""V3 native scalar-head AR/TF profile, isolated from all frozen profiles.

Before NumPy, prepared data code, wrappers or frameworks import, the CLI checks
allocation and the externally frozen protocol/source hashes using only stdlib.
Probe/pilot/budget/reload never predict test labels; final is explicitly once.
"""
from pathlib import Path
import argparse,ast,gc,json,os,sys,time,math,hashlib,datetime

ROOT=Path(__file__).resolve().parents[2]
OWN=Path(__file__).resolve().parent
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
EXPECTED_PREPARED_SHA='955f25cf30a173c578eaa620cce6a0fae1e3d89bf072b0e21c5177e3854f3013'
PROFILE='multifield_v3_native_scalar_heads'
STRATEGIES=('ar_rollout_training','one_step_teacher_forcing')
RESOURCE_MODES=('inspect','probe','stream-probe')
FIT_MODES=('budget','reload','final')


def source_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def resolve_source(path):
    value=Path(path)
    return value.resolve() if value.is_absolute() else (ROOT/value).resolve()


def preflight(args):
    assert args.allocated,'Root GPU/memory allocation required before all data/framework imports'
    assert args.domain and args.manifest and args.normalization,'Runtime requires domain, manifest and normalization'
    assert args.protocol and args.protocol_sha256,'Externally frozen protocol path and SHA256 required'
    assert len(args.protocol_sha256)==64 and all(c in '0123456789abcdef' for c in args.protocol_sha256)
    protocol_path=Path(args.protocol).resolve()
    assert source_sha(protocol_path)==args.protocol_sha256,'Frozen protocol hash mismatch'
    protocol=json.loads(protocol_path.read_text(encoding='utf8'))
    assert protocol['profile']==PROFILE
    assert math.isfinite(args.train_budget_seconds) and args.train_budget_seconds>0
    assert args.mode!='pilot','R2 uses distinct-case stream probes; a public one-epoch pilot is forbidden'
    assert protocol['phase'] in ('resource_probes','fit')
    if protocol['phase']=='resource_probes':
        assert args.mode in RESOURCE_MODES,'All six functional and stream probes must precede any fit'
        assert protocol['training_budget_seconds'] is None,'Resource probes cannot claim an accepted training budget'
    else:
        assert args.mode in RESOURCE_MODES+FIT_MODES
        cap=protocol['training_budget_seconds']
        assert isinstance(cap,(int,float)) and math.isfinite(cap) and cap>0,'Fit protocol needs a finite positive common cap'
        if args.mode in FIT_MODES:
            assert args.train_budget_seconds==cap,'Training budget differs from frozen fit protocol'
    rows=protocol['sources'];assert isinstance(rows,list) and rows
    seen={}
    for row in rows:
        path=resolve_source(row['path']);assert path not in seen,'Duplicate source contract path'
        assert source_sha(path)==row['sha256'],f'Frozen source changed: {path}'
        seen[path]=row['sha256']
    for path in [Path(__file__).resolve(),OWN/'run_domain_temporal.py',OWN/'native_scalar_rollout_v3.py',
                 OWN/'multifield_protocol_guard.py',OWN/'profile_temporal_case_stream.py']:
        assert path.resolve() in seen,f'Required source absent from protocol: {path}'
    assert seen[(OWN/'run_domain_temporal.py').resolve()]==EXPECTED_PREPARED_SHA
    domain=protocol['domains'][args.domain]
    for key,arg in [('manifest',args.manifest),('normalization',args.normalization)]:
        path=Path(arg).resolve()
        assert path==resolve_source(domain[key+'_path']),f'{key} path differs from protocol'
        assert source_sha(path)==domain[key+'_sha256'],f'Frozen {key} changed'
    ready=Path(args.manifest).resolve().with_name('READY.json')
    assert ready==resolve_source(domain['ready_path']),'READY inventory path differs from protocol'
    assert source_sha(ready)==domain['ready_sha256'],'Frozen READY label inventory changed'
    assert domain['channels']==(7 if args.domain=='bumper' else 6)
    readiness=None
    if args.mode in FIT_MODES:
        from multifield_protocol_guard import verify_fit_readiness
        assert Path(sys.modules['multifield_protocol_guard'].__file__).resolve()==(OWN/'multifield_protocol_guard.py').resolve()
        readiness=verify_fit_readiness(protocol)
        assert isinstance(readiness,dict),'Readiness guard must return its verified receipt'
    return protocol_path,protocol,readiness


def source_selfcheck():
    source=Path(__file__).read_text(encoding='utf8');tree=ast.parse(source)
    modules=[]
    for node in tree.body:
        if isinstance(node,ast.Import):modules.extend(item.name.split('.')[0] for item in node.names)
        elif isinstance(node,ast.ImportFrom):modules.append(node.module.split('.')[0])
    assert set(modules)<={'pathlib','argparse','ast','gc','json','os','sys','time','math','hashlib','datetime'}
    assert source_sha(OWN/'run_domain_temporal.py')==EXPECTED_PREPARED_SHA
    wrapper=OWN/'native_scalar_rollout_v3.py'
    if wrapper.exists():ast.parse(wrapper.read_text(encoding='utf8'))
    for name in ['multifield_protocol_guard.py','profile_temporal_case_stream.py']:
        helper=OWN/name
        if helper.exists():ast.parse(helper.read_text(encoding='utf8'))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            assert not any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='prepared' and t.attr=='build' for t in targets)
    main=next(node for node in tree.body if isinstance(node,ast.If) and "__main__" in ast.unparse(node.test))
    text=ast.unparse(main)
    assert text.index('preflight(args)')<text.index('import numpy')<text.index('import run_domain_temporal')<text.index('import torch')
    gate=ast.unparse(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='preflight'))
    assert gate.index("source_sha(path) == row['sha256']")<gate.index('from multifield_protocol_guard import verify_fit_readiness')
    assert 'verify_fit_readiness(protocol)' in gate and "protocol['phase'] == 'resource_probes'" in gate
    assert not any(name.split('.')[0] in {'numpy','torch','physicsnemo'} for name in sys.modules)
    print(json.dumps({'status':'stdlib_source_selfcheck_pass','profile':PROFILE,'driver_sha256':source_sha(__file__),
        'prepared_source_sha256':EXPECTED_PREPARED_SHA,'no_data_or_framework_import':True,'no_cache_arrays_created':True}))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def field_diagnostics(prediction,truth,names):
    fields={}
    for i,name in enumerate(names):
        if 'plastic' not in name.lower() and 'epsp' not in name.lower():continue
        p,t=prediction[...,i].astype(np.float64),truth[...,i].astype(np.float64)
        positive=t>0;exactzero=t==0;negative=t<0;nonpositive=t<=0;ppositive=p>0;error=p-t
        tp=int(np.count_nonzero(positive&ppositive));fp=int(np.count_nonzero(~positive&ppositive));fn=int(np.count_nonzero(positive&~ppositive))
        def history(value):
            delta=np.diff(value,axis=2);decrease=-delta[delta<0]
            return {'strict_negative_increment_count':int(decrease.size),'minimum_delta':float(delta.min()),'increment_count':int(delta.size),
                    'maximum_time_decrease':float(decrease.max()) if decrease.size else 0.,
                    'decrease_magnitude_quantiles':{str(q):float(np.quantile(decrease,q)) for q in [.5,.9,.95,.99,1.]} if decrease.size else None}
        fields[name]={'reference_exact_positive_count':int(positive.sum()),'reference_exact_zero_count':int(exactzero.sum()),
                      'reference_negative_count':int(negative.sum()),'reference_nonpositive_support_count':int(nonpositive.sum()),
                      'prediction_negative_count':int(np.count_nonzero(p<0)),
                      'positive_support_rmse':float(np.sqrt(np.mean(error[positive]**2))) if positive.any() else None,
                      'zero_support_rmse':float(np.sqrt(np.mean(error[exactzero]**2))) if exactzero.any() else None,
                      'nonpositive_support_rmse':float(np.sqrt(np.mean(error[nonpositive]**2))) if nonpositive.any() else None,
                      'positive_support_precision':tp/(tp+fp) if tp+fp else None,'positive_support_recall':tp/(tp+fn) if tp+fn else None,
                      'positive_support_iou':tp/(tp+fp+fn) if tp+fp+fn else None,
                      'prediction_history':history(p),'reference_history':history(t),'float32_machine_epsilon':float(np.finfo(np.float32).eps),
                      'reference_absolute_peak_times_float32_epsilon':float(np.max(np.abs(t))*np.finfo(np.float32).eps)}
    return {'definition':'Posthoc raw fixed-node EPSP history and strict>0 support; same truth/prediction diagnostics, no clipping/correction/tuned threshold. Strict negative increments include roundoff; magnitude distribution retained.',
            'epsp_output_present':bool(fields),'fields':fields,'position_only_reason_if_empty':'No EPSP field in the requested output names' if not fields else None}

def contract(data):
    result=data.contract()
    result['budget_profile']=PROFILE
    result['prepared_source_sha256']=prepared.sha(Path(prepared.__file__))
    result['budget_source_sha256']=prepared.sha(Path(__file__))
    result['frozen_protocol']={'path':str(PROTOCOL_PATH),'sha256':PROTOCOL_SHA256}
    result['protocol_phase']=PROTOCOL['phase']
    result['fit_readiness_checked_before_data_import']=FIT_READINESS is not None
    result['native_scalar_adaptation']={'official_C3_wrappers_unmodified':True,
        'source_initial_velocity_divided_by_common_length':True,'scalar_feedback':False,
        'raw_head_layout':'rows0:3 normalized acceleration; rows3:C direct normalized scalar fields',
        'output_layout':'integrated normalized positions0:3 plus native normalized scalar fields3:C',
        'training_loss':'one joint normalized loss over all C6/C7 fields and all ten future times'}
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

def build(data,strategy,device):
    assert strategy in STRATEGIES and data.c in (6,7)
    dp,wr=prepared.original.official('crash' if data.name=='bumper' else 'drop_test')
    config=dict(n_layers=2,n_hidden=48,n_head=4,mlp_ratio=2,slice_num=8,dropout=0.,use_te=False,time_input=False,
                functional_dim=4,out_dim=data.c,geometry_dim=3,
                global_dim=len(data.cache.meta['global_keys']),include_local_features=False,structured_shape=None,
                plus=False,concrete_dropout=False,state_mixing_mode='weighted',attention_type='GALE',num_time_steps=11,
                dt=data.dt,initial_vel=torch.tensor(data.velocity/data.length,device=device).reshape(1,3))
    model=native_scalar.make_model(wr.GeoTransolver,strategy,**config).to(device)
    serial={**config,'initial_vel':config['initial_vel'].cpu().tolist()}
    core=Path(inspect.getfile(wr.GeoTransolver))
    assert core.resolve()==resolve_source(PROTOCOL['core_source_path']),'Imported GeoTransolver core path differs from frozen protocol'
    official=OWN/'sources'/('crash' if data.name=='bumper' else 'drop_test')/'rollout.py'
    wrapper=Path(native_scalar.__file__)
    frozen={resolve_source(row['path']):row['sha256'] for row in PROTOCOL['sources']}
    for path in [core,official,wrapper,Path(prepared.original.__file__)]:
        assert frozen[path.resolve()]==prepared.sha(path),f'Active model source differs from protocol: {path}'
    source={'wrapper_file':str(wrapper),'wrapper_sha256':prepared.sha(wrapper),
            'official_wrapper_file':str(official),'official_wrapper_sha256':prepared.sha(official),
            'core_class_file':str(core),'core_class_sha256':prepared.sha(core),
            'physicsnemo_version':importlib.metadata.version('nvidia-physicsnemo'),
            'driver_sha256':prepared.sha(Path(__file__)),'prepared_driver_sha256':EXPECTED_PREPARED_SHA,
            'protocol_file':str(PROTOCOL_PATH),'protocol_sha256':PROTOCOL_SHA256,
            'adaptation':'Separate native C6/C7 scalar-head wrapper over the installed official GeoTransolver core; official C3 wrapper bytes unchanged'}
    return model,dp,data.c,serial,source

def output_head(model,channels):
    block=model.ln_mlp_out[0]
    assert isinstance(block,torch.nn.Sequential) and len(block)==2
    assert isinstance(block[0],torch.nn.LayerNorm) and isinstance(block[1],torch.nn.Linear)
    head=block[1]
    assert head.in_features==48 and head.out_features==channels and head.bias is not None
    return head

def original_position_equivalence(data,strategy,model,dp,channels,sample):
    # This reference build is confined to the train-only probe, never a fit.
    original,original_dp,original_channels,_,original_source=prepared.build(data,strategy,device)
    assert original_channels==3
    reference_head=output_head(original,3);extended_head=output_head(model,channels)
    original_state=original.state_dict();extended_state=model.state_dict()
    assert original_state.keys()==extended_state.keys()
    head_name=next(name for name,module in original.named_modules() if module is reference_head)
    assert next(name for name,module in model.named_modules() if module is extended_head)==head_name
    widened={head_name+'.weight',head_name+'.bias'}
    copied=[]
    with torch.no_grad():
        for name,value in original_state.items():
            target=extended_state[name]
            if name in widened:
                assert value.shape[0]==3 and target.shape[0]==channels and value.shape[1:]==target.shape[1:]
                target[:3].copy_(value)
            else:
                assert value.shape==target.shape,f'Unexpected adaptation shape change: {name}'
                target.copy_(value)
            copied.append(name)
    original.eval();model.eval()
    original_sample=data.sample(original_dp,data.cache.split['train'][0],3)
    with torch.no_grad():
        reference=original(original_sample,data.stats_t)
        extended=model(sample,data.stats_t)[...,:3]
    assert reference.shape==extended.shape==(data.n,data.t,3)
    # Outputs are normalized positions; compare displacement difference / train field std.
    scale=torch.as_tensor(data.length/data.std[:3],device=device)
    delta=(extended-reference).abs()*scale
    assert torch.isfinite(delta).all() and float(delta.max())<=1e-5
    report={'original_channels':3,'extended_channels':channels,'first_train_case_only':True,
            'shared_state_tensors_copied':len(copied),'first_three_head_rows_copied':True,
            'scalar_head_rows_kept_own_initialization':True,'pre_training':True,
            'tolerance_in_training_scale_units':1e-5,'max_abs_difference_in_training_scale_units':float(delta.max()),
            'per_position_field_physical_max_abs_difference':((extended-reference).abs().amax(dim=(0,1))*data.length).cpu().tolist(),
            'original_source':original_source,'used_for_fit':False}
    del original,original_state,extended_state,reference,extended,original_sample
    gc.collect()
    return report

def run_probe(data,strategy):
    out,private=paths(data,strategy,'probe')
    assert not any(out.iterdir()) and not any(private.iterdir()),'Probe requires empty immutable evidence directories'
    model,dp,c,config,source=build(data,strategy,device)
    case=data.cache.split['train'][0];sample=data.sample(dp,case,c)
    assert sample.node_target.shape==(data.n,10,c)
    equivalence=original_position_equivalence(data,strategy,model,dp,c,sample)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    head=output_head(model,c);times=[];losses=[];gradient_rows=[];change_rows=[];bias_gradients=[];bias_changes=[];clipped_norms=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        model.train();opt.zero_grad(set_to_none=True);prepared.sync(device);start=time.perf_counter()
        before=head.weight.detach().clone();before_bias=head.bias.detach().clone()
        pred=model(sample,data.stats_t)
        assert pred.shape==(data.n,10,c) and torch.isfinite(pred).all()
        loss=data.loss(pred,sample.node_target);assert torch.isfinite(loss)
        loss.backward()
        assert head.weight.grad is not None and head.bias.grad is not None
        grad=head.weight.grad.detach();bias_grad=head.bias.grad.detach()
        norms=torch.linalg.vector_norm(grad,dim=1)
        assert torch.isfinite(norms).all() and torch.isfinite(bias_grad).all() and bool((norms>0).all()),'Every acceleration/scalar output weight row needs a finite nonzero joint-loss gradient'
        gradient_rows.append(norms.cpu().tolist());bias_gradients.append(bias_grad.cpu().tolist())
        clipped_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(clipped_norm);clipped_norms.append(float(clipped_norm))
        opt.step();prepared.sync(device)
        delta=torch.linalg.vector_norm(head.weight.detach()-before,dim=1);bias_delta=head.bias.detach()-before_bias
        assert torch.isfinite(delta).all() and torch.isfinite(bias_delta).all() and bool((delta>0).all()),'Every acceleration/scalar output weight row must change after the joint Adam update'
        change_rows.append(delta.cpu().tolist());bias_changes.append(bias_delta.cpu().tolist());times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
    poisoned=dp.SimSample(node_features=dict(sample.node_features),node_target=torch.full_like(sample.node_target,12345.),global_features=sample.global_features)
    model.eval()
    with torch.no_grad():
        inference=model(sample,data.stats_t);poisoned_inference=model(poisoned,data.stats_t)
    assert torch.isfinite(inference).all() and torch.equal(inference,poisoned_inference),'Eval outputs must be independent of all target values'
    teacher_poison=None
    if strategy=='one_step_teacher_forcing':
        target=sample.node_target.clone();target[...,3:]=12345.
        scalar_poison=dp.SimSample(node_features=dict(sample.node_features),node_target=target,global_features=sample.global_features)
        assert torch.equal(target[...,:3],sample.node_target[...,:3])
        model.train()
        with torch.no_grad():
            teacher=model(sample,data.stats_t);poisoned_teacher=model(scalar_poison,data.stats_t)
        assert torch.isfinite(teacher).all() and torch.equal(teacher,poisoned_teacher),'TF training must never read scalar target values'
        teacher_poison={'first_train_case_only':True,'past_position_targets_preserved':True,
                        'all_scalar_target_values_replaced':True,'all_outputs_exactly_equal':True,
                        'max_abs_difference':float((teacher-poisoned_teacher).abs().max())}
    report={'status':'actual_native_scalar_three_joint_Adam_updates_train_only_probe','domain':data.name,'strategy':strategy,
            'profile':PROFILE,'channels':c,'full_nodes':data.n,'steps':3,'future_frames_per_update':10,'case':case,
            'seconds':times,'losses':losses,'head_layout':'ln_mlp_out[0]: Sequential(LayerNorm, Linear(48,C)); rows0:3 acceleration, rows3:C native scalars',
            'head_weight_gradient_l2_by_row':gradient_rows,'head_weight_update_l2_by_row':change_rows,
            'head_bias_gradient_by_row':bias_gradients,'head_bias_update_by_row':bias_changes,
            'pre_clip_global_gradient_norm':clipped_norms,'head_acceleration_row_range':[0,3],'head_scalar_row_range':[3,c],
            'target_value_counts_per_update':{'acceleration_position_loss_values':data.n*10*3,'scalar_loss_values':data.n*10*(c-3)},
            'all_output_rows_finite_nonzero_gradient_and_changed_each_update':True,
            'original_C3_position_equivalence_before_training':equivalence,
            'eval_all_target_value_poisoning':{'first_train_case_only':True,'all_outputs_exactly_equal':True,'max_abs_difference':float((inference-poisoned_inference).abs().max())},
            'TF_training_scalar_target_poisoning':teacher_poison,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rss_observed_bytes':psutil.Process().memory_info().rss,
            'test_predicted':False,'test_target_values_accessed':False,'validation_target_values_accessed':False,
            'contract':contract(data),'constructor':config,'source':source,'torch':torch.__version__,
            'timing_scope':'Three functional updates reuse the same GPU-resident first-train-case sample; these cached-sample times do not set the common fit budget',
            'budget_decision_timing_source':'Separate fresh-model stream-probe over five distinct train cases'}
    prepared.write(out/'probe.json',report)
    print(json.dumps({key:report[key] for key in ['domain','strategy','channels','full_nodes','seconds','cuda_peak_allocated_bytes','rss_observed_bytes','test_predicted']}),flush=True)

def stream_probe(data,strategy):
    out,private=paths(data,strategy,'stream-probe')
    assert not any(out.iterdir()) and not any(private.iterdir()),'Stream probe requires empty immutable evidence directories'
    model,dp,c,config,source=build(data,strategy,device)
    from profile_temporal_case_stream import measure_stream
    assert Path(inspect.getfile(measure_stream)).resolve()==(OWN/'profile_temporal_case_stream.py').resolve()
    report=measure_stream(data,strategy,model,dp,c,device,source,PROTOCOL_SHA256,time_indices=None)
    report.update({'domain':data.name,'strategy':strategy,'profile':PROFILE,'channels':c,
                   'contract':contract(data),'constructor':config})
    prepared.write(out/'stream-profile.json',report)
    print(json.dumps({'domain':data.name,'strategy':strategy,'channels':c,'profile':PROFILE,
                      'status':'distinct_train_case_stream_probe_written','test_predicted':False}),flush=True)

def fit(data,strategy,budget,pilot=False):
    assert not pilot and PROTOCOL['phase']=='fit' and FIT_READINESS is not None
    out,private=paths(data,strategy,'pilot' if pilot else 'fit')
    assert not any(out.iterdir()) and not any(private.iterdir()),'Fit requires new empty output and checkpoint directories; existing evidence is immutable'
    phase={case:0 for case in data.cache.split['train']}
    model,dp,c,config,source=build(data,strategy,device)
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
            index=None  # Both native scalar strategies consume all ten times per update.
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
        if epoch_complete and val<best:
            best=val;best_epoch=epoch
            torch.save({'state_dict':model.state_dict(),'contract':contract(data),'constructor':config,'source':source,
                        'best_epoch':epoch,'profile':PROFILE,'phase_offsets':phase,'training_budget_seconds':budget},private/'best.pt')
        row={'epoch':epoch,'epoch_complete':epoch_complete,'eligible_for_checkpoint_selection':epoch_complete,'training_mse':float(np.mean(losses)),
             'validation_all_time_mse':val,'updates':updates,'cumulative_train_seconds':train_seconds,
             'epoch_train_seconds':train_seconds-before,'epoch_validation_seconds':vtime}
        history.append(row);prepared.write(out/('pilot-history.json' if pilot else 'history.json'),history)
        print(json.dumps({'domain':data.name,'strategy':strategy,**row}),flush=True)
        if pilot:break
    # Reproduce checkpoint on validation before touching the held-out test.
    validation_metrics=mean_metrics=zero_metrics=ballistic_metrics=diagnostics=None
    if best_epoch is not None:
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
        diagnostics=field_diagnostics(p,t,data.names[:c])
    last_complete=max([r['epoch'] for r in history if r['epoch_complete']],default=None)
    complete_epochs=sum(row['epoch_complete'] for row in history)
    exposure_gate=updates>=100 and complete_epochs>=2
    selected_row=next((row for row in history if row['epoch']==best_epoch),None)
    selected_exposure={'optimizer_updates':selected_row['updates'] if selected_row else None,
                       'complete_case_epochs':best_epoch,
                       'meets_executed_exposure_threshold':bool(selected_row and selected_row['updates']>=100 and best_epoch>=2),
                       'selection_policy_unchanged':True}
    report={'status':('one_epoch_resource_pilot_validation_only' if pilot else 'budget_fit_validation_only_test_not_evaluated') if best_epoch is not None else 'budget_exhausted_before_one_complete_epoch_no_selected_checkpoint',
            'domain':data.name,'strategy':strategy,'channels':c,'profile':PROFILE,'epochs_started':epoch,'updates':updates,
            'intended_optimizer_training_budget_seconds':budget,'actual_optimizer_training_seconds':train_seconds,
            'all_time_validation_seconds':val_seconds,'total_wall_seconds':time.perf_counter()-wallstart,
            'budget_definition':'Includes per-update input transfer, forward/backward, gradient clip, optimizer and synchronization; excludes validation and one-off loading/import/setup. At most one optimizer update overshoot.',
            'pilot_budget_policy':'A resource pilot completes exactly one whole epoch even if it exceeds the nominal comparison budget; pilot artifacts are separate and never overwritten by the budget fit.',
            'best_validation_all_time_mse':best if best_epoch is not None else None,'best_epoch':best_epoch,'latest_validation_all_time_mse':history[-1]['validation_all_time_mse'],
            'best_validation_physical_metrics':validation_metrics,'validation_train_mean_baseline':mean_metrics,
            'validation_zero_baseline':zero_metrics,'validation_ballistic_displacement_only_baseline':ballistic_metrics,
            'ballistic_scope':'Known source initial velocity multiplied by target time, no contact/deformation update and no stress or strain prediction; included because rigid translation can dominate global displacement.',
            'minimum_at_last_complete_epoch':best_epoch==last_complete if best_epoch is not None else None,'last_complete_epoch':last_complete,'convergence_claimed':False,
            'quality_exposure_gate':exposure_gate,
            'executed_exposure':{'optimizer_updates':updates,'complete_case_epochs':complete_epochs,
                                 'minimum_optimizer_updates':100,'minimum_complete_case_epochs':2},
            'quality_exposure_gate_definition':'Minimum executed work only; at least100 joint optimizer updates and two complete case epochs. This does not establish convergence or accuracy.',
            'selected_checkpoint_exposure':selected_exposure,
            'checkpoint_selection_policy':'Only complete sampled-case epochs are candidates; terminal partial-epoch updates and validation are retained but cannot select the checkpoint.',
            'validation_field_diagnostics':diagnostics,
            'test_predicted':False,'test_used_for_selection':False,'constructor':config,'source':source,'contract':contract(data),
            'time_sampling':{'definition':'Native scalar AR/TF consume all ten future times for every optimizer update',
                             'seed':20261015,'phase_offsets':phase,'per_case_time_label_exposures':exposures,
                             'sampled_epoch_interpretation':'One complete epoch visits every train case once and every one of its ten C6/C7 target frames once.',
                             'validation':'Every validation case, every stored future time; no temporal subsampling.'},
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'rss_observed_peak_bytes':rss,
            'checkpoint_sha256':prepared.sha(private/'best.pt') if best_epoch is not None else None,'validation_prediction_sha256':prepared.sha(private/'validation-predictions.npz') if best_epoch is not None else None,
            'task_limit':'V3 original-wrapper adaptation jointly predicts C7/C6 displacement and native scalar fields; scalars do not feed the positional rollout.'}
    result_path=out/('pilot-result.json' if pilot else 'result.json')
    prepared.write(result_path,report)
    if best_epoch is not None:
        freeze={'checkpoint_sha256':prepared.sha(private/'best.pt'),'validation_prediction_sha256':prepared.sha(private/'validation-predictions.npz'),
                'constructor_sha256':fingerprint(config),'contract_sha256':fingerprint(contract(data)),'source_metadata_sha256':fingerprint(source),
                'normalization_sha256':data.normalization_sha256,'prepared_source_sha256':EXPECTED_PREPARED_SHA,'budget_source_sha256':prepared.sha(Path(__file__)),
                'result_sha256':prepared.sha(result_path),'history_sha256':prepared.sha(out/('pilot-history.json' if pilot else 'history.json')),
                'selected_complete_epoch':best_epoch,'test_predicted':False}
        prepared.write(out/'selection-freeze.json',freeze)
    print(json.dumps({k:report[k] for k in ['domain','strategy','channels','updates','actual_optimizer_training_seconds','all_time_validation_seconds','best_epoch','test_predicted']}),flush=True)

def load_selected(data,strategy):
    out,private=paths(data,strategy)
    ckpt=torch.load(private/'best.pt',map_location=device,weights_only=False)
    freeze=json.loads((out/'selection-freeze.json').read_text())
    assert freeze['checkpoint_sha256']==prepared.sha(private/'best.pt')
    assert freeze['validation_prediction_sha256']==prepared.sha(private/'validation-predictions.npz')
    assert freeze['normalization_sha256']==data.normalization_sha256 and freeze['contract_sha256']==fingerprint(contract(data))
    assert freeze['budget_source_sha256']==prepared.sha(Path(__file__)) and freeze['prepared_source_sha256']==prepared.sha(Path(prepared.__file__))
    assert freeze['result_sha256']==prepared.sha(out/'result.json') and freeze['history_sha256']==prepared.sha(out/'history.json')
    assert ckpt['contract']==contract(data),'Data/profile/source contract changed'
    model,dp,c,config,source=build(data,strategy,device)
    assert config==ckpt['constructor']
    assert freeze['constructor_sha256']==fingerprint(config) and freeze['source_metadata_sha256']==fingerprint(source)
    for key in ['wrapper_sha256','official_wrapper_sha256','core_class_sha256','physicsnemo_version','driver_sha256','prepared_driver_sha256','protocol_sha256']:assert source[key]==ckpt['source'][key]
    model.load_state_dict(ckpt['state_dict'])
    return out,private,model,dp,c,ckpt

def reload_validation(data,strategy):
    out,private,model,dp,c,ckpt=load_selected(data,strategy)
    assert not (out/'fresh-process-validation-reload.json').exists(),'Reload evidence already exists; preserve it'
    _,p,t=prepared.evaluate(data,model,dp,c,data.cache.split['validation'],keep=True)
    original=np.load(private/'validation-predictions.npz');assert np.array_equal(original['truth'],t)
    delta=np.abs((original['prediction']-p)/data.std[:c]);assert np.isfinite(delta).all() and delta.max()<=1e-5
    prepared.write(out/'fresh-process-validation-reload.json',{'domain':data.name,'strategy':strategy,'test_predicted':False,
        'fresh_process':True,'checkpoint_sha256':prepared.sha(private/'best.pt'),'source_constructor_contract_gate_passed':True,
        'selection_freeze_sha256':prepared.sha(out/'selection-freeze.json'),'contract_sha256':fingerprint(contract(data)),
        'normalization_sha256':data.normalization_sha256,
        'prediction_max_abs_difference':float(np.abs(original['prediction']-p).max()),'max_abs_difference_in_training_scale_units':float(delta.max()),
        'per_field_physical_max_abs_difference':np.max(np.abs(original['prediction']-p),axis=(0,1,2)).tolist()})

def final_test(data,strategy):
    out,private,model,dp,c,ckpt=load_selected(data,strategy)
    fit_result=json.loads((out/'result.json').read_text(encoding='utf8'))
    exposure=fit_result['executed_exposure']
    assert fit_result['quality_exposure_gate'] is True and exposure['optimizer_updates']>=100 and exposure['complete_case_epochs']>=2,'Minimum executed training exposure not met; final test is blocked'
    selected=fit_result['selected_checkpoint_exposure']
    assert selected['optimizer_updates']>0 and selected['complete_case_epochs']>=1 and selected['complete_case_epochs']==ckpt['best_epoch']>=1,'Selected checkpoint must have trained updates and one complete epoch; an initial epoch0 checkpoint cannot authorize test evaluation'
    assert not (out/'final-test.json').exists() and not (private/'test-predictions.npz').exists() and not (out/'final-evaluation-started.json').exists(),'Final evaluation already started/exists; never retune or overwrite'
    replay=json.loads((out/'fresh-process-validation-reload.json').read_text())
    assert replay['source_constructor_contract_gate_passed'] and replay['test_predicted'] is False
    assert replay['selection_freeze_sha256']==prepared.sha(out/'selection-freeze.json') and replay['checkpoint_sha256']==prepared.sha(private/'best.pt')
    assert replay['contract_sha256']==fingerprint(contract(data)) and replay['normalization_sha256']==data.normalization_sha256
    assert math.isfinite(replay['max_abs_difference_in_training_scale_units']) and replay['max_abs_difference_in_training_scale_units']<=1e-5
    # Exclusive creation also prevents two fresh final processes racing the gate.
    with (out/'final-evaluation-started.json').open('x',encoding='utf8') as sentinel:
        json.dump({'checkpoint_sha256':prepared.sha(private/'best.pt'),
                   'selection_freeze_sha256':prepared.sha(out/'selection-freeze.json'),
                   'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},sentinel,indent=2,allow_nan=False)
    prepared.sync(device);start=time.perf_counter();loss,p,t=prepared.evaluate(data,model,dp,c,data.cache.split['test'],keep=True);prepared.sync(device)
    inference_seconds=time.perf_counter()-start
    metrics=prepared.physical_metrics(p,t,data.names[:c],data.units[:c],data.cache.split['test'],data.cache.meta['time_values'])
    np.savez_compressed(private/'test-predictions.npz',prediction=p,truth=t)
    prepared.write(out/'final-test.json',{'status':'single_final_heldout_evaluation_after_validation_checkpoint_selection',
        'domain':data.name,'strategy':strategy,'profile':PROFILE,'selected_epoch':ckpt['best_epoch'],'normalized_mse':loss,
        'inference_seconds':inference_seconds,'metrics':metrics,'contract':contract(data),
        'quality_exposure_gate':True,'executed_exposure':exposure,'selected_checkpoint_exposure':fit_result['selected_checkpoint_exposure'],
        'checkpoint_sha256':prepared.sha(private/'best.pt'),'prediction_sha256':prepared.sha(private/'test-predictions.npz'),
        'field_diagnostics':field_diagnostics(p,t,data.names[:c]),
        'test_used_for_selection':False,'no_retuning_after_this_evaluation':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--domain',choices=['bumper','phone_full'])
    ap.add_argument('--manifest');ap.add_argument('--normalization')
    ap.add_argument('--strategy',choices=STRATEGIES,default='ar_rollout_training')
    ap.add_argument('--mode',choices=['source-check','inspect','probe','stream-probe','budget','reload','final'],required=True)
    ap.add_argument('--train-budget-seconds',type=float,default=180.)
    ap.add_argument('--allocated',action='store_true')
    ap.add_argument('--protocol');ap.add_argument('--protocol-sha256')
    args=ap.parse_args()
    if args.mode=='source-check':
        source_selfcheck();raise SystemExit(0)
    PROTOCOL_PATH,PROTOCOL,FIT_READINESS=preflight(args)
    PROTOCOL_SHA256=args.protocol_sha256
    os.environ['PHYSICSNEMO_FORCE_TE']='False';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    import numpy as np
    import run_domain_temporal as prepared
    import native_scalar_rollout_v3 as native_scalar
    import importlib.metadata,inspect
    assert prepared.sha(Path(prepared.__file__))==EXPECTED_PREPARED_SHA
    data=prepared.Domain(args.domain,Path(args.manifest),Path(args.normalization))
    expected=PROTOCOL['domains'][args.domain]
    assert data.cache.split==expected['split'] and data.names==expected['field_names'] and data.units==expected['field_units']
    assert data.c==expected['channels']
    assert data.ready['ready_sha256']==expected['ready_sha256']
    if args.mode=='inspect':
        out=OUT/args.domain/PROFILE/'prepared-contract.json'
        assert not out.exists(),'Inspect evidence already exists; preserve it'
        prepared.write(out,{'contract':contract(data),'no_framework_import':True,'allocation_and_protocol_gate_passed':True,'test_predicted':False})
        print(json.dumps({'domain':data.name,'status':'allocated_numpy_inspect_pass','profile':PROFILE}));raise SystemExit(0)
    import torch,psutil
    prepared.torch=torch;prepared.psutil=psutil
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(20261015)
    device=torch.device('cuda');data.attach(device)
    if args.mode=='probe':run_probe(data,args.strategy)
    elif args.mode=='stream-probe':stream_probe(data,args.strategy)
    elif args.mode=='budget':fit(data,args.strategy,args.train_budget_seconds)
    elif args.mode=='reload':reload_validation(data,args.strategy)
    else:final_test(data,args.strategy)

