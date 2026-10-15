"""Allocated protocol/core gate around the byte-frozen original V2 TC driver."""
import argparse
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
OWN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies'
PROFILE = 'budget_v2_complete_epoch'
V2 = OWN / 'run_budget_temporal_v2.py'
V2_SHA = '985d5f3dc3b74d0414824fe4feed2427a8ee1e4a096c357b849b3fabe0d182b1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))


def preflight(args):
    assert args.allocated, 'Explicit root allocation required before data/framework imports'
    assert len(args.protocol_sha256) == 64 and all(c in '0123456789abcdef' for c in args.protocol_sha256)
    protocol_path = Path(args.protocol).resolve()
    assert sha(protocol_path) == args.protocol_sha256
    protocol = json.loads(protocol_path.read_text())
    sources = {resolve(row['path']): row['sha256'] for row in protocol['sources']}
    assert len(sources) == len(protocol['sources'])
    for path, digest in sources.items():
        assert sha(path) == digest, path
    assert sources[Path(__file__).resolve()] == sha(__file__)
    assert sources[V2.resolve()] == V2_SHA == sha(V2)
    spec = protocol['domains'][args.domain]
    for kind, path in [('manifest',Path(args.manifest).resolve()),
                       ('normalization',Path(args.normalization).resolve()),
                       ('ready',Path(args.manifest).resolve().with_name('READY.json'))]:
        assert path == resolve(spec[kind+'_path']) and sha(path) == spec[kind+'_sha256']
    assert math.isfinite(args.train_budget_seconds) and args.train_budget_seconds > 0
    if args.mode in ['probe','stream-probe']:
        assert protocol['phase'] == 'resource_probes' and protocol['training_budget_seconds'] is None
    else:
        assert protocol['phase'] == 'fit' and args.train_budget_seconds == protocol['training_budget_seconds']
        from multifield_protocol_guard import verify_fit_readiness
        verify_fit_readiness(protocol)
    return protocol_path, protocol, sources


def fit_outcome(domain, protocol_sha):
    directory = OUT / domain / 'time_conditional' / PROFILE / 'fit'
    report_path = directory/'result.json'; history_path = directory/'history.json'
    report = json.loads(report_path.read_text()); history = json.loads(history_path.read_text())
    freeze_path = directory/'selection-freeze.json'
    freeze = json.loads(freeze_path.read_text()) if freeze_path.exists() else None
    if freeze:
        assert freeze['result_sha256'] == sha(report_path) and freeze['history_sha256'] == sha(history_path)
        assert freeze['selected_complete_epoch'] == report['best_epoch']
    completed = sum(bool(row['epoch_complete']) for row in history)
    selected = next((row for row in history if row['epoch'] == report['best_epoch']), None)
    return {'domain':domain,'strategy':'time_conditional','profile':PROFILE,'protocol_sha256':protocol_sha,
            'result_sha256':sha(report_path),'history_sha256':sha(history_path),
            'selection_freeze_sha256':sha(freeze_path) if freeze else None,
            'executed_updates':report['updates'],'complete_case_epochs':completed,
            'minimum_executed_updates':100,'minimum_complete_case_epochs':2,
            'minimum_executed_training_exposure_met':report['updates'] >= 100 and completed >= 2,
            'selected_epoch':report['best_epoch'],'selected_checkpoint_updates':selected['updates'] if selected else None,
            'selected_checkpoint_has_complete_updated_epoch':bool(selected and selected['epoch_complete'] and selected['epoch'] >= 1 and selected['updates'] > 0),
            'selected_checkpoint_minimum_exposure_met':bool(selected and selected['updates'] >= 100 and selected['epoch'] >= 2),
            'selected_checkpoint_policy':'Validation may select an earlier complete updated epoch; selected exposure need not reach terminal minimum. Epoch0/zero-update state is an untrained failed attempt, not a learned predictor and is not tested.',
            'quality_scope':'Executed training exposure only; no convergence, accuracy or physics validity PASS.',
            'test_predicted':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',required=True,choices=['phone_full','bumper'])
    parser.add_argument('--manifest',required=True); parser.add_argument('--normalization',required=True)
    parser.add_argument('--strategy',choices=['time_conditional'],default='time_conditional')
    parser.add_argument('--mode',required=True,choices=['probe','stream-probe','budget','reload','final'])
    parser.add_argument('--protocol',required=True); parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--train-budget-seconds',type=float,default=180.)
    parser.add_argument('--allocated',action='store_true')
    args = parser.parse_args()
    protocol_path, protocol, sources = preflight(args)
    receipts = OWN/'tc-shim-evidence-v3'
    receipt_path = receipts/f'{args.domain}--{args.mode}.json'
    assert not receipt_path.exists(), 'TC shim stage evidence already exists'
    if args.mode in ['reload','final']:
        budget_receipt = json.loads((receipts/f'{args.domain}--budget.json').read_text())
        assert budget_receipt['protocol_sha256'] == args.protocol_sha256
        expected = json.loads((receipts/f'{args.domain}--budget-outcome.json').read_text())
        assert expected == fit_outcome(args.domain,args.protocol_sha256)
        if args.mode == 'final':
            assert expected['minimum_executed_training_exposure_met'], 'No test before minimum actual training exposure'
            assert expected['selected_checkpoint_has_complete_updated_epoch'], 'No test for an epoch0/zero-update untrained state'
    if args.mode == 'probe':
        assert not (OUT/args.domain/'time_conditional'/'probe.json').exists()
    os.environ['PHYSICSNEMO_FORCE_TE']='False';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    # Imported only after explicit allocation and all stdlib source/data gates.
    from physicsnemo.models.geotransolver import GeoTransolver
    core_path = Path(inspect.getfile(GeoTransolver)).resolve()
    assert core_path == resolve(protocol['core_source_path']) and sha(core_path) == sources[core_path]
    write_once(receipt_path, {'domain':args.domain,'mode':args.mode,'profile':PROFILE,
                              'protocol_path':str(protocol_path),'protocol_sha256':args.protocol_sha256,
                              'active_core_path':str(core_path),'active_core_sha256':sha(core_path),
                              'shim_sha256':sha(__file__),'frozen_v2_sha256':sha(V2),'pid':os.getpid(),
                              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'actual_core_bound_before_v2_model_use':True})
    if args.mode == 'stream-probe':
        import torch
        import psutil
        import run_domain_temporal as prepared
        from profile_temporal_case_stream import measure_stream
        prepared.torch=torch;prepared.psutil=psutil
        assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(20261015)
        device=torch.device('cuda')
        data=prepared.Domain(args.domain,Path(args.manifest),Path(args.normalization))
        assert data.ready['ready_sha256'] == protocol['domains'][args.domain]['ready_sha256']
        data.attach(device)
        model,dp,c,config,source=prepared.build(data,'time_conditional',device)
        source.update({'protocol_sha256':args.protocol_sha256,'entry_shim_sha256':sha(__file__)})
        result=measure_stream(data,'time_conditional',model,dp,c,device,source,args.protocol_sha256)
        result['constructor']=config
        result.setdefault('contract',data.contract())['frozen_protocol']={'path':str(protocol_path),'sha256':args.protocol_sha256}
        result['contract']['budget_profile']=PROFILE
        target=OUT/args.domain/'time_conditional'/PROFILE/'stream-probe'/'stream-profile.json'
        write_once(target,result)
    else:
        command=[str(V2),'--domain',args.domain,'--manifest',args.manifest,'--normalization',args.normalization,
                 '--strategy','time_conditional','--mode',args.mode,'--train-budget-seconds',str(args.train_budget_seconds),'--allocated']
        previous=sys.argv
        try:
            sys.argv=command
            runpy.run_path(str(V2),run_name='__main__')
        finally:
            sys.argv=previous
        if args.mode == 'probe':
            original=OUT/args.domain/'time_conditional'/'probe.json'
            result=json.loads(original.read_text());result.update({'profile':PROFILE,
                'original_probe_path':original.relative_to(ROOT).as_posix(),'original_probe_sha256':sha(original),
                'case':protocol['domains'][args.domain]['split']['train'][0],
                'test_predicted':False,'test_target_values_accessed':False,'validation_target_values_accessed':False})
            result['source'].update({'protocol_sha256':args.protocol_sha256,'entry_shim_sha256':sha(__file__)})
            result['contract']['frozen_protocol']={'path':str(protocol_path),'sha256':args.protocol_sha256}
            result['contract']['budget_profile']=PROFILE
            result['timing_scope']='Three updates of the same already-transferred train sample; use separate different-case stream profile for training-budget estimates.'
            result.pop('same_dataset_30_epoch_estimate_seconds_excluding_validation',None)
            write_once(OUT/args.domain/'time_conditional'/PROFILE/'probe'/'probe.json',result)
        elif args.mode == 'budget':
            write_once(receipts/f'{args.domain}--budget-outcome.json',fit_outcome(args.domain,args.protocol_sha256))


if __name__ == '__main__':
    main()
