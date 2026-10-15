"""CPU-only deterministic sampler / evidence-preservation guard checks."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys,subprocess
import numpy as np
import run_budget_temporal_v2 as profile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
assert 'torch' not in sys.modules
coverage=[]
for count in [5,112]:
    cases=[f'case{i}' for i in range(count)]
    phase=profile.phases(cases)
    seen=np.zeros((count,10),dtype='int32')
    for epoch in range(10):
        for i,c in enumerate(cases):seen[i,(epoch+phase[c])%10]+=1
    assert np.all(seen==1)
    per_epoch=np.bincount(list(phase.values()),minlength=10)
    assert per_epoch.max()-per_epoch.min()<=1
    coverage.append({'cases':count,'ten_epochs_each_case_time_pair_exactly_once':True,'first_epoch_time_counts':per_epoch.tolist()})

# Change only this isolated process's artifact roots, never production files.
scratch=ROOT/'work/expanded-temporal/guard-selfcheck'
profile.OUT=scratch/'outputs';profile.OWN=scratch/'private'
guards=[]
for index,(where,filename) in enumerate([('out','history.json'),('private','best.pt'),('out','final-test.json'),('out','final-evaluation-started.json')]):
    data=SimpleNamespace(name=f'guard_{index}')
    target,private=profile.paths(data,'time_conditional')
    path=(target if where=='out' else private)/filename
    path.write_text('{}')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        profile.fit(data,'time_conditional',180.)
        raise RuntimeError('Existing evidence unexpectedly accepted')
    except AssertionError as error:
        assert 'new empty output' in str(error)
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    guards.append({'existing_artifact':filename,'fit_rejected_before_framework_use':True,'original_bytes_unchanged':True})
assert 'torch' not in sys.modules
finite_guards=[]
for value in ['inf','nan','0','-1']:
    result=subprocess.run([sys.executable,str(Path(profile.__file__)),'--domain','bumper','--manifest','unused',
                           '--normalization','unused','--mode','inspect','--train-budget-seconds',value],capture_output=True,text=True)
    assert result.returncode!=0 and 'math.isfinite(args.train_budget_seconds)' in result.stderr
    finite_guards.append({'value':value,'cli_rejected_before_dataset_or_framework':True,'exit_code':result.returncode})
report={'status':'pass','scope':'Only deterministic cyclic sampler and artifact-preservation entry guards; no framework import, optimizer, GPU or test prediction.',
        'profile_source_sha256':hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest(),
        'sampler_coverage':coverage,'preservation_guards':guards,'finite_budget_guards':finite_guards,
        'not_executed_here':['GPU resource probe','one-epoch training','SHA-freeze/reload numerical inference','final-test inference']}
(OUT/'budget-profile-cpu-selfcheck.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'status':'pass','coverage_cases':[5,112],'guard_cases':len(guards),'finite_budget_guard_cases':len(finite_guards),'framework_imported':False}))
