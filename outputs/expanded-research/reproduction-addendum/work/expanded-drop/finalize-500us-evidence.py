"""Freeze compact 0.5 ms evidence without copying binary models, decks or weights."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/expanded-research/benchmarks/drop_test'
DEST=BASE/'500us-final'
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
summary=json.loads((DEST/'summary.json').read_text())
assert summary['completed_to_requested_time'] and summary['last_actual_frame_s']>=4.9999e-4
assert summary['all_saved_cell_EPSP_monotone']
canonical=BASE/'doe/canonical'
ready=json.loads((canonical/'dataset_READY_evidence.json').read_text())
checks={}
for relative,expected in ready['file_hashes'].items():
    path=canonical/relative
    actual=digest(path)
    checks[relative]={'expected_sha256':expected['sha256'],'actual_sha256':actual,'unchanged':actual==expected['sha256'],'bytes':path.stat().st_size}
assert all(value['unchanged'] for value in checks.values())
(DEST/'frozen20us-AI-cache-preservation.json').write_text(json.dumps({'all_frozen_files_unchanged':True,'file_checks':checks},indent=2))
expected={
 'work/expanded-drop/source/Cell_Phone_Drop_0000.rad':'435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2',
 'work/expanded-drop/100us-run/Cell_Phone_Drop_0002_0001.rst':'e56c1f0764f5daa4946c313458d126f8df4230f19b7917b87de4843603de1dee',
 'work/expanded-drop/500us-run/Cell_Phone_Drop_0003_0001_I.rst':'8434daa6788ff7ca094eb3ad02270e9cb06281ca162affca1a335cd67bd58aca',
 'work/opencourant-experiment/solver/OpenCourant/exec/engine_win64_sp.exe':'13a9d6551fb9206b6c27c8af819e752edf3550e4725e446a2c3042c6c1a2b9d1'}
source_checks={path:{'sha256':digest(ROOT/path),'expected_sha256':value} for path,value in expected.items()}
assert all(value['sha256']==value['expected_sha256'] for value in source_checks.values())
for source,name in [
 (BASE/'500us/restart-boundary-bitwise-check.json','100us-to-first500us-boundary-bitwise.json'),
 (BASE/'500us-resume03/source-control-and-restart-audit.json','source-control-and-restart-audit.json'),
 (BASE/'500us-resume03/execution-metrics.json','resume03-execution-metrics.json'),
 (BASE/'500us-resume03/frozen-inputs-and-guards.json','resume03-frozen-inputs-and-guards.json'),
 (BASE/'500us/execution-metrics.json','first500us-guard-stop-metrics.json'),
 (BASE/'500us-resume02/execution-metrics.json','failed-RERUN02-metrics.json')]:
    shutil.copy2(source,DEST/name)
metrics=json.loads((DEST/'resume03-execution-metrics.json').read_text(encoding='utf-8-sig'))
assert metrics['exit_code']==0 and metrics['stopped_reason'] is None
for name in ['summarize-500us-bounded.py','plot-500us-bounded.py','finalize-500us-evidence.py','run-engine-500us-resume03.ps1','resource_gate.py','check-500us-final-boundary.py','run-final-boundary-check.ps1']:
    shutil.copy2(ROOT/'work/expanded-drop'/name,DEST/name)
run=ROOT/'work/expanded-drop/500us-resume03-run'
for name in ['engine.stdout.log','engine.stderr.log','Cell_Phone_Drop_0004.out']:
    shutil.copy2(run/name,DEST/name)
files={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(DEST.iterdir()) if p.is_file() and p.name not in ['artifact-freeze.json','job-progress.json']}
freeze={'freeze_utc':datetime.now(timezone.utc).isoformat(),'actual_FEM_completed_to_0_5ms':True,
        'source_and_checkpoint_hash_checks':source_checks,'all_frozen20us_AI_files_unchanged':True,
        'files':files,'binary_policy':'Raw ANIM/RST/TH, model decks and weights remain local under work; compact code/log/json/csv/PNG evidence only in this directory.',
        'restart_boundary_scope':'The100us boundary is bitwise verified. Guard-stop and next restart lack identical-time saved ANIM, so no new field-bitwise identity claim is made across unequal-time snapshots.'}
boundary=DEST/'final-checkpoint-boundary-bitwise.json'
if boundary.exists():
    result=json.loads(boundary.read_text())
    freeze['final_checkpoint_reload_same_time_all_native_fields_bitwise']=bool(result['same_actual_time_s'] and result['all_native_fields_bitwise_equal'])
(DEST/'artifact-freeze.json').write_text(json.dumps(freeze,indent=2))
(DEST/'job-progress.json').write_text(json.dumps({'status':'completed','requested_final_time_s':.0005,'last_actual_frame_s':summary['last_actual_frame_s'],'summary':'summary.json','artifact_freeze':'artifact-freeze.json'},indent=2))
print(json.dumps({'status':'completed','frozen_evidence_files':len(files),'unchanged_AI_files':len(checks)},indent=2))
