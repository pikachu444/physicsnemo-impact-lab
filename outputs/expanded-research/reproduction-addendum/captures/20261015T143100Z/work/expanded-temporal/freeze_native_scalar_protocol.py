"""Freeze the bounded six-fit plan using stdlib text/JSON/hash access only."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies'
PROFILE = 'multifield_v3_native_scalar_heads'
CORE = 'work/gpu-experiment/venv/Lib/site-packages/physicsnemo/models/geotransolver/geotransolver.py'
SOURCES = [
    'work/expanded-temporal/run_budget_temporal_v3.py',
    'work/expanded-temporal/native_scalar_rollout_v3.py',
    'work/expanded-temporal/run_budget_temporal_v2.py',
    'work/expanded-temporal/run_domain_temporal.py',
    'work/expanded-temporal/run_temporal.py',
    'work/expanded-ml/benchmark_data.py',
    'work/expanded-temporal/sources/crash/rollout.py',
    'work/expanded-temporal/sources/crash/datapipe.py',
    'work/expanded-temporal/sources/drop_test/rollout.py',
    'work/expanded-temporal/sources/drop_test/datapipe.py',
    'work/expanded-ml/venv/Lib/site-packages/zz_reuse_physicsnemo.pth',
    'work/gpu-experiment/venv/Lib/site-packages/nvidia_physicsnemo-2.2.2.dist-info/METADATA',
    CORE,
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    assert not any(name in sys.modules for name in ['numpy','torch','physicsnemo'])
    path = OUT / 'native-scalar-v3-protocol.json'
    assert not path.exists(), 'Protocol already exists; preserve it'
    for relative, expected in {
        'work/expanded-temporal/run_budget_temporal_v2.py': '985d5f3dc3b74d0414824fe4feed2427a8ee1e4a096c357b849b3fabe0d182b1',
        'work/expanded-temporal/run_domain_temporal.py': '955f25cf30a173c578eaa620cce6a0fae1e3d89bf072b0e21c5177e3854f3013',
        'work/expanded-temporal/run_temporal.py': 'baabebfe4198678a0915263edc348285a23bb02930ba45c52208be62b98950af',
        'work/expanded-temporal/sources/crash/rollout.py': '59d797f4fdf52473baac61d134923465b2620f344f5af3a2888f8b71bafa0afb',
        'work/expanded-temporal/sources/drop_test/rollout.py': '5b349a87aa6b5fcf994499d5848a655965fbec9ad51e1eeb384af56cfe548354',
    }.items():
        assert sha(ROOT / relative) == expected, relative
    domains = {}
    for name, cache, norm, nodes in [
        ('phone_full','phone-full-cache-v2','phone-full-normalization.json',76065),
        ('bumper','multifield7-cache','bumper7-normalization.json',13675),
    ]:
        manifest = ROOT / 'work/expanded-ml' / cache / 'manifest.json'
        ready_path = manifest.with_name('READY.json')
        normalization = ROOT / 'work/expanded-ml' / norm
        meta = json.loads(manifest.read_text()); ready = json.loads(ready_path.read_text())
        assert ready['status'] == 'ready' and ready['writer_exit_code'] == 0
        assert ready['manifest_sha256'] == sha(manifest)
        channels = len(meta['field_names'])
        assert channels == (6 if name == 'phone_full' else 7) and len(meta['time_values']) == 10
        assert len(meta['runs']) == ready['expected_cases']
        domains[name] = {'manifest_path': manifest.relative_to(ROOT).as_posix(), 'manifest_sha256': sha(manifest),
                         'ready_path': ready_path.relative_to(ROOT).as_posix(), 'ready_sha256': sha(ready_path),
                         'normalization_path': normalization.relative_to(ROOT).as_posix(), 'normalization_sha256': sha(normalization),
                         'split': meta['split'], 'field_names': meta['field_names'], 'field_units': meta['field_units'],
                         'channels': channels, 'nodes': nodes, 'time_values': meta['time_values']}
    protocol = {'schema_version': 3, 'profile': PROFILE,
                'status': 'pretraining_protocol_no_new_domain_fit_or_test_yet',
                'seed': 20261015, 'training_budget_seconds': 180.,
                'sources': [{'path': relative, 'sha256': sha(ROOT/relative)} for relative in SOURCES],
                'core_source_path': CORE, 'domains': domains,
                'current_six_fit_plan': {'time_conditional': {'profile':'budget_v2_complete_epoch','fits':2,'channels':'native full C6/C7'},
                                         'ar_rollout_training': {'profile':PROFILE,'fits':2,'channels':'native full C6/C7'},
                                         'one_step_teacher_forcing': {'profile':PROFILE,'fits':2,'channels':'native full C6/C7'}},
                'original_large_domain_C3_fit_count': 0,
                'original_evidence_policy': 'Original C3 coupon training, support/failure probes and V2/prepared sources remain immutable. Large-domain original C3 is used only as a train-case position-equivalence reference in the V3 probe.',
                'head_layout': {'raw_first_three':'normalized acceleration',
                                'raw_suffix':'native train-normalized scalar fields in domain field_names[3:] order',
                                'returned_first_three':'integrated normalized position',
                                'returned_suffix':'same direct normalized native scalars',
                                'scalar_feedback':False,'recurrent_geometry_channels':3},
                'teacher_forcing': 'Training only, past position pairs only. Native scalar labels never enter forward inputs. Evaluation uses source initial state and predicted position pairs; no target values are read.',
                'loss': 'Joint mean of squared field errors in the identical train-only physical normalization; same C6/C7 channel policy as TC. No clipping, monotonic correction or test-tuned weighting.',
                'checkpoint_selection': 'Minimum all-case/all-time free-rollout validation normalized MSE among complete sampled-case epochs only; earliest retained on an exact tie. Terminal partial work and validation are retained but ineligible.',
                'exposure_policy': 'Report selected checkpoint and all executed/terminal-partial work separately. TC covers all ten times per case once per ten completed sampled epochs; V3 AR/TF each update consumes all ten times.',
                'budget_scope': 'Transfer, forward, backward, clip and optimizer synchronization included; validation/setup excluded, at most one-update overshoot. This first bounded comparison does not claim convergence. Static caps exclude transfer and cannot be ranked as identical timing budgets.',
                'probe': 'Train first case only: original C3 position equivalence, three real joint head Adam updates with per-row finite gradients and changed weights, all-target eval poison invariance, TF scalar-target training poison invariance. Actual resource measurements required after allocation.',
                'final_gate': 'Freeze source/protocol/config/normalization/split/checkpoint/history/validation-prediction hashes; fresh-process validation reload <=1e-5 train-standard-deviation units; then one sentinel-guarded held-out free rollout. Further metrics only from saved final NPZ.',
                'no_test_selection':True,'no_new_paid_resources':True,
                'native_field_limit': 'VM outputs are scalars, not stress tensor3. Phone Frob is native projected total strain; bumper has only native EPSP and no native total-strain label. Displacement-derived strain is not substituted.',
                'launcher_receipt_note': 'Launcher source hash and this protocol hash are frozen in the separate independent review receipt to avoid a circular self-source hash.'}
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(protocol, indent=2, allow_nan=False))
    assert not any(name in sys.modules for name in ['numpy','torch','physicsnemo'])
    print(json.dumps({'status':'frozen','path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
                      'source_files':len(SOURCES),'framework_or_numpy_imported':False,'array_access':False}))


if __name__ == '__main__':
    freeze()
