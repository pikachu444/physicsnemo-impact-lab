"""Current hybrid-plan exposure accounting and saved-final-prediction arithmetic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies'
PROFILES = {'time_conditional': 'budget_v2_complete_epoch',
            'ar_rollout_training': 'multifield_v3_native_scalar_heads',
            'one_step_teacher_forcing': 'multifield_v3_native_scalar_heads'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with path.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))


def complete_exposures(cases, phase, epochs, strategy):
    result = {case: [0] * 10 for case in cases}
    for case in cases:
        for epoch in range(epochs):
            if strategy == 'time_conditional':
                result[case][(epoch + phase[case]) % 10] += 1
            else:
                result[case] = [value + 1 for value in result[case]]
    return result


def difference(left, right):
    result = {case: [a-b for a,b in zip(left[case], right[case])] for case in left}
    assert all(value >= 0 for row in result.values() for value in row)
    return result


def exposure_summary(mapping, nodes, channels):
    counts = [value for row in mapping.values() for value in row]
    total = sum(counts)
    return {'per_case_time_label_exposures': mapping, 'case_time_label_exposures': total,
            'node_time_label_exposures': total * nodes,
            'scalar_target_exposures': total * nodes * channels,
            'minimum_exposures_per_case_time': min(counts),
            'maximum_exposures_per_case_time': max(counts),
            'unseen_case_time_pairs': counts.count(0),
            'complete_ten_time_coverage_cycles_per_case_minimum': min(counts)}


def exposures(domain, strategy):
    profile = PROFILES[strategy]
    directory = OUT / domain / strategy / profile / 'fit'
    report_path = directory / 'result.json'
    report = json.loads(report_path.read_text())
    history_path = directory / 'history.json'
    history = json.loads(history_path.read_text())
    cases = report['contract']['split']['train']
    phase = report['time_sampling']['phase_offsets']
    actual = report['time_sampling']['per_case_time_label_exposures']
    selected_epoch = report['best_epoch']
    freeze_path = directory / 'selection-freeze.json'
    private = OWN / 'domain-checkpoints' / domain / strategy / profile / 'fit'
    freeze_hash = None
    if selected_epoch is not None:
        freeze = json.loads(freeze_path.read_text())
        assert sha(report_path) == freeze['result_sha256'] and sha(history_path) == freeze['history_sha256']
        assert selected_epoch == freeze['selected_complete_epoch']
        assert sha(private/'best.pt') == freeze['checkpoint_sha256'] == report['checkpoint_sha256']
        assert sha(private/'validation-predictions.npz') == freeze['validation_prediction_sha256'] == report['validation_prediction_sha256']
        freeze_hash = sha(freeze_path)
    else:
        assert not freeze_path.exists() and not (private/'best.pt').exists()
    selected = complete_exposures(cases, phase, selected_epoch or 0, strategy)
    last_complete = complete_exposures(cases, phase, report['last_complete_epoch'] or 0, strategy)
    nodes, _, _ = report['contract']['shape']
    summarize = lambda mapping: exposure_summary(mapping, nodes, report['channels'])
    assert sum(map(sum, actual.values())) == report['updates'] * (1 if strategy == 'time_conditional' else 10)
    selected_row = next((row for row in history if row['epoch'] == selected_epoch), None)
    result = {'domain': domain, 'strategy': strategy, 'profile': profile,
              'source_result_sha256': sha(report_path), 'source_history_sha256': sha(history_path),
              'selection_freeze_sha256': freeze_hash,
              'source_script_sha256': sha(__file__), 'selected_epoch': selected_epoch,
              'selected_checkpoint_updates': selected_row['updates'] if selected_row else 0,
              'selected_checkpoint_train_seconds': selected_row['cumulative_train_seconds'] if selected_row else 0,
              'all_executed_updates': report['updates'],
              'all_executed_train_seconds': report['actual_optimizer_training_seconds'],
              'checkpoint_selected_from_complete_epoch': bool(selected_row and selected_row['epoch_complete']),
              'selected_checkpoint': summarize(selected),
              'all_executed_including_terminal_partial': summarize(actual),
              'work_after_selected_checkpoint': summarize(difference(actual, selected)),
              'terminal_partial_epoch_only': summarize(difference(actual, last_complete)),
              'interpretation': 'TC ten completed sampled epochs cover every case-time pair once. V3 AR/TF consume all ten C6/C7 native-field labels per case update. Counts are optimization exposure, not independent examples.',
              'budget_comparison_limit': 'Temporal budget includes input transfer; static 240s comparison excludes transfer. Validation and one-off setup are excluded from both temporal training budget and exposure counts.',
              'test_used_for_selection': False}
    write_once(directory / 'exposure-accounting.json', result)
    return result


def posthoc(domain, strategy):
    # Framework-free; labels are read from the already frozen final NPZ only.
    import numpy as np
    sys.path.insert(0, str(ROOT / 'work/expanded-ml'))
    from benchmark_data import physical_metrics
    profile = PROFILES[strategy]
    directory = OUT / domain / strategy / profile / 'fit'
    private = OWN / 'domain-checkpoints' / domain / strategy / profile / 'fit'
    final_path = directory / 'final-test.json'
    final = json.loads(final_path.read_text())
    prediction_path = private / 'test-predictions.npz'
    assert sha(prediction_path) == final['prediction_sha256']
    with np.load(prediction_path) as saved:
        prediction, truth = saved['prediction'], saved['truth']
    contract = final['contract']
    channels = prediction.shape[-1]
    args = (contract['names'][:channels], contract['units'][:channels], contract['split']['test'], contract['time_values'])
    metric = physical_metrics(prediction, truth, *args)
    assert metric == final['metrics'], 'Saved raw final prediction metrics changed'
    manifest_path = ROOT / 'work/expanded-ml' / ('phone-full-cache-v2/manifest.json' if domain == 'phone_full' else 'multifield7-cache/manifest.json')
    assert sha(manifest_path) == contract['cache_ready']['manifest_sha256']
    meta = json.loads(manifest_path.read_text())
    assert meta['split'] == contract['split']
    ready_path = manifest_path.with_name('READY.json')
    assert sha(ready_path) == contract['cache_ready']['ready_sha256']
    ready = json.loads(ready_path.read_text())
    assert ready['status'] == 'ready' and ready['manifest_sha256'] == contract['cache_ready']['manifest_sha256']
    inventory = {row['name']: row['sha256'] for row in ready['files']}
    train_mean = np.zeros(truth.shape[1:], np.float64)
    mean_sources = []
    for case in contract['split']['train']:
        relative = meta['runs'][case]['target']; target_path = manifest_path.parent / relative
        digest = sha(target_path)
        assert digest == inventory[relative]
        train_mean += np.load(target_path, mmap_mode='r')[..., :channels]
        mean_sources.append({'case': case, 'target_sha256': digest})
    train_mean /= len(mean_sources)
    ballistic = np.broadcast_to(np.asarray(contract['initial_velocity'], np.float64)[None,None,None,:]
                               * np.asarray(contract['time_values'], np.float64)[None,None,:,None], truth[...,:3].shape)
    raw_signs = {name: {'reference_strict_positive_count': int(np.count_nonzero(truth[...,i] > 0)),
                        'reference_exact_zero_count': int(np.count_nonzero(truth[...,i] == 0)),
                        'reference_strict_negative_count': int(np.count_nonzero(truth[...,i] < 0)),
                        'prediction_strict_negative_count': int(np.count_nonzero(prediction[...,i] < 0))}
                 for i,name in enumerate(args[0]) if i >= 3}
    result = {'domain': domain, 'strategy': strategy, 'profile': profile,
              'final_result_sha256': sha(final_path), 'prediction_sha256': sha(prediction_path),
              'checkpoint_sha256': final['checkpoint_sha256'], 'baseline_manifest_sha256': sha(manifest_path),
              'baseline_ready_sha256': sha(ready_path),
              'training_mean_sources': mean_sources,
              'training_mean_policy': 'Arithmetic pointwise mean of exactly the frozen training cases, reconstructed from SHA-verified cache targets; no validation or test targets contribute.',
              'definition': 'Posthoc arithmetic on saved single-pass final prediction and truth. No model loading, new inference, training, clipping, threshold selection or test tuning.',
              'raw_final_metrics_recomputed_identically': True,
              'native_scalar_raw_sign_counts': raw_signs,
              'sign_count_note': 'Exact zeros and negative references are separated here. In the inherited V2 EPSP support diagnostic, the field named reference_exact_zero_count denotes the non-positive complement of strict positive support.',
              'train_mean_baseline': physical_metrics(np.broadcast_to(train_mean, truth.shape), truth, *args),
              'zero_baseline': physical_metrics(np.zeros_like(truth), truth, *args),
              'source_velocity_ballistic_displacement_only_baseline': physical_metrics(ballistic, truth[...,:3], *tuple([args[0][:3], args[1][:3], args[2], args[3]])),
              'ballistic_scope': 'Source initial velocity times nominal target time; no contact, stress or strain model.',
              'centroid_translation_truth_energy_fraction': float(1 - np.sum((truth[...,:3].astype(np.float64)-truth[...,:3].mean(axis=1,keepdims=True,dtype=np.float64))**2) / np.sum(truth[...,:3].astype(np.float64)**2)),
              'raw_outputs_preserved': True, 'test_used_for_selection': False}
    write_once(directory / 'saved-prediction-posthoc.json', result)
    return result


def selfcheck():
    cases = ['a','b']; phases = {'a':0,'b':5}
    for strategy in ['time_conditional','ar_rollout_training']:
        ten = complete_exposures(cases, phases, 10, strategy)
        assert all(value == (1 if strategy == 'time_conditional' else 10) for row in ten.values() for value in row)
        selected = complete_exposures(cases, phases, 3, strategy)
        delta = difference(ten, selected)
        assert sum(map(sum, delta.values())) == (14 if strategy == 'time_conditional' else 140)
        summary = exposure_summary(selected, 4, 3)
        assert summary['scalar_target_exposures'] == (72 if strategy == 'time_conditional' else 720)
    assert 'numpy' not in sys.modules and 'torch' not in sys.modules
    print(json.dumps({'status':'pass','scope':'standard-library exposure accounting only','framework_imported':False,'numpy_imported':False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--domain', choices=['phone_full','bumper'])
    parser.add_argument('--strategy', choices=['time_conditional','ar_rollout_training','one_step_teacher_forcing'])
    parser.add_argument('--posthoc', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        assert args.domain and args.strategy
        value = posthoc(args.domain,args.strategy) if args.posthoc else exposures(args.domain,args.strategy)
        print(json.dumps({'status':'written','domain':args.domain,'strategy':args.strategy,'posthoc':args.posthoc}))
