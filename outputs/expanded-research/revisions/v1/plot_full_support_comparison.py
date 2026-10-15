"""Plot saved JSON only, after twelve completed tests and an explicit root CPU window.

Preparation/import/check-inputs is stdlib-only. Rendering lazily imports Matplotlib.
No arrays, weights, model calls, fitting, tests, clipping, or metric recomputation.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PROTOCOL = BASE/'full-support-comparison-execution-protocol.json'
PNG = BASE/'full-support-algorithm-comparison.png'
JSON = BASE/'full-support-algorithm-comparison.json'
MODELS = ('transolver', 'flare', 'geotransolver', 'geoflare', 'meshgraphnet', 'figconvunet')
LABELS = ('Transolver', 'FLARE', 'GeoTransolver GALE', 'GeoTransolver GALE_FA', 'MeshGraphNet', 'FIGConvUNet', 'Training mean', 'POD + RBF', 'POD + Ridge', 'Direct RBF')
RUNS = {'phone': BASE/'benchmarks/drop_test/models-full-c6', 'bumper': BASE/'benchmarks/bumper/models-c7'}
ROM = {'phone': 'phone9-full', 'bumper': 'bumper7'}
GIB = 1024**3
MIB = 1024**2


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


class Inputs:
    """Read small JSON/source bytes; explicitly reject arrays and model checkpoints."""
    def __init__(self):
        self.files = {}

    def track(self, path, expected=None):
        path = Path(path).resolve()
        assert path.suffix in ('.json', '.py'), f'Forbidden non-metadata input: {path}'
        assert path.stat().st_size <= 2*MIB, f'Metadata exceeds bounded size: {path}'
        digest = sha(path)
        if expected is not None:
            assert digest == expected, f'Frozen metadata/source hash changed: {path}'
        if str(path) in self.files:
            assert self.files[str(path)]['sha256'] == digest, f'Input changed during reading: {path}'
        self.files[str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}
        return digest

    def read(self, path):
        self.track(path)
        def reject(value):
            raise ValueError(f'Nonfinite JSON constant: {value}')
        return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)

    def unchanged(self):
        for name, expected in self.files.items():
            assert sha(name) == expected['sha256'], f'Input changed during chart generation: {name}'


def by_name(mapping, name):
    values = [value for key, value in mapping.items() if Path(key).name == name]
    assert values and len(set(values)) == 1, f'Missing/ambiguous recorded digest: {name}'
    return values[0]


def readiness():
    rows = []
    for domain, folder in RUNS.items():
        for model in MODELS:
            path = folder/model/'test_report.json'
            status = 'missing'
            if path.exists():
                assert path.stat().st_size <= 2*MIB
                status = json.loads(path.read_text(encoding='utf-8-sig')).get('status', 'missing_status')
            rows.append({'domain': domain, 'model': model, 'status': status, 'path': str(path.relative_to(BASE))})
    return {'checked_utc': utc(), 'all_twelve_tests_completed': all(x['status'] == 'completed' for x in rows),
            'completed_count': sum(x['status'] == 'completed' for x in rows), 'rows': rows,
            'rendered': False, 'frameworks_imported': False}


def metric_value(metrics, field, key, factor, unit):
    # Missing keys are an error. A recorded null remains undefined, never zero.
    raw = metrics[field][key]
    assert raw is None or isinstance(raw, (int, float)) and not isinstance(raw, bool)
    assert raw is None or math.isfinite(raw) and raw >= 0
    return {'source_field': field, 'source_key': key, 'source_value': raw,
            'display_factor': factor, 'value': None if raw is None else raw*factor,
            'unit': unit, 'display_status': 'undefined_null' if raw is None else ('exact_zero_annotation' if raw == 0 else 'positive_log_coordinate')}


def panel_specs(domain):
    specs = [
        ('Displacement vector\nrelative L2 [%]', [('displacement_vector', 'relative_l2', 100., '%', 'single')]),
        ('Centroid-removed vector\nrelative L2 [%]', [('displacement_vector', 'centroid_removed_relative_l2', 100., '%', 'single')])]
    if domain == 'phone':
        specs += [
            ('GPS nodal VM\nRMSE [MPa]', [('von_mises_GPS', 'rmse', 1., 'MPa', 'single')]),
            ('GPS nodal VM\nrelative L2 [%]', [('von_mises_GPS', 'relative_l2', 100., '%', 'single')]),
            ('Total-strain Frobenius\nrelative L2 [%]', [('total_strain_frobenius', 'relative_l2', 100., '%', 'single')]),
            ('EPSP\nrelative L2 [%]', [('epsp', 'relative_l2', 100., '%', 'single')])]
    else:
        specs += [
            ('Projected layer VM\nRMSE [MPa]', [(f'inplane_vm_{layer}', 'rmse', 1000., 'MPa', layer) for layer in ('upper', 'lower')]),
            ('Projected layer VM\nrelative L2 [%]', [(f'inplane_vm_{layer}', 'relative_l2', 100., '%', layer) for layer in ('upper', 'lower')]),
            ('Upper EPSP\nrelative L2 [%]', [('EPSP_upper', 'relative_l2', 100., '%', 'upper')]),
            ('Lower EPSP\nrelative L2 [%]', [('EPSP_lower', 'relative_l2', 100., '%', 'lower')])]
    return specs


def collect(inputs):
    gate = readiness()
    assert gate['all_twelve_tests_completed'], 'All twelve completed test_report.json files are required; no other dataset/support substitution.'
    mean_all = inputs.read(BASE/'full-support-mean-baselines.json')
    domains = {}
    for domain, folder in RUNS.items():
        mean = mean_all['domains'][domain]
        assert mean['status'] == 'computed_and_self_checked'
        pod = inputs.read(BASE/'benchmarks/rom'/ROM[domain]/'metrics.json')
        direct = inputs.read(BASE/'benchmarks/rom'/(ROM[domain]+'-direct-rbf')/'metrics.json')
        assert pod['status'] == 'actual_POD_fit_validation_selection_and_heldout_evaluation_completed'
        assert direct['status'] == 'rbf_direct_field_training_and_heldout_evaluation_completed'
        contract = pod['contract']; assert contract == direct['contract']
        shape = [len(contract['split']['test'])]+contract['shape']
        assert mean['shape'] == shape and mean['test_cases'] == contract['split']['test']
        assert mean['training_cases'] == contract['split']['train']
        assert mean['field_names'] == contract['fields'] and mean['field_units'] == contract['units'] and mean['time_values'] == contract['times']
        assert mean['provenance']['manifest_sha256'] == contract['manifest_sha256']
        assert mean['provenance']['ready_sha256'] == pod['dataset_freeze']['ready_evidence_sha256'] == direct['dataset_freeze']['ready_sha256']
        assert mean['common_normalization_file_sha256'] == pod['shared_normalization_sha256'] == direct['shared_normalization_sha256']
        assert not mean['raw_predictions_clipped']
        methods = []
        for model, label in zip(MODELS, LABELS):
            run = folder/model
            report = inputs.read(run/'test_report.json')
            frozen = inputs.read(run/'checkpoint-frozen.json')
            training = inputs.read(run/'training_report.json')
            assert report['status'] == 'completed' and report['test_performance_passes'] == 1
            assert frozen['status'] == 'checkpoint_fixed_and_validation_roundtrip_passed'
            assert training['status'].startswith('training_completed_')
            inputs.track(run/'training_report.json', frozen['training_report_sha256'])
            inputs.track(run/'checkpoint-frozen.json', by_name(report['frozen_evidence_sha256'], 'checkpoint-frozen.json'))
            assert report['checkpoint_sha256'] == frozen['checkpoint_sha256']
            assert report['source_sha256'] == frozen['source_sha256'] == training['source_sha256']
            for name, expected in report['source_sha256'].items():
                inputs.track(run/'source'/Path(name).name, expected)
            native = report['model_source']
            if native.get('source_file') and native.get('source_sha256'):
                inputs.track(native['source_file'], native['source_sha256'])
            assert report['shape'] == shape and report['cases'] == mean['test_cases']
            assert report['field_names'] == contract['fields'] and report['field_units'] == contract['units'] and report['time_values'] == contract['times']
            assert report['time_unit'] == mean['time_unit']
            assert report['frozen_data_verification']['manifest_sha256'] == contract['manifest_sha256']
            assert report['frozen_data_verification']['ready_sha256'] == mean['provenance']['ready_sha256']
            assert by_name(report['frozen_evidence_sha256'], 'normalization.json') == mean['common_normalization_file_sha256']
            assert by_name(report['frozen_evidence_sha256'], 'training_mean_field.npy') == mean['mean_sha256']
            assert by_name(report['output_sha256'], 'truth.npy') == mean['saved_truth_sha256']
            assert not report['raw_predictions_clipped']
            budget = training['budget_accounting']
            methods.append({'id': model, 'label': label, 'kind': 'neural', 'metrics': report['physical_metrics'],
                            'seed': training['seed'], 'quality_budget_passed': report['quality_gates']['gate_passed'],
                            'optimizer_updates': budget['total_updates'], 'case_observations': budget['total_samples'],
                            'optimizer_compute_seconds': budget['train_seconds'], 'convergence': training['convergence'],
                            'evidence': str((run/'test_report.json').relative_to(BASE))})
        assert len({method['seed'] for method in methods}) == 1
        methods += [
            {'id': 'training_mean', 'label': LABELS[6], 'kind': 'mean', 'metrics': mean['physical_metrics'], 'fit_note': 'Training-only mean, not an independent neural fit'},
            {'id': 'pod_rbf', 'label': LABELS[7], 'kind': 'rom', 'metrics': pod['test_metrics']['pod_rbf'], 'fit_note': 'Shares train POD decomposition with POD+Ridge', 'selected': pod['selected_by_validation_only']['pod_rbf']},
            {'id': 'pod_ridge', 'label': LABELS[8], 'kind': 'rom', 'metrics': pod['test_metrics']['pod_ridge'], 'fit_note': 'Shares train POD decomposition with POD+RBF', 'selected': pod['selected_by_validation_only']['pod_ridge']},
            {'id': 'direct_rbf', 'label': LABELS[9], 'kind': 'rom', 'metrics': direct['metrics']['rbf_direct_field'], 'fit_note': 'Phone full-rank POD+RBF and direct RBF are nearly equivalent here; not two independent superiority results'}]
        panels = []
        for title, series in panel_specs(domain):
            panel = {'title': title, 'series': []}
            for field, key, factor, unit, layer in series:
                panel['series'].append({'layer': layer, 'values': [metric_value(method['metrics'], field, key, factor, unit) for method in methods]})
            panels.append(panel)
        for method in methods:
            del method['metrics']
        domains[domain] = {'shape': shape, 'test_cases': mean['test_cases'], 'field_names': contract['fields'],
                           'field_units': contract['units'], 'time_values': contract['times'], 'time_unit': mean['time_unit'],
                           'manifest_sha256': contract['manifest_sha256'], 'READY_sha256': mean['provenance']['ready_sha256'],
                           'normalization_file_sha256': mean['common_normalization_file_sha256'],
                           'methods': methods, 'panels': panels}
    return domains


class HostMemory(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('load', wintypes.DWORD)] + [(x, ctypes.c_ulonglong) for x in ('tp', 'ap', 'tc', 'ac', 'tv', 'av', 'ex')]


class ProcessMemory(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in ('peak_rss', 'rss', 'pp', 'p', 'pnp', 'np', 'pf', 'ppf')]


def memory_sample():
    host = HostMemory(); host.size = ctypes.sizeof(host)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(host))
    proc = ProcessMemory(); proc.size = ctypes.sizeof(proc)
    getproc = ctypes.windll.kernel32.GetCurrentProcess; getproc.restype = ctypes.c_void_p
    getmem = ctypes.windll.psapi.GetProcessMemoryInfo
    getmem.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    assert getmem(getproc(), ctypes.byref(proc), proc.size)
    return {'utc': utc(), 'host_available_bytes': int(host.ap), 'rss_bytes': int(proc.rss), 'peak_rss_bytes': int(proc.peak_rss)}


def single_cpu():
    getproc = ctypes.windll.kernel32.GetCurrentProcess; getproc.restype = ctypes.c_void_p
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    getmask = ctypes.windll.kernel32.GetProcessAffinityMask
    getmask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    assert getmask(getproc(), ctypes.byref(process), ctypes.byref(system))
    chosen = process.value & -process.value
    setmask = ctypes.windll.kernel32.SetProcessAffinityMask
    setmask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    assert setmask(getproc(), chosen)
    return chosen


def monitor(state, stop, lock):
    try:
        while not stop.wait(.25):
            sample = memory_sample()
            with lock:
                state['resource_samples'].append(sample)
                if sample['host_available_bytes'] < 512*MIB or sample['peak_rss_bytes'] > 250*MIB:
                    state.update(status='resource_guard_stop', stop_sample=sample, PNG_valid=False,
                                 reason='Own postprocessor stopped at512MiB host floor or250MiB process peak RSS; no model work was performed.')
                    write_json(JSON, state)
                    os._exit(74)
    except Exception as error:
        with lock:
            state.update(status='resource_monitor_failed', PNG_valid=False, monitor_error=repr(error))
            write_json(JSON, state)
        os._exit(76)


def render(domains):
    # This is the only third-party import path; authorization/preflight precede it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 6, figsize=(23, 12), sharey='row')
    fig.subplots_adjust(left=.115, right=.99, top=.835, bottom=.20, wspace=.32, hspace=.48)
    colors = ['#245a81']*6+['#222222', '#bc7620', '#80528e', '#278773']
    for row, domain in enumerate(('phone', 'bumper')):
        data = domains[domain]
        labels = [m['label']+(' †' if m.get('quality_budget_passed') is False else '') for m in data['methods']]
        for column, panel in enumerate(data['panels']):
            axis = axes[row, column]
            positives = [v['value'] for series in panel['series'] for v in series['values'] if v['value'] is not None and v['value'] > 0]
            axis.set_xscale('log')
            axis.set_xlim(min(positives)/1.5, max(positives)*1.5) if positives else axis.set_xlim(1, 10)
            for series in panel['series']:
                marker = '^' if series['layer'] == 'lower' else 'o'
                offset = (-.12 if series['layer'] == 'upper' else .12) if len(panel['series']) > 1 else 0
                for i, value in enumerate(series['values']):
                    y = i+offset
                    if value['value'] is None or value['value'] == 0:
                        axis.text(.02, y, 'null' if value['value'] is None else '0 exact', transform=axis.get_yaxis_transform(), va='center', fontsize=8, color=colors[i])
                    else:
                        axis.scatter(value['value'], y, s=37, marker=marker, color=colors[i], linewidth=.5, edgecolor='white', zorder=3)
            axis.set_ylim(9.6, -.6); axis.set_yticks(list(range(10)), labels)
            axis.tick_params(axis='y', labelleft=column == 0, labelsize=11)
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=3)); axis.xaxis.set_major_formatter(LogFormatterSciNotation())
            axis.xaxis.set_minor_locator(NullLocator()); axis.grid(axis='x', color='#d5dce1', linewidth=.7, alpha=.8)
            axis.set_title(panel['title'], fontsize=12, pad=11)
    fig.suptitle('Full-support saved-FEM benchmark comparison', fontsize=21, y=.985)
    fig.text(.115, .945, 'Physical pooled test errors | fixed geometry | single seed | bounded NN optimizer budget', fontsize=12)
    fig.text(.115, .905, 'PHONE — 76,065 corners; 2 test cases × 10 nominal 2–20 µs times', fontsize=14, weight='bold')
    fig.text(.115, .515, 'BUMPER — 13,675 nodes; 7 test cases × 10 nominal 10–100 ms times', fontsize=14, weight='bold')
    notes = [
        'Metric x axes are logarithmic and have separate physical units. Bumper VM: circle = upper layer; triangle = lower layer.',
        'Phone total strain = Frobenius norm of projected tensor; EPSP is separate. Bumper EPSP is plastic strain, not total strain.',
        'Nodal heads/projections are not native cell/IP peaks. null is omitted; exact zero is text-only, never clipped onto a log axis.',
        'NN: bounded 240 s compute, one seed; convergence is unverified. † marks an insufficient quality budget, if present.',
        'ROM predictors share train-only POD bases; phone full-rank POD+RBF ≈ direct RBF. Reloads/variants are not extra algorithm families.'
    ]
    for i, note in enumerate(notes): fig.text(.03, .135-i*.026, note, fontsize=10.5)
    fig.savefig(PNG, dpi=160, facecolor='white')
    actual_axes = []
    for row, domain in enumerate(('phone', 'bumper')):
        for column, panel in enumerate(domains[domain]['panels']):
            axis = axes[row, column]; lo, hi = axis.get_xlim()
            ticks = [(float(value), label.get_text()) for value, label in zip(axis.get_xticks(), axis.get_xticklabels()) if lo <= value <= hi]
            values = [v['value'] for series in panel['series'] for v in series['values']]
            actual_axes.append({'domain': domain, 'panel_title': panel['title'], 'x_limits': [float(lo), float(hi)],
                                'visible_major_ticks': ticks, 'scale': axis.get_xscale(),
                                'null_annotations': sum(x is None for x in values),
                                'exact_zero_annotations': sum(x == 0 for x in values if x is not None)})
    plt.close(fig)
    return {'matplotlib_version': matplotlib.__version__, 'PNG_pixels': [3680, 1920], 'layout': '2domain rows ×6separate metric axes,10methods per axis', 'metric_axis_scale': 'log_x', 'actual_axes': actual_axes, 'zero_policy': 'Exact zero annotated at axis fraction, not assigned a false log coordinate', 'null_policy': 'Recorded null annotated and excluded from numeric axis'}


def selfcheck():
    assert metric_value({'x': {'v': .5}}, 'x', 'v', 100., '%')['value'] == 50
    assert metric_value({'x': {'v': None}}, 'x', 'v', 1., '1')['value'] is None
    assert metric_value({'x': {'v': 0}}, 'x', 'v', 1., '1')['display_status'] == 'exact_zero_annotation'
    assert metric_value({'x': {'v': .25}}, 'x', 'v', 1000., 'MPa')['value'] == 250
    try: metric_value({'x': {}}, 'x', 'v', 1., '1')
    except KeyError: pass
    else: raise AssertionError('Missing metric silently replaced')
    assert len(panel_specs('phone')) == len(panel_specs('bumper')) == 6
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true', help='Stdlib-only readiness report; no chart, guards or imports')
    parser.add_argument('--cpu-window-authorized', action='store_true')
    parser.add_argument('--expected-script-sha256')
    args = parser.parse_args()
    selfcheck()
    if args.check_inputs:
        print(json.dumps(readiness(), indent=2)); return 0
    if not args.cpu_window_authorized:
        raise SystemExit('Root CPU-window authorization missing: no framework import or chart generation occurred.')
    assert args.expected_script_sha256 and sha(__file__) == args.expected_script_sha256
    inputs = Inputs(); protocol = inputs.read(PROTOCOL)
    assert protocol['script_sha256'] == args.expected_script_sha256
    inputs.track(__file__, args.expected_script_sha256)
    assert not PNG.exists() and not JSON.exists(), 'One chart execution only; never overwrite existing output evidence.'
    affinity = single_cpu(); preflight = memory_sample()
    if preflight['host_available_bytes'] < int(1.5*GIB) or preflight['peak_rss_bytes'] > 250*MIB:
        print(json.dumps({'status': 'preflight_deferred_no_import_or_output', 'sample': preflight})); return 75
    domains = collect(inputs)
    pre_import = memory_sample()
    if pre_import['host_available_bytes'] < int(1.5*GIB):
        print(json.dumps({'status': 'pre_import_deferred_no_import_or_output', 'sample': pre_import})); return 75
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    state = {'status': 'running', 'started_utc': utc(), 'source_sha256': args.expected_script_sha256,
             'root_cpu_window_authorized': True, 'CPU_affinity_single_bit': affinity,
             'resource_policy': protocol['resource_policy'], 'resource_samples': [preflight, pre_import],
             'inputs': inputs.files, 'domains': domains, 'model_calls': 0,
             'array_or_checkpoint_files_opened': 0, 'metric_work': 'Saved scalar JSON values with display-only unit conversions',
             'provenance_scope': 'Saved metadata digests compared; original NPY/NPZ/PT arrays/weights never opened.',
             'visual_review': {'status': 'pending_external_view_image', 'required': True,
                               'instruction': 'Root must open the actual PNG with view_image and record readability/units/legend QA before final delivery.'}}
    with JSON.open('x', encoding='utf-8') as stream: json.dump(state, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stop = threading.Event(); lock = threading.Lock()
    watcher = threading.Thread(target=monitor, args=(state, stop, lock), daemon=True); watcher.start()
    try:
        display = render(domains)
        inputs.unchanged()
        final = memory_sample()
        assert final['host_available_bytes'] >= 512*MIB and final['peak_rss_bytes'] <= 250*MIB
        with lock:
            state['resource_samples'].append(final)
            state.update(status='rendered_pending_actual_visual_review', finished_utc=utc(), display=display,
                         PNG_valid=True, PNG_sha256=sha(PNG), PNG_bytes=PNG.stat().st_size,
                         minimum_sampled_host_available_bytes=min(x['host_available_bytes'] for x in state['resource_samples']),
                         maximum_process_peak_rss_bytes=max(x['peak_rss_bytes'] for x in state['resource_samples']))
    except Exception as error:
        with lock: state.update(status='failed', PNG_valid=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        stop.set(); watcher.join(timeout=2)
        with lock: write_json(JSON, state)
    print(json.dumps({'status': state['status'], 'PNG': str(PNG), 'JSON': str(JSON)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
