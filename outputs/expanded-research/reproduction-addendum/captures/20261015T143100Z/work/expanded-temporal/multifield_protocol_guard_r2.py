"""Validate frozen JSON/source evidence before importing any array or ML runtime."""

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies'
PROFILE = 'multifield_v3_native_scalar_heads'
STRATEGIES = ('time_conditional', 'ar_rollout_training', 'one_step_teacher_forcing')
DOMAINS = {'phone_full': 6, 'bumper': 7}
DECISION_SCOPE = 'actual_executed_work_selected_checkpoint_reported_separately'
FIT_EXTRAS = {'phase', 'training_budget_seconds', 'preprobe_protocol', 'probe_evidence',
              'budget_decision', 'fit_readiness'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f'{label} must be finite and positive')
    return float(value)


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f'{label} must be finite')
    return float(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value):
    require(isinstance(value, str) and bool(value), 'Reference path must be a nonempty relative path')
    path = Path(value)
    require(not path.is_absolute() and not path.drive, f'Absolute reference forbidden: {value}')
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT.resolve()), f'Reference escapes workspace: {value}')
    return resolved


def read_json(path):
    require(Path(path).suffix.lower() == '.json', 'Only JSON evidence may be read')
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    value = json.loads(Path(path).read_text(encoding='utf8'), parse_constant=invalid_constant)
    require(isinstance(value, dict), f'JSON object required: {path}')
    return value


def read_reference(reference, label):
    require(isinstance(reference, dict), f'{label} reference required')
    digest = reference.get('sha256')
    require(isinstance(digest, str) and len(digest) == 64
            and all(c in '0123456789abcdef' for c in digest), f'{label} needs a lowercase SHA256')
    path = relative_path(reference.get('path'))
    require(path.suffix.lower() == '.json', f'{label} must reference JSON, never an array or checkpoint')
    require(sha(path) == digest, f'{label} hash mismatch: {path}')
    return read_json(path)


def profile_for(strategy):
    return 'budget_v2_complete_epoch' if strategy == 'time_conditional' else PROFILE


def artifact_path(domain, strategy, kind):
    stage, filename = ('probe', 'probe.json') if kind == 'functional' else ('stream-probe', 'stream-profile.json')
    return OUT / domain / strategy / profile_for(strategy) / stage / filename


def _same_number(actual, expected, label):
    require(math.isclose(number(actual, label), expected, rel_tol=1e-9, abs_tol=1e-12),
            f'{label} does not match measured steps')


def _frozen_digest(sources, relative):
    path = relative_path(relative)
    require(path in sources, f'Required source absent from frozen inventory: {relative}')
    return sources[path]


def _source_contract(report, preprobe, preprobe_sha, domain, strategy, frozen_sources):
    source = report.get('source')
    require(isinstance(source, dict) and source.get('protocol_sha256') == preprobe_sha,
            'Artifact source must bind the preprobe protocol SHA256')
    if 'protocol_sha256' in report:
        require(report['protocol_sha256'] == preprobe_sha, 'Artifact protocol SHA256 mismatch')
    expected_wrapper = ('work/expanded-temporal/sources/'
                        + ('crash' if domain == 'bumper' else 'drop_test') + '/rollout.py'
                        if strategy == 'time_conditional' else 'work/expanded-temporal/native_scalar_rollout_v3.py')
    for file_key, hash_key, expected_path in [
        ('wrapper_file', 'wrapper_sha256', expected_wrapper),
        ('core_class_file', 'core_class_sha256', preprobe['core_source_path']),
    ]:
        reported = Path(source.get(file_key, ''))
        reported = reported.resolve() if reported.is_absolute() else relative_path(str(reported))
        expected = relative_path(expected_path)
        require(reported == expected and source.get(hash_key) == _frozen_digest(frozen_sources, expected_path),
                f'Artifact {file_key} differs from frozen source')
    driver = 'run_domain_temporal.py' if strategy == 'time_conditional' else 'run_budget_temporal_v3.py'
    require(source.get('driver_sha256') == _frozen_digest(frozen_sources, 'work/expanded-temporal/' + driver),
            'Artifact driver differs from frozen source')
    for key in ('prepared_driver_sha256', 'entry_shim_sha256'):
        if key in source:
            require(source[key] in frozen_sources.values(), f'Artifact {key} is not a frozen source')
    contract = report.get('contract', {})
    metadata = preprobe['domains'][domain]
    require(contract.get('domain') == domain and contract.get('split') == metadata['split'],
            'Artifact domain/split differs from frozen train/validation/test metadata')
    require(contract.get('normalization_sha256') == metadata['normalization_sha256'],
            'Artifact normalization differs from preprobe')
    require(contract.get('cache_ready', {}).get('manifest_sha256') == metadata['manifest_sha256'],
            'Artifact manifest differs from preprobe')
    require(contract.get('frozen_protocol', {}).get('sha256') == preprobe_sha,
            'Artifact contract must bind the preprobe protocol')
    require(contract.get('budget_profile') == profile_for(strategy), 'Artifact contract profile mismatch')


def _identity(report, domain, strategy):
    require(report.get('domain') == domain and report.get('strategy') == strategy,
            'Artifact domain/strategy mismatch')
    require(report.get('profile') == profile_for(strategy), 'Artifact profile mismatch')
    require(report.get('channels') == DOMAINS[domain], 'Artifact must use native C6/C7 channels')
    for key in ('test_predicted', 'test_target_values_accessed', 'validation_target_values_accessed'):
        require(report.get(key) is False, f'Artifact must explicitly record {key}=false')


def _functional(report, preprobe, domain, strategy):
    require(report.get('steps') == 3, 'Functional probe must execute three updates')
    require(report.get('case') == preprobe['domains'][domain]['split']['train'][0],
            'Functional probe must use the first frozen train case')
    require(len(report.get('seconds', [])) == 3, 'Functional probe needs three measured update times')
    for value in report['seconds']:
        positive(value, 'Functional update seconds')
    if strategy == 'time_conditional':
        require(report.get('status') == 'actual_FEM_forward_backward_optimizer_probe_only',
                'TC needs the actual FEM optimizer probe')
        positive(report.get('changed_parameters'), 'TC changed parameters')
        require(number(report.get('inference_target_value_independence_max_abs'), 'TC target independence') == 0,
                'TC inference changed under target poisoning')
        return
    require(report.get('status') == 'actual_native_scalar_three_joint_Adam_updates_train_only_probe',
            'Native scalar functional probe status mismatch')
    require(report.get('all_output_rows_finite_nonzero_gradient_and_changed_each_update') is True,
            'Every native output row must have a finite nonzero gradient and change each update')
    channels = DOMAINS[domain]
    for key in ('head_weight_gradient_l2_by_row', 'head_weight_update_l2_by_row'):
        rows = report.get(key, [])
        require(isinstance(rows, list) and len(rows) == 3, f'{key} needs three measured rows')
        for row in rows:
            require(isinstance(row, list) and len(row) == channels, f'{key} needs every native output channel')
            for value in row:
                positive(value, key)
    poison = report.get('eval_all_target_value_poisoning', {})
    require(poison.get('all_outputs_exactly_equal') is True
            and number(poison.get('max_abs_difference'), 'Eval poison difference') == 0,
            'All evaluation outputs must be exactly invariant to target values')
    if strategy == 'one_step_teacher_forcing':
        poison = report.get('TF_training_scalar_target_poisoning', {})
        require(isinstance(poison, dict) and poison.get('all_outputs_exactly_equal') is True
                and poison.get('past_position_targets_preserved') is True
                and poison.get('all_scalar_target_values_replaced') is True
                and number(poison.get('max_abs_difference'), 'TF scalar poison difference') == 0,
                'TF scalar-target poison invariance failed')
    equivalence = report.get('original_C3_position_equivalence_before_training', {})
    tolerance = positive(equivalence.get('tolerance_in_training_scale_units'), 'C3 equivalence tolerance')
    difference = number(equivalence.get('max_abs_difference_in_training_scale_units'), 'C3 equivalence difference')
    require(tolerance <= 1e-5 and 0 <= difference <= tolerance
            and equivalence.get('original_channels') == 3
            and equivalence.get('extended_channels') == channels
            and equivalence.get('first_train_case_only') is True
            and equivalence.get('pre_training') is True and equivalence.get('used_for_fit') is False,
            'Original C3 train-case position equivalence failed')


def stream_projection(report, train_count, budget):
    """Recompute estimates from separate wall-clock update measurements, never loss values."""
    measured = [positive(step.get('measured_update_wall_seconds'), 'Measured stream update')
                for step in report['steps']]
    warm = measured[1:]
    required = max(100, 2 * train_count)
    conservative = max(required * max(warm), measured[0] + (required - 1) * max(warm))
    return {'required_updates': required, 'minimum_updates': 100, 'minimum_complete_case_epochs': 2,
            'train_case_count': train_count, 'warm_median_update_wall_seconds': statistics.median(warm),
            'warm_max_update_wall_seconds': max(warm), 'cold_first_update_wall_seconds': measured[0],
            'median_estimate_seconds': statistics.median(warm) * required,
            'conservative_estimate_seconds': conservative,
            'proposed_budget_seconds': budget, 'projects_minimum_executed_work': budget >= conservative,
            'estimates_only': True,
            'uncertainties': ['Five measured train cases may not cover later case or thermal variability.',
                              'Validation and setup are outside the optimizer budget.',
                              'Actual updates and complete case epochs must be checked after the fit; selected checkpoint exposure is separate.']}


def _stream(report, preprobe, domain, budget):
    require(report.get('status') == 'actual_five_distinct_train_case_optimizer_stream_profile',
            'Stream profile must be an actual distinct-train-case optimizer measurement')
    train = preprobe['domains'][domain]['split']['train']
    cases = report.get('case_ids', [])
    require(isinstance(cases, list) and len(cases) >= 5 and len(cases) == len(set(cases))
            and set(cases) <= set(train), 'Stream profile needs at least five distinct frozen train cases')
    require(report.get('selected_cases_train_only') is True and report.get('batch_size') == 1,
            'Stream profile must use train-only batch size one')
    require(report.get('train_case_ids') == train and report.get('train_case_count') == len(train),
            'Stream full train-case metadata differs from preprobe')
    steps = report.get('steps', [])
    require(isinstance(steps, list) and len(steps) == len(cases)
            and [step.get('case') for step in steps] == cases, 'Stream measured steps/cases mismatch')
    require(report.get('updates') == len(steps) and report.get('changed_parameter_updates') == len(steps),
            'Stream must execute real parameter-changing updates on every measured case')
    for step in steps:
        require(step.get('batch_size') == 1 and step.get('input_equivalence_bitwise') is True
                and step.get('head_weight_changed') is True, 'Stream sample equivalence or actual optimizer change failed')
        positive(step.get('head_weight_delta_l2'), 'Stream head weight change')
        for key in ('cpu_prepare_seconds', 'h2d_wall_seconds', 'h2d_cuda_event_seconds',
                    'compute_wall_seconds', 'compute_cuda_event_seconds', 'measured_update_wall_seconds'):
            positive(step.get(key), key)
        for phase in ('forward_loss', 'backward', 'gradient_clip', 'adam_step'):
            timings = step.get('phase_seconds', {}).get(phase, {})
            positive(timings.get('wall_seconds'), phase + ' wall seconds')
            positive(timings.get('cuda_event_seconds'), phase + ' CUDA seconds')
    projection = stream_projection(report, len(train), budget)
    claimed = report.get('projection', {})
    require(claimed.get('required_updates') == projection['required_updates']
            and claimed.get('minimum_optimizer_updates') == 100
            and claimed.get('minimum_case_epochs') == 2 and claimed.get('estimates_only') is True,
            'Stream required-work projection must use max(100, 2 * train cases)')
    for key in ('warm_median_update_wall_seconds', 'warm_max_update_wall_seconds', 'cold_first_update_wall_seconds'):
        _same_number(claimed.get(key), projection[key], 'Stream projection ' + key)
    return projection


def verify_fit_readiness(protocol) -> dict:
    """Return a resource-only readiness receipt, or raise before any optional imports."""
    require(isinstance(protocol, dict) and protocol.get('phase') == 'fit', 'Only a final fit protocol can authorize a fit')
    budget = positive(protocol.get('training_budget_seconds'), 'Common training budget')
    preprobe_ref = protocol.get('preprobe_protocol')
    preprobe = read_reference(preprobe_ref, 'Preprobe protocol')
    require(preprobe.get('phase') == 'resource_probes' and preprobe.get('training_budget_seconds') is None,
            'Preprobe protocol must authorize resource probes without an accepted fit budget')
    require({k: v for k, v in protocol.items() if k not in FIT_EXTRAS}
            == {k: v for k, v in preprobe.items() if k not in FIT_EXTRAS},
            'Fit source/domain/seed/config differs from the frozen preprobe protocol')
    require(protocol.get('profile') == PROFILE and set(protocol.get('domains', {})) == set(DOMAINS),
            'Fit protocol must cover the two native scalar domains')
    sources = {}
    for row in preprobe.get('sources', []):
        path = relative_path(row.get('path'))
        require(path not in sources, 'Duplicate frozen source path')
        require(path.suffix.lower() in {'.py', '.json', '.pth'} or path.name == 'METADATA',
                'Frozen source list accepts only source text, JSON, package .pth text or METADATA')
        require(path.suffix.lower() != '.pth' or path.name == 'zz_reuse_physicsnemo.pth',
                'Only the declared package-reuse .pth text is a source; model checkpoint .pth files are forbidden')
        require(sha(path) == row.get('sha256'), f'Frozen source changed: {path}')
        sources[path] = row['sha256']
    require(bool(sources), 'Frozen source inventory required')
    for domain, channels in DOMAINS.items():
        metadata = preprobe['domains'][domain]
        require(metadata.get('channels') == channels, 'Frozen domain native channel count differs')
        for key in ('manifest', 'ready', 'normalization'):
            read_reference({'path': metadata[key + '_path'], 'sha256': metadata[key + '_sha256']}, domain + ' ' + key)
    decision = protocol.get('budget_decision', {})
    require(decision.get('root_authorized') is True and decision.get('basis') == 'train_only_resource_stream'
            and decision.get('test_or_validation_accuracy_used') is False,
            'An explicit root resource-only budget decision is required')
    require(positive(decision.get('uniform_budget_seconds'), 'Decision common budget') == budget and decision.get('minimum_updates') == 100
            and decision.get('minimum_complete_case_epochs') == 2
            and decision.get('quality_gate_scope') == DECISION_SCOPE,
            'Budget decision differs from common budget or actual-work minimums')
    rows = protocol.get('probe_evidence', [])
    require(isinstance(rows, list) and len(rows) == 6, 'Exactly six functional-plus-stream evidence pairs required')
    seen = set()
    projections = []
    for row in rows:
        pair = (row.get('domain'), row.get('strategy'))
        require(pair[0] in DOMAINS and pair[1] in STRATEGIES and pair not in seen,
                'Evidence must contain six distinct domain/strategy pairs')
        seen.add(pair)
        domain, strategy = pair
        for kind in ('functional', 'stream'):
            reference = row.get(kind)
            report = read_reference(reference, f'{domain}/{strategy}/{kind}')
            require(relative_path(reference['path']) == artifact_path(domain, strategy, kind).resolve(),
                    'Evidence path differs from canonical profile artifact')
            _identity(report, domain, strategy)
            _source_contract(report, preprobe, preprobe_ref['sha256'], domain, strategy, sources)
            if kind == 'functional':
                _functional(report, preprobe, domain, strategy)
            else:
                projection = _stream(report, preprobe, domain, budget)
                helper = 'work/expanded-temporal/profile_temporal_case_stream.py'
                require(report.get('profile_helper_sha256') == _frozen_digest(sources, helper), 'Stream profiler source hash mismatch')
                projections.append({'domain': domain, 'strategy': strategy, 'profile': profile_for(strategy), **projection})
    require(seen == {(domain, strategy) for domain in DOMAINS for strategy in STRATEGIES},
            'A required domain/strategy evidence pair is missing')
    if 'resource_projections' in decision:
        require(decision['resource_projections'] == projections, 'Saved budget projections differ from measured stream evidence')
    feasible = all(p['projects_minimum_executed_work'] for p in projections)
    if 'all_projected_minimums_feasible' in decision:
        require(decision['all_projected_minimums_feasible'] is feasible, 'Saved budget feasibility differs from measured evidence')
    return {'status': 'all_six_train_only_resource_probe_pairs_verified', 'phase': 'fit',
            'preprobe_protocol_sha256': preprobe_ref['sha256'], 'evidence_pairs': 6, 'artifact_hashes_verified': 12,
            'training_budget_seconds': budget, 'quality_gate_scope': DECISION_SCOPE,
            'projections': projections,
            'all_projected_minimums_feasible': feasible,
            'convergence_or_accuracy_pass_claimed': False}


def selfcheck():
    """Exercise positive and fail-closed paths using temporary source/JSON fixtures only."""
    import copy
    import sys
    import tempfile

    global ROOT, OUT
    original_root, original_out = ROOT, OUT
    before = set(sys.modules)
    checks = 0
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False), encoding='utf8')
        return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    def rejected(value):
        nonlocal checks
        try:
            verify_fit_readiness(value)
        except (ValueError, KeyError, TypeError, FileNotFoundError):
            checks += 1
        else:
            raise AssertionError('Invalid fit readiness fixture was accepted')
    try:
        with tempfile.TemporaryDirectory(prefix='temporal-guard-') as directory:
            ROOT = Path(directory)
            OUT = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies'
            paths = ['work/expanded-temporal/' + name for name in [
                'run_domain_temporal.py', 'run_budget_temporal_v3.py', 'native_scalar_rollout_v3.py',
                'profile_temporal_case_stream.py', 'sources/crash/rollout.py', 'sources/drop_test/rollout.py']]
            paths.append('core.py')
            sources = []
            for path in paths:
                target = ROOT / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('# mock source only\n', encoding='utf8')
                sources.append({'path': path, 'sha256': sha(target)})
            pre = {'profile': PROFILE, 'phase': 'resource_probes', 'training_budget_seconds': None,
                   'seed': 20261015, 'sources': sources, 'core_source_path': 'core.py', 'domains': {}}
            for domain, channels in DOMAINS.items():
                metadata = {'channels': channels, 'split': {'train': [domain + str(i) for i in range(5)],
                                                          'validation': [domain + 'val'], 'test': [domain + 'test']}}
                for kind in ('manifest', 'ready', 'normalization'):
                    ref = write(ROOT / domain / (kind + '.json'), {'fixture': kind})
                    metadata.update({kind + '_path': ref['path'], kind + '_sha256': ref['sha256']})
                pre['domains'][domain] = metadata
            pre_ref = write(OUT / 'preprobe.json', pre)
            protocol = copy.deepcopy(pre)
            protocol.update({'phase': 'fit', 'training_budget_seconds': 200., 'preprobe_protocol': pre_ref,
                             'budget_decision': {'root_authorized': True, 'basis': 'train_only_resource_stream',
                                 'test_or_validation_accuracy_used': False, 'uniform_budget_seconds': 200.,
                                 'minimum_updates': 100, 'minimum_complete_case_epochs': 2,
                                 'quality_gate_scope': DECISION_SCOPE}, 'probe_evidence': []})
            for domain, channels in DOMAINS.items():
                metadata = pre['domains'][domain]
                train = metadata['split']['train']
                for strategy in STRATEGIES:
                    wrapper = ('work/expanded-temporal/sources/' + ('crash' if domain == 'bumper' else 'drop_test') + '/rollout.py'
                               if strategy == 'time_conditional' else 'work/expanded-temporal/native_scalar_rollout_v3.py')
                    source = {'protocol_sha256': pre_ref['sha256'], 'wrapper_file': wrapper,
                              'wrapper_sha256': sha(ROOT / wrapper), 'core_class_file': 'core.py',
                              'core_class_sha256': sha(ROOT / 'core.py'), 'driver_sha256': sources[0]['sha256']}
                    common = {'domain': domain, 'strategy': strategy, 'profile': profile_for(strategy), 'channels': channels,
                              'test_predicted': False, 'test_target_values_accessed': False,
                              'validation_target_values_accessed': False, 'source': source,
                              'contract': {'domain': domain, 'split': metadata['split'], 'budget_profile': profile_for(strategy),
                                  'normalization_sha256': metadata['normalization_sha256'],
                                  'cache_ready': {'manifest_sha256': metadata['manifest_sha256']},
                                  'frozen_protocol': {'sha256': pre_ref['sha256']}}}
                    functional = {**common, 'status': 'actual_FEM_forward_backward_optimizer_probe_only',
                                  'steps': 3, 'case': train[0], 'seconds': [1., 1., 1.], 'changed_parameters': 1,
                                  'inference_target_value_independence_max_abs': 0.}
                    if strategy != 'time_conditional':
                        functional.update({'status': 'actual_native_scalar_three_joint_Adam_updates_train_only_probe',
                            'all_output_rows_finite_nonzero_gradient_and_changed_each_update': True,
                            'head_weight_gradient_l2_by_row': [[1.] * channels for _ in range(3)],
                            'head_weight_update_l2_by_row': [[1.] * channels for _ in range(3)],
                            'eval_all_target_value_poisoning': {'all_outputs_exactly_equal': True, 'max_abs_difference': 0.},
                            'TF_training_scalar_target_poisoning': {'all_outputs_exactly_equal': True, 'max_abs_difference': 0.,
                                'past_position_targets_preserved': True, 'all_scalar_target_values_replaced': True},
                            'original_C3_position_equivalence_before_training': {'tolerance_in_training_scale_units': 1e-5,
                                'max_abs_difference_in_training_scale_units': 0., 'original_channels': 3,
                                'extended_channels': channels, 'first_train_case_only': True, 'pre_training': True, 'used_for_fit': False}})
                    steps = [{'case': case, 'batch_size': 1, 'input_equivalence_bitwise': True, 'head_weight_changed': True,
                              'head_weight_delta_l2': 1., **{key: 1. for key in ('cpu_prepare_seconds', 'h2d_wall_seconds',
                                  'h2d_cuda_event_seconds', 'compute_wall_seconds', 'compute_cuda_event_seconds', 'measured_update_wall_seconds')},
                              'phase_seconds': {phase: {'wall_seconds': 1., 'cuda_event_seconds': 1.}
                                  for phase in ('forward_loss', 'backward', 'gradient_clip', 'adam_step')}} for case in train]
                    stream = {**common, 'status': 'actual_five_distinct_train_case_optimizer_stream_profile',
                              'protocol_sha256': pre_ref['sha256'], 'profile_helper_sha256': sources[3]['sha256'],
                              'case_ids': train, 'train_case_ids': train, 'train_case_count': 5,
                              'selected_cases_train_only': True, 'batch_size': 1, 'steps': steps,
                              'updates': 5, 'changed_parameter_updates': 5,
                              'projection': {'required_updates': 100, 'minimum_optimizer_updates': 100,
                                  'minimum_case_epochs': 2, 'estimates_only': True, 'warm_median_update_wall_seconds': 1.,
                                  'warm_max_update_wall_seconds': 1., 'cold_first_update_wall_seconds': 1.}}
                    protocol['probe_evidence'].append({'domain': domain, 'strategy': strategy,
                        'functional': write(artifact_path(domain, strategy, 'functional'), functional),
                        'stream': write(artifact_path(domain, strategy, 'stream'), stream)})
            receipt = verify_fit_readiness(protocol)
            require(receipt['artifact_hashes_verified'] == 12 and receipt['all_projected_minimums_feasible'], 'Valid fixture failed')
            checks += 1
            for key, value in [('phase', 'resource_probes'), ('training_budget_seconds', 0.),
                               ('training_budget_seconds', float('nan')), ('seed', -1)]:
                invalid = copy.deepcopy(protocol)
                invalid[key] = value
                rejected(invalid)
            for key, value in [('root_authorized', False), ('test_or_validation_accuracy_used', True),
                               ('minimum_updates', 99), ('minimum_complete_case_epochs', 1), ('uniform_budget_seconds', 180.)]:
                invalid = copy.deepcopy(protocol)
                invalid['budget_decision'][key] = value
                rejected(invalid)
            invalid = copy.deepcopy(protocol)
            invalid['probe_evidence'][-1] = invalid['probe_evidence'][0]
            rejected(invalid)
            invalid = copy.deepcopy(protocol)
            invalid['probe_evidence'][0]['functional']['sha256'] = '0' * 64
            rejected(invalid)
            stream_ref = protocol['probe_evidence'][0]['stream']
            stream_path = ROOT / stream_ref['path']
            valid_stream = read_json(stream_path)
            for mutate in [lambda value: value['case_ids'].__setitem__(0, 'held-out'),
                           lambda value: value['steps'][0].__setitem__('h2d_wall_seconds', 0.),
                           lambda value: value.__setitem__('test_predicted', True),
                           lambda value: value['projection'].__setitem__('required_updates', 99),
                           lambda value: value.__setitem__('changed_parameter_updates', 0)]:
                changed = copy.deepcopy(valid_stream)
                mutate(changed)
                invalid = copy.deepcopy(protocol)
                invalid['probe_evidence'][0]['stream'] = write(stream_path, changed)
                rejected(invalid)
            write(stream_path, valid_stream)
            for missing, hash_key in [('core.py', 'core_class_sha256'),
                                     ('work/expanded-temporal/run_domain_temporal.py', 'driver_sha256'),
                                     ('work/expanded-temporal/sources/drop_test/rollout.py', 'wrapper_sha256')]:
                missing_map = {relative_path(row['path']): row['sha256'] for row in pre['sources'] if row['path'] != missing}
                invalid_report = read_json(ROOT / protocol['probe_evidence'][0]['functional']['path'])
                invalid_report['source'][hash_key] = None
                try:
                    _source_contract(invalid_report, pre, pre_ref['sha256'], 'phone_full', 'time_conditional', missing_map)
                except ValueError:
                    checks += 1
                else:
                    raise AssertionError('Missing required source and null artifact hash were accepted')
            import freeze_temporal_fit_decision as freezer
            previous_guard = freezer.guard
            try:
                freezer.guard = sys.modules[__name__]
                frozen = freezer.freeze(ROOT / pre_ref['path'], pre_ref['sha256'], 200., True)
                final_path = ROOT / frozen['path']
                final_sha = sha(final_path)
                require(verify_fit_readiness(read_json(final_path))['all_projected_minimums_feasible'],
                        'Generated final protocol failed its own guard')
                try:
                    freezer.freeze(ROOT / pre_ref['path'], pre_ref['sha256'], 200., True)
                except FileExistsError:
                    require(sha(final_path) == final_sha, 'Existing final fit protocol changed')
                    checks += 2
                else:
                    raise AssertionError('Generator overwrote an existing protocol')
            finally:
                freezer.guard = previous_guard
            require(not any(name.split('.')[0] in {'numpy', 'torch', 'physicsnemo'} for name in set(sys.modules) - before),
                    'Self-check imported an array/framework runtime')
    finally:
        ROOT, OUT = original_root, original_out
    return {'status': 'stdlib_mock_readiness_selfcheck_pass', 'checks': checks,
            'framework_or_numpy_imported': False, 'array_or_checkpoint_access': False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selfcheck', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(selfcheck(), allow_nan=False))
