"""External Windows Python -S guard for the unchanged R4 train-only probe stage."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[2]
SELF = 'work/expanded-temporal/run_multifield_guarded.py'
SHARED = 'work/expanded-clean-runtime/run_guarded_r2.py'
SHARED_SHA = '45d6bfb697b95f8df6721d2d42146c53095dc81e062e885f9fa91d17b5fdda1a'
PROTOCOL = 'outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-v3-resource-protocol-r4.json'
PROTOCOL_SHA = '7a3ea4d8c4d96b611a05797bd446856fe28de3f059534ef7359f6f9032352326'
PLAN = 'outputs/expanded-research/benchmarks/temporal_strategies/temporal-multifield-current-plan-r4.json'
PLAN_SHA = '951655f0f99947870ddebe4a3254c4f6f6ce35449e19233a523da2da66a852c0'
STAGE = 'work/expanded-temporal/run_multifield_stage.py'
PYTHON = 'work/expanded-ml/venv/Scripts/python.exe'
DOMAINS = ['phone_full', 'bumper']
STRATEGIES = ['time_conditional', 'ar_rollout_training', 'one_step_teacher_forcing']
MODES = ['probe', 'stream-probe']
STANDARD_POLICY = {'preflight_min_available_bytes': 1610612736, 'preflight_observations': 2,
          'preflight_interval_seconds': 10, 'running_min_available_bytes': 536870912,
          'family_rss_cap_bytes': 2147483648, 'wall_cap_seconds': 600,
          'monitor_interval_seconds': 1, 'cleanup_timeout_seconds': 30,
          'no_automatic_retry': True}
POLICIES = {'standard-resource-v1': STANDARD_POLICY,
            'cold-calibration-v1': {**STANDARD_POLICY, 'preflight_min_available_bytes': 536870912,
                                    'wall_cap_seconds': 120, 'monitor_interval_seconds': .2}}
REASONS = {'standard-resource-v1': 'Standard retained 1.5GiB operational margin for train-only resource probes.',
           'cold-calibration-v1': '1.5GiB is root operational margin borrowed from clean install, not measured small-probe minimum; one cold resource calibration under retained512MiBfloor'}
INVOCATION_CONTEXT = {'shell': 'cmd.exe', 'login': False, 'parent_python_no_site': True,
                      'reason': 'Reduce extra resident shell memory during cold resource calibration'}
ENVIRONMENT = {'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'PHYSICSNEMO_FORCE_TE': 'False'}
CREATE_SUSPENDED = 4
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(value, relative=False):
    require(isinstance(value, (str, Path)), 'A source/reference path is required')
    path = Path(value)
    if relative:
        require(not path.is_absolute() and not path.drive, 'Lock source paths must be relative')
    result = (path if path.is_absolute() else ROOT / path).resolve()
    require(result.is_relative_to(ROOT.resolve()), 'Reference escapes the workspace')
    return result


def read_json(path):
    require(Path(path).suffix.lower() == '.json', 'Only JSON metadata may be read')
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, 'Duplicate JSON key: ' + key)
            value[key] = item
        return value
    value = json.loads(Path(path).read_text(encoding='utf8'), parse_constant=invalid_constant,
                       object_pairs_hook=unique_object)
    require(isinstance(value, dict), 'JSON object required')
    return value


def checked_reference(reference, path, digest):
    require(reference == {'path': path, 'sha256': digest}, 'Lock reference differs from the pinned R4 contract')
    target = inside(path, relative=True)
    require(sha(target) == digest, 'Pinned metadata/source changed: ' + path)
    return target


def verify_lock(args):
    """Read only named source files and six metadata JSON files before hardware access."""
    require(args.allocated is True and isinstance(args.allocation, str) and args.allocation.strip(),
            'Explicit root allocation and a nonempty allocation note are required')
    require(args.domain in DOMAINS and args.strategy in STRATEGIES and args.mode in MODES,
            'Only the six domain/strategy pairs and resource modes are authorized')
    require(inside(args.protocol) == inside(PROTOCOL, relative=True)
            and args.protocol_sha256 == PROTOCOL_SHA, 'Only the pinned R4 resource protocol is authorized')
    lock_path = inside(args.execution_lock)
    require(isinstance(args.execution_lock_sha256, str) and len(args.execution_lock_sha256) == 64
            and all(c in '0123456789abcdef' for c in args.execution_lock_sha256), 'Execution lock needs lowercase SHA256')
    require(sha(lock_path) == args.execution_lock_sha256, 'Execution lock SHA256 mismatch')
    lock = read_json(lock_path)
    required_keys = {'schema_version', 'status', 'root_authorized', 'protocol', 'plan', 'sources', 'policy', 'policy_id',
                     'allowed_domains', 'allowed_strategies', 'allowed_modes', 'allowed_pairs', 'maximum_actual_attempts',
                     'reason', 'invocation_context', 'no_fit', 'no_test', 'training_budget_seconds'}
    require(set(lock) == required_keys, 'Execution lock has missing or unknown fields')
    require(lock['schema_version'] == 1 and lock['status'] == 'root_authorized_train_only_multifield_resource_guard'
            and lock['root_authorized'] is True, 'Explicit root execution lock authorization required')
    policy_id = lock['policy_id']
    require(policy_id in POLICIES and lock['policy'] == POLICIES[policy_id], 'Execution lock policy differs from an exact authorized policy')
    calibration = policy_id == 'cold-calibration-v1'
    domains, strategies, modes = (['phone_full'], ['time_conditional'], ['probe']) if calibration else (DOMAINS, STRATEGIES, MODES)
    pairs = [{'domain': domain, 'strategy': strategy} for domain in domains for strategy in strategies]
    require(lock['allowed_domains'] == domains and lock['allowed_strategies'] == strategies
            and lock['allowed_modes'] == modes and lock['allowed_pairs'] == pairs
            and lock['maximum_actual_attempts'] == (1 if calibration else None)
            and lock['reason'] == REASONS[policy_id] and lock['invocation_context'] == INVOCATION_CONTEXT,
            'Execution lock scope, attempt limit or invocation context differs from its authorized policy')
    require({'domain': args.domain, 'strategy': args.strategy} in pairs and args.mode in modes,
            'This pair/mode is outside the selected policy')
    require(lock['no_fit'] is True and lock['no_test'] is True
            and lock['training_budget_seconds'] is None, 'Execution lock must authorize train-only probes with no fit cap')
    protocol_path = checked_reference(lock['protocol'], PROTOCOL, PROTOCOL_SHA)
    plan_path = checked_reference(lock['plan'], PLAN, PLAN_SHA)
    protocol, plan = read_json(protocol_path), read_json(plan_path)
    require(protocol.get('revision') == 4 and protocol.get('schema_version') == 3
            and protocol.get('profile') == 'multifield_v3_native_scalar_heads'
            and protocol.get('phase') == 'resource_probes' and protocol.get('training_budget_seconds') is None
            and protocol.get('seed') == 20261015, 'Pinned protocol is not the R4 no-fit resource contract')
    require(plan.get('revision') == 4 and plan.get('authoritative_current_plan') is True
            and plan.get('planned_full_fit_count') == 6 and plan.get('training_budget_seconds') is None
            and plan.get('resource_protocol_sha256') == PROTOCOL_SHA
            and plan.get('original_large_domain_C3_full_fits_scheduled') == 0
            and plan.get('domains') == DOMAINS and set(plan.get('strategies', {})) == set(STRATEGIES),
            'Pinned plan does not describe the approved six-fit resource preparation')
    rows = protocol.get('sources')
    require(isinstance(rows, list) and len(rows) == 20, 'R4 requires all twenty source rows')
    frozen = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'sha256'}, 'Malformed R4 source row')
        path = inside(row['path'], relative=True)
        require(path not in frozen, 'Duplicate R4 source path')
        require(path.suffix.lower() in {'.py', '.json'} or path.name in {'METADATA', 'zz_reuse_physicsnemo.pth'},
                'Array/checkpoint access is forbidden during external preflight')
        frozen[path] = row['sha256']
    expected = {**frozen, inside(SELF, relative=True): None, inside(SHARED, relative=True): SHARED_SHA,
                protocol_path: PROTOCOL_SHA, plan_path: PLAN_SHA}
    require(len(expected) == 24 and inside(STAGE, relative=True) in frozen, 'R4 exact source inventory is incomplete')
    require(inside(SELF, relative=True) == Path(__file__).resolve(), 'Executing guard is outside its declared workspace path')
    lock_sources = {}
    require(isinstance(lock['sources'], list) and len(lock['sources']) == 24, 'Execution lock requires exactly 24 unique source/metadata rows')
    for row in lock['sources']:
        require(isinstance(row, dict) and set(row) == {'path', 'sha256'}, 'Malformed execution lock source row')
        path = inside(row['path'], relative=True)
        digest = row['sha256']
        require(path not in lock_sources and isinstance(digest, str) and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'Duplicate or invalid execution lock source row')
        lock_sources[path] = digest
    require(set(lock_sources) == set(expected), 'Execution lock source paths differ from the exact 24 required paths')
    for path, expected_digest in expected.items():
        if expected_digest is not None:
            require(lock_sources[path] == expected_digest, 'Lock digest differs from the pinned R4/shared contract')
        require(sha(path) == lock_sources[path], 'Locked source changed: ' + str(path))
    metadata_receipts = []
    require(set(protocol.get('domains', {})) == set(DOMAINS), 'Both frozen domains must be verified')
    for domain in DOMAINS:
        metadata = protocol['domains'][domain]
        require(metadata.get('channels') == (6 if domain == 'phone_full' else 7), 'Native domain channel count changed')
        loaded = {}
        for kind in ('manifest', 'normalization', 'ready'):
            path = inside(metadata[kind + '_path'], relative=True)
            require(path.suffix.lower() == '.json' and sha(path) == metadata[kind + '_sha256'],
                    domain + ' ' + kind + ' metadata hash changed')
            loaded[kind] = read_json(path)
            metadata_receipts.append({'domain': domain, 'kind': kind,
                                      'path': path.relative_to(ROOT).as_posix(), 'sha256': metadata[kind + '_sha256']})
        require(loaded['ready'].get('status') == 'ready' and loaded['ready'].get('writer_exit_code') == 0
                and loaded['ready'].get('manifest_sha256') == metadata['manifest_sha256'], 'READY/manifest metadata mismatch')
        require(loaded['manifest'].get('split') == metadata['split']
                and loaded['manifest'].get('field_names') == metadata['field_names']
                and loaded['manifest'].get('time_values') == metadata['time_values'], 'Manifest frozen domain metadata mismatch')
    require((ROOT / PYTHON).is_file(), 'Original R4 ML interpreter is missing')
    return {'execution_lock': {'path': lock_path.relative_to(ROOT).as_posix(), 'sha256': args.execution_lock_sha256},
            'protocol': lock['protocol'], 'plan': lock['plan'], 'guard_sha256': lock_sources[inside(SELF, relative=True)],
            'shared_guard_sha256': SHARED_SHA, 'verified_source_count': 24, 'metadata': metadata_receipts,
            'policy_id': policy_id, 'policy': dict(POLICIES[policy_id]), 'reason': lock['reason'],
            'invocation_context': lock['invocation_context']}


def load_shared_api():
    path = inside(SHARED, relative=True)
    require(sha(path) == SHARED_SHA, 'Shared guard changed before importlib load')
    spec = importlib.util.spec_from_file_location('_multifield_pinned_job_api', path)
    require(spec is not None and spec.loader is not None, 'Cannot load pinned job API')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # The shared main is never invoked.
    return types.SimpleNamespace(ProcessJob=module.ProcessJob, available_memory=module.available_memory, rss=module.rss,
                                 popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic)


def stage_command(args):
    return [str(ROOT / PYTHON), str(ROOT / STAGE), '--domain', args.domain, '--strategy', args.strategy,
            '--mode', args.mode, '--protocol', str(inside(PROTOCOL, relative=True)),
            '--protocol-sha256', PROTOCOL_SHA, '--allocated']


def cleanup(job, process, api, record, terminate):
    """Attempt every cleanup operation even if another one fails; inspect only this job."""
    errors = []
    def attempt(label, function):
        try:
            return function()
        except BaseException as error:
            errors.append({'operation': label, 'type': type(error).__name__, 'message': str(error)})
            return None
    if job is not None:
        members = attempt('job_pids_before_cleanup', job.pids)
        record['job_pids_before_cleanup'] = members
        if terminate or members:
            attempt('terminate_own_job', job.terminate)
            record['own_job_termination_requested'] = True
    if process is not None:
        # Assignment can fail or be interrupted while the child is still suspended and unassigned.
        if not record['contained_before_resume'] and attempt('poll_unassigned_child', process.poll) is None:
            attempt('terminate_owned_unassigned_child_handle', process.terminate)
            record['unassigned_child_handle_termination_requested'] = True
        attempt('reap_owned_child', lambda: process.wait(timeout=record['policy']['cleanup_timeout_seconds']))
        record['child_reaped'] = process.returncode is not None
        record['child_exit_code'] = process.returncode
    remaining = None
    if job is not None:
        deadline = api.clock() + record['policy']['cleanup_timeout_seconds']
        while True:
            remaining = attempt('job_pids_during_cleanup', job.pids)
            if not remaining or api.clock() >= deadline:
                break
            attempt('cleanup_wait', lambda: api.sleep(.1))
        record['job_pids_at_end'] = remaining
        record['job_empty_confirmed'] = remaining == []
        attempt('close_own_job', job.close)
    else:
        record['job_pids_at_end'] = []
        record['job_empty_confirmed'] = True
    record['cleanup_errors'] = errors
    if errors or remaining or (process is not None and not record['child_reaped']):
        record['status'] = 'failed_cleanup'


def run_attempt(args, verified, api, output_directory=None):
    """Hardware/process access is injected so self-checks never inspect the real host."""
    policy_id = verified['policy_id']
    require(policy_id in POLICIES and verified.get('policy') == POLICIES[policy_id], 'Attempt requires an exact verified policy')
    policy = POLICIES[policy_id]
    require(policy_id != 'cold-calibration-v1' or
            (args.domain, args.strategy, args.mode) == ('phone_full', 'time_conditional', 'probe'),
            'Calibration is limited to one phone TC functional probe')
    started = now()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt_id = f'{args.domain}--{args.strategy}--{args.mode}--{stamp}--{os.getpid()}'
    directory = Path(output_directory) if output_directory is not None else ROOT / 'work/expanded-temporal/guarded-resource-attempts'
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / (attempt_id + '.json')
    record = {'attempt': attempt_id, 'started_utc': started, 'domain': args.domain, 'strategy': args.strategy,
              'mode': args.mode, 'root_allocation_note': args.allocation, 'allocated': args.allocated,
              'preflight_only': args.preflight_only, 'policy_id': policy_id, 'policy': dict(policy), 'verified_contract': verified,
              'invocation_context': verified['invocation_context'], 'actual_sys_argv': list(sys.argv),
              'actual_sys_executable': sys.executable, 'actual_parent_no_site': bool(sys.flags.no_site),
              'invocation_context_limit': 'The lock records the expected host shell/login context; this child process cannot independently attest the host tool shell.',
              'no_test': True, 'no_fit': True, 'training_budget_seconds': None,
              'parent_numpy_framework_imports': 0, 'automatic_retries': 0,
              'job_created': False, 'child_started': False, 'contained_before_resume': False,
              'child_reaped': False, 'preflight_samples': [], 'runtime_samples': [],
              'observed_own_job_pids': [], 'peak_sampled_family_rss_bytes': 0,
              'minimum_sampled_available_bytes': None, 'status': 'preflight',
              'measurement_limit': 'Nominal periodic samples of own-job working-set sums can miss transient peaks. The 2GiB RSS policy is a sampled stop threshold, not an OS memory hard limit.'}
    process = job = None
    log_stream = None
    begin = api.clock()
    code = 1
    try:
        if policy_id == 'cold-calibration-v1':
            sentinel = directory / 'cold-calibration-v1-attempt-started.json'
            with sentinel.open('x', encoding='utf8') as stream:
                stream.write(json.dumps({'attempt': attempt_id, 'started_utc': started, 'policy_id': policy_id,
                                         'execution_lock': verified['execution_lock'], 'root_allocation_note': args.allocation,
                                         'no_automatic_retry': True}, indent=2, allow_nan=False) + '\n')
            record['calibration_attempt_sentinel'] = str(sentinel)
        for index in range(policy['preflight_observations']):
            free = api.available_memory()
            require(type(free) is int and free >= 0, 'Host memory observation is unavailable or invalid')
            record['preflight_samples'].append({'utc': now(), 'available_bytes': free})
            record['minimum_sampled_available_bytes'] = min(row['available_bytes'] for row in record['preflight_samples'])
            if free < policy['preflight_min_available_bytes']:
                record.update({'status': 'deferred_resource_preflight', 'guard_reason': 'first_preflight_memory' if index == 0 else 'second_preflight_memory'})
                return_code = 3
                break
            if index == 0:
                api.sleep(policy['preflight_interval_seconds'])
        else:
            return_code = None
        if return_code is not None:
            code = return_code
        elif args.preflight_only:
            record['status'] = 'preflight_passed_no_child_requested'
            code = 0
        else:
            command = stage_command(args)
            environment = {**os.environ, **ENVIRONMENT}
            record.update({'command': command, 'cwd': str(ROOT), 'environment_overrides': dict(ENVIRONMENT),
                           'creationflags': BELOW_NORMAL_PRIORITY_CLASS | CREATE_SUSPENDED,
                           'priority': 'BelowNormal', 'parent_no_site': bool(sys.flags.no_site)})
            job = api.ProcessJob()
            record['job_created'] = True
            log = directory / (attempt_id + '.log')
            log_stream = log.open('x', encoding='utf8')
            record['log'] = str(log)
            runtime_start = api.clock()
            pending_interrupt = []
            previous_handler = signal.signal(signal.SIGINT, lambda signum, frame: pending_interrupt.append(signum))
            try:
                process = api.popen(command, cwd=ROOT, env=environment, stdout=log_stream, stderr=subprocess.STDOUT,
                                    creationflags=BELOW_NORMAL_PRIORITY_CLASS | CREATE_SUSPENDED)
                record.update({'child_started': True, 'child_pid': process.pid, 'child_started_utc': now(),
                               'status': 'running', 'child_framework_import_count': 'not_observed_by_stdlib_parent',
                               'SIGINT_deferred_during_suspended_creation_and_assignment': True})
                job.assign_and_resume(process)
                record['contained_before_resume'] = True
            finally:
                signal.signal(signal.SIGINT, previous_handler)
            if pending_interrupt:
                raise KeyboardInterrupt('SIGINT deferred until owned child assignment completed')
            while True:
                members = job.pids()
                observed = set(record['observed_own_job_pids']) | set(members)
                record['observed_own_job_pids'] = sorted(observed)
                rss_rows = []
                missing_live = []
                for pid in members:
                    memory = api.rss(pid)
                    if memory is None:
                        if pid in job.pids():
                            missing_live.append(pid)
                    else:
                        require(type(memory) is int and memory >= 0, 'Invalid own-job RSS observation')
                        rss_rows.append({'pid': pid, 'rss_bytes': memory})
                free = api.available_memory()
                require(type(free) is int and free >= 0, 'Invalid running host memory observation')
                family = sum(row['rss_bytes'] for row in rss_rows)
                elapsed = api.clock() - runtime_start
                sample = {'utc': now(), 'wall_seconds': elapsed, 'own_job_pids': members,
                          'rss_by_own_job_pid': rss_rows, 'family_rss_bytes': family,
                          'unmeasurable_live_own_job_pids': missing_live, 'host_available_bytes': free}
                record['runtime_samples'].append(sample)
                record['peak_sampled_family_rss_bytes'] = max(record['peak_sampled_family_rss_bytes'], family)
                record['minimum_sampled_available_bytes'] = min(record['minimum_sampled_available_bytes'], free)
                reason = ('own_job_RSS_observation_unavailable' if missing_live else
                          'host_available_memory_floor' if free < policy['running_min_available_bytes'] else
                          'own_job_family_RSS_threshold' if family > policy['family_rss_cap_bytes'] else
                          'per_probe_wall_cap' if elapsed >= policy['wall_cap_seconds'] else None)
                if reason:
                    record.update({'status': 'stopped_resource_guard', 'guard_reason': reason})
                    break
                exit_code = process.poll()
                if exit_code is not None:
                    if job.pids():
                        record.update({'status': 'stopped_lingering_descendants', 'guard_reason': 'own_job_members_after_stage_exit'})
                    else:
                        record['status'] = 'completed' if exit_code == 0 else 'failed_original_stage'
                    break
                api.sleep(policy['monitor_interval_seconds'])
            cleanup(job, process, api, record, terminate=record['status'] != 'completed')
            job = None
            code = 0 if record['status'] == 'completed' and record.get('job_empty_confirmed') and record.get('child_reaped') else 1
    except BaseException as error:
        record.update({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed_guard_or_observation',
                       'guard_reason': type(error).__name__, 'error': str(error)})
        cleanup(job, process, api, record, terminate=True)
        job = None
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if job is not None:
            cleanup(job, process, api, record, terminate=True)
        if log_stream is not None:
            log_stream.close()
            record['log_sha256'] = sha(Path(record['log']))
        record.update({'finished_utc': now(), 'wall_seconds': api.clock() - begin, 'guard_exit_code': code,
                       'receipt_path': str(receipt)})
        with receipt.open('x', encoding='utf8') as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return code, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=DOMAINS, required=True)
    parser.add_argument('--strategy', choices=STRATEGIES, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--execution-lock', required=True)
    parser.add_argument('--execution-lock-sha256', required=True)
    parser.add_argument('--allocation', required=True)
    parser.add_argument('--allocated', action='store_true', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    require(os.name == 'nt' and sys.flags.no_site, 'Use an external Windows Python -S process')
    require(not any(name.split('.')[0] in {'numpy', 'torch', 'physicsnemo'} for name in sys.modules),
            'The external parent must remain standard-library only')
    verified = verify_lock(args)
    api = load_shared_api()
    code, record = run_attempt(args, verified, api)
    print(json.dumps(record, allow_nan=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
