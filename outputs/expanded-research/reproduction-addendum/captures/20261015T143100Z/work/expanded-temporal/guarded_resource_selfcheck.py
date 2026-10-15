"""Deterministic standard-library checks; run with Python -S, without hardware access."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


SOURCE = Path(__file__).with_name('run_multifield_guarded.py')
SPEC = importlib.util.spec_from_file_location('_guarded_resource_selfcheck_target', SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)
MIB = 1024 ** 2
GIB = 1024 ** 3
FORBIDDEN = {'numpy', 'torch', 'physicsnemo', 'tensorflow', 'jax'}


class FakeProcess:
    pid = 101

    def __init__(self, api):
        self.api = api
        self.returncode = None
        self.resumed = False

    def poll(self):
        self.api.events.append('poll')
        if self.resumed and self.returncode is None:
            self.returncode = 0
            self.api.members = []
        return self.returncode

    def terminate(self):
        self.api.events.append('handle_terminate')
        self.returncode = -2
        self.api.members = []

    def wait(self, timeout):
        self.api.events.append('wait')
        assert timeout == 30
        assert self.returncode is not None, 'A live child reached reap without termination'
        return self.returncode


class FakeJob:
    def __init__(self, api):
        self.api = api
        self.closed = False

    def assign_and_resume(self, process):
        self.api.events.append('assign')
        if self.api.scenario == 'assignment_before':
            raise OSError('mock assignment failure before containment')
        self.api.members = [process.pid, 102] if self.api.scenario == 'family_rss' else [process.pid]
        self.api.events.append('assigned')
        if self.api.scenario == 'assignment_after':
            raise OSError('mock failure after assignment and before resume')
        self.api.events.append('resume')
        process.resumed = True

    def pids(self):
        self.api.events.append('pids')
        return list(self.api.members)

    def terminate(self):
        self.api.events.append('job_terminate')
        self.api.members = []
        # Simulate asynchronous termination during assignment failure so the
        # independently owned process handle must also stop the suspended child.
        if self.api.scenario not in {'assignment_before', 'assignment_after'}:
            self.api.process.returncode = -1

    def close(self):
        self.api.events.append('close')
        self.closed = True


class FakeAPI:
    def __init__(self, memory, scenario='success'):
        self.memory = iter(memory)
        self.scenario = scenario
        self.events = []
        self.memory_calls = 0
        self.sleeps = []
        self.members = []
        self.process = None
        self.job = None
        self.elapsed = 0

    def available_memory(self):
        self.events.append('memory')
        self.memory_calls += 1
        return next(self.memory)

    def rss(self, pid):
        self.events.append('rss')
        assert pid in {101, 102}
        if self.scenario == 'keyboard_interrupt':
            raise KeyboardInterrupt('mock monitor interrupt')
        if self.scenario == 'rss_unavailable':
            return None
        if self.scenario == 'rss_vanished':
            self.members = []
            return None
        if self.scenario == 'wall_cap':
            self.elapsed += 600
        return GIB + 1 if self.scenario == 'family_rss' else 128 * MIB

    def ProcessJob(self):
        self.events.append('job')
        assert self.job is None
        self.job = FakeJob(self)
        return self.job

    def popen(self, command, **kwargs):
        self.events.append('popen')
        assert self.process is None
        assert kwargs['creationflags'] == guard.CREATE_SUSPENDED | guard.BELOW_NORMAL_PRIORITY_CLASS
        assert kwargs['creationflags'] & 4
        assert kwargs['cwd'] == guard.ROOT
        assert kwargs['stderr'] == guard.subprocess.STDOUT
        assert kwargs['stdout'].writable()
        assert command[-1] == '--allocated'
        assert command[command.index('--mode') + 1] in guard.MODES
        for name, value in guard.ENVIRONMENT.items():
            assert kwargs['env'][name] == value
        self.process = FakeProcess(self)
        return self.process

    def sleep(self, seconds):
        self.events.append('sleep')
        self.sleeps.append(seconds)
        self.elapsed += seconds

    def clock(self):
        self.events.append('clock')
        return self.elapsed


def arguments(**changes):
    values = dict(domain='phone_full', strategy='ar_rollout_training', mode='probe',
                  allocated=True, allocation='Explicit mock root allocation', preflight_only=False)
    values.update(changes)
    return types.SimpleNamespace(**values)


def verified(policy_id='standard-resource-v1'):
    return {'policy_id': policy_id, 'policy': copy.deepcopy(guard.POLICIES[policy_id]),
            'invocation_context': copy.deepcopy(guard.INVOCATION_CONTEXT),
            'execution_lock': {'path': 'mock-execution-lock.json', 'sha256': '0' * 64}}


def run_case(directory, name, api, args=None, contract=None):
    destination = directory / name
    # The launcher owns no real process or hardware API in these checks.
    with patch.object(guard.signal, 'signal', return_value=None):
        code, record = guard.run_attempt(args or arguments(), contract or verified(), api,
                                         output_directory=destination)
    receipt = Path(record['receipt_path'])
    assert receipt.parent == destination
    saved = json.loads(receipt.read_text(encoding='utf8'))
    assert saved['guard_exit_code'] == code
    assert saved['status'] == record['status']
    assert saved['parent_numpy_framework_imports'] == 0
    assert saved['automatic_retries'] == 0
    assert saved['no_fit'] is True and saved['no_test'] is True
    assert saved.get('cleanup_errors', []) == []
    return code, record


def no_child(api, record):
    assert api.job is None and api.process is None
    assert 'job' not in api.events and 'popen' not in api.events
    assert not record['job_created'] and not record['child_started']


def cleaned(api, record):
    assert record['job_empty_confirmed'] is True
    assert record['job_pids_at_end'] == [] and api.members == []
    assert record['child_reaped'] is True and api.process.returncode is not None
    assert api.job.closed and api.events.count('wait') == 1
    assert api.events.index('wait') < api.events.index('close')
    assert record['cleanup_errors'] == []


def rejected(directory, name, args, contract):
    api = FakeAPI([])
    try:
        guard.run_attempt(args, contract, api, output_directory=directory / name)
    except ValueError:
        pass
    else:
        raise AssertionError(name + ' was accepted')
    assert api.events == [], 'Policy rejection touched the fake hardware/process API'
    assert not (directory / name).exists()


def main():
    assert sys.flags.no_site, 'Run this self-check with Python -S'
    assert not FORBIDDEN.intersection(name.split('.')[0] for name in sys.modules)
    minimum = guard.STANDARD_POLICY['preflight_min_available_bytes']
    floor = guard.STANDARD_POLICY['running_min_available_bytes']
    count = 0
    with tempfile.TemporaryDirectory(prefix='guarded-resource-selfcheck-') as temporary:
        directory = Path(temporary)
        api = FakeAPI([minimum - 1])
        code, record = run_case(directory, 'first_low', api)
        assert code == 3 and record['guard_reason'] == 'first_preflight_memory'
        assert record['status'] == 'deferred_resource_preflight'
        assert api.memory_calls == 1 and api.sleeps == []
        no_child(api, record)
        count += 1

        api = FakeAPI([minimum, minimum - 1])
        code, record = run_case(directory, 'second_low', api)
        assert code == 3 and record['guard_reason'] == 'second_preflight_memory'
        assert api.memory_calls == 2 and api.sleeps == [10]
        no_child(api, record)
        count += 1

        api = FakeAPI([minimum, minimum])
        code, record = run_case(directory, 'preflight_only', api, arguments(strategy='time_conditional', preflight_only=True))
        assert code == 0 and record['status'] == 'preflight_passed_no_child_requested'
        assert api.memory_calls == 2 and api.sleeps == [10]
        no_child(api, record)
        count += 1

        api = FakeAPI([minimum, minimum, floor])
        code, record = run_case(directory, 'success', api)
        assert code == 0 and record['status'] == 'completed' and record['child_exit_code'] == 0
        assert record['contained_before_resume'] is True
        assert api.events.index('job') < api.events.index('popen') < api.events.index('assign')
        assert api.events.index('assign') < api.events.index('assigned') < api.events.index('resume')
        assert api.events.index('resume') < api.events.index('rss') < api.events.index('poll')
        assert 'job_terminate' not in api.events and 'handle_terminate' not in api.events
        cleaned(api, record)
        count += 1

        for scenario, runtime_memory, reason in (
                ('host_floor', floor - 1, 'host_available_memory_floor'),
                ('family_rss', floor, 'own_job_family_RSS_threshold'),
                ('wall_cap', floor, 'per_probe_wall_cap'),
                ('rss_unavailable', floor, 'own_job_RSS_observation_unavailable')):
            api = FakeAPI([minimum, minimum, runtime_memory], scenario)
            code, record = run_case(directory, scenario, api)
            assert code == 1 and record['status'] == 'stopped_resource_guard'
            assert record['guard_reason'] == reason
            assert record['own_job_termination_requested'] is True
            assert api.events.index('job_terminate') < api.events.index('wait')
            if scenario == 'family_rss':
                assert record['peak_sampled_family_rss_bytes'] == 2 * GIB + 2
            if scenario == 'wall_cap':
                assert record['runtime_samples'][0]['wall_seconds'] == 600
            if scenario == 'rss_unavailable':
                assert record['runtime_samples'][0]['unmeasurable_live_own_job_pids'] == [101]
            cleaned(api, record)
            count += 1

        api = FakeAPI([minimum, minimum], 'keyboard_interrupt')
        code, record = run_case(directory, 'keyboard_interrupt', api)
        assert code == 130 and record['status'] == 'interrupted'
        assert record['guard_reason'] == 'KeyboardInterrupt'
        assert record['own_job_termination_requested'] is True
        cleaned(api, record)
        count += 1

        for scenario in ('assignment_before', 'assignment_after'):
            api = FakeAPI([minimum, minimum], scenario)
            code, record = run_case(directory, scenario, api)
            assert code == 1 and record['status'] == 'failed_guard_or_observation'
            assert record['guard_reason'] == 'OSError' and record['contained_before_resume'] is False
            assert 'resume' not in api.events
            assert record['unassigned_child_handle_termination_requested'] is True
            assert api.events.index('job_terminate') < api.events.index('handle_terminate') < api.events.index('wait')
            assert ('assigned' in api.events) == (scenario == 'assignment_after')
            cleaned(api, record)
            count += 1

        api = FakeAPI([minimum, minimum, floor], 'rss_vanished')
        code, record = run_case(directory, 'rss_vanished', api)
        assert code == 0 and record['status'] == 'completed'
        assert record['runtime_samples'][0]['unmeasurable_live_own_job_pids'] == []
        assert record['runtime_samples'][0]['rss_by_own_job_pid'] == []
        cleaned(api, record)
        count += 1

        calibration = verified('cold-calibration-v1')
        calibration_directory = directory / 'calibration_one_shot'
        api = FakeAPI([512 * MIB - 1])
        code, record = run_case(directory, 'calibration_one_shot', api, arguments(strategy='time_conditional'), contract=calibration)
        assert code == 3 and record['guard_reason'] == 'first_preflight_memory'
        sentinel = calibration_directory / 'cold-calibration-v1-attempt-started.json'
        assert sentinel.is_file() and Path(record['calibration_attempt_sentinel']) == sentinel
        sentinel_bytes = sentinel.read_bytes()
        assert json.loads(sentinel_bytes)['execution_lock'] == calibration['execution_lock']
        assert api.memory_calls == 1 and api.sleeps == []
        no_child(api, record)
        count += 1

        api = FakeAPI([])
        code, record = run_case(directory, 'calibration_one_shot', api, arguments(strategy='time_conditional'), contract=calibration)
        assert code == 1 and record['guard_reason'] == 'FileExistsError'
        assert api.memory_calls == 0 and api.sleeps == []
        assert sentinel.read_bytes() == sentinel_bytes
        no_child(api, record)
        count += 1

        for name, changes in (('wrong_domain', {'domain': 'bumper'}),
                              ('wrong_strategy', {'strategy': 'ar_rollout_training'}),
                              ('wrong_mode', {'mode': 'stream-probe'})):
            rejected(directory, name, arguments(**changes), calibration)
            count += 1
        wrong_policy = verified('cold-calibration-v1')
        wrong_policy['policy_id'] = 'unapproved-policy'
        rejected(directory, 'wrong_policy', arguments(), wrong_policy)
        count += 1
        rejected(directory, 'standard_calibration_bypass', arguments(strategy='time_conditional'), verified())
        count += 1
        downgraded = verified('cold-calibration-v1')
        downgraded['policy']['preflight_min_available_bytes'] = 512 * MIB - 1
        rejected(directory, 'downgraded_floor', arguments(), downgraded)
        count += 1

    assert not FORBIDDEN.intersection(name.split('.')[0] for name in sys.modules)
    print(json.dumps({'selfcheck': 'guarded_resource', 'tests_passed': count,
                      'status': 'passed', 'numpy_framework_imports': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
