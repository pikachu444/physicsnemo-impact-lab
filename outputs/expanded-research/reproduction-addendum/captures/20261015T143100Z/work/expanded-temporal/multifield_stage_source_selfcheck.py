"""Synthetic standard-library monitoring check; never queries hardware or imports frameworks."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import hashlib
import run_multifield_stage as stage


class Process:
    pid = 42
    info = {'name': 'synthetic', 'memory_info': S(rss=300*1024**2)}
    def children(self, recursive):
        assert recursive
        return []
    def memory_info(self):
        return S(rss=123, vms=456)
    def cpu_times(self):
        return S(user=2., system=3.)


fake = S(Error=RuntimeError,
         virtual_memory=lambda: S(available=1000,percent=25.),
         swap_memory=lambda: S(used=77),
         process_iter=lambda attrs: [Process()])
with patch.object(stage.subprocess, 'run', return_value=S(stdout='fake GPU, 10, 100, 5\n', returncode=0)):
    row = stage.snapshot(fake, Process())
    assert row['process_tree_rss_bytes'] == 123 and row['process_tree_virtual_bytes'] == 456
    assert row['process_tree_cpu_seconds'] == 5. and row['host_available_bytes'] == 1000
    assert row['host_consumers_above_256MiB'][0]['pid'] == 42
    assert row['gpu_query_exit_code'] == 0
with patch.object(stage.subprocess, 'run', side_effect=OSError('synthetic unavailable query')):
    row = stage.snapshot(fake, Process())
    assert row['gpu_query_exit_code'] is None


class NoSuchProcess(RuntimeError):
    pass


class Child:
    pid = 42
    def __init__(self, fast_exit=False):
        self.returncode = 7 if fast_exit else None
        self.terminated = False
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        assert self.returncode is not None
        return self.returncode
    def terminate(self):
        self.terminated = True
        self.returncode = -9


scratch = stage.OWN / 'guard-selfcheck'
scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='current-plan-selfcheck-', dir=scratch) as temp:
    assert Path(temp).resolve().is_relative_to(scratch.resolve())
    root = Path(temp)
    sources = []
    for filename in ['run_tc_frozen_v2.py', 'run_budget_temporal_v3.py']:
        path = root / 'work/expanded-temporal' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic inert source')
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    domains = {}
    for name,(manifest,normalization) in stage.DATA.items():
        domains[name] = {}
        for kind,path in [('manifest',root/'work/expanded-ml'/manifest),
                          ('normalization',root/'work/expanded-ml'/normalization),
                          ('ready',(root/'work/expanded-ml'/manifest).with_name('READY.json'))]:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text('synthetic inert metadata')
            domains[name][kind+'_path'] = path.relative_to(root).as_posix()
            domains[name][kind+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_path = root / 'protocol.json'
    protocol_path.write_text(json.dumps({'sources': sources, 'domains':domains, 'phase':'resource_probes','training_budget_seconds': None}))
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    with patch.object(stage, 'ROOT', root), patch.object(stage, 'OWN', root/'work/expanded-temporal'):
        for strategy in ['time_conditional','ar_rollout_training','one_step_teacher_forcing']:
            command, digest = stage.command_for('phone_full', strategy, 'probe', None, protocol_path, protocol_hash)
            expected = 'run_tc_frozen_v2.py' if strategy == 'time_conditional' else 'run_budget_temporal_v3.py'
            assert Path(command[1]).name == expected
            assert '--protocol' in command and '--protocol-sha256' in command
            assert digest == sources[0]['sha256']
        try:
            stage.command_for('bumper','ar_rollout_training','budget',300.,protocol_path,protocol_hash)
            raise AssertionError('Budget drift accepted')
        except AssertionError as error:
            assert str(error) != 'Budget drift accepted'
for fast_exit in [True, False]:
    child = Child(fast_exit)
    process_constructor = lambda pid: Process()
    if fast_exit:
        def process_constructor(pid):
            raise NoSuchProcess('synthetic already exited')
    fake_psutil = S(Process=process_constructor, NoSuchProcess=NoSuchProcess)
    with tempfile.TemporaryDirectory(prefix='stage-selfcheck-', dir=scratch) as temp:
        assert Path(temp).resolve().is_relative_to(scratch.resolve())
        with patch.object(stage, 'OWN', Path(temp)), patch.dict(sys.modules, {'psutil':fake_psutil}), \
             patch.object(stage.subprocess, 'Popen', return_value=child), \
             patch.object(stage, 'command_for', return_value=(['synthetic inert command'], 'fake-driver-sha')), \
             patch.object(stage, 'snapshot', side_effect=RuntimeError('synthetic monitor failure')), \
             patch.object(sys, 'argv', ['stage', '--domain','bumper','--strategy','time_conditional','--mode','probe','--protocol','synthetic','--protocol-sha256','synthetic','--allocated']), \
             patch('builtins.print'):
            try:
                stage.main()
                raise AssertionError('Expected sys.exit')
            except SystemExit as error:
                assert error.code == (7 if fast_exit else 1)
        evidence = json.loads(next((Path(temp)/'multifield-execution-v3').glob('*.exit.json')).read_text())
        assert evidence['exit_code'] == (7 if fast_exit else -9)
        assert bool(evidence['runner_error']) is (not fast_exit)
        assert child.terminated is (not fast_exit)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules and 'psutil' not in sys.modules
print(json.dumps({'status':'pass','scope':'synthetic V2-TC/V3-AR-TF command selection, protocol/budget guard, resource observations and child cleanup','hardware_queried':False,'numpy_or_framework_imported':False}))
