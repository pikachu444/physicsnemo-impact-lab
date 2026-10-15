"""Synthetic standard-library monitoring check; never queries hardware or imports frameworks."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import run_domain_stage as stage


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
             patch.object(stage, 'snapshot', side_effect=RuntimeError('synthetic monitor failure')), \
             patch.object(sys, 'argv', ['stage', '--domain','bumper','--strategy','time_conditional','--mode','probe','--allocated']), \
             patch('builtins.print'):
            try:
                stage.main()
                raise AssertionError('Expected sys.exit')
            except SystemExit as error:
                assert error.code == (7 if fast_exit else 1)
        evidence = json.loads(next((Path(temp)/'domain-execution').glob('*.exit.json')).read_text())
        assert evidence['exit_code'] == (7 if fast_exit else -9)
        assert bool(evidence['runner_error']) is (not fast_exit)
        assert child.terminated is (not fast_exit)
assert 'numpy' not in sys.modules and 'torch' not in sys.modules and 'psutil' not in sys.modules
print(json.dumps({'status':'pass','scope':'synthetic resource observations, fast child exit and monitor-error reap/exit receipt','hardware_queried':False,'numpy_or_framework_imported':False}))
