"""Check metadata forwarding/restoration with plain objects, without frameworks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('metadata_observer', HERE/'stage-observer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    token = object()
    calls = []

    class Fake:
        def solve(self, *args, **kwargs):
            calls.append((args, kwargs))
            if kwargs.get('fail'):
                raise ValueError('expected self-check exception')
            return token

    class Logger:
        def step(self):
            return {'iteration': 1, 'elapsed': .1, 'loss_interface': .01}

    old_solve, old_step = Fake.solve, Logger.step
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        journal = module.StageJournal(Path(directory)/'events.jsonl', time.perf_counter())
        with journal.observe(Fake, Logger):
            assert Fake().solve('same', key=token) is token
            assert Fake().solve(2) is token
            row = Logger().step()
        assert calls == [(('same',), {'key': token}), ((2,), {})]
        assert Fake.solve is old_solve and Logger.step is old_step
        events = [json.loads(line) for line in journal.path.read_text().splitlines()]
        assert [event['event'] for event in events] == ['solve_started', 'solve_completed', 'solve_started', 'solve_completed', 'interface_iteration_completed']
        assert events[0]['stage'] == 'monolithic' and events[2]['schwarz_iteration'] == 1
        assert events[-1]['metric'] == row
        try:
            with journal.observe(Fake, Logger):
                Fake().solve(fail=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Exception must propagate')
        assert Fake.solve is old_solve and Logger.step is old_step
        assert json.loads(journal.path.read_text().splitlines()[-1])['event'] == 'solve_failed'
    assert not any(name == 'torch' or name == 'numpy' or name.startswith('fpddm') for name in sys.modules)
    record = {'status': 'passed', 'framework_imported': False, 'scientific_solver_executed': False,
              'checks': ['identical_arguments', 'identical_return_identity', 'original_methods_restored',
                         'ordered_stage_and_metric_persistence', 'exception_propagation_and_restoration']}
    (HERE/'stage-observer-selfcheck.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
