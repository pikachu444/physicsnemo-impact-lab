"""Standard-library observation only; delegates every official call unchanged."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


class StageJournal:
    def __init__(self, path, origin):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.origin = origin
        self.solver_calls = 0
        self.durations = []
        self.rows = []

    def record(self, event, **values):
        row = {'event': event, 'pid': os.getpid(),
               'utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_from_driver_start_seconds': time.perf_counter()-self.origin,
               **values}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        return row

    def observe(self, solver_class, metric_logger_class):
        """Temporarily time official methods without changing arguments/results."""
        journal = self
        original_solve = solver_class.solve
        original_step = metric_logger_class.step

        def timed_solve(solver, *args, **kwargs):
            journal.solver_calls += 1
            index = journal.solver_calls
            stage = 'monolithic' if index == 1 else 'patch_batch'
            started = time.perf_counter()
            journal.record('solve_started', stage=stage, solver_call=index,
                           schwarz_iteration=None if index == 1 else index-1)
            try:
                result = original_solve(solver, *args, **kwargs)
            except BaseException as exc:
                journal.record('solve_failed', stage=stage, solver_call=index,
                               duration_seconds=time.perf_counter()-started,
                               exception_type=type(exc).__name__)
                raise
            duration = time.perf_counter()-started
            journal.durations.append({'stage': stage, 'solver_call': index, 'duration_seconds': duration})
            journal.record('solve_completed', stage=stage, solver_call=index,
                           schwarz_iteration=None if index == 1 else index-1,
                           duration_seconds=duration)
            return result

        def timed_step(logger, *args, **kwargs):
            started = time.perf_counter()
            row = original_step(logger, *args, **kwargs)
            journal.rows.append(dict(row))
            journal.record('interface_iteration_completed', metric=dict(row),
                           metric_persistence_duration_seconds=time.perf_counter()-started)
            return row

        class Observation:
            def __enter__(self):
                solver_class.solve = timed_solve
                metric_logger_class.step = timed_step
                return journal

            def __exit__(self, exc_type, exc, traceback):
                solver_class.solve = original_solve
                metric_logger_class.step = original_step
                return False

        return Observation()
