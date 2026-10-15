"""Current six-fit plan: frozen V2 TC plus separate V3 native-scalar AR/TF."""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN = Path(__file__).resolve().parent
PYTHON = ROOT / 'work/expanded-ml/venv/Scripts/python.exe'
PROTOCOL = ROOT / 'outputs/expanded-research/benchmarks/temporal_strategies/native-scalar-v3-protocol.json'
EXPECTED_PROTOCOL = '6b41e5f6369c682b3dda2c0c1a294218452d0f055a9f093b8ecffa28c212ed45'
DATA = {
    'phone_full': ('phone-full-cache-v2/manifest.json', 'phone-full-normalization.json'),
    'bumper': ('multifield7-cache/manifest.json', 'bumper7-normalization.json'),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command_for(domain, strategy, mode, budget):
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == EXPECTED_PROTOCOL, 'Frozen current-plan protocol changed'
    protocol = json.loads(PROTOCOL.read_text())
    sources = {row['path']: row['sha256'] for row in protocol['sources']}
    for relative, digest in sources.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    driver = OWN / ('run_budget_temporal_v2.py' if strategy == 'time_conditional' else 'run_budget_temporal_v3.py')
    driver_hash = sources[driver.relative_to(ROOT).as_posix()]
    assert budget == protocol['training_budget_seconds']
    manifest, normalization = DATA[domain]
    domain_contract = protocol['domains'][domain]
    for kind, path in [('manifest', ROOT/'work/expanded-ml'/manifest),
                       ('normalization', ROOT/'work/expanded-ml'/normalization),
                       ('ready', (ROOT/'work/expanded-ml'/manifest).with_name('READY.json'))]:
        assert path.resolve() == (ROOT/domain_contract[kind+'_path']).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == domain_contract[kind+'_sha256'], kind
    command = [str(PYTHON), str(driver), '--domain', domain, '--manifest', str(ROOT/'work/expanded-ml'/manifest),
               '--normalization', str(ROOT/'work/expanded-ml'/normalization), '--strategy', strategy,
               '--mode', mode, '--train-budget-seconds', str(budget), '--allocated']
    if strategy != 'time_conditional':
        command += ['--protocol', str(PROTOCOL), '--protocol-sha256', EXPECTED_PROTOCOL]
    return command, driver_hash


def snapshot(psutil, child):
    processes = [child]
    try:
        processes += child.children(recursive=True)
    except psutil.Error:
        pass
    rss = vms = 0
    cpu = 0.
    for process in processes:
        try:
            memory = process.memory_info(); times = process.cpu_times()
            rss += memory.rss; vms += memory.vms; cpu += times.user + times.system
        except psutil.Error:
            pass
    host = psutil.virtual_memory(); swap = psutil.swap_memory()
    consumers = []
    for process in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            memory = process.info['memory_info']
            if memory and memory.rss >= 256 * 1024**2:
                consumers.append({'pid': process.pid, 'name': process.info['name'], 'rss_bytes': memory.rss})
        except psutil.Error:
            pass
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
        gpu_output, gpu_code = gpu.stdout.strip(), gpu.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        gpu_output, gpu_code = str(error), None
    return {'utc': now(), 'process_tree_rss_bytes': rss, 'process_tree_virtual_bytes': vms,
            'process_tree_cpu_seconds': cpu, 'host_available_bytes': host.available,
            'host_used_percent': host.percent, 'host_swap_used_bytes': swap.used,
            'host_consumers_above_256MiB': consumers,
            'gpu_csv_name_usedMiB_totalMiB_utilizationPercent': gpu_output,
            'gpu_query_exit_code': gpu_code}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=list(DATA))
    parser.add_argument('--strategy', required=True, choices=['time_conditional', 'ar_rollout_training', 'one_step_teacher_forcing'])
    parser.add_argument('--mode', required=True, choices=['probe', 'pilot', 'budget', 'reload', 'final'])
    parser.add_argument('--train-budget-seconds', type=float, default=180.)
    parser.add_argument('--allocated', action='store_true')
    args = parser.parse_args()
    assert args.allocated, 'Explicit root GPU/memory allocation required before monitoring/import/process creation'
    command, driver_hash = command_for(args.domain, args.strategy, args.mode, args.train_budget_seconds)
    import psutil
    directory = OWN / 'multifield-execution-v3'
    directory.mkdir(exist_ok=True)
    stem = directory / f'{args.domain}--{args.strategy}--{args.mode}'
    paths = [Path(str(stem) + suffix) for suffix in ['.command.json', '.log', '.resources.jsonl', '.exit.json']]
    assert not any(path.exists() for path in paths), 'Existing stage evidence is immutable'
    started = now(); start = time.perf_counter()
    with paths[0].open('x', encoding='utf8') as receipt:
        receipt.write(json.dumps({'command': command, 'cwd': str(ROOT), 'started_utc': started,
                                    'driver_sha256': driver_hash, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    'protocol_sha256': EXPECTED_PROTOCOL,
                                    'seed': 20261015, 'root_allocation_required': True}, indent=2))
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PHYSICSNEMO_FORCE_TE': 'False'}
    samples = []; process = None; code = None; runner_error = None
    try:
        with paths[1].open('x', encoding='utf8') as log, paths[2].open('x', encoding='utf8') as resources:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                monitored = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                monitored = None
            while monitored is not None and process.poll() is None:
                row = snapshot(psutil, monitored); samples.append(row)
                resources.write(json.dumps(row) + '\n'); resources.flush()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            code = process.wait()
    except BaseException as error:
        runner_error = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            code = process.returncode
    runner_code = 1 if runner_error else code
    result = {'domain': args.domain, 'strategy': args.strategy, 'mode': args.mode, 'exit_code': code,
              'runner_exit_code': runner_code, 'runner_error': runner_error,
              'started_utc': started, 'finished_utc': now(), 'wall_seconds': time.perf_counter()-start,
              'nominal_poll_wait_seconds': 5, 'resource_sample_count': len(samples),
              'process_tree_rss_observed_peak_bytes': max((row['process_tree_rss_bytes'] for row in samples), default=0),
              'process_tree_virtual_observed_peak_bytes': max((row['process_tree_virtual_bytes'] for row in samples), default=0),
              'host_available_observed_minimum_bytes': min((row['host_available_bytes'] for row in samples), default=None),
              'command_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
              'log_sha256': hashlib.sha256(paths[1].read_bytes()).hexdigest() if paths[1].exists() else None,
              'resources_sha256': hashlib.sha256(paths[2].read_bytes()).hexdigest() if paths[2].exists() else None,
              'measurement_limit': 'Each observation adds GPU-query time before a nominal 5-second wait. Use sample UTC timestamps for actual cadence. Observations can miss transient peaks; driver records CUDA allocation peak separately.'}
    with paths[3].open('x', encoding='utf8') as receipt:
        receipt.write(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)
    sys.exit(runner_code)


if __name__ == '__main__':
    main()
