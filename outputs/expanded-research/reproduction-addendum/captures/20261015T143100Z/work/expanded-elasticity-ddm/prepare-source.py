"""Prepare an isolated numerical-only adapter from verified official sources."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'work/expanded-elasticity/source/examples/tcad/fp_ddm'
MANIFEST = ROOT / 'outputs/expanded-research/benchmarks/elasticity_numerical/source-manifest.json'
PIN = '11685edb3ec1fb6f25665e96cf7a8c4d11a23f16'


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['commit'] == PIN
    selected = ['__init__.py', 'domain.py', 'elasticity.py', 'elasticity_example.py',
                'interfaces.py', 'observers.py', 'schwarz.py']
    records = []
    for name in selected:
        relative = 'examples/tcad/fp_ddm/fpddm/' + name
        entry = next(row for row in manifest['files'] if row['path'] == relative)
        data = (ORIGINAL / 'fpddm' / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], name
        changes = []
        if name == 'schwarz.py':
            for line in (b'from .model import thermal_losses\n',
                         b'from .solvers import NeuralDomainSolver\n'):
                assert data.count(line) == 1
                data = data.replace(line, b'')
                changes.append({'removed_unused_import': line.decode().strip()})
        destination = HERE / 'source/fpddm' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert destination.read_bytes() == data, destination
        else:
            destination.write_bytes(data)
        records.append({'path': relative, 'url': entry['url'],
                        'official_sha256': entry['sha256'],
                        'executed_sha256': hashlib.sha256(data).hexdigest(),
                        'changes': changes})
    record = {'official_pin': PIN, 'source_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              'scope': 'Actual official run_elasticity, Q1 solver, domain, interface exchange, metrics, and Schwarz/EarlyStopping. Only two unused thermal/FNO imports removed from isolated Schwarz copy. No official CLI claim.',
              'files': records}
    (HERE / 'adapter-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'prepared', 'files': len(records), 'framework_imports': False}))


if __name__ == '__main__':
    main()
