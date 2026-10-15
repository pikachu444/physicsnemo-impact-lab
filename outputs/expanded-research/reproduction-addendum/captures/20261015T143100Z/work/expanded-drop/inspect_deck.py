from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / 'work/expanded-drop/source/Cell_Phone_Drop_0000.rad'
output = root / 'outputs/expanded-research/benchmarks/drop_test/deck-inspection.json'
counts = Counter()
cards = Counter()
includes = []
bbox = [[float('inf')] * 3, [float('-inf')] * 3]
active = ''
header = []
materials = []
parts = []
with source.open(encoding='utf-8') as stream:
    for line_number, line in enumerate(stream, 1):
        text = line.strip()
        if line_number <= 12:
            header.append(text)
        if text.startswith('/'):
            active = text
            cards[active.split('/')[1]] += 1
            if active.startswith(('/INCLUDE', '/SUBMODEL')):
                includes.append({'line': line_number, 'card': active})
            if active.startswith('/MAT/'):
                materials.append({'card': active, 'line': line_number})
            if active.startswith('/PART/'):
                parts.append({'card': active, 'line': line_number})
            continue
        if not text or text.startswith('#'):
            continue
        fields = text.split()
        if active == '/NODE':
            assert len(fields) == 4, (line_number, fields)
            xyz = list(map(float, fields[1:]))
            counts['nodes'] += 1
            for axis in range(3):
                bbox[0][axis] = min(bbox[0][axis], xyz[axis])
                bbox[1][axis] = max(bbox[1][axis], xyz[axis])
        elif active.startswith('/TETRA10/') and len(fields) == 1:
            counts['tetra10'] += 1
        elif active.startswith('/BRICK/'):
            assert len(fields) == 9, (line_number, fields)
            counts['brick'] += 1

assert counts['nodes'] > 200000
assert counts['tetra10'] > 100000
assert len(materials) == 5
assert not includes
result = {
    'source_path': str(source.relative_to(root)),
    'source_bytes': source.stat().st_size,
    'source_sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
    'license_header': header[:4],
    'units': {'mass': 'Mg', 'length': 'mm', 'time': 's'},
    'counts': {**dict(counts), 'solid_elements': counts['tetra10'] + counts['brick'],
               'parts': len(parts), 'materials': len(materials)},
    'coordinate_bounds_mm': {'min': bbox[0], 'max': bbox[1]},
    'card_counts': dict(cards),
    'includes_or_submodels': includes,
    'materials': materials,
    'parts': parts,
    'assertions_passed': True,
}
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
