"""Summarize actual Run224 FEM logs; no ML or GPU work is performed here."""
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
WORK = ROOT / 'work' / 'bumper-fem-rerun'
RUN = WORK / 'run-extended'
STEM = 'Bumper_Beam_AP_meshed'

def summary(path):
    text = path.read_text(errors='replace')
    rows = []
    for line in text.splitlines():
        match = re.match(r'^\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+(NODE|SHELL)\s+(\d+)\s+([-\d.]+)%\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)', line)
        if match:
            values = list(match.groups())
            fields = ['cycle_index', 'model_time_ms', 'time_step_ms', 'controlling_entity_type', 'controlling_entity_id', 'energy_error_percent', 'internal_energy_J', 'translational_kinetic_energy_J', 'rotational_kinetic_energy_J', 'external_work_J', 'mass_error_fraction', 'total_mass_kg', 'added_mass_kg']
            rows.append({key: value if key == 'controlling_entity_type' else float(value) for key, value in zip(fields, values)})
    get = lambda pattern: [float(value) for value in re.findall(pattern, text)]
    if rows:
        rows[-1]['reported_mass_error_percent'] = 100 * rows[-1]['mass_error_fraction']
        rows[-1]['added_mass_ratio_to_printed_total_mass'] = rows[-1]['added_mass_kg'] / rows[-1]['total_mass_kg'] if rows[-1]['total_mass_kg'] else None
    return {'normal_termination': 'NORMAL TERMINATION' in text,
            'precision_header': next((line.strip('* ').strip() for line in text.splitlines() if 'Precision Version' in line), None),
            'threads_per_domain': int(get(r'NUMBER OF THREADS PER DOMAIN\s+(\d+)')[0]) if get(r'NUMBER OF THREADS PER DOMAIN\s+(\d+)') else None,
            'spmd_domains': int(get(r'NUMBER OF SPMD DOMAINS\s+(\d+)')[0]) if get(r'NUMBER OF SPMD DOMAINS\s+(\d+)') else None,
            'solver_reported_elapsed_seconds': get(r'ELAPSED TIME\s*=\s*([\d.]+)\s*s')[-1] if get(r'ELAPSED TIME\s*=\s*([\d.]+)\s*s') else None,
            'solver_reported_memory_MB': get(r'TOTAL MEMORY USED\s*\.*:\s*([\d.]+)\s*MB')[-1] if get(r'TOTAL MEMORY USED\s*\.*:\s*([\d.]+)\s*MB') else None,
            'summary_total_cycles': get(r'TOTAL NUMBER OF CYCLES\s*:\s*(\d+)')[-1] if get(r'TOTAL NUMBER OF CYCLES\s*:\s*(\d+)') else None,
            'last_printed_record': rows[-1] if rows else None}

source = summary(WORK / (STEM + '_0001.out'))
current = summary(RUN / (STEM + '_0001.out'))
manifest = json.loads((OUT / 'source-manifest.json').read_text(encoding='utf-8-sig'))
metrics = json.loads((OUT / 'execution-metrics.json').read_text(encoding='utf-8-sig'))
starter = (RUN / (STEM + '_0000.out')).read_text(errors='replace')
source_starter = (WORK / (STEM + '_0000.out')).read_text(errors='replace')
features = json.loads((ROOT / 'work' / 'physicsnemo-experiment' / 'data' / 'CURATED_DATA_VTP' / 'GLOBAL_FEATURES.json').read_text())['Run224']
result = {
    'scope': 'Actual local Windows rerun of the same public held-out Run224 bumper FEM input; FEM-to-FEM comparison, not AI accuracy.',
    'dataset_revision': manifest['dataset_revision'],
    'dataset_path': 'RAW_DATA/VALIDATION_DATA/Run224',
    'global_features_from_curated_dataset': features,
    'current_solver_release': 'OpenCourant latest-20261011',
    'current_solver_commit': 'cb760c206e346eff12161a2268b134ef9d48a4ef',
    'source_solver': 'OpenRadioss; precise binary release/build not identified in supplied log',
    'source_threads': source['threads_per_domain'],
    'current_threads': current['threads_per_domain'],
    'model_units': {'mass': 'kg', 'length': 'mm', 'time': 'ms', 'velocity': 'mm/ms', 'stress': 'GPa', 'energy': 'J'},
    'unit_evidence': '/BEGIN input and work kg/mm/ms, steel density 7.85e-6 kg/mm^3, E=210 GPa',
    'source_engine': source,
    'current_engine': current,
    'process_measurements': metrics,
    'run_artifacts_bytes_including_raw_and_converted_local_files': sum(path.stat().st_size for path in RUN.iterdir() if path.is_file()),
    'current_starter_warning_ids': sorted(set(re.findall(r'WARNING ID\s*:\s*(\d+)', starter))),
    'current_starter_reported_warning_count': int(re.findall(r'^\s*(\d+)\s+WARNING\(S\)', starter, re.MULTILINE)[-1]),
    'current_starter_error_ids': re.findall(r'ERROR ID\s*:\s*(\d+)', starter),
    'source_starter_warning_ids': sorted(set(re.findall(r'WARNING ID\s*:\s*(\d+)', source_starter))),
    'source_starter_reported_warning_count': int(re.findall(r'^\s*(\d+)\s+WARNING\(S\)', source_starter, re.MULTILINE)[-1]),
    'initial_penetration_warning': {'source_penetrations': re.findall(r'THERE ARE\s+(\d+)\s+INITIAL PENETRATIONS', source_starter), 'current_penetrations': re.findall(r'THERE ARE\s+(\d+)\s+INITIAL PENETRATIONS', starter), 'source_concerned_nodes': re.findall(r'CONCERNING\s+(\d+)\s+NODES', source_starter), 'current_concerned_nodes': re.findall(r'CONCERNING\s+(\d+)\s+NODES', starter), 'interpretation': 'Inherited public-model warning; recorded without changing the physical input or judging validity from warning count alone.'},
    'input_hash_unchanged': {},
    'timing_limitations': 'Source and rerun differ in solver build and threads. Other GPU training/inference tasks ran concurrently during local rerun; timing is not an isolated controlled benchmark.',
}
for item in manifest['files']:
    if item['name'].endswith('.rad'):
        expected = item['sha256']
        actual = hashlib.sha256((RUN / item['name']).read_bytes()).hexdigest()
        result['input_hash_unchanged'][item['name']] = {'sha256': actual, 'matches_pinned_source': actual == expected}
assert all(entry['matches_pinned_source'] for entry in result['input_hash_unchanged'].values())
if current['normal_termination']:
    assert current['last_printed_record']['model_time_ms'] >= 100
    result['final_global_differences'] = {}
    for key in ['internal_energy_J', 'translational_kinetic_energy_J', 'rotational_kinetic_energy_J', 'external_work_J', 'total_mass_kg', 'added_mass_kg']:
        original = source['last_printed_record'][key]
        rerun = current['last_printed_record'][key]
        result['final_global_differences'][key] = {'source_printed': original, 'rerun_printed': rerun, 'difference': rerun - original, 'relative_difference': (rerun - original) / abs(original) if original else None}
else:
    result['final_global_differences'] = None
(OUT / 'fem-rerun-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

# Preserve original local logs in work; publish only text with machine paths/hostnames redacted.
for prefix, directory in [('source', WORK), ('rerun', RUN)]:
    for name in [STEM + '_0000.out', STEM + '_0001.out', 'starter.stdout.log', 'starter.stderr.log', 'engine.stdout.log', 'engine.stderr.log']:
        path = directory / name
        if not path.exists():
            continue
        text = path.read_text(errors='replace').replace(str(ROOT), '<WORKSPACE>')
        text = re.sub(r'(?m)^(\s*)(?:DESKTOP|LAPTOP)-[^\s]+', r'\1<HOST>', text)
        text = re.sub(r'[A-Za-z]:\\Users\\[^\r\n]*?(?=Bumper_Beam_AP_meshed|$)', '<SOURCE_WORKSPACE>/', text, flags=re.MULTILINE)
        (OUT / (prefix + '-' + name)).write_text(text, encoding='utf-8')
print(json.dumps({'source': source, 'rerun': current, 'warnings': result['current_starter_warning_ids']}, indent=2))
