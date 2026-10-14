"""Check the unmodified MiniQA run and plot its FEM results, without training ML."""
import csv
import hashlib
import json
import pathlib
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parent
run = root / 'run'
lines = (run / 'BOX1BL_0001.out').read_text(errors='replace').splitlines()
records = []
for line in lines:
    match = re.match(r'^\s+(\d+)\s+([\d.E+-]+)\s+([\d.E+-]+)\s+SHELL\s+(\d+)\s+([-\d.]+)%\s+([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)', line)
    if match:
        records.append([float(value) for value in match.groups()])
last = records[-1]
reference_fields = {}
for line in (root / 'original-input' / 'ref.extract').read_text().splitlines():
    fields = line.split()
    if len(fields) >= 3:
        reference_fields[fields[1]] = float(fields[2])
reference = {key: reference_fields[field] for key, field in {
    'last_cycle_index': 'CYCLE', 'time': 'TIME', 'time_step': 'TSTEP',
    'element_id': 'ELTID', 'energy_error_percent': 'ERROR',
    'internal_energy': 'IENERGY', 'kinetic_energy_translation': 'KENERGYT',
    'kinetic_energy_rotation': 'KENERGYR'}.items()}
actual = dict(zip(reference, last))
comparisons = {key: {'reference': value, 'actual_printed': actual[key],
                     'absolute_difference': abs(value - actual[key]),
                     'matches_at_reported_precision': bool(np.isclose(value, actual[key], rtol=1e-12, atol=1e-12))}
               for key, value in reference.items()}
assert all(entry['matches_at_reported_precision'] for entry in comparisons.values())
assert 'NORMAL TERMINATION' in '\n'.join(lines)
with (run / 'BOX1BLT01.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
time = np.array([float(row['time']) for row in rows])

def vtk_geometry(path):
    tokens = path.read_text().split()
    start = tokens.index('POINTS')
    count = int(tokens[start + 1])
    xyz = np.asarray(tokens[start + 3:start + 3 + count * 3], dtype=float).reshape(-1, 3)
    start = tokens.index('CELLS')
    ncells = int(tokens[start + 1])
    offset = start + 3
    cells = []
    for _ in range(ncells):
        count = int(tokens[offset])
        cells.append(np.asarray(tokens[offset + 1:offset + count + 1], dtype=int))
        offset += count + 1
    return xyz, cells

initial, cells = vtk_geometry(run / 'BOX1BLA001.vtk')
final, final_cells = vtk_geometry(run / 'BOX1BLA011.vtk')
assert initial.shape == final.shape
assert len(cells) == len(final_cells)
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)
for field, label in [('INTERNAL ENERGY', 'Internal'), ('PLASTIC WORK', 'Plastic work'), ('HOURGLASS ENERGY', 'Hourglass')]:
    axes[0].plot(time, [float(row[field]) for row in rows], label=label, linewidth=1.8)
axes[0].set(xlabel='Time (model time unit)', ylabel='Energy (model energy unit)', title='FEM time histories')
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.2)
kinetic_axis = axes[0].twinx()
kinetic_axis.plot(time, [float(row['KINETIC ENERGY']) for row in rows], color='#bb3d42', linestyle='--', linewidth=1.6, label='Kinetic (right axis)')
kinetic_axis.set_ylabel('Kinetic energy (model energy unit)', color='#bb3d42')
kinetic_axis.tick_params(axis='y', colors='#bb3d42')
kinetic_axis.legend(loc='lower right', frameon=False)
for xyz, color, label in [(initial, '#8392a5', 'Initial'), (final, '#165fa4', 'Final (model time 0.1001)')]:
    # The VTK also contains visual/rigid entities. Only quadrilateral shell cells are shown.
    first = True
    for cell in cells:
        if len(cell) != 4 or np.any(cell >= 155):
            continue
        p = xyz[np.r_[cell, cell[0]]]
        axes[1].plot(p[:, 0], p[:, 1], color=color, alpha=0.7, linewidth=0.6,
                     label=label if first else None)
        first = False
axes[1].set(xlabel='x (model length unit)', ylabel='y (model length unit)', title='Deformed shell mesh, XY view')
axes[1].set_aspect('equal', adjustable='datalim')
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.2)
fig.suptitle('OpenCourant MiniQA BOX1BL | actual FEM run, no AI prediction', fontsize=13)
fig.savefig(root / 'fem-results.png', dpi=180)
plt.close(fig)

summary = {
    'claim_scope': 'Unmodified small public FEM regression model executed locally on Windows; no AI training or accuracy claim.',
    'solver_commit': 'cb760c206e346eff12161a2268b134ef9d48a4ef',
    'solver_release': 'latest-20261011',
    'reference_comparison': comparisons,
    'printed_energy_balance_error_percent': actual['energy_error_percent'],
    'summary_total_number_of_cycles': 1103,
    'last_cycle_index': 1102,
    'vtk_snapshots': len(list(run.glob('*.vtk'))),
    'vtk_points_including_visual_entities': len(initial),
    'vtk_cells_including_visual_entities': len(cells),
    'time_history_samples': len(rows),
    'last_time_history_sample_model_time': float(time[-1]),
    'time_history_csv_final_internal_energy_model_unit': float(rows[-1]['INTERNAL ENERGY']),
    'time_history_csv_final_plastic_work_model_unit': float(rows[-1]['PLASTIC WORK']),
    'time_history_csv_final_hourglass_energy_model_unit': float(rows[-1]['HOURGLASS ENERGY']),
    'units': {'solver_process_duration': 'wall-clock seconds', 'model_length': 'unverified model length unit', 'model_time': 'unverified model time unit', 'model_energy': 'unverified model energy unit', 'declared_begin_input_work_units': 'kg m s / kg m s', 'caution': 'Legacy QA values rho=7.8e-6, E=205, yield=0.25 do not constitute physically validated steel SI values despite the /BEGIN declaration; no conversion to physical product units is claimed.'},
    'starter_warnings': ['100214: unsupported /IOFLAG field in legacy input', '1220: strain-rate effect without activated rate filtering'],
    'source_input_license': 'CC BY-NC 4.0; preserve header. Solver code is AGPL-3.0.',
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / 'original-input').glob('*'))},
    'raw_results_bytes': sum(p.stat().st_size for p in run.glob('*') if p.is_file()),
}
(root / 'validation-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in summary.items() if key not in ['reference_comparison', 'input_sha256']}, indent=2))
