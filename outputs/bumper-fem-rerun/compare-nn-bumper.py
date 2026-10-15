"""Reuse saved full-mesh NN predictions and compare Run224 with the new FEM run."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import json
import argparse
from pathlib import Path
import runpy
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--training-dir', type=Path, default=ROOT / 'outputs/experiment/fulltraining', help='Directory containing heldout_predictions.npz, metrics.json and mesh_contract.json')
parser.add_argument('--output-dir', type=Path, default=HERE, help='Write comparison JSON/PNG here; use a fresh work/ directory to preserve recorded results')
args = parser.parse_args()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
TRAINING = args.training_dir.resolve()
WORK = ROOT / 'work/bumper-fem-rerun'
RUN = WORK / 'run-extended'
helper = runpy.run_path(str(HERE / 'compare-bumper.py'))
digest = helper['sha256']
artifact = TRAINING / 'heldout_predictions.npz'
stored = np.load(artifact)
training_metrics = json.loads((TRAINING / 'metrics.json').read_text())
contract = json.loads((TRAINING / 'mesh_contract.json').read_text())
runs = training_metrics['split']['test']
assert runs == [f'Run{i}' for i in range(224, 231)]
run_index = runs.index('Run224')
indices = stored['node_indices'].astype(np.int64)
assert len(indices) == len(np.unique(indices)) == 13675
assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() == contract['node_indices_int64_le_sha256']
times = stored['time'].astype(np.float64)
assert np.array_equal(times, np.arange(10, 101, 10, dtype=float))
reference_path = ROOT / 'work/physicsnemo-experiment/data/CURATED_DATA_VTP/VALIDATION_DATA/Run224.vtp'
reader = vtkXMLPolyDataReader()
reader.SetFileName(str(reference_path))
reader.Update()
mesh = reader.GetOutput()
xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
thickness = vtk_to_numpy(mesh.GetPointData().GetArray('thickness'))
assert (thickness[indices] > 0).all()
assert np.array_equal(stored['coords'][run_index], xyz[indices].astype(np.float32))
target = np.stack([vtk_to_numpy(mesh.GetPointData().GetArray(f'displacement_t{time:.3f}')) for time in times], axis=1)[indices].astype(np.float32)
assert np.array_equal(stored['truth'][run_index].reshape(len(indices), len(times), 3), target), 'Saved truth does not exactly match Run224 VTP in node/time order'
prediction = stored['transolver'][run_index].reshape(target.shape).astype(np.float64)
target = target.astype(np.float64)
rad_path = WORK / 'Bumper_Beam_AP_meshed_0000.rad'
rad_ids, rad_xyz = helper['read_nodes'](rad_path)
rad_index, rad_distance, accepted = helper['coordinate_map'](xyz[indices], rad_xyz, .001)
assert accepted.all() and len(np.unique(rad_index)) == len(indices)
mapped_ids = rad_ids[rad_index]
frames = [helper['frame'](path) for path in sorted(RUN.glob('*.vtk'))]
initial = min(frames, key=lambda frame: abs(frame['time']))
assert abs(initial['time']) < 1e-8
initial_lookup = helper['unique_id_lookup'](initial['ids'])
initial_index = np.asarray([initial_lookup[int(node)] for node in mapped_ids])
actual_frames, actual_times, aligned = [], [], []
for nominal in times:
    frame = min(frames, key=lambda item: abs(item['time'] - nominal))
    assert abs(frame['time'] - nominal) <= .1
    lookup = helper['unique_id_lookup'](frame['ids'])
    frame_index = np.asarray([lookup[int(node)] for node in mapped_ids])
    assert np.array_equal(frame['ids'][frame_index], mapped_ids)
    displacement = frame['displacement'][frame_index] if frame['displacement'] is not None else frame['xyz'][frame_index] - initial['xyz'][initial_index]
    assert displacement.shape == (13675, 3) and np.isfinite(displacement).all()
    aligned.append(displacement)
    actual_times.append(frame['time'])
    actual_frames.append({'file': frame['name'], 'sha256': digest(RUN / frame['name']), 'nominal_ms': float(nominal), 'actual_ms': frame['time'], 'difference_ms': frame['time'] - float(nominal)})
actual = np.stack(aligned, axis=1)
assert actual.shape == prediction.shape == target.shape == (13675, 10, 3)
assert np.isfinite(prediction).all() and np.isfinite(target).all()

def metrics(predicted, reference):
    error = predicted - reference
    norms = np.linalg.norm(error, axis=-1)
    return {'component_rmse_mm': float(np.sqrt(np.mean(error ** 2))),
            'vector_rmse_mm': float(np.sqrt(np.mean(np.sum(error ** 2, axis=-1)))),
            'vector_mae_mm': float(np.mean(norms)),
            'vector_error_p95_mm': float(np.quantile(norms, .95)),
            'vector_error_max_mm': float(np.max(norms)),
            'relative_displacement_l2': float(np.linalg.norm(error) / np.linalg.norm(reference))}

pairs = {'nn_vs_original_fem': (prediction, target), 'nn_vs_new_fem': (prediction, actual), 'new_fem_vs_original_fem': (actual, target)}
result = {
    'scope': 'Saved full-training PhysicsNeMo Transolver Run224 predictions compared with independently rerun OpenCourant FEM; no additional training or NN inference executed by this script.',
    'run': 'Run224', 'test_run_index': run_index, 'test_run_order': runs,
    'prediction_file': artifact.relative_to(ROOT).as_posix() if ROOT in artifact.parents else artifact.name, 'prediction_sha256': digest(artifact),
    'training_metrics_sha256': digest(TRAINING / 'metrics.json'),
    'checkpoint_sha256_from_mesh_contract': contract['checkpoint_sha256'],
    'reference_vtp_sha256': digest(reference_path), 'rad_sha256': digest(rad_path),
    'verified_assertions': ['Heldout run names and order exactly Run224..Run230', 'NPZ node_indices hash equals full-mesh contract', 'All 13675 points have positive thickness and unique node indices', 'NPZ coords bitwise equal Run224 VTP coordinates cast float32 at node_indices', 'NPZ truth bitwise equal Run224 VTP displacement arrays cast float32 in identical node/time/component order', 'Initial-coordinate RAD mapping accepted every selected point uniquely within 0.001mm', 'All frame NODE_ID mappings exact, unique, present and finite', 'Nominal times exactly 10,20,...,100ms; each actual FIELD TIME differs by <=0.1ms'],
    'shape_nodes_times_components': list(actual.shape),
    'excluded_reference_points': int(len(xyz) - len(indices)),
    'source_mapping_max_distance_mm': float(np.max(rad_distance[:, 0])),
    'source_mapping_min_second_nearest_distance_mm': float(np.min(rad_distance[:, 1])),
    'actual_frame_times': actual_frames,
    'metric_definitions': {'component_rmse': 'sqrt(mean over nodes,time,xyz of squared displacement error)', 'vector_rmse': 'sqrt(mean over nodes,time of squared Euclidean displacement error)', 'relative_l2': 'L2 norm of displacement error / L2 norm of pair reference displacement'},
    'all_ten_snapshots': {name: metrics(predicted, reference) for name, (predicted, reference) in pairs.items()},
    'final_100ms': {name: metrics(predicted[:, -1], reference[:, -1]) for name, (predicted, reference) in pairs.items()},
    'per_time_component_rmse_mm': {name: [metrics(predicted[:, i], reference[:, i])['component_rmse_mm'] for i in range(10)] for name, (predicted, reference) in pairs.items()},
    'limits': ['This is one heldout run in a fixed public bumper family, not a full appliance or new-geometry validation.', 'New FEM snapshots are nearest actual times, not interpolated; actual/nominal differences are reported.', 'Original FEM conversion history and original solver build remain partly unverified.', 'FEM repeatability error and NN approximation error are reported separately; NN forward timing and FEM wall time do not have identical output scope.']
}
(OUT / 'nn-vs-rerun-run224.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for name, label in [('nn_vs_original_fem', 'NN vs original FEM'), ('nn_vs_new_fem', 'NN vs new FEM')]:
    axes[0].plot(times, result['per_time_component_rmse_mm'][name], label=label)
axes[0].set(xlabel='Nominal time (ms)', ylabel='Component displacement RMSE (mm)', title='Saved NN prediction on Run224')
axes[0].legend(frameon=False)
axes[0].grid(alpha=.2)
axes[1].plot(times, result['per_time_component_rmse_mm']['new_fem_vs_original_fem'], color='#167459')
axes[1].set(xlabel='Nominal time (ms)', ylabel='Component displacement RMSE (mm)', title='New FEM vs original FEM, separate scale')
axes[1].grid(alpha=.2)
fig.suptitle('Run224 | 13,675 verified node IDs | 10 saved times | distinct NN and FEM errors')
fig.savefig(OUT / 'nn-vs-rerun-run224.png', dpi=180)
plt.close(fig)
print(json.dumps({'all_ten_snapshots': result['all_ten_snapshots'], 'final_100ms': result['final_100ms']}, indent=2))
