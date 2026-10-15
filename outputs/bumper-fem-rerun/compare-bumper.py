"""Compare a public Run224 trajectory with an actual FEM rerun, never AI output.

Requires NumPy, SciPy, VTK and Matplotlib. The default paths preserve this
repository's outputs/ and work/ layout. --inspect-only needs no VTK frames.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_nodes(path):
    ids, xyz, active = [], [], False
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('/'):
            active = line.strip() == '/NODE'
        elif active and line.strip() and not line.startswith('#'):
            ids.append(int(line[:10]))
            xyz.append([float(line[start:start + 20]) for start in (10, 30, 50)])
    ids, xyz = np.asarray(ids, dtype=np.int64), np.asarray(xyz)
    if len(ids) < 2 or len(np.unique(ids)) != len(ids) or not np.isfinite(xyz).all():
        raise ValueError('RAD /NODE block is empty, contains duplicate IDs, or has nonfinite coordinates')
    return ids, xyz


def unique_id_lookup(ids):
    values, counts = np.unique(ids, return_counts=True)
    duplicate = set(values[counts > 1].tolist())
    return {int(value): i for i, value in enumerate(ids) if int(value) not in duplicate}


def coordinate_map(query, source, tolerance):
    distance, index = cKDTree(source).query(query, k=2, workers=1)
    accepted = np.isfinite(distance[:, 0]) & (distance[:, 0] <= tolerance) & (distance[:, 1] > tolerance)
    values, counts = np.unique(index[accepted, 0], return_counts=True)
    repeated = values[counts > 1]
    accepted &= ~np.isin(index[:, 0], repeated)
    return index[:, 0], distance, accepted


def describe(values):
    values = np.asarray(values)
    return {'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'p99': float(np.quantile(values, .99)), 'max': float(np.max(values))} if values.size else None


def frame(path):
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    mesh = reader.GetOutput()
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError('VTK frame has no points: ' + path.name)
    fields = mesh.GetFieldData()
    time_array = next((fields.GetArray(i) for i in range(fields.GetNumberOfArrays())
                       if fields.GetArrayName(i).upper() == 'TIME'), None)
    point_data = mesh.GetPointData()
    id_array = next((point_data.GetArray(i) for i in range(point_data.GetNumberOfArrays())
                     if point_data.GetArrayName(i).upper() == 'NODE_ID'), None)
    if time_array is None or id_array is None:
        raise ValueError('VTK frame lacks FIELD TIME or point NODE_ID: ' + path.name)
    ids_float = vtk_to_numpy(id_array).reshape(-1)
    if not np.isfinite(ids_float).all() or not np.equal(ids_float, np.rint(ids_float)).all():
        raise ValueError('VTK NODE_ID contains nonfinite or noninteger values: ' + path.name)
    ids = ids_float.astype(np.int64)
    xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    if len(ids) != len(xyz):
        raise ValueError('VTK point and NODE_ID counts disagree: ' + path.name)
    displacement, displacement_name = None, None
    for i in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(i)
        normalized = re.sub('[^a-z]', '', name.lower())
        array = point_data.GetArray(i)
        if normalized in ('displacement', 'displacements', 'displacementvector') and array.GetNumberOfComponents() == 3:
            displacement, displacement_name = vtk_to_numpy(array).astype(np.float64), name
            break
    actual_time = float(time_array.GetTuple1(0))
    if not np.isfinite(actual_time):
        raise ValueError('VTK TIME is nonfinite: ' + path.name)
    return {'name': path.name, 'time': actual_time, 'ids': ids, 'xyz': xyz,
            'displacement': displacement, 'displacement_name': displacement_name,
            'point_arrays': [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]}


def id_statistics(data):
    ids = data['ids']
    return {'points': len(ids), 'negative_node_ids': int(np.sum(ids < 0)),
            'zero_node_ids': int(np.sum(ids == 0)),
            'duplicate_id_occurrences': int(len(ids) - len(np.unique(ids))),
            'nonfinite_coordinate_points': int(np.sum(~np.isfinite(data['xyz']).all(axis=1)))}


def solver_log_metadata(path):
    if not path.exists():
        return {'available': False}
    text = path.read_text(errors='replace')
    precision = next((line.strip('* ').strip() for line in text.splitlines()[:25] if 'Precision Version' in line), None)
    banner = next((line.strip('* ').strip() for line in text.splitlines()[:25] if re.search(r'Open(?:Radioss|Courant) (?:Starter|Engine)', line)), None)
    threads = re.search(r'NUMBER OF THREADS PER DOMAIN[^\d\n]*(\d+)', text)
    return {'available': True, 'file': path.name, 'sha256': sha256(path),
            'solver_banner': banner, 'precision_banner': precision,
            'threads_per_domain': int(threads.group(1)) if threads else None,
            'exact_build_version': 'Not identified by these log banners; an input deck version is not a solver build version.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'work/bumper-fem-rerun/run-extended')
    parser.add_argument('--rad', type=Path, default=ROOT / 'work/bumper-fem-rerun/Bumper_Beam_AP_meshed_0000.rad')
    parser.add_argument('--reference', type=Path, default=ROOT / 'work/physicsnemo-experiment/data/CURATED_DATA_VTP/VALIDATION_DATA/Run224.vtp')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--tolerance-mm', type=float, default=.001)
    parser.add_argument('--target-ms', type=float, default=100.)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    if args.tolerance_mm <= 0 or not np.isfinite(args.tolerance_mm):
        raise ValueError('Mapping tolerance must be positive and finite')
    report = {'claim_scope': 'FEM-to-FEM comparison: original public OpenRadioss-derived Run224 versus a new OpenCourant rerun; no AI accuracy claim.',
              'status': 'source_mapping_inspected',
              'precision_scope': 'Observed initial-coordinate differences are compatible with limited output precision; the complete conversion provenance is unverified. Solver build, precision, output timing, conversion and numerical settings can cause differences.',
              'units': {'length': 'mm', 'time': 'ms', 'displacement': 'mm'},
              'mapping_tolerance_mm': args.tolerance_mm,
              'reference_file': args.reference.name, 'reference_sha256': sha256(args.reference),
              'rad_file': args.rad.name, 'rad_sha256': sha256(args.rad)}
    report['original_solver_log'] = solver_log_metadata(args.rad.with_name('Bumper_Beam_AP_meshed_0001.out'))
    report['rerun_solver_log'] = solver_log_metadata(args.run / 'Bumper_Beam_AP_meshed_0001.out')
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(args.reference))
    reader.Update()
    mesh = reader.GetOutput()
    if not mesh.GetPoints():
        raise ValueError('Reference VTP contains no points')
    ref_xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    point_data = mesh.GetPointData()
    target_name = f'displacement_t{args.target_ms:.3f}'
    target_array = point_data.GetArray(target_name)
    if target_array is None:
        raise ValueError('Reference VTP lacks ' + target_name)
    reference = vtk_to_numpy(target_array).astype(np.float64)
    thickness_array = point_data.GetArray('thickness')
    thickness = vtk_to_numpy(thickness_array) if thickness_array is not None else np.full(len(ref_xyz), np.nan)
    node_ids, rad_xyz = read_nodes(args.rad)
    rad_index, distances, accepted = coordinate_map(ref_xyz, rad_xyz, args.tolerance_mm)
    mapped_ids = np.full(len(ref_xyz), np.iinfo(np.int64).min, dtype=np.int64)
    mapped_ids[accepted] = node_ids[rad_index[accepted]]
    report['source_mapping'] = {
        'vtp_points': len(ref_xyz), 'rad_nodes': len(node_ids),
        'vtp_point_arrays': [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())],
        'exact_duplicate_vtp_coordinates': len(ref_xyz) - len(np.unique(ref_xyz, axis=0)),
        'exact_duplicate_rad_coordinates': len(rad_xyz) - len(np.unique(rad_xyz, axis=0)),
        'accepted_to_rad': int(accepted.sum()),
        'accepted_initial_distance_mm': describe(distances[accepted, 0]),
        'all_nearest_distance_mm': describe(distances[:, 0]),
        'minimum_second_candidate_distance_mm': float(distances[:, 1].min()),
        'unresolved_points': [{'vtp_index': int(i), 'coordinate_mm': ref_xyz[i].tolist(),
                               'thickness_mm': float(thickness[i]) if np.isfinite(thickness[i]) else None,
                               'nearest_rad_node_id': int(node_ids[rad_index[i]]),
                               'nearest_distance_mm': float(distances[i, 0])} for i in np.flatnonzero(~accepted)]}
    if args.inspect_only:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    paths = sorted(args.run.glob('*.vtk'))
    if not paths:
        raise ValueError('No converted VTK frames available; use --inspect-only until conversion completes')
    frames = [frame(path) for path in paths]
    initial = min(frames, key=lambda item: item['time'])
    final = min(frames, key=lambda item: abs(item['time'] - args.target_ms))
    if max(item['time'] for item in frames) < args.target_ms - .1:
        raise ValueError('Available VTK frames do not yet reach the target time within 0.1 ms; comparison not written')
    if abs(initial['time']) > .001:
        raise ValueError('Earliest frame is not a time-zero reference (tolerance 0.001 ms)')
    initial_lookup, final_lookup = unique_id_lookup(initial['ids']), unique_id_lookup(final['ids'])
    fallback_index, fallback_distance, fallback_ok = coordinate_map(ref_xyz, initial['xyz'], args.tolerance_mm)
    fallback = ~accepted & fallback_ok
    fallback &= np.array([int(initial['ids'][index]) in initial_lookup for index in fallback_index])
    mapped_ids[fallback] = initial['ids'][fallback_index[fallback]]
    accepted |= fallback
    values, counts = np.unique(mapped_ids[accepted], return_counts=True)
    repeated = values[counts > 1]
    accepted &= ~np.isin(mapped_ids, repeated)
    selected = np.flatnonzero(accepted)
    has_both = np.array([int(mapped_ids[i]) in initial_lookup and int(mapped_ids[i]) in final_lookup for i in selected])
    missing_count = int((~has_both).sum())
    selected = selected[has_both]
    first_index = np.array([initial_lookup[int(mapped_ids[i])] for i in selected], dtype=np.int64)
    final_index = np.array([final_lookup[int(mapped_ids[i])] for i in selected], dtype=np.int64)
    prediction = (final['displacement'][final_index] if final['displacement'] is not None
                  else final['xyz'][final_index] - initial['xyz'][first_index])
    finite = np.isfinite(prediction).all(axis=1) & np.isfinite(reference[selected]).all(axis=1)
    finite &= np.isfinite(initial['xyz'][first_index]).all(axis=1) & np.isfinite(final['xyz'][final_index]).all(axis=1)
    nonfinite_count = int((~finite).sum())
    selected, prediction = selected[finite], prediction[finite]
    first_index = first_index[finite]
    if not len(selected):
        raise ValueError('No finite, uniquely aligned points remain')
    target = reference[selected]
    difference = prediction - target
    error_norm = np.linalg.norm(difference, axis=1)
    ref_norm, actual_norm = np.linalg.norm(target, axis=1), np.linalg.norm(prediction, axis=1)
    denominator = float(np.linalg.norm(target))
    initial_error = np.linalg.norm(initial['xyz'][first_index] - ref_xyz[selected], axis=1)
    report.update({
        'status': 'fem_to_fem_compared', 'target_reference_array': target_name,
        'target_time_ms': args.target_ms, 'actual_time_ms': final['time'],
        'time_difference_ms': final['time'] - args.target_ms,
        'time_alignment': 'Closest available actual FIELD TIME; no temporal interpolation or time relabeling.',
        'initial_frame': initial['name'], 'initial_time_ms': initial['time'],
        'selected_frame': final['name'], 'selected_frame_sha256': sha256(args.run / final['name']),
        'available_frames': [{'name': item['name'], 'actual_time_ms': item['time']} for item in frames],
        'initial_frame_statistics': id_statistics(initial), 'final_frame_statistics': id_statistics(final),
        'final_frame_point_arrays': final['point_arrays'],
        'displacement_source': ('VTK point vector ' + final['displacement_name']) if final['displacement'] is not None else 'Final minus initial VTK coordinates aligned by NODE_ID',
        'effective_counts': {'reference_points': len(ref_xyz), 'initial_vtk_recovered_points': int(fallback.sum()),
                             'unique_mapping_points': int(accepted.sum()), 'missing_in_either_frame': missing_count,
                             'nonfinite_aligned_points': nonfinite_count, 'compared_points': len(selected),
                             'excluded_points': len(ref_xyz) - len(selected)},
        'excluded_vtp_indices': sorted(set(range(len(ref_xyz))) - set(selected.tolist())),
        'initial_vtk_to_vtp_distance_mm': describe(initial_error),
        'metrics': {
            'vector_rmse_mm': float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))),
            'vector_rmse_definition': 'sqrt(mean over nodes of squared Euclidean displacement error)',
            'component_rmse_mm': float(np.sqrt(np.mean(difference ** 2))),
            'component_rmse_definition': 'sqrt(mean over all nodes and x/y/z components of squared displacement error)',
            'component_rmse_xyz_mm': np.sqrt(np.mean(difference ** 2, axis=0)).tolist(),
            'vector_mae_mm': float(np.mean(error_norm)),
            'component_mae_mm': float(np.mean(np.abs(difference))),
            'vector_error_p95_mm': float(np.quantile(error_norm, .95)),
            'vector_error_max_mm': float(error_norm.max()),
            'relative_displacement_l2': float(np.linalg.norm(difference) / denominator) if denominator else None,
            'relative_l2_definition': 'L2 norm of displacement error divided by L2 norm of reference displacement, on identical compared nodes',
            'reference_displacement_norm_mm': describe(ref_norm), 'rerun_displacement_norm_mm': describe(actual_norm)},
        'comparison_limitations': ['This compares two numerical FEM trajectories, not a physical impact experiment or an AI prediction.',
                                   'The original public trajectory and the new OpenCourant rerun can differ in solver version, precision, settings and conversion.',
                                   'Spatial correspondence uses undeformed coordinates and persistent node IDs; unresolved points are explicitly excluded.',
                                   'The nearest saved time can differ from nominal 100 ms; review the recorded difference before interpreting errors.']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    limits = (float(min(ref_norm.min(), actual_norm.min())), float(max(ref_norm.max(), actual_norm.max())))
    axes[0].scatter(ref_norm, actual_norm, s=3, alpha=.4, rasterized=True)
    axes[0].plot(limits, limits, color='black', linewidth=.8)
    axes[0].set(xlabel='Public reference |displacement| (mm)', ylabel='OpenCourant rerun |displacement| (mm)', title='FEM displacement magnitude')
    axes[0].set_aspect('equal', adjustable='box')
    image = axes[1].scatter(ref_xyz[selected, 1], ref_xyz[selected, 2], c=error_norm, s=4, cmap='magma', rasterized=True)
    axes[1].set(xlabel='Initial y (mm)', ylabel='Initial z (mm)', title='Vector error on initial geometry')
    fig.colorbar(image, ax=axes[1], label='Displacement error norm (mm)')
    axes[2].hist(error_norm, bins=50, color='#2b638a')
    axes[2].set(xlabel='Displacement error norm (mm)', ylabel='Compared nodes', title=f'{len(selected):,} aligned finite points')
    fig.suptitle(f'Public Run224 vs actual FEM rerun | target {args.target_ms:g} ms, saved {final["time"]:.6f} ms | no AI prediction')
    args.output.mkdir(parents=True, exist_ok=True)
    plot = args.output / 'fem-to-fem-run224-comparison.png'
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    report['plot_file'] = plot.name
    destination = args.output / 'fem-to-fem-run224-comparison.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'json': destination.name, 'plot': plot.name,
                      'effective_counts': report['effective_counts'], 'metrics': report['metrics']}, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__,
                          'error': str(error).replace(str(ROOT), '<WORKSPACE>')}, indent=2), file=sys.stderr)
        raise SystemExit(1)
