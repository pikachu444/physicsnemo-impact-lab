"""Run the actual pinned numerical elasticity/Schwarz recipe on CPU only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print('Prepared; framework imports require an allocated --execute run.')
        return 0
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.dont_write_bytecode = True
    started = time.perf_counter()
    import psutil
    process = psutil.Process()
    if psutil.virtual_memory().available < 1024**3:
        raise RuntimeError('Pre-import available memory below allocated v2 1 GiB gate')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device('cpu')
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    manifest = json.loads((HERE / 'adapter-manifest.json').read_text())
    for row in manifest['files']:
        source = HERE / 'source/fpddm' / Path(row['path']).name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row['executed_sha256']
    sys.path.insert(0, str(HERE / 'source'))
    from fpddm.elasticity_example import run_elasticity
    from fpddm.elasticity import Elasticity2DSolver
    imported_seconds = time.perf_counter() - started
    print(json.dumps({'event': 'cpu_run_started', 'pid': os.getpid(),
                      'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
                      'size': 32, 'patches': [2, 2], 'overlap': 4, 'max_iterations': 100}), flush=True)
    solve_started = time.perf_counter()
    result = run_elasticity(args.out, size=32, overlap=4, max_iterations=100,
                            tolerance=1e-8, device='cpu', visualize=False)
    solve_seconds = time.perf_counter() - solve_started
    solver = Elasticity2DSolver(spacing=(1/31, 1/31), device='cpu', dtype=torch.float64)
    arrays = {'reference': result.reference_displacement[None], 'ddm': result.displacement[None]}
    diagnostics = {}
    field_values = {}
    for name, displacement in arrays.items():
        strain = solver.strain(displacement)[0]
        stress = solver.stress(displacement, 120., .25)[0]
        von_mises = solver.von_mises(displacement, 120., .25)[0]
        residual = solver.residual(displacement, 120., .25, torch.zeros_like(displacement))
        volume = torch.full((1, 1, 32, 32), 1/31**2, dtype=torch.float64)
        volume[..., (0, -1), :] *= .5
        volume[..., :, (0, -1)] *= .5
        internal_force = volume * residual
        mask = torch.ones_like(displacement, dtype=torch.bool)
        mask[..., 1:-1, 1:-1] = False
        reactions = torch.where(mask, internal_force, torch.zeros_like(internal_force))
        internal_norm = float(torch.linalg.vector_norm(internal_force))
        energy = .5 * float((displacement * internal_force).sum())
        final_boundary_work = float((displacement * reactions).sum())
        diagnostics[name] = {
            'free_dof_force_residual_relative': float(torch.linalg.vector_norm(internal_force[~mask])) / internal_norm,
            'net_reaction_y_x': reactions.sum((0, 2, 3)).tolist(),
            'net_reaction_relative': float(torch.linalg.vector_norm(reactions.sum((0, 2, 3)))) / internal_norm,
            'strain_energy_from_official_internal_force': energy,
            'final_boundary_force_displacement_product': final_boundary_work,
            'two_energy_vs_final_boundary_work_relative': abs(2*energy-final_boundary_work) / max(abs(2*energy), abs(final_boundary_work), 1e-30),
        }
        field_values[name] = [displacement[0, 0], displacement[0, 1], strain[0, 0], strain[1, 1],
                              strain[0, 1], stress[0, 0], stress[1, 1], stress[0, 1], von_mises]
    args.out.mkdir(parents=True, exist_ok=True)
    suffixes = ['u_y', 'u_x', 'strain_yy', 'strain_xx', 'strain_yx', 'stress_yy', 'stress_xx', 'stress_yx', 'von_mises']
    header = ['y', 'x'] + [name+'_'+suffix for name in ('reference', 'ddm') for suffix in suffixes]
    with (args.out / 'fields.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for iy in range(32):
            for ix in range(32):
                writer.writerow([iy/31, ix/31] + [float(field[iy, ix]) for name in ('reference', 'ddm') for field in field_values[name]])
    def relative(first, second):
        return float(torch.linalg.vector_norm(first-second) / torch.linalg.vector_norm(second).clamp_min(1e-30))
    forbidden = [name for name in sys.modules if name == 'physicsnemo' or name.startswith('physicsnemo.') or name in ('fpddm.model', 'fpddm.solvers', 'fpddm.thermal', 'fpddm.training', 'fpddm.data')]
    all_cpu = all(t.device.type == 'cpu' for t in (result.displacement, result.reference_displacement,
                                                  result.stress, result.reference_stress, result.exact_displacement))
    record = {
        'status': 'full_numerical_ddm_executed', 'official_source_pin': manifest['official_pin'],
        'execution_scope': manifest['scope'],
        'algorithm': 'Actual official 2x2 overlapping Schwarz with Q1/BiCGSTAB local numerical solves; no neural model or training.',
        'source_adapter_manifest_sha256': hashlib.sha256((HERE/'adapter-manifest.json').read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pid': os.getpid(), 'torch_version': torch.__version__, 'device': 'cpu', 'cpu_threads': torch.get_num_threads(),
        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'torch_cuda_build_version': torch.version.cuda,
        'cuda_initialized': torch.cuda.is_initialized(), 'all_solve_tensors_on_cpu': all_cpu,
        'inactive_thermal_fno_modules_imported': forbidden,
        'grid': [32, 32], 'patch_grid': [2, 2], 'patch_size': [18, 18], 'overlap': 4,
        'maximum_schwarz_iterations': 100, 'interface_tolerance': 1e-8,
        'schwarz_iterations': len(result.metrics), 'converged': result.converged,
        'final_interface_rmse': result.metrics[-1]['loss_interface'],
        'numerical_solver_batch_calls': 1 + len(result.metrics),
        'numerical_patch_problems_including_monolithic': 1 + 4*len(result.metrics), 'neural_models_trained': 0,
        'displacement_ddm_vs_monolithic_relative_L2': result.displacement_relative_error,
        'strain_ddm_vs_monolithic_relative_L2': relative(solver.strain(result.displacement[None]), solver.strain(result.reference_displacement[None])),
        'stress_ddm_vs_monolithic_relative_L2': result.stress_relative_error,
        'von_mises_ddm_vs_monolithic_relative_L2': relative(field_values['ddm'][-1], field_values['reference'][-1]),
        'diagnostics': diagnostics,
        'units': 'Consistent model length and stress units; official example declares no SI material units.',
        'stress_strain_scope': 'Derived small linear strain and plane-stress elastic Cauchy stress; not native nonlinear/contact/dynamic/plastic/fracture FEM labels.',
        'timings_seconds': {'imports': imported_seconds, 'official_full_ddm_recipe': solve_seconds, 'total': time.perf_counter()-started},
        'peak_working_set_bytes': getattr(process.memory_info(), 'peak_wset', process.memory_info().rss),
        'competing_workload_context': 'A separate GPU/static-learning suite was active in this shared host; timings are concurrent execution evidence, not isolated performance comparisons.',
    }
    (args.out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    assert all_cpu and not forbidden and not torch.cuda.is_initialized()
    print(json.dumps(record), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
