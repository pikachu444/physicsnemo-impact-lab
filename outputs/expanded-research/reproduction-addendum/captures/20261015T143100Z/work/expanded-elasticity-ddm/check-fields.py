"""Independent standard-library manufactured-field and Q1 energy checks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def relative(values, targets):
    return math.sqrt(sum((v-t)**2 for v, t in zip(values, targets)) / max(sum(t*t for t in targets), 1e-30))


def q1_energy(uy, ux, h):
    """Integrate bilinear displacement energy at four Gauss points per cell."""
    total = 0.
    points = ((1-1/math.sqrt(3))/2, (1+1/math.sqrt(3))/2)
    for iy in range(len(uy)-1):
        for ix in range(len(uy[0])-1):
            for s in points:
                for t in points:
                    derivatives = []
                    for values in (uy, ux):
                        a, b = values[iy][ix], values[iy][ix+1]
                        c, d = values[iy+1][ix], values[iy+1][ix+1]
                        derivatives.append((((1-t)*(c-a)+t*(d-b))/h,
                                            ((1-s)*(b-a)+s*(d-c))/h))
                    eyy, exx = derivatives[0][0], derivatives[1][1]
                    eyx = (derivatives[0][1]+derivatives[1][0])/2
                    total += .5*(128*eyy**2+128*exx**2+64*eyy*exx+192*eyx**2)*h*h/4
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader((args.out/'fields.csv').open())]
    execution = json.loads((args.out/'execution.json').read_text())
    assert len(rows) == 32**2 and all(math.isfinite(v) for row in rows for v in row.values())
    h = 1/31
    # Independent affine energy sanity check for the quadrature implementation.
    affine_y = [[0., .0003], [.001, .0013]]
    affine_x = [[0., .002], [.0003, .0023]]
    assert abs(q1_energy(affine_y, affine_x, 1.)-.00039264) < 1e-15
    q1_target = .031744-(.00512/12)*h*h
    metrics = {}
    checks = {'finite_32_by_32_fields': True, 'cpu_only_runtime': execution['all_solve_tensors_on_cpu'] and not execution['cuda_initialized'],
              'no_thermal_or_fno_imports': not execution['inactive_thermal_fno_modules_imported'],
              'schwarz_converged': execution['converged'] and execution['final_interface_rmse'] < 1e-8,
              'independent_affine_energy_integrator_selfcheck': True}
    for name in ('reference', 'ddm'):
        observed = {key: [] for key in ('u', 'strain', 'stress', 'von_mises')}
        exact = {key: [] for key in observed}
        continuum = {key: [] for key in observed}
        boundary_error = 0.
        uy, ux = [], []
        for iy in range(32):
            uy.append([]); ux.append([])
            for ix in range(32):
                row = rows[iy*32+ix]
                y, x = row['y'], row['x']
                assert abs(y-iy*h) < 1e-15 and abs(x-ix*h) < 1e-15
                displacement = [.01*y*y, -.032*x*y]
                boundary_shift = .01*h if iy == 0 else (-.01*h if iy == 31 else 0.)
                strain = [.02*y+boundary_shift, -.032*y, -.016*x, -.016*x]
                stress = [1.536*y+128*boundary_shift, -3.456*y+32*boundary_shift, -1.536*x, -1.536*x]
                vm = math.sqrt(stress[0]**2-stress[0]*stress[1]+stress[1]**2+3*stress[2]**2)
                actual_u = [row[name+'_u_y'], row[name+'_u_x']]
                actual_strain = [row[name+'_strain_yy'], row[name+'_strain_xx'], row[name+'_strain_yx'], row[name+'_strain_yx']]
                actual_stress = [row[name+'_stress_yy'], row[name+'_stress_xx'], row[name+'_stress_yx'], row[name+'_stress_yx']]
                for key, values, target, continuous in (
                    ('u', actual_u, displacement, displacement),
                    ('strain', actual_strain, strain, [.02*y, -.032*y, -.016*x, -.016*x]),
                    ('stress', actual_stress, stress, [1.536*y, -3.456*y, -1.536*x, -1.536*x]),
                    ('von_mises', [row[name+'_von_mises']], [vm], [math.sqrt(19.611648*y*y+7.077888*x*x)])):
                    observed[key].extend(values); exact[key].extend(target); continuum[key].extend(continuous)
                uy[-1].append(actual_u[0]); ux[-1].append(actual_u[1])
                if iy in (0, 31) or ix in (0, 31):
                    boundary_error = max(boundary_error, *(abs(a-b) for a, b in zip(actual_u, displacement)))
        energy = q1_energy(uy, ux, h)
        errors = {key: relative(observed[key], exact[key]) for key in observed}
        metrics[name] = {'manufactured_boundary_aware_relative_L2': errors,
                         'uncorrected_continuum_relative_L2': {key: relative(observed[key], continuum[key]) for key in observed},
                         'physical_boundary_displacement_max_abs_error': boundary_error,
                         'independent_q1_gauss_energy': energy,
                         'energy_vs_analytic_q1_relative': abs(energy-q1_target)/q1_target,
                         'energy_vs_continuum_relative': abs(energy-.031744)/.031744}
        u_limit, field_limit, energy_limit = (1e-7, 1e-6, 1e-6) if name == 'reference' else (1e-4, 5e-4, 1e-5)
        checks[name+'_manufactured_u'] = errors['u'] <= u_limit
        for field in ('strain', 'stress', 'von_mises'):
            checks[name+'_manufactured_'+field] = errors[field] <= field_limit
        checks[name+'_physical_bc'] = boundary_error <= 1e-12
        checks[name+'_independent_q1_energy'] = abs(energy-q1_target)/q1_target <= energy_limit
        checks[name+'_independent_vs_official_energy'] = abs(energy-execution['diagnostics'][name]['strain_energy_from_official_internal_force'])/energy <= 1e-10
        checks[name+'_free_equilibrium'] = execution['diagnostics'][name]['free_dof_force_residual_relative'] <= (1e-6 if name == 'reference' else 1e-3)
    checks['ddm_vs_monolithic_displacement'] = execution['displacement_ddm_vs_monolithic_relative_L2'] <= 1e-4
    for field in ('strain', 'stress', 'von_mises'):
        checks['ddm_vs_monolithic_'+field] = execution[field+'_ddm_vs_monolithic_relative_L2'] <= 5e-4
    result = {'status': 'passed' if all(checks.values()) else 'failed',
              'independent_checker': 'Standard library only; analytic manufactured fields and separately implemented Q1 energy quadrature. No solver/framework import.',
              'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fields_sha256': hashlib.sha256((args.out/'fields.csv').read_bytes()).hexdigest(),
              'analytic_reference': {'u_y': '.01*y^2', 'u_x': '-.032*x*y',
                 'strain_y_x': [['.02*y', '-.016*x'], ['-.016*x', '-.032*y']],
                 'stress_y_x': [['1.536*y', '-1.536*x'], ['-1.536*x', '-3.456*y']],
                 'von_mises': 'sqrt(19.611648*y^2+7.077888*x^2)', 'body_force': [0, 0],
                 'continuum_unit_square_energy': .031744, 'exact_nodal_q1_interpolant_energy': q1_target},
              'boundary_recovery_note': 'Official one-sided outer-y derivatives have epsilon_yy error +/- .01*h, sigma_yy +/-1.28*h, sigma_xx +/-.32*h even for exact nodal displacement. Boundary-aware target includes this known discretization error.',
              'metrics': metrics, 'checks': checks}
    (args.out/'independent-checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'checks_passed': sum(checks.values()), 'checks_total': len(checks), 'failed': [k for k, v in checks.items() if not v]}))
    assert all(checks.values()), 'See saved failed numerical checks'


if __name__ == '__main__':
    main()
