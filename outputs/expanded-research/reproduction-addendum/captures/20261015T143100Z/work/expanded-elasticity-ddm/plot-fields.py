"""Render saved CPU elasticity fields after the numerical process has exited."""
import argparse
import csv
import json
import math
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import psutil
    if psutil.virtual_memory().available < int(1.5*1024**3):
        raise RuntimeError('Preserve fields; plotting waits for1.5GiB available RAM')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader((args.out/'fields.csv').open())]
    result = json.loads((args.out/'execution.json').read_text())
    assert len(rows) == 32*32
    fields = {}
    for name in ('reference', 'ddm'):
        fields[name] = [
            [math.hypot(row[name+'_u_y'], row[name+'_u_x']) for row in rows],
            [math.sqrt(row[name+'_strain_yy']**2+row[name+'_strain_xx']**2+2*row[name+'_strain_yx']**2) for row in rows],
            [math.sqrt(row[name+'_stress_yy']**2+row[name+'_stress_xx']**2+2*row[name+'_stress_yx']**2) for row in rows],
            [row[name+'_von_mises'] for row in rows],
        ]
    labels = ['Displacement magnitude', 'Small strain tensor norm', 'Plane stress tensor norm', 'Von Mises stress']
    figure, axes = plt.subplots(4, 3, figsize=(11, 12), constrained_layout=True)
    for iy, label in enumerate(labels):
        first, second = fields['reference'][iy], fields['ddm'][iy]
        maximum = max(max(first), max(second))
        values = (first, second, [abs(a-b) for a, b in zip(first, second)])
        for ix, array in enumerate(values):
            image = axes[iy, ix].imshow([array[row*32:(row+1)*32] for row in range(32)],
                                        origin='lower', extent=(0, 1, 0, 1), cmap='magma',
                                        vmin=0, vmax=maximum if ix < 2 else None)
            axes[iy, ix].set_title(('Monolithic', 'Numerical DDM', '|DDM - monolithic|')[ix]+'\n'+label, fontsize=10)
            axes[iy, ix].set_xlabel('x'); axes[iy, ix].set_ylabel('y')
            figure.colorbar(image, ax=axes[iy, ix], fraction=.046)
    figure.suptitle(f"32 x32 plane-stress elasticity |2 x2 Schwarz |{result['schwarz_iterations']} iterations\nDerived linear fields; consistent model units", fontsize=13)
    figure.savefig(args.out/'comparison.png', dpi=160)
    plt.close(figure)
    print(str(args.out/'comparison.png'))


if __name__ == '__main__':
    main()
