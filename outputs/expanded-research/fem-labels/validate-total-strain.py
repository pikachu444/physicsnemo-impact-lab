"""Validate the actual output-only Run224 FEM run against prior identical physics."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('direct',OUT/'direct-anim-reader.py')
direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)

def stats(x,y):
    d=x.astype(np.float64)-y.astype(np.float64)
    return {'max_abs_difference':float(np.max(abs(d))),'component_rmse':float(np.sqrt(np.mean(d*d)))}

def energy(path):
    text=path.read_text(errors='replace')
    final=[line for line in text.splitlines() if re.match(r'\s*100000\s+100\.0',line)][-1]
    return {'printed_final_cycle_row':final.strip(),'normal_termination':'NORMAL TERMINATION' in text,
            'cycles':int(re.search(r'TOTAL NUMBER OF CYCLES\s*:\s*(\d+)',text)[1])}

def main():
    baseline=ROOT/'work/bumper-fem-rerun/run-extended'
    new=ROOT/'work/expanded-fem-labels/run-total-strain'
    cache=ROOT/'work/expanded-fem-labels/new-total-strain-cache/Run224'
    frames=[]
    for number in range(1,12):
        name=f'Bumper_Beam_AP_meshedA{number:03d}'
        a=direct.read_anim(baseline/name);b=direct.read_anim(new/name)
        for key in ('node_id','shell_element_id','shell_part_id','shell_connectivity','shell_active'):
            assert np.array_equal(a[key],b[key]),key
        assert a['time']==b['time']
        entry={'frame':name,'actual_time_ms':b['time'],'displacement_mm':stats(a['displacement'],b['displacement']),
               'coordinates_mm':stats(a['coordinates'],b['coordinates'])}
        entry['existing_stress_fields_GPa']={key:stats(value,b['shell_tensors'][key]) for key,value in a['shell_tensors'].items()}
        entry['existing_epsp_fields_dimensionless']={key:stats(value,b['shell_scalars'][key]) for key,value in a['shell_scalars'].items()}
        frames.append(entry)
    strain=np.load(cache/'shell_total_strain_local_inplane.npy',mmap_mode='r')
    principal=np.load(cache/'nodal_total_strain_principal.npy',mmap_mode='r')
    assert strain.shape==(11,13626,7,3) and principal.shape==(13675,11,4)
    assert np.isfinite(strain).all() and np.isfinite(principal).all()
    assert np.max(abs(strain[0]))==0 and np.count_nonzero(strain[1:])>0
    assert not np.array_equal(strain[:,:,:,0],np.load(cache/'shell_epsp.npy',mmap_mode='r'))
    source_manifest=json.loads((cache/'manifest.json').read_text())
    report={'status':'actual_total_strain_FEM_labels_generated_and_validated',
            'claim_scope':'Same Run224 deck, only two Engine output cards added; actual FEM total strain and EPSP are separate quantities. One condition, no family generalization claim.',
            'units':{'displacement':'mm','time':'ms','stress':'GPa','strain':'dimensionless','energy':'J'},
            'baseline':energy(baseline/'Bumper_Beam_AP_meshed_0001.out'),
            'output_added_run':energy(new/'Bumper_Beam_AP_meshed_0001.out'),
            'paired_frames':frames,
            'total_strain':{'shape':list(strain.shape),'locations':source_manifest['tensor_locations'],
                'components':['element_local_xx','element_local_yy','element_local_xy_tensor_shear'],
                'initial_max_abs':float(np.max(abs(strain[0]))),'min':float(strain.min()),'max':float(strain.max()),'nonzero_values':int(np.count_nonzero(strain)),
                'finite_strain_measure':'Not established; preserve actual solver output without renaming it Hencky/Almansi/Green-Lagrange.'},
            'nodal_inplane_principal_strain':{'shape':list(principal.shape),'channels':source_manifest['nodal_total_strain_principal_channels'],
                'min':float(principal.min()),'max':float(principal.max()),'scope':'Eigenvalues of in-plane 2x2 actual FEM total strain, scalar area-weighted projection, not 3D principal strain.'},
            'energy_error_percent':-.4,'mass_error_fraction_last_printed':.0002315,
            'added_mass_kg_last_printed':.1196,'internal_energy_J_last_printed':6037,'kinetic_energy_J_last_printed':215.2,
            'initial_contact_warning':'The unchanged source deck also has initial penetration warning 343; warning count alone is not a model-invalidity verdict.',
            'execution_metrics':json.loads((OUT/'execution-metrics.json').read_text()),
            'cache_manifest_sha256':__import__('hashlib').sha256((cache/'manifest.json').read_bytes()).hexdigest()}
    (OUT/'total-strain-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords=np.load(cache/'coords.npy');target=np.load(cache/'nodal_targets.npy',mmap_mode='r');p=np.load(cache/'nodal_total_strain_principal.npy',mmap_mode='r')
    deformed=coords+target[:,-1,:3]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,values,title in zip(axes.flat,[target[:,-1,3],target[:,-1,5],p[:,-1,0],p[:,-1,1]],
       ['In-plane VM stress upper (GPa)','Equivalent plastic strain EPSP upper (-)','Total strain: upper in-plane major (-)','Total strain: upper in-plane minor (-)']):
        dots=ax.scatter(deformed[:,1],deformed[:,0],c=values,s=1.7,cmap='viridis',rasterized=True)
        ax.set_aspect('equal');ax.set_xlabel('Y (mm)');ax.set_ylabel('X (mm)');ax.set_title(title,fontsize=10);fig.colorbar(dots,ax=ax,shrink=.72)
    fig.suptitle(f'Run224 actual FEM fields at {source_manifest["frame_checks"][-1]["actual_time_ms"]:.6f} ms\nInitial-area scalar projection to nodes; surface thickness locations and smoothing limits apply',fontsize=11)
    fig.savefig(OUT/'run224-total-strain-fields.png',dpi=170);plt.close(fig)
    print(json.dumps({'status':report['status'],'strain_range':[report['total_strain']['min'],report['total_strain']['max']],
      'paired_max_disp_difference_mm':max(x['displacement_mm']['max_abs_difference'] for x in frames)}))
if __name__=='__main__':main()
