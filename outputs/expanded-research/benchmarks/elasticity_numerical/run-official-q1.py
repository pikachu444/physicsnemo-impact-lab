"""Validate the untouched official Q1/BiCGSTAB solver after a resource slot.

Without --execute this script imports only the standard library. The affine
patch has independent analytical u/strain/stress/tractions/energy. The second
body-force problem checks equilibrium and work, not exact displacement accuracy.
"""
import argparse
import csv
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--size',type=int,default=9)
    p.add_argument('--out',type=Path,default=OUT/'cpu-execution')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({'status':'prepared_not_executed','required_action':'Run with --execute only after the root resource slot is assigned.'}));return 0
    if a.size<3:raise ValueError('Need at least one free interior node')
    if (a.out/'execution.json').exists():raise ValueError('Preserve existing execution; select a new --out directory')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    sys.dont_write_bytecode=True
    start=time.perf_counter()
    import torch
    import psutil
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import_seconds=time.perf_counter()-start
    source=ROOT/'work/expanded-elasticity/source/examples/tcad/fp_ddm/fpddm/elasticity.py'
    manifest=json.loads((OUT/'source-manifest.json').read_text())
    entry=next(x for x in manifest['files'] if x['path'].endswith('/fpddm/elasticity.py'))
    digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==entry['sha256']
    spec=importlib.util.spec_from_file_location('untouched_official_fpddm_elasticity',source)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    size=a.size;dy=1./(size-1);dx=2./(size-1)
    solver=official.Elasticity2DSolver(spacing=(dy,dx),max_iter=1000,tolerance=1e-12,device=a.device,dtype=torch.float64)
    device=solver.device;dtype=solver.dtype
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def sync():
        if a.device=='cuda':torch.cuda.synchronize(device)
    y=torch.linspace(0.,1.,size,device=device,dtype=dtype);x=torch.linspace(0.,2.,size,device=device,dtype=dtype)
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    volume=torch.full((1,1,size,size),dy*dx,device=device,dtype=dtype)
    volume[...,(0,-1),:]*=.5;volume[...,:,(0,-1)]*=.5
    def number(t):return float(t.detach().cpu())
    def metrics(u,force,mask):
        zero=torch.zeros_like(force)
        internal=solver.residual(u,120.,.25,zero)
        residual=internal-force
        nodal_internal=volume*internal;nodal_external=volume*force
        reaction=torch.where(mask,volume*residual,torch.zeros_like(u))
        scale=max(number(torch.linalg.vector_norm(nodal_internal)),number(torch.linalg.vector_norm(nodal_external)),1e-30)
        free_relative=number(torch.linalg.vector_norm((volume*residual)[~mask]))/scale
        force_balance=(reaction+nodal_external).sum((0,2,3))
        moment=number((xx*(reaction+nodal_external)[:,0]-yy*(reaction+nodal_external)[:,1]).sum())
        energy=.5*number((u*nodal_internal).sum())
        final_force_product=number((u*(reaction+nodal_external)).sum())
        return {'free_dof_force_residual_relative':free_relative,'reaction_sum_y_x':reaction.sum((0,2,3)).cpu().tolist(),
          'body_force_sum_y_x':nodal_external.sum((0,2,3)).cpu().tolist(),'net_force_balance_y_x':force_balance.cpu().tolist(),
          'net_force_balance_relative':number(torch.linalg.vector_norm(force_balance))/scale,
          'net_moment_balance_z':moment,'net_moment_balance_relative':abs(moment)/(scale*(5.**.5)),
          'strain_energy_per_unit_thickness':energy,'final_force_displacement_product_body_plus_reactions':final_force_product,
          'two_energy_vs_final_force_displacement_product_relative':abs(2*energy-final_force_product)/max(abs(2*energy),abs(final_force_product),1e-30)},reaction
    with torch.no_grad():
        exact=torch.stack((.001*yy+.0003*xx,.0003*yy+.002*xx))[None]
        epsilon=exact.new_tensor([[.001,.0003],[.0003,.002]])[None,:,:,None,None]
        sigma=exact.new_tensor([[.192,.0288],[.0288,.288]])[None,:,:,None,None]
        mask=torch.ones_like(exact,dtype=torch.bool);mask[...,1:-1,1:-1]=False
        bc=torch.where(mask,exact,torch.zeros_like(exact));force=torch.zeros_like(exact)
        sync();t=time.perf_counter();u=solver.solve(torch.tensor(120.),torch.tensor(.25),force,bc,mask);sync();patch_seconds=time.perf_counter()-t
        strain=solver.strain(u);stress=solver.stress(u,120.,.25)
        patch,reaction=metrics(u,force,mask)
        traction=torch.zeros_like(u);wy=torch.full((size,),dy,device=device,dtype=dtype);wx=torch.full((size,),dx,device=device,dtype=dtype)
        wy[0]*=.5;wy[-1]*=.5;wx[0]*=.5;wx[-1]*=.5
        for side,sign in ((0,-1.),(-1,1.)):
            traction[0,:,side,:]+=sign*sigma[0,:,0,0,0,None]*wx[None,:]
            traction[0,:,:,side]+=sign*sigma[0,:,1,0,0,None]*wy[None,:]
        patch.update({'displacement_relative_L2':number(torch.linalg.vector_norm(u-exact)/torch.linalg.vector_norm(exact)),
          'boundary_displacement_max_abs_error':number(abs(u[mask]-bc[mask]).max()),
          'strain_max_abs_error':number(abs(strain-epsilon).max()),'stress_max_abs_error':number(abs(stress-sigma).max()),
          'analytic_boundary_traction_nodal_force_max_abs_error':number(abs(reaction-traction).max()),
          'analytic_energy_per_unit_thickness':.00078528,
          'energy_vs_analytic_relative':abs(patch['strain_energy_per_unit_thickness']-.00078528)/.00078528})
        a.out.mkdir(parents=True,exist_ok=True)
        with (a.out/'affine-patch-fields.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['y_L','x_L','u_y_L','u_x_L','strain_yy','strain_xx','strain_yx','stress_yy_S','stress_xx_S','stress_yx_S'])
            values=torch.stack([yy,xx,u[0,0],u[0,1],strain[0,0,0],strain[0,1,1],strain[0,0,1],stress[0,0,0],stress[0,1,1],stress[0,0,1]],dim=-1).cpu().reshape(-1,10).tolist()
            writer.writerows(values)
        body_force=torch.zeros_like(exact);body_force[:,0]=-.01
        clamp=torch.zeros_like(mask);clamp[...,:,0]=True;clamp_bc=torch.zeros_like(exact)
        sync();t=time.perf_counter();body_u=solver.solve(torch.tensor(120.),torch.tensor(.25),body_force,clamp_bc,clamp);sync();body_seconds=time.perf_counter()-t
        body,_=metrics(body_u,body_force,clamp)
        body['boundary_displacement_max_abs_error']=number(abs(body_u[clamp]).max())
        body['max_displacement_magnitude']=number(torch.linalg.vector_norm(body_u,dim=1).max())
        body['independent_expected_total_body_force_y_x']=[-.02,0.]
        body['independent_expected_total_reaction_y_x']=[.02,0.]
        body['accuracy_scope']='No closed-form displacement accuracy claim; independently prescribed constant force, free equilibrium, support reaction balance and energy/work are checked.'
    checks={'affine_u':patch['displacement_relative_L2']<1e-8,'affine_strain':patch['strain_max_abs_error']<1e-9,
      'affine_stress':patch['stress_max_abs_error']<1e-7,'affine_traction':patch['analytic_boundary_traction_nodal_force_max_abs_error']<1e-8,
      'affine_energy':patch['energy_vs_analytic_relative']<1e-8,
      'affine_free_equilibrium':patch['free_dof_force_residual_relative']<1e-8,
      'body_free_equilibrium':body['free_dof_force_residual_relative']<1e-8,'body_force_balance':body['net_force_balance_relative']<1e-8,
      'body_moment_balance':body['net_moment_balance_relative']<1e-8,
      'body_energy_identity':body['two_energy_vs_final_force_displacement_product_relative']<1e-8,
      'body_positive_energy':body['strain_energy_per_unit_thickness']>0,
      'body_independent_force_sum':max(abs(body['body_force_sum_y_x'][0]+.02),abs(body['body_force_sum_y_x'][1]))<1e-12,
      'body_clamp':body['boundary_displacement_max_abs_error']==0}
    try:runtime_physicsnemo_metadata=metadata.version('nvidia-physicsnemo')
    except metadata.PackageNotFoundError:runtime_physicsnemo_metadata=None
    result={'status':'official_q1_validation_passed' if all(checks.values()) else 'official_q1_validation_checks_failed',
      'algorithm':'Untouched official current-source Q1 finite elements + matrix-free preconditioned BiCGSTAB; no neural network, optimizer or training.',
      'official_source_pin':manifest['commit'],'official_source_version':'2.3.0a0','official_solver_sha256':digest,
      'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'source_manifest_sha256':hashlib.sha256((OUT/'source-manifest.json').read_bytes()).hexdigest(),
      'runtime_physicsnemo_metadata_version':runtime_physicsnemo_metadata,'torch_version':torch.__version__,'device':str(device),
      'torch_cuda_build_version':torch.version.cuda,'CUDA_VISIBLE_DEVICES':os.getenv('CUDA_VISIBLE_DEVICES'),
      'cuda_initialized':torch.cuda.is_initialized(),
      'all_solve_tensors_on_cpu':all(v.device.type=='cpu' for v in (u,body_u,body_force,force,bc)) if a.device=='cpu' else None,
      'competing_workload_context':'Root reported separate FIG GPU training at slot allocation; this process does not monitor that workload. Timings are not an isolated-hardware benchmark.',
      'dtype':str(dtype),'cpu_threads':torch.get_num_threads(),'grid':[size,size],'domain_lengths_y_x':[1.,2.],
      'units':'Consistent model length L and stress S; no SI/material unit declared by source. Energy per unit thickness has S L^2 units.',
      'affine_independent_reference':{'u_y':'.001*y+.0003*x','u_x':'.0003*y+.002*x','strain_y_x':[[.001,.0003],[.0003,.002]],'stress_y_x':[[.192,.0288],[.0288,.288]],'body_force':[0,0]},
      'affine_patch':patch,'constant_body_force_clamp':body,'checks':checks,
      'timings_seconds':{'imports':import_seconds,'affine_solve':patch_seconds,'body_solve':body_seconds,'total':time.perf_counter()-start},
      'peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
      'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if a.device=='cuda' else None,
      'physicsnemo_imported':any(x=='physicsnemo' or x.startswith('physicsnemo.') for x in sys.modules),
      'not_claimed':['Official full 2x2 DDM run','PhysicsNeMo 2.2.2 example-version reproduction','AI learning or prediction accuracy','nonlinear/contact/dynamic/plastic/fracture/appliance validation']}
    (a.out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':checks,'seconds':result['timings_seconds']['total'],'peak_working_set_bytes':result['peak_working_set_bytes']}))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
