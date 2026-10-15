"""Extract pinned FEM ANIM labels without averaging local tensor components.

Canonical tensors remain element/local/through-thickness quantities. Only
in-plane von Mises invariants and EPSP scalars are projected to reference nodes.
One VTK frame is resident at a time; outputs are float32 NumPy mmap files.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import psutil
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

ROOT = Path(__file__).resolve().parents[3]
LOCATIONS = ['upper', 'lower'] + [f'layer_{i}' for i in range(1, 6)]
CHANNELS = ['displacement_x', 'displacement_y', 'displacement_z',
            'inplane_vm_upper', 'inplane_vm_lower', 'EPSP_upper', 'EPSP_lower']

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write_manifest(path,value):
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def input_mesh(path):
    ids, coords, elements, parts = [], [], [], []
    active, part = '', None
    for line in Path(path).read_text().splitlines():
        if line.startswith('/'):
            fields = line.strip().split('/')
            active = fields[1]
            part = int(fields[2]) if active in ('SHELL', 'SH3N') else None
        elif line.strip() and not line.startswith('#'):
            if active == 'NODE':
                ids.append(int(line[:10]))
                coords.append([float(line[j:j+20]) for j in (10, 30, 50)])
            elif active in ('SHELL', 'SH3N'):
                values = [int(x) for x in line.split()]
                e, nodes = values[0], values[1:]
                if active == 'SH3N': nodes.append(nodes[-1])
                elements.append([e, *nodes]); parts.append(part)
    ids = np.asarray(ids, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    order = np.argsort(elements[:, 0])
    return ids, coords, elements[order], np.asarray(parts, dtype=np.int64)[order]

def unique_lookup(values, label):
    values = np.asarray(values, dtype=np.int64)
    u, counts = np.unique(values, return_counts=True)
    duplicate = set(u[counts > 1].tolist())
    return {int(v): i for i, v in enumerate(values) if int(v) not in duplicate}

def curated(path):
    reader = vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
    mesh = reader.GetOutput(); pd = mesh.GetPointData()
    xyz = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    thickness = vtk_to_numpy(pd.GetArray('thickness')).reshape(-1).astype(np.float32)
    mask = np.flatnonzero(thickness > 0)
    truth = {}
    for i in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(i)
        if name.startswith('displacement_t'):
            truth[float(name.split('_t')[1])] = vtk_to_numpy(pd.GetArray(i))[mask].copy()
    return xyz[mask], thickness[mask], mask, truth

def array(data, name):
    a = data.GetArray(name)
    if a is None: raise ValueError('Missing FEM field ' + name)
    result = vtk_to_numpy(a)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite FEM field ' + name)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anim-dir', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--rad', type=Path, default=ROOT/'work/bumper-fem-rerun/Bumper_Beam_AP_meshed_0000.rad')
    p.add_argument('--curated-root', type=Path, default=ROOT/'work/physicsnemo-experiment/data/CURATED_DATA_VTP')
    p.add_argument('--converter', type=Path, default=ROOT/'work/opencourant-experiment/solver/OpenCourant/exec/anim_to_vtk_win64.exe')
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--include-total-strain', action='store_true')
    p.add_argument('--backend',choices=['direct','vtk'],default='direct')
    p.add_argument('--provenance',choices=['original_public_OpenRadioss_ANIM','new_output_only_OpenCourant_Run224'],default='original_public_OpenRadioss_ANIM')
    a = p.parse_args(); started = time.perf_counter(); a.output_dir.mkdir(parents=True, exist_ok=True)
    destination=a.output_dir.resolve()
    write_manifest(destination/'manifest.json',{'status':'building','run':a.run_name})
    a.output_dir=destination/('building-'+str(os.getpid()))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    frames = sorted(f for f in a.anim_dir.glob('*A[0-9][0-9][0-9]') if f.is_file())
    frames = [f for f in frames if int(f.name[-3:]) <= 11]
    if not frames or (len(frames) != 11 and not a.allow_partial): raise ValueError('Expected exactly 11 ANIM frames A001..A011')
    if a.include_total_strain and a.provenance!='new_output_only_OpenCourant_Run224':
        raise ValueError('New total-strain execution must be explicitly identified with --provenance')
    reference = list(a.curated_root.glob('*/'+a.run_name+'.vtp'))
    if len(reference) != 1: raise ValueError('Expected one curated VTP for '+a.run_name)
    coords, thickness, selected, truth = curated(reference[0])
    rad_ids, rad_xyz, shells, shell_parts = input_mesh(a.rad)
    distances, indices = cKDTree(rad_xyz).query(coords, k=2, workers=1)
    assert np.all(distances[:, 0] <= .001) and np.all(distances[:, 1] > .001)
    assert len(np.unique(indices[:, 0])) == len(coords)
    node_ids = rad_ids[indices[:, 0]]
    rad_lookup = unique_lookup(rad_ids, 'RAD node'); target_lookup = unique_lookup(node_ids, 'target node')
    connectivity = np.array([[rad_lookup[int(v)] for v in row[1:]] for row in shells], dtype=np.int64)
    points = rad_xyz[connectivity]
    areas = .5*np.linalg.norm(np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0]),axis=1)
    areas += .5*np.linalg.norm(np.cross(points[:,2]-points[:,0], points[:,3]-points[:,0]),axis=1)
    assert np.all(areas > 0)
    # Equal weight per unique corner, multiplied by the initial geometric area.
    dst, src, weights = [], [], []
    for e, shell in enumerate(shells):
        corners = list(dict.fromkeys(shell[1:].tolist()))
        for node in corners:
            if node in target_lookup:
                dst.append(target_lookup[node]); src.append(e); weights.append(areas[e]/len(corners))
    dst=np.asarray(dst); src=np.asarray(src); weights=np.asarray(weights)
    denominator=np.bincount(dst,weights=weights,minlength=len(node_ids))
    assert np.all(denominator > 0), 'Positive-thickness target node without real-shell support'
    constant=np.bincount(dst,weights=weights/denominator[dst],minlength=len(node_ids))
    assert np.max(abs(constant-1)) < 1e-12
    T,N,E=len(frames),len(node_ids),len(shells)
    np.save(a.output_dir/'node_id.npy',node_ids); np.save(a.output_dir/'coords.npy',coords.astype(np.float32))
    np.save(a.output_dir/'curated_node_indices.npy',selected.astype(np.int64)); np.save(a.output_dir/'thickness.npy',thickness)
    np.save(a.output_dir/'shell_element_id.npy',shells[:,0]); np.save(a.output_dir/'shell_part_id.npy',shell_parts)
    np.save(a.output_dir/'shell_node_id.npy',shells[:,1:]); np.save(a.output_dir/'shell_initial_area_mm2.npy',areas.astype(np.float32))
    targets=np.lib.format.open_memmap(a.output_dir/'nodal_targets.npy',mode='w+',dtype='float32',shape=(N,T,7))
    stress=np.lib.format.open_memmap(a.output_dir/'shell_stress_local_inplane.npy',mode='w+',dtype='float32',shape=(T,E,7,3))
    epsp=np.lib.format.open_memmap(a.output_dir/'shell_epsp.npy',mode='w+',dtype='float32',shape=(T,E,7))
    status=np.lib.format.open_memmap(a.output_dir/'shell_animation_status.npy',mode='w+',dtype='uint8',shape=(T,E))
    strain=np.lib.format.open_memmap(a.output_dir/'shell_total_strain_local_inplane.npy',mode='w+',dtype='float32',shape=(T,E,7,3)) if a.include_total_strain else None
    principal=np.lib.format.open_memmap(a.output_dir/'nodal_total_strain_principal.npy',mode='w+',dtype='float32',shape=(N,T,4)) if a.include_total_strain else None
    solver_vm=np.lib.format.open_memmap(a.output_dir/'shell_solver_von_mises.npy',mode='w+',dtype='float32',shape=(T,E)) if a.include_total_strain else None
    all_times=[]; checks=[]; sources=[]; vmcheck=[]
    if a.backend=='direct':
        spec=importlib.util.spec_from_file_location('direct_anim_reader',Path(__file__).with_name('direct-anim-reader.py'))
        direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)
    env=os.environ.copy()
    solver=a.converter.parent.parent
    env['PATH']=';'.join(str(solver/'extlib'/x/'win64') for x in ('hm_reader','h3d/lib','intelOneAPI_runtime'))+';'+env['PATH']
    for t,path in enumerate(frames):
        frame_start=time.perf_counter(); vtk_path=a.output_dir/'frame.tmp.vtk'
        if a.backend=='direct':
            raw=direct.read_anim(path)
            frame_ids=raw['node_id'];element_ids=raw['shell_element_id'];part_ids=raw['shell_part_id']
            cells=raw['shell_connectivity'];cell_type=np.array([5 if len(set(row.tolist()))==3 else 9 if len(set(row.tolist()))==4 else -1 for row in cells])
            scalars={'2DELEM_'+k.replace(' ','_'):v for k,v in raw['shell_scalars'].items()}
            tensors={'2DELEM_'+k.replace(' ','_'):v for k,v in raw['shell_tensors'].items()}
            tm=raw['time'];point_disp=raw['displacement']
            cell_status=raw['shell_active'].astype(np.uint8)
        else:
            with vtk_path.open('wb') as out:
                conversion=subprocess.run([str(a.converter.resolve()),str(path.resolve())],stdout=out,stderr=subprocess.PIPE,env=env)
            if conversion.returncode: raise RuntimeError(conversion.stderr.decode(errors='replace'))
            r=vtkUnstructuredGridReader(); r.SetFileName(str(vtk_path)); r.ReadAllScalarsOn();r.ReadAllVectorsOn();r.ReadAllTensorsOn();r.ReadAllFieldsOn();r.Update()
            mesh=r.GetOutput(); pd=mesh.GetPointData();cd=mesh.GetCellData()
            frame_ids=array(pd,'NODE_ID').reshape(-1).astype(np.int64)
            element_ids=array(cd,'ELEMENT_ID').reshape(-1).astype(np.int64)
            part_ids=array(cd,'PART_ID').reshape(-1).astype(np.int64)
            cells=[[mesh.GetCell(i).GetPointId(k) for k in range(mesh.GetCell(i).GetNumberOfPoints())] for i in range(mesh.GetNumberOfCells())]
            cell_type=np.array([mesh.GetCellType(i) for i in range(mesh.GetNumberOfCells())])
            scalars={cd.GetArrayName(i):vtk_to_numpy(cd.GetArray(i)) for i in range(cd.GetNumberOfArrays()) if cd.GetArray(i).GetNumberOfComponents()==1}
            tensors={cd.GetArrayName(i):vtk_to_numpy(cd.GetArray(i)).reshape(-1,3,3)[:,[0,1,0],[0,1,1]] for i in range(cd.GetNumberOfArrays()) if cd.GetArray(i).GetNumberOfComponents()==9}
            tm=float(mesh.GetFieldData().GetArray('TIME').GetTuple1(0));point_disp=array(pd,'Displacement')
            cell_status=array(cd,'EROSION_STATUS').reshape(-1).astype(np.uint8)
        node_lookup=unique_lookup(frame_ids,'ANIM node')
        ni=np.array([node_lookup[int(v)] for v in node_ids])
        # Element IDs are reused across element families (e.g. spring/shell).
        # Key only real triangle/quad cells by both part and shell element ID.
        element_lookup={}; required_shell_keys=set(zip(shells[:,0].tolist(),shell_parts.tolist()))
        for i,(e,pid) in enumerate(zip(element_ids,part_ids)):
            if cell_type[i] not in (5,9): continue
            key=(int(e),int(pid))
            if key not in required_shell_keys: continue
            if key in element_lookup: raise ValueError('Duplicate shell ELEMENT_ID/PART_ID')
            element_lookup[key]=i
        ei=np.array([element_lookup[(int(v),int(pid))] for v,pid in zip(shells[:,0],shell_parts)])
        assert np.all(np.isin(cell_type[ei],[5,9])),'Unsupported real-shell cell corner count'
        assert np.array_equal(part_ids[ei].astype(np.int64),shell_parts)
        status[t]=cell_status[ei]
        # Prove each real element is the same connectivity, keyed by NODE_ID.
        for e,i in enumerate(ei):
            actual={int(frame_ids[k]) for k in cells[int(i)]}
            assert actual==set(shells[e,1:].tolist()), 'Element connectivity mismatch'
            actual_order=list(dict.fromkeys(int(frame_ids[k]) for k in cells[int(i)]))
            expected_order=list(dict.fromkeys(shells[e,1:].tolist()))
            assert actual_order==expected_order,'Ordered corner mismatch: element local frame may differ'
        all_times.append(tm)
        displacement=point_disp[ni].astype(np.float64)
        targets[:,t,:3]=displacement
        for j,location in enumerate(LOCATIONS):
            tensor_name='2DELEM_Stress_('+location+')' if j<2 else '2DELEM_Stress_(layer__'+str(j-1)+')_'
            stress[t,:,j,:]=tensors[tensor_name][ei]
            ep_name='2DELEM_Plastic_Strain_'+location.capitalize() if j<2 else '2DELEM_Plast_Strn_Layer___'+str(j-1)+'_'
            epsp[t,:,j]=scalars[ep_name].reshape(-1)[ei]
            if strain is not None:
                st_name=tensor_name.replace('Stress','Strain')
                strain[t,:,j,:]=tensors[st_name][ei]
                if j<2:
                    xx_s,yy_s,xy_s=np.asarray(strain[t,:,j,:],dtype=np.float64).T
                    mean_s=(xx_s+yy_s)/2
                    radius_s=np.sqrt(((xx_s-yy_s)/2)**2+xy_s**2)
                    for k,scalar_s in enumerate((mean_s+radius_s,mean_s-radius_s)):
                        principal[:,t,2*j+k]=np.bincount(dst,weights=weights*scalar_s[src],minlength=N)/denominator
            if j<2:
                xx,yy,xy=stress[t,:,j,:].T
                vm=np.sqrt(np.maximum(xx*xx-xx*yy+yy*yy+3*xy*xy,0))
                for channel,scalar in ((3+j,vm),(5+j,np.asarray(epsp[t,:,j],dtype=np.float64))):
                    targets[:,t,channel]=np.bincount(dst,weights=weights*scalar[src],minlength=N)/denominator
        assert np.isfinite(targets[:,t]).all()
        if solver_vm is not None: solver_vm[t]=scalars['2DELEM_Von_Mises'].reshape(-1)[ei]
        nominal=float((int(path.name[-3:])-1)*10)
        delta=displacement-truth[nominal]
        checks.append({'anim':path.name,'actual_time_ms':tm,'nominal_time_ms':nominal,'time_delta_ms':tm-nominal,
                       'shell_status_zero_count':int(np.sum(status[t]==0)),
                       'source_anim_vs_curated_displacement_component_rmse_mm':float(np.sqrt(np.mean(delta*delta))),
                       'max_abs_projected_value_per_channel':np.max(abs(targets[:,t]),axis=0).tolist(),
                       'conversion_extraction_seconds':time.perf_counter()-frame_start})
        sources.append({'name':path.name,'bytes':path.stat().st_size,'sha256':digest(path)})
        # No persistent VTK duplication of the ANIM cache.
        if a.backend=='vtk': del mesh,r,pd,cd;vtk_path.unlink()
        else: del raw
        del scalars,tensors;gc.collect()
    targets.flush();stress.flush();epsp.flush();status.flush()
    if strain is not None: strain.flush()
    if principal is not None: principal.flush();solver_vm.flush()
    assert np.all(np.diff(all_times)>0)
    np.save(a.output_dir/'time_ms.npy',np.asarray(all_times,dtype=np.float64))
    nominal_times=np.array([(int(x.name[-3:])-1)*10 for x in frames],dtype=np.float64)
    np.save(a.output_dir/'nominal_time_ms.npy',nominal_times)
    report={'status':'complete' if T==11 else 'partial_pilot_complete','schema_version':2,'run':a.run_name,'dataset_revision':'78f1302fba4d75926c6fdd22d01d5133efc14742',
      'provenance':a.provenance,'extractor_sha256':digest(Path(__file__)),
      'direct_reader_sha256':digest(Path(__file__).with_name('direct-anim-reader.py')) if a.backend=='direct' else None,
      'upstream_reader_commit':direct.SOURCE_COMMIT if a.backend=='direct' else None,
      'converter_sha256':digest(a.converter) if a.backend=='vtk' else None,
      'license':'Altair Engineering Inc. 2022 model: CC BY-NC 4.0; derived FEM fields retain attribution/noncommercial constraints.',
      'source_files':sources,'curated_vtp_sha256':digest(reference[0]),'rad_mapping_sha256':digest(a.rad),
      'reader_backend':a.backend,'reader_precision':'binary float32' if a.backend=='direct' else 'official converter six-significant-digit ASCII',
      'nodes':N,'real_shell_elements':E,'frames':T,'tensor_locations':LOCATIONS,'tensor_components':['local_xx','local_yy','local_xy'],
      'nodal_target_shape':[N,T,7],'nodal_target_channels':CHANNELS,'nodal_target_units':['mm']*3+['GPa']*2+['dimensionless']*2,
      'canonical_stress_shape':[T,E,7,3],'canonical_stress_unit':'GPa','canonical_epsp_shape':[T,E,7],
      'total_strain_available':strain is not None,'total_strain_unit':'dimensionless','total_strain_finite_measure':'not yet established; raw solver total strain, distinct from EPSP and displacement-gradient strain',
      'nodal_total_strain_principal_shape':[N,T,4] if strain is not None else None,
      'nodal_total_strain_principal_channels':['upper_inplane_major','upper_inplane_minor','lower_inplane_major','lower_inplane_minor'] if strain is not None else None,
      'principal_strain_scope':'Eigenvalues of the 2x2 in-plane FEM total-strain tensor, scalar area-weighted nodal projection; not full 3D principal strain.' if strain is not None else None,
      'solver_von_mises_scope':'Raw /ANIM/SHELL/VONM scalar, default shell aggregation; not used as upper/lower labels.' if strain is not None else None,
      'tensor_coordinate_system':'element local; no rotation to global; only in-plane xx,yy,xy stored by converter',
      'shear_convention':'strain xy is tensor shear epsilon_xy=engineering_gamma_xy/2; stress xy unchanged',
      'projection':'For each real shell: initial geometric area divided equally among unique corners; area-weighted scalar invariant/EPSP average over adjacent shells. No local tensor component averaging.',
      'projection_limits':'Nodal scalar averaging smooths element discontinuities and hotspots, including shared part boundaries; canonical cell fields preserved.',
      'shell_status_policy':'Raw ANIM status (official VTK EROSION_STATUS), uint8 [T,E], is preserved. Fixed-initial-topology scalar projection includes every real shell, including status 0; it does not remove stale/eroded output or claim damage validity. Per-frame status-zero counts are reported.',
      'inplane_vm_formula':'sqrt(xx^2-xx*yy+yy^2+3*xy^2), from actual FEM shell in-plane local stress; not an independent solver VONM field',
      'upper_lower_semantics':'Stress/EPSP upper/lower outermost integration points; total strain upper/lower extrapolated outer shell surfaces; layer_1..5 also preserved.',
      'isolated_positive_thickness_nodes':int(np.sum(denominator==0)),'projection_constant_test_max_abs_error':float(np.max(abs(constant-1))),
      'mapping_max_initial_RAD_to_curated_distance_mm':float(distances[:,0].max()),'minimum_second_candidate_distance_mm':float(distances[:,1].min()),
      'node_id_sha256':hashlib.sha256(node_ids.astype('<i8').tobytes()).hexdigest(),
      'curated_node_indices_sha256':hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest(),
      'frame_checks':checks,'elapsed_seconds':time.perf_counter()-started,
      'python_peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
      'not_claimed':['full 3D stress tensor','total strain for original 131-case dataset','global tensor component prediction','nodal hotspot conservation']}
    report['output_files']=[{'name':f.name,'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(a.output_dir.glob('*.npy'))]
    report['commit_policy']='Write arrays in per-process staging directory; close mmap handles; atomic-replace each NPY; commit complete manifest last. Readers must require globally frozen readiness before training.'
    for mapped in (targets,stress,epsp,status,strain,principal,solver_vm):
        if mapped is not None: mapped._mmap.close()
    del targets,stress,epsp,status,strain,principal,solver_vm
    gc.collect()
    for f in sorted(a.output_dir.glob('*.npy')):
        f.replace(destination/f.name)
    write_manifest(destination/'manifest.json',report)
    a.output_dir.rmdir()
    print(json.dumps({'run':a.run_name,'status':report['status'],'N':N,'T':T,'E':E,'seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__': main()
