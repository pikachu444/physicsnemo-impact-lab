"""Read produced VTUs with independent meshio5.3.5, then compare frozen data."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
here=Path(__file__).resolve().parent;out=here.parent;root=out.parents[3]
sys.path.insert(0,str(root/'work/expanded-drop/vtu-validation-deps'))
import meshio
cache=out/'doe/canonical'
manifest=json.loads((cache/'manifest-full-assembly-corners.json').read_text())
ref=np.load(cache/'shared/coordinates_full.npy').astype(np.float64)
ids=np.load(cache/'shared/node_ids_full.npy')
raw=np.load(cache/'shared/solid_connectivity_8.npy')
reports=[]
for run in [row for row in manifest['runs'] if row['split']=='test']:
    file=here/f"{run['run_id']}_t20us_FEM.vtu"
    mesh=meshio.read(file)
    target=np.load(cache/run['target'],mmap_mode='r')[:,9,:]
    assert np.array_equal(mesh.points,ref+target[:,:3].astype(np.float64))
    pd=mesh.point_data
    assert np.array_equal(pd['NODE_ID'].ravel(),ids)
    assert np.array_equal(pd['ReferenceCoordinates_mm'],ref)
    for name,values in [('FEM_U_mm',target[:,:3]),('FEM_GPS_VM_MPa',target[:,3]),('FEM_TOTAL_STRAIN_FROB',target[:,4]),('FEM_EPSP',target[:,5])]:
        assert np.array_equal(pd[name].reshape(values.shape),values),name
    assert not any(name.startswith(('AI_','ERROR_','ABS_ERROR')) for name in pd)
    indexes=np.concatenate(mesh.cell_data['ANIM_SOLID_INDEX']).ravel()
    assert np.array_equal(indexes,np.arange(len(raw)))
    total=0;counts={}
    for block,cell_indices,parts in zip(mesh.cells,mesh.cell_data['ANIM_SOLID_INDEX'],mesh.cell_data['PART_ID']):
        cell_indices=cell_indices.ravel()
        assert block.type in ('tetra','hexahedron','wedge')
        assert block.data.shape[1]=={'tetra':4,'hexahedron':8,'wedge':6}[block.type]
        assert np.all(np.diff(np.sort(block.data,axis=1),axis=1)>0)
        # Set comparison against native8slot connectivity, without importing writer.
        for displayed,native in zip(block.data,raw[cell_indices]):
            assert set(displayed)==set(native)
        assert np.all(pd['PART_ID'].ravel()[block.data]==parts.reshape(-1,1))
        counts[block.type]=counts.get(block.type,0)+len(block.data);total+=len(block.data)
    assert counts=={'tetra':109600,'hexahedron':20150,'wedge':468} and total==130218
    assert float(mesh.field_data['TimeValue'].ravel()[0])==2e-5
    assert float(mesh.field_data['GeometryDeformationScale'].ravel()[0])==1
    reports.append({'run_id':run['run_id'],'read_success':True,'reader':f'meshio{meshio.__version__}',
        'points':len(mesh.points),'cells':total,'cell_type_counts':counts,'source_node_order_bitwise_equal':True,
        'all_FEM_point_fields_bitwise_equal_to_frozen_nominal20us_target':True,'coordinates_equal_reference_plus_FEM_u_scale1':True,
        'all_cell_corner_sets_equal_native':True,'point_and_cell_PART_ID_consistent':True,'AI_or_error_fields_present':False,
        'bytes':file.stat().st_size,'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
(here/'independent-reader-validation.json').write_text(json.dumps({'reader_version':meshio.__version__,'reports':reports,'limits':'Independent VTU XML parser validation; ParaView GUI was not opened. VTU uses standard VTKXML and cell types10/12/13.'},indent=2))
print(json.dumps(reports,indent=2))
