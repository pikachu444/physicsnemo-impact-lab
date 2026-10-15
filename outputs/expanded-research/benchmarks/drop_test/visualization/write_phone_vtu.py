"""Export the frozen full-corner FEM support, with optional aligned AI arrays.

NumPy + standard library only. VTU is local data, excluded from the public package.
The default mesh is already deformed by FEM u at scale1; no second warp is needed.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
from xml.sax.saxutils import quoteattr
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
CACHE=OUT/'doe/canonical'
HEX_EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
WEDGE_EDGES=[(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)]
HEX_TETS=[(0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6)]
WEDGE_TETS=[(0,1,2,3),(1,4,2,3),(2,4,5,3)]

def signed_volume(points, cells, decompositions):
    values=np.zeros(len(cells),np.float64)
    for a,b,c,d in decompositions:
        p=points[cells[:,[a,b,c,d]]]
        values+=np.linalg.det(p[:,1:]-p[:,:1])/6
    return values

def geometry():
    shared=CACHE/'shared'
    points=np.load(shared/'coordinates_full.npy').astype(np.float64)
    node_ids=np.load(shared/'node_ids_full.npy')
    conn=np.load(shared/'solid_connectivity_8.npy')
    sorted_conn=np.sort(conn,axis=1)
    unique=np.ones(conn.shape,bool);unique[:,1:]=sorted_conn[:,1:]!=sorted_conn[:,:-1]
    counts=unique.sum(axis=1)
    tetra=counts==4;hexa=counts==8;wedge=counts==6
    assert [int(tetra.sum()),int(hexa.sum()),int(wedge.sum())]==[109600,20150,468]
    converted=conn.copy()
    tet=sorted_conn[tetra][unique[tetra]].reshape(-1,4)
    tet_volume=signed_volume(points,tet,[(0,1,2,3)])
    reverse=tet_volume<0
    tet[reverse]=tet[reverse][:,[0,2,1,3]]
    converted[tetra,:4]=tet
    assert np.all(signed_volume(points,tet,[(0,1,2,3)])>0)
    # All actual degenerate cells have repeated corners2=3 and6=7.
    raw_wedge=conn[wedge]
    assert np.all(raw_wedge[:,2]==raw_wedge[:,3]) and np.all(raw_wedge[:,6]==raw_wedge[:,7])
    wed=raw_wedge[:,[0,1,2,4,5,6]].copy()
    for old,new in zip(raw_wedge,wed):
        old_edges={tuple(sorted((int(old[a]),int(old[b])))) for a,b in HEX_EDGES if old[a]!=old[b]}
        new_edges={tuple(sorted((int(new[a]),int(new[b])))) for a,b in WEDGE_EDGES}
        assert len(old_edges)==9 and old_edges==new_edges
    wedge_volume=signed_volume(points,wed,WEDGE_TETS)
    reverse=wedge_volume<0
    wed[reverse]=wed[reverse][:,[0,2,1,3,5,4]]
    converted[wedge,:6]=wed
    assert np.all(signed_volume(points,wed,WEDGE_TETS)>0)
    hex_volume=signed_volume(points,conn[hexa],HEX_TETS)
    assert np.all(hex_volume>0), 'Unexpected HEX orientation or invalid corner cell'
    # Compare wedge volume with the source eight-slot HEX decomposition.
    native_wedge_volume=signed_volume(points,raw_wedge,HEX_TETS)
    wedge_relative_error=np.abs(np.abs(wedge_volume)-np.abs(native_wedge_volume))/np.abs(native_wedge_volume)
    # The two decompositions choose different diagonals on near-planar quads.
    # Stored float32 corners produce up to~4e-6 relative volume difference.
    assert wedge_relative_error.max()<1e-4
    keep=np.arange(8)[None,:]<counts[:,None]
    flat=converted[keep].astype(np.int32)
    offsets=np.cumsum(counts,dtype=np.int32)
    types=np.where(tetra,10,np.where(hexa,12,13)).astype(np.uint8)
    assert np.array_equal(np.unique(flat),np.arange(len(points)))
    legacy=np.load(OUT/'20us/phone-last-frame-truth.npz')
    part_ids=np.empty(len(conn),np.int32);begin=0
    for end,name in zip(legacy['solid_part_end_indices'],legacy['solid_part_names']):
        part_ids[begin:int(end)]=int(str(name).lstrip('_').split(':')[0]);begin=int(end)
    assert begin==len(conn)
    owners=np.repeat(part_ids,counts)
    point_part=np.zeros(len(points),np.int32)
    min_part=np.full(len(points),np.iinfo(np.int32).max,np.int32)
    np.maximum.at(point_part,flat,owners);np.minimum.at(min_part,flat,owners)
    assert np.array_equal(point_part,min_part), 'Shared node belongs to multiple physical parts; use a multi-label policy'
    audit={'nodes':len(points),'solid_cells':len(conn),'cell_type_counts':{'VTK_TETRA4_10':int(tetra.sum()),'VTK_HEXAHEDRON8_12':int(hexa.sum()),'VTK_WEDGE6_13':int(wedge.sum())},
        'wedge_conversion':'All468cells have native8slots [a,b,c,c,d,e,f,f]; mapped to6corners[a,b,c,d,e,f]. All9unique physical edges are preserved.',
        'max_relative_wedge_reference_volume_difference':float(wedge_relative_error.max()),
        'wedge_volume_note':'Node coordinates and all9edges preserved exactly. Alternate tetrahedral diagonals on nearly planar float32 quadrilateral faces leave a small nonzero volume difference; this is recorded, not claimed as bitwise geometric integration equality.',
        'tet_orientation':'Sorted corner IDs reoriented to positive VTK tetra volume; cell order and NODE_ID point order stay fixed.',
        'positive_reference_corner_volumes':True,'node_part_unique':True,
        'coverage':'76065physicalsolidcornernodes only;181034originalTETRA10midsidenodes excluded. No artificialwall display cells.',
        'linearization':'TETRA10 is shown as TETRA4 corners; original quadratic FEM formulation is not preserved in the visualization.',
        'VTK_type_reference':'https://vtk.org/doc/nightly/html/vtkCellType_8h.html',
        'VTU_XML_reference':'https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html'}
    return points,node_ids,converted,counts,flat,offsets,types,part_ids,point_part,audit

def data_array(stream,name,values,components=1):
    values=np.asarray(values)
    dtype={('f',4):'Float32',('f',8):'Float64',('i',4):'Int32',('i',8):'Int64',('u',1):'UInt8'}[(values.dtype.kind,values.dtype.itemsize)]
    values=np.ascontiguousarray(values,dtype=values.dtype.newbyteorder('<'))
    payload=values.tobytes()
    encoded=base64.b64encode(struct.pack('<Q',len(payload))+payload).decode('ascii')
    tuples=values.size//components
    stream.write(f'<DataArray type="{dtype}" Name={quoteattr(name)} NumberOfComponents="{components}" NumberOfTuples="{tuples}" format="binary">{encoded}</DataArray>\n')

def write_case(run_id, destination=None, prediction=None, prediction_node_ids=None):
    """Prediction is optional[N,6] on exact NODE_ID order; never synthesize AI values."""
    manifest=json.loads((CACHE/'manifest-full-assembly-corners.json').read_text())
    run=next(row for row in manifest['runs'] if row['run_id']==run_id)
    assert manifest['ready']
    target=np.load(CACHE/run['target'],mmap_mode='r')[:,9,:].copy()
    metadata=json.loads((CACHE/run['metadata']).read_text())
    ref,ids,converted,counts,flat,offsets,types,cell_parts,point_parts,audit=geometry()
    assert target.shape==(len(ref),6) and np.isfinite(target).all()
    deformed=ref+target[:,:3].astype(np.float64)
    deformed_volumes=np.empty(len(types),np.float64)
    for vtk_type,columns,decomposition in [(10,4,[(0,1,2,3)]),(12,8,HEX_TETS),(13,6,WEDGE_TETS)]:
        mask=types==vtk_type
        deformed_volumes[mask]=signed_volume(deformed,converted[mask,:columns],decomposition)
    assert np.all(deformed_volumes>0), 'Linearized displayed cell inverted'
    point_data={'NODE_ID':ids,'PART_ID':point_parts,'ReferenceCoordinates_mm':ref,'FEM_U_mm':target[:,:3],
        'FEM_GPS_VM_MPa':target[:,3],'FEM_TOTAL_STRAIN_FROB':target[:,4],'FEM_EPSP':target[:,5]}
    if prediction is not None:
        prediction=np.asarray(prediction,dtype=np.float32)
        assert prediction_node_ids is not None and np.array_equal(prediction_node_ids,ids), 'Prediction NODE_ID order must match exactly'
        assert prediction.shape==target.shape and np.isfinite(prediction).all()
        error=prediction-target
        for prefix,value in [('AI',prediction),('ERROR_AI_MINUS_FEM',error),('ABS_ERROR',np.abs(error))]:
            point_data.update({f'{prefix}_U_mm':value[:,:3],f'{prefix}_GPS_VM_MPa':value[:,3],f'{prefix}_TOTAL_STRAIN_FROB':value[:,4],f'{prefix}_EPSP':value[:,5]})
        point_data['ERROR_U_NORM_mm']=np.linalg.norm(error[:,:3],axis=1)
    destination=Path(destination) if destination else HERE/f'{run_id}_t20us_FEM.vtu'
    with destination.open('w',encoding='ascii',newline='\n') as stream:
        stream.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n<UnstructuredGrid>\n<FieldData>\n')
        data_array(stream,'TimeValue',np.array([2e-5],np.float64))
        data_array(stream,'NominalTime_s',np.array([2e-5],np.float64))
        data_array(stream,'GeometryDeformationScale',np.array([1.0],np.float64))
        stream.write(f'</FieldData>\n<Piece NumberOfPoints="{len(ref)}" NumberOfCells="{len(types)}">\n<PointData Scalars="FEM_GPS_VM_MPa" Vectors="FEM_U_mm">\n')
        for name,value in point_data.items():data_array(stream,name,value,3 if value.ndim==2 else 1)
        stream.write('</PointData>\n<CellData Scalars="PART_ID">\n')
        data_array(stream,'PART_ID',cell_parts)
        data_array(stream,'ANIM_SOLID_INDEX',np.arange(len(types),dtype=np.int32))
        data_array(stream,'LinearizedCornerCount',counts.astype(np.uint8))
        stream.write('</CellData>\n<Points>\n');data_array(stream,'DeformedCoordinates_mm',deformed,3)
        stream.write('</Points>\n<Cells>\n')
        for name,value in [('connectivity',flat),('offsets',offsets),('types',types)]:data_array(stream,name,value)
        stream.write('</Cells>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')
    audit.update(run_id=run_id,split=run['split'],globals=run['globals'],nominal_time_s=2e-5,time_bracket=metadata['time_brackets'][9],
        target_full_sha256=metadata['target_full_sha256'],point_order='Exact shared/node_ids_full.npy order;76065points unchanged.',
        geometry='ReferenceCoordinates_mm + FEM_U_mm, deformation scale1. Already deformed; do not warp by FEM_U a second time.',
        field_units={'FEM_U_mm':'mm','FEM_GPS_VM_MPa':'MPa','FEM_TOTAL_STRAIN_FROB':'1','FEM_EPSP':'1'},
        field_definitions='GPSVM from nodalGPS tensor. Strain validIPmean→corner-volume weighted globaltensor→Frob. EPSP corner-volume projected. These are canonical nodal targets, not nativecell/IP hotspot values.',
        AI_arrays_present=prediction is not None,all_displayed_corner_volumes_positive=True,
        VTU_bytes=destination.stat().st_size,VTU_sha256=hashlib.file_digest(destination.open('rb'),'sha256').hexdigest(),
        publication='VTU local-only. Publicpackage should containwriter,JSONhash/audit andPNGpreview; no VTU or rawmesh/data.')
    destination.with_suffix('.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'file':str(destination),'bytes':audit['VTU_bytes'],'sha256':audit['VTU_sha256'],'AI_arrays_present':audit['AI_arrays_present']},indent=2))
    return destination

if __name__=='__main__':
    manifest=json.loads((CACHE/'manifest-full-assembly-corners.json').read_text())
    tests=[row['run_id'] for row in manifest['runs'] if row['split']=='test']
    assert len(tests)==2
    for name in tests:write_case(name)
