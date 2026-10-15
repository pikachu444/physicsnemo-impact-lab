"""Read-only coordinate/input collision audit of the frozen phone FEM cache.

CPU one thread. NumPy/SciPy/psutil only; no ML framework/GPU or source writes.
Exact equality means byte-identical stored coordinates. Near pairs use the
Euclidean distance in archival RAD float64 coordinates, in millimetres.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.spatial import cKDTree
import psutil

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
FIELDS=['u_x','u_y','u_z','von_mises_GPS','total_strain_frobenius','epsp']
UNITS=['mm','mm','mm','MPa','dimensionless','dimensionless']

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def exact_pairs(values):
    x=np.ascontiguousarray(values)
    keys=x.view('V'+str(x.dtype.itemsize*x.shape[1])).reshape(-1)
    _,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    groups=[np.flatnonzero(inverse==g) for g in np.flatnonzero(counts>1)]
    pairs=np.array([p for group in groups for p in itertools.combinations(group.tolist(),2)],dtype=np.int64).reshape(-1,2)
    return pairs,{'groups':len(groups),'involved_nodes':int(sum(len(g) for g in groups)),
      'unordered_pairs':len(pairs),'max_multiplicity':int(counts.max())}

def original_materials(path):
    parts={};materials={};active='';payload=[];nodes=ties=walls=0
    for line in path.open():
        text=line.strip()
        if text.startswith('/'):
            active=text;payload=[]
            ties+=int(text.startswith('/INTER/TYPE2/'))
            walls+=int(text.startswith('/RWALL/'))
        elif text and not text.startswith('#'):
            if active=='/NODE':nodes+=1
            elif active.startswith('/PART/'):
                payload.append(text)
                if len(payload)==2:
                    prop,mat,*_=map(int,payload[1].split())
                    parts[int(active.split('/')[2])]={'name':payload[0],'property_id':prop,'material_id':mat}
            elif active.startswith('/MAT/') and not payload:
                payload.append(text)
                materials[int(active.split('/')[3])]={'model':active.split('/')[2],'name':text}
    return parts,materials,{'original_node_count':nodes,'type2_tied_interfaces':ties,'rigid_wall_cards':walls}

def values_example(i,j,run,time_s,target,coords,node_ids,part,material,feature,parts,materials):
    pi,pj=int(part[i]),int(part[j]);mi,mj=int(material[i]),int(material[j])
    return {'run':run,'nominal_time_s':float(time_s),'node_indices':[int(i),int(j)],
      'node_ids':[int(node_ids[i]),int(node_ids[j])], 'part_ids':[pi,pj],
      'part_names':[parts[pi]['name'],parts[pj]['name']], 'material_ids':[mi,mj],
      'material_names':[materials[mi]['name'],materials[mj]['name']],
      'source_coordinates_mm':[coords[i].tolist(),coords[j].tolist()],
      'source_coordinate_distance_mm':float(np.linalg.norm(coords[i]-coords[j])),
      'static_feature_mm':[float(feature[i]),float(feature[j])],
      'full_C6_values':[target[i].tolist(),target[j].tolist()],
      'absolute_C6_difference':abs(target[i].astype(np.float64)-target[j].astype(np.float64)).tolist()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,default=ROOT/'outputs/expanded-research/benchmarks/drop_test/doe/canonical')
    p.add_argument('--rad',type=Path,default=ROOT/'work/expanded-drop/source/Cell_Phone_Drop_0000.rad')
    p.add_argument('--part-source',type=Path,default=ROOT/'outputs/expanded-research/benchmarks/drop_test/20us/phone-last-frame-truth.npz')
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--tolerance-mm',type=float,default=1e-6)
    a=p.parse_args();started=time.perf_counter();a.out.mkdir(parents=True,exist_ok=True)
    rp=a.cache/'dataset_READY_evidence.json';mp=a.cache/'manifest-full-assembly-corners.json'
    ready_bytes=rp.read_bytes();manifest_bytes=mp.read_bytes()
    ready=json.loads(ready_bytes);manifest=json.loads(manifest_bytes)
    ready['file_hashes']={k.replace('\\','/'):v for k,v in ready['file_hashes'].items()}
    assert ready['ready'] and ready['cases']==9 and manifest['ready']
    assert manifest['field_names']==FIELDS and manifest['shape_per_run']==[76065,10,6]
    shared=a.cache/'shared'
    required=['coordinates_full.npy','coordinates_source_float64_full.npy','node_ids_full.npy',
      'solid_connectivity_8.npy','static_feature_full.npy','connected_component_full.npy','edge_index_full.npy']
    evidence={}
    for name in required:
        path=shared/name;record=ready['file_hashes']['shared/'+name]
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
        evidence['shared/'+name]=record
    coords=np.load(shared/'coordinates_source_float64_full.npy')
    model_coords=np.load(shared/'coordinates_full.npy')
    node_ids=np.load(shared/'node_ids_full.npy')
    feature=np.load(shared/'static_feature_full.npy').reshape(-1)
    conn=np.load(shared/'solid_connectivity_8.npy',mmap_mode='r')
    components=np.load(shared/'connected_component_full.npy')
    assert coords.dtype==np.float64 and model_coords.dtype==np.float32
    assert coords.shape==model_coords.shape==(76065,3) and len(np.unique(node_ids))==76065
    assert np.array_equal(coords.astype(np.float32),model_coords)
    parts,materials,deck_counts=original_materials(a.rad)
    legacy=np.load(a.part_source,allow_pickle=False)
    ends=legacy['solid_part_end_indices'];names=legacy['solid_part_names'];legacy.close()
    assert len(ends)==17 and ends[-1]==130218
    part=np.full(len(coords),-1,np.int32);begin=0
    for end,name in zip(ends,names):
        pid=int(re.search(r'(\d+):',str(name))[1]);indices=np.unique(conn[begin:int(end)])
        assert np.all(part[indices]==-1),'A node belongs to more than one part'
        part[indices]=pid;begin=int(end)
    assert np.all(part>=0)
    material=np.array([parts[int(x)]['material_id'] for x in part],dtype=np.int32)
    assert len(np.unique(part))==17 and len(np.unique(material))==5
    pair64,summary64=exact_pairs(coords)
    pair32,summary32=exact_pairs(model_coords)
    full_features=np.column_stack([model_coords,feature]).astype(np.float32)
    full_pairs,summaryfull=exact_pairs(full_features)
    tree=cKDTree(coords)
    count=(int(tree.count_neighbors(tree,a.tolerance_mm))-len(coords))//2
    assert count<1000000,'Use streamed pair audit if multiplicities grow'
    near=tree.query_pairs(a.tolerance_mm,output_type='ndarray')
    assert len(near)==count
    categories={'source_float64_bitwise_exact':pair64,'model_float32_bitwise_exact':pair32,
                'source_float64_distance_within_1e-6_mm':near,'model_coords_and_static_float32_bitwise_exact':full_pairs}
    reports={}
    for name,pairs in categories.items():
        pdiff=part[pairs[:,0]]!=part[pairs[:,1]] if len(pairs) else np.array([],bool)
        mdiff=material[pairs[:,0]]!=material[pairs[:,1]] if len(pairs) else np.array([],bool)
        fdiff=abs(feature[pairs[:,0]].astype(np.float64)-feature[pairs[:,1]].astype(np.float64)) if len(pairs) else np.array([])
        group_counts={}
        for x,y in part[pairs]:
            key=str(min(int(x),int(y)))+'-'+str(max(int(x),int(y)))
            group_counts[key]=group_counts.get(key,0)+1
        reports[name]={'unordered_pairs':len(pairs),'different_node_id_pairs':int(np.sum(node_ids[pairs[:,0]]!=node_ids[pairs[:,1]])),
          'different_part_pairs':int(pdiff.sum()),'different_material_pairs':int(mdiff.sum()),'part_pair_counts':group_counts,
          'static_feature_abs_difference_mm':{'min':float(fdiff.min()),'median':float(np.median(fdiff)),'max':float(fdiff.max())} if len(pairs) else None,
          'target_differences':{'max_abs_per_channel':[0.]*6,'pairs_nonzero_in_any_case_time_per_channel':[0]*6,'max_examples_per_channel':[None]*6,
            'displacement_vector_max_mm':0.,'displacement_vector_max_example':None},
          'per_case':[]}
    pair_nonzero={k:np.zeros((len(v),6),bool) for k,v in categories.items()}
    case_evidence=[]
    for row in manifest['runs']:
        run=row['run_id'];path=a.cache/row['target'];record=ready['file_hashes'][row['target']]
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256']
        target=np.load(path,mmap_mode='r');assert target.shape==(76065,10,6) and np.isfinite(target).all()
        case_evidence.append({'run':run,'split':row['split'],'globals':row['globals'],'file':row['target'],'sha256':record['sha256']})
        for name,pairs in categories.items():
            if not len(pairs):continue
            left=target[pairs[:,0]].astype(np.float64);right=target[pairs[:,1]].astype(np.float64)
            diff=abs(left-right);nonzero=np.any(diff!=0,axis=1)
            pair_nonzero[name]|=nonzero
            td=reports[name]['target_differences']
            maximum=diff.max(axis=(0,1));count_nonzero=nonzero.sum(axis=0)
            reports[name]['per_case'].append({'run':run,'max_abs_per_channel':maximum.tolist(),'pairs_nonzero_per_channel':count_nonzero.tolist()})
            for c in range(6):
                if maximum[c]>td['max_abs_per_channel'][c]:
                    q,t=np.unravel_index(int(np.argmax(diff[:,:,c])),diff[:,:,c].shape)
                    i,j=map(int,pairs[q]);ex=values_example(i,j,run,manifest['time_values'][t],target[:,t,:],coords,node_ids,part,material,feature,parts,materials)
                    ex['channel']=FIELDS[c];ex['channel_unit']=UNITS[c];ex['max_abs_difference']=float(maximum[c])
                    td['max_examples_per_channel'][c]=ex;td['max_abs_per_channel'][c]=float(maximum[c])
            vector=np.linalg.norm(left[:,:,:3]-right[:,:,:3],axis=2)
            if float(vector.max())>td['displacement_vector_max_mm']:
                q,t=np.unravel_index(int(np.argmax(vector)),vector.shape);i,j=map(int,pairs[q])
                td['displacement_vector_max_mm']=float(vector[q,t]);td['displacement_vector_max_example']=values_example(i,j,run,manifest['time_values'][t],target[:,t,:],coords,node_ids,part,material,feature,parts,materials)
        del target
    for name in categories:
        reports[name]['target_differences']['pairs_nonzero_in_any_case_time_per_channel']=pair_nonzero[name].sum(axis=0).tolist()
        reports[name]['target_differences']['pairs_nonzero_in_any_channel_case_time']=int(np.any(pair_nonzero[name],axis=1).sum())
    edges=np.load(shared/'edge_index_full.npy',mmap_mode='r')
    assert edges.shape==(2,561184)
    cross_edges=int(np.sum(part[edges[0]]!=part[edges[1]]))
    component_parts=[{'component':int(c),'nodes':int(np.sum(components==c)),'part_ids':np.unique(part[components==c]).tolist()} for c in np.unique(components)]
    assert len(component_parts)==17 and all(len(x['part_ids'])==1 for x in component_parts)
    assert rp.read_bytes()==ready_bytes and mp.read_bytes()==manifest_bytes,'Frozen source changed during audit'
    forbidden=[x for x in sys.modules if x.split('.')[0] in ('torch','tensorflow','vtk','vtkmodules','jax','physicsnemo')]
    assert not forbidden
    record={'status':'read_only_input_audit_completed','scope':'Nine frozen FEM conditions, full linearized solid-corner support, 2..20us nominal labels. No prediction-error or model-impossibility verdict.',
      'coordinate_pair_definitions':{'exact':'Byte-identical stored coordinates, distinct NODE_IDs, unordered i<j pairs; float64 archival and float32 model representations audited separately.',
        'near':'Euclidean separation <= 1e-6 mm in original RAD float64 coordinates; inclusive of exact pairs.',
        'coords_plus_static':'Byte-identical concatenation of model float32 xyz and the single existing float32 static feature. Globals and nominal time are common to both nodes within each evaluated case/time.'},
      'source_ready_sha256':hashlib.sha256(ready_bytes).hexdigest(),'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
      'original_rad_sha256':sha(a.rad),'part_npz_sha256':sha(a.part_source),'shared_file_evidence':evidence,'case_target_evidence':case_evidence,
      'fields':FIELDS,'field_units':UNITS,'nominal_time_s':manifest['time_values'],'global_keys':manifest['global_keys'],
      'coordinate_equality_summaries':{'source_float64':summary64,'model_float32':summary32,'model_coords_plus_static_float32':summaryfull},
      'pair_results':reports,'parts':parts,'materials':materials,
      'node_coverage':{**deck_counts,'corner_nodes':76065,'excluded_nodes':181034,'excluded_are_TETRA10_midside_only':True,
         'coverage_evidence':'drop_test/active-flag-and-node-coverage-correction.json; 109600 TETRA10/20618 BRICK, original quadratic midside DOFs are not model-input support.'},
      'graph':{'directed_edges':561184,'connected_components':17,'cross_part_edges':cross_edges,'component_part_mapping':component_parts,
        'missing_coupling':'14 TYPE2 tied interfaces and wall-contact couplings exist in FEM but are not added as cross-part graph edges. Seventeen graph components do not mean the FEM parts are physically uncoupled.'},
      'static_feature':{'definition':'(sum of incident reference linearized-corner cell volumes / count of incident unique-corner cells)^(1/3)',
        'unit':'mm','is_thickness':False,'explicit_part_material_nodeID_channels':False,
        'scope':'A local mesh/geometry length proxy. Measured differences can distinguish current coincident pairs but it does not semantically encode part or material identity.'},
      'interpretation_limits':['Different FEM outputs at the same stored location are factual multi-part observations, not a neural-model prediction test.',
        'No identical xyz+existing-static-feature float32 pairs were found if the reported count is zero; xyz-only collisions must not be called full-input collisions.',
        'Near-equal coordinates within 1e-6 mm are not byte-identical inputs.',
        'C6 stress is GPS-derived nodal von Mises; total strain target is the Frobenius norm of a volume-projected tensor; EPSP is distinct. This audit does not compare all tensor components.',
        'Different node IDs/parts/materials and connected graph neighbourhoods may provide distinguishable structure when a model actually receives them; this audit does not assess model expressivity.',
        'No new topology, geometry, long-duration drop or damage prediction is validated.'],
      'execution':{'cpu_threads_environment':1,'framework_modules_imported':forbidden,'gpu_used':False,'elapsed_seconds':time.perf_counter()-started,
        'peak_working_set_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None)}}
    (a.out/'assembly-input-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    with (a.out/'coordinate-pairs.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['index_a','index_b','node_id_a','node_id_b','part_a','part_b','material_a','material_b','distance_mm','bitwise_equal_source_float64','bitwise_equal_model_float32','static_feature_a_mm','static_feature_b_mm'])
        exact64set={tuple(x) for x in pair64.tolist()};exact32set={tuple(x) for x in pair32.tolist()}
        for i,j in sorted(map(tuple,near.tolist())):
            writer.writerow([i,j,int(node_ids[i]),int(node_ids[j]),int(part[i]),int(part[j]),int(material[i]),int(material[j]),np.linalg.norm(coords[i]-coords[j]),(i,j) in exact64set,(i,j) in exact32set,float(feature[i]),float(feature[j])])
    print(json.dumps({'status':record['status'],'exact_pairs':len(pair64),'near_pairs':len(near),'exact_coords_plus_static_pairs':len(full_pairs),'seconds':record['execution']['elapsed_seconds'],'peak_working_set_bytes':record['execution']['peak_working_set_bytes']}))
if __name__=='__main__':main()
