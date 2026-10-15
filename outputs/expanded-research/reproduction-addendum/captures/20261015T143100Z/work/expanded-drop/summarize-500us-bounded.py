"""CPU-only, mmap-based native FEM diagnostics; leaves frozen AI targets intact."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import csv
import gc
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import subprocess
from resource_gate import guard as resource_guard
def guard(stage):
    result=resource_guard(stage,minimum_available=768*1024**2)
    result['configured_minimum_available_bytes']=768*1024**2
    return result
guard('before_numpy')
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/expanded-research/benchmarks/drop_test'
CHUNK=8192
POINTS=('111','112','121','211')

class Frame:
    def __init__(self,path):
        self.path=Path(path)
        self.stream=self.path.open('rb')
        self.map=mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
        self.pos=0
        self.magic,self.time=struct.unpack_from('>if',self.map,0)
        assert self.magic==0x542c
        self.pos=8+81*3
        flags=self.ints(10)
        self.nnode,nface,npart,nscalar,nescalar,nvector,ntensor,nskew=self.ints(8)
        self.skip(nskew*12)
        self.coords=self.pos;self.skip(self.nnode*12)
        self.skip(nface*17+npart*54+self.nnode*6)
        self.names(nscalar+nescalar,81)
        self.skip((self.nnode*nscalar+nface*nescalar)*4)
        self.vector_names=self.names(nvector,81)
        self.vector_offsets={name:self.pos+i*self.nnode*12 for i,name in enumerate(self.vector_names)}
        self.skip(nvector*self.nnode*12+ntensor*81+nface*ntensor*12)
        if flags[0]:self.skip((nface+self.nnode)*4)
        self.node_ids=self.pos if flags[1] else None
        if flags[1]:self.skip((nface+self.nnode)*4)
        if flags[4]:self.skip(npart*12)
        assert flags[2]
        self.nelem,nsolidpart,nsolidscalar,nsolidtensor=self.ints(4)
        self.connectivity=self.pos;self.skip(self.nelem*32)
        self.active=self.pos;self.skip(self.nelem)
        self.ends=self.ints(nsolidpart)
        self.part_names=self.names(nsolidpart,50)
        scalar_names=self.names(nsolidscalar,81)
        self.scalars={name:self.pos+i*self.nelem*4 for i,name in enumerate(scalar_names)}
        self.skip(nsolidscalar*self.nelem*4)
        tensor_names=self.names(nsolidtensor,81)
        self.tensors={name:self.pos+i*self.nelem*24 for i,name in enumerate(tensor_names)}
        self.skip(nsolidtensor*self.nelem*24)
        assert self.pos<=len(self.map)
        assert self.nnode==257104 and self.nelem==130218 and len(self.ends)==17
        assert self.ends[-1]==self.nelem
    def skip(self,size):self.pos+=int(size)
    def ints(self,count):
        values=struct.unpack_from('>'+str(count)+'i',self.map,self.pos)
        self.skip(count*4)
        return values
    def names(self,count,width):
        result=[self.map[self.pos+i*width:self.pos+(i+1)*width].split(b'\0',1)[0].decode('ascii').replace(' ','_') for i in range(count)]
        self.skip(count*width)
        return result
    def array(self,offset,shape,dtype='>f4'):
        return np.ndarray(shape,dtype=dtype,buffer=self.map,offset=offset)
    def close(self):
        self.map.close();self.stream.close()

def parts_from_deck():
    result={};active='';body=[]
    for line in (ROOT/'work/expanded-drop/source/Cell_Phone_Drop_0000.rad').open():
        text=line.strip()
        if text.startswith('/'):
            active=text;body=[]
        elif text and not text.startswith('#') and active.startswith('/PART/'):
            body.append(text)
            if len(body)==2:
                prop,material,*_=map(int,text.split())
                result[int(active.split('/')[2])]={'name':body[0],'material':material,'property':prop}
    assert len(result)==17
    return result

PARTS=parts_from_deck()

def statistics(path):
    f=Frame(path)
    eps=f.array(f.scalars['Plastic_Strain'],(f.nelem,))
    vm=f.array(f.scalars['Von_Mises'],(f.nelem,))
    flags=f.array(f.active,(f.nelem,),'u1')
    assert np.isfinite(eps).all() and np.isfinite(vm).all()
    assert eps.min()>=0 and vm.min()>=0 and np.all(flags==1)
    row={'file':str(Path(path).relative_to(ROOT)),'actual_time_s':f.time,'solid_elements':f.nelem,
         'native_cell_vm_max_MPa':float(vm.max()),'native_cell_vm_min_MPa':float(vm.min()),
         'native_cell_epsp_max':float(eps.max()),'native_cell_epsp_min':float(eps.min()),
         'plastic_cells':int(np.count_nonzero(eps)),'all_solids_active':True,'valid_IP_strain':{}}
    start=0;support=[]
    for end,name in zip(f.ends,f.part_names):
        pid=int(name.lstrip('_').split(':')[0])
        count=int(np.count_nonzero(eps[start:end]))
        support.append({'part_id':pid,'name':PARTS[pid]['name'],'material':PARTS[pid]['material'],
                        'cells':end-start,'plastic_cells':count,'epsp_max':float(eps[start:end].max())})
        assert PARTS[pid]['material']==1 or count==0,('EPSP in elastic part',pid,count)
        start=end
    row['plastic_support']=support
    for point in POINTS:
        name=next(key for key in f.tensors if key.startswith('Stra_') and 'Point'+point in key)
        tensor=f.array(f.tensors[name],(f.nelem,6))
        mins=np.full(6,np.inf);maxs=np.full(6,-np.inf);normmax=0.;nonzero=0
        for start in range(0,f.nelem,CHUNK):
            value=tensor[start:start+CHUNK].astype(np.float64)
            assert np.isfinite(value).all()
            mins=np.minimum(mins,value.min(axis=0));maxs=np.maximum(maxs,value.max(axis=0))
            norm=np.sqrt(np.sum(value[:,:3]**2,axis=1)+2*np.sum(value[:,3:]**2,axis=1))
            normmax=max(normmax,float(norm.max()));nonzero+=int(np.count_nonzero(np.any(value!=0,axis=1)))
        row['valid_IP_strain'][point]={'min_6':mins.tolist(),'max_6':maxs.tolist(),'max_frobenius':normmax,'nonzero_cells':nonzero}
    row['total_strain_all_valid_IP_max_frobenius']=max(v['max_frobenius'] for v in row['valid_IP_strain'].values())
    row['total_strain_IP111_max_frobenius']=row['valid_IP_strain']['111']['max_frobenius']
    row['file_bytes']=Path(path).stat().st_size
    with Path(path).open('rb') as stream:row['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    del eps,vm,flags,tensor,value,norm
    f.close();gc.collect()
    row['resource']=guard('frame_statistics_complete')
    return row

def histories(runs,destination):
    converter=ROOT/'work/opencourant-experiment/solver/OpenCourant/exec/th_to_csv_win64.exe'
    scratch=ROOT/'work/expanded-drop/500us-postprocess-history'
    scratch.mkdir(exist_ok=True)
    paths=[BASE/'20us/Cell_Phone_DropT01.csv',BASE/'100us/Cell_Phone_DropT02.csv']
    for i,run in enumerate(runs):
        if run.name in ('20us-run','100us-run'):continue
        for source in sorted(run.glob('Cell_Phone_DropT[0-9][0-9]')):
            folder=scratch/f'{i:02d}-{run.name}';folder.mkdir(exist_ok=True)
            copy=folder/source.name;shutil.copy2(source,copy)
            result=subprocess.run([str(converter),copy.name],cwd=folder,capture_output=True,text=True,timeout=60)
            (destination/f'{run.name}-{source.name}-converter.log').write_text(result.stdout+'\n'+result.stderr)
            assert result.returncode==0
            paths.append(Path(str(copy)+'.csv'))
    merged={}
    for path in paths:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                merged[float(row['time'])]={key:float(value) for key,value in row.items() if key and value}
    rows=[merged[key] for key in sorted(merged)]
    keys=list(rows[0])
    with (destination/'global-history-0-to500us.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selfcheck',action='store_true')
    parser.add_argument('--partial',action='store_true')
    parser.add_argument('--destination',type=Path,default=BASE/'500us-final')
    parser.add_argument('--run',action='append',type=Path)
    args=parser.parse_args()
    destination=args.destination;destination.mkdir(exist_ok=True,parents=True)
    if args.selfcheck:
        check=statistics(ROOT/'work/expanded-drop/100us-run/Cell_Phone_DropA021')
        expected=json.loads((BASE/'100us/summary.json').read_text())['peak_observed_cell_vm_frame']
        assert check['native_cell_vm_max_MPa']==expected['von_mises_max_MPa']
        assert check['native_cell_epsp_max']==expected['epsp_max']
        assert check['plastic_cells']==expected['epsp_nonzero_elements']
        assert np.isclose(check['total_strain_IP111_max_frobenius'],expected['strain_111_max_frobenius'],rtol=1e-12)
        (destination/'bounded-reader-selfcheck.json').write_text(json.dumps(check,indent=2))
        print(json.dumps({'status':'selfcheck_passed','resource':guard('selfcheck_passed')}));return
    runs=args.run or [ROOT/'work/expanded-drop'/name for name in ['20us-run','100us-run','500us-run','500us-resume03-run']]
    if not args.partial:
        stdout=(runs[-1]/'engine.stdout.log').read_text()
        assert 'NORMAL TERMINATION' in stdout and 'USER BREAK' not in stdout
    frames={}
    for run in runs:
        for path in sorted(run.glob('Cell_Phone_DropA[0-9][0-9][0-9]')):
            row=statistics(path);frames[row['actual_time_s']]=row
    stats=[frames[key] for key in sorted(frames)]
    assert args.partial or stats[-1]['actual_time_s']>=4.9999e-4
    previous=None;monotone=[]
    for row in stats:
        f=Frame(ROOT/row['file'])
        current=f.array(f.scalars['Plastic_Strain'],(f.nelem,)).copy()
        if previous is not None:
            decrement=previous-current
            monotone.append({'actual_time_s':row['actual_time_s'],'decreased_cells':int(np.count_nonzero(decrement>0)),
                             'max_decrement':float(max(0,decrement.max()))})
        previous=current;f.close()
    assert all(row['decreased_cells']==0 for row in monotone),monotone
    old=Frame(ROOT/'work/expanded-drop/20us-run/Cell_Phone_DropA001')
    final=Frame(ROOT/stats[-1]['file'])
    topology={'node_ids_bitwise_equal_original':bool(np.array_equal(old.array(old.node_ids,(old.nnode,),'>i4'),final.array(final.node_ids,(final.nnode,),'>i4'))),
              'solid_connectivity_bitwise_equal_original':bool(np.array_equal(old.array(old.connectivity,(old.nelem,8),'>i4'),final.array(final.connectivity,(final.nelem,8),'>i4')))}
    assert all(topology.values())
    normal=np.asarray(json.loads((BASE/'contact-time-estimate.json').read_text())['plane_unit_normal'])
    corner_normal=final.array(final.vector_offsets['Velocity'],(final.nnode,3))[:76065]@normal
    local_velocity={'support':'76065 physical solid corner nodes, excludes midsides and wall display','unit':'mm/s',
                    'min':float(corner_normal.min()),'max':float(corner_normal.max()),
                    'outward_positive_nodes':int(np.count_nonzero(corner_normal>0)),
                    'interpretation':'Local corner motion is distinct from global COM rebound.'}
    old.close();final.close();del previous,current,corner_normal
    rows=histories(runs,destination)
    normal=np.asarray(json.loads((BASE/'contact-time-estimate.json').read_text())['plane_unit_normal'])
    time=np.array([r['time'] for r in rows])
    velocity=np.array([[r[k] for k in ['X-MOMENTUM','Y-MOMENTUM','Z-MOMENTUM']] for r in rows])@normal/np.array([r['MASS']*1000 for r in rows])
    accounted=np.array([sum(r[k] for k in ['KINETIC ENERGY','INTERNAL ENERGY','ROTATION ENERGY','SPRING ENERGY','CONTACT ENERGY','HOURGLASS ENERGY'])-r['EXTERNAL WORK'] for r in rows])*.001
    outward=np.flatnonzero(velocity>0)
    peak=max(stats,key=lambda r:r['native_cell_vm_max_MPa'])
    summary={'completed_to_requested_time':not args.partial,'requested_final_time_s':.0005,'last_actual_frame_s':stats[-1]['actual_time_s'],
      'last_history_time_s':float(time[-1]),'saved_frames':len(stats),'last_frame':stats[-1],
      'observed_maximum_cell_vm_frame':peak,'observed_peak_is_last_saved_frame':peak==stats[-1],
      'observed_maximum_EPSP':max(r['native_cell_epsp_max'] for r in stats),
      'observed_maximum_total_strain_all_valid_IP_frobenius':max(r['total_strain_all_valid_IP_max_frobenius'] for r in stats),
      'history_initial_normal_COM_velocity_m_per_s':float(velocity[0]),'history_last_normal_COM_velocity_m_per_s':float(velocity[-1]),
      'source_initial_velocity_mm_per_s':[-3200.900476,-3511.241739,-2618.285239],
      'initial_momentum_note':'Actual saved global P/M differs slightly from nominal INIVEL speed. The original20us ANIM did not store t0 nodal velocity, and the cause was not independently isolated. Rebound uses actual P/M sign history.',
      'any_outward_normal_COM_velocity':bool(len(outward)),'first_saved_outward_normal_COM_time_s':float(time[outward[0]]) if len(outward) else None,
      'last_kinetic_energy_J':rows[-1]['KINETIC ENERGY']*.001,'last_internal_energy_J':rows[-1]['INTERNAL ENERGY']*.001,
      'last_plastic_work_J':rows[-1]['PLASTIC WORK']*.001,'last_external_work_J':rows[-1]['EXTERNAL WORK']*.001,
      'initial_accounted_energy_J':float(accounted[0]),'last_accounted_energy_J':float(accounted[-1]),
      'accounted_energy_drift_percent':float(100*(accounted[-1]/accounted[0]-1)),
      'last_added_mass_percent':rows[-1]['PERCENTAGE ADDED MASS'],
      'all_saved_cell_EPSP_monotone':True,'topology_checks':topology,'last_corner_normal_velocity':local_velocity,
      'accounted_energy_definition':'KE+IE+rotation+spring+contact+hourglass-external work; plastic work is already in IE.',
      'units':{'source':'Mg-mm-s','native_stress':'MPa','native_strain':'dimensionless tensor strain, shear not engineering gamma','history_energy_to_J':.001,'history_mass_to_kg':1000,'momentum_to_kg_m_per_s':1},
      'strain_support':'All130218 active solids at IP111; TETRA10 additionally IP112/121/211; nonphysical BRICK extra IP zero outputs excluded from interpretation.',
      'plasticity_check':'All nonzero native cell EPSP confined to original material1 PLAS_TAB. The four other materials are elastic.',
      'AI_scope':'Frozen nine-condition2..20us labels unchanged. Single-center0.5ms FEM diagnostic only.',
      'model_scope':'Public CC BY-NC4.0 phone benchmark; not a commercial production model or a validated consumer product.',
      'limits':'Snapshot maximum is not guaranteed continuous-time peak. COM sign reversal is a saved global momentum observation; no separation/contact-force criterion or product-failure validation.',
      'resource':guard('statistics_complete')}
    (destination/'frame-field-statistics.json').write_text(json.dumps(stats,indent=2))
    (destination/'EPSP-monotonicity-check.json').write_text(json.dumps(monotone,indent=2))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2))
    with (destination/'field-trends.csv').open('w',newline='') as stream:
        keys=['actual_time_s','native_cell_vm_max_MPa','native_cell_epsp_max','plastic_cells','total_strain_all_valid_IP_max_frobenius','total_strain_IP111_max_frobenius']
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in stats)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('last_frame','observed_maximum_cell_vm_frame')},indent=2))

if __name__=='__main__':main()
