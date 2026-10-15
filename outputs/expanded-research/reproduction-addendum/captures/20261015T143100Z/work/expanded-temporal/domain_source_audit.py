"""Source-only initial-state audit for future structural temporal runs."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def card_fields(path,names):
    active=''; selected={name:[] for name in names}
    for line in path.open():
        text=line.strip()
        if text.startswith('/'):
            active=text
        elif active in selected and text and not text.startswith('#'):
            selected[active].append(text.split())
    return selected
rad=ROOT/'work/bumper-fem-rerun/Bumper_Beam_AP_meshed_0000.rad'
cards=card_fields(rad,['/INIVEL/TRA/1','/GRNOD/GRNOD/26','/GRNOD/NODE/25','/GRNOD/PART/24'])
velocity=next(row for row in cards['/INIVEL/TRA/1'] if len(row)==5)
v0=list(map(float,velocity[:3])); group,skew=map(int,velocity[3:])
assert v0==[-5.,0.,0.] and group==26 and skew==0
integer_rows=lambda rows:[int(v) for row in rows if all(v.lstrip('-').isdigit() for v in row) for v in row]
assert integer_rows(cards['/GRNOD/GRNOD/26'])==[25,24]
assert integer_rows(cards['/GRNOD/NODE/25'])==[14153]
parts=set(integer_rows(cards['/GRNOD/PART/24']))
assert parts==set([4,5,6,7,8,9,10,15,16,17,18,19,20,21])
nodes=set();active=''
for line in rad.open():
    text=line.strip()
    if text.startswith('/'):active=text
    elif text and not text.startswith('#') and active.startswith(('/SHELL/','/SH3N/')):
        if int(active.rsplit('/',1)[1]) in parts:
            fields=text.split()
            if fields and all(f.lstrip('-').isdigit() for f in fields): nodes.update(map(int,fields[1:]))
cacheids=set(np.load(ROOT/'work/expanded-fem-labels/cache/Run224/node_id.npy').tolist())
assert nodes==cacheids and len(nodes)==13675

phone=ROOT/'work/expanded-drop/source/Cell_Phone_Drop_0000.rad'
phonecards=card_fields(phone,['/INIVEL/TRA/1','/GRNOD/NODE/1'])
pvel=next(row for row in phonecards['/INIVEL/TRA/1'] if len(row)==5)
pv=list(map(float,pvel[:3])); pg,ps=map(int,pvel[3:])
assert pv==[-3200.900476,-3511.241739,-2618.285239] and pg==1 and ps==0
pgroup=set(integer_rows(phonecards['/GRNOD/NODE/1']))
pid=np.load(ROOT/'outputs/expanded-research/benchmarks/drop_test/doe/canonical/shared/node_ids_full.npy')
assert len(pgroup)==257099 and set(pid.tolist()).issubset(pgroup) and len(pid)==76065
report={'status':'Source initial velocity and support audited; new-domain framework probes/training not executed',
 'bumper':{'source_rad_sha256':sha(rad),'source_card':'/INIVEL/TRA/1','initial_velocity':v0,'velocity_unit':'mm/ms',
           'source_group':26,'group_skew':0,'nested_groups':[25,24],'group25_nodes':[14153],
           'group24_parts':sorted(parts),'shell_node_union':len(nodes),'cache_node_id_set_matches_source_group':True,
           'all_131_RAD_independently_audited':False,
           'family_initial_velocity_basis':'Direct Run224 source plus constant velocity_x=-5 in family global metadata; other130 RAD decks not independently inspected',
           'dt':10.,'time_unit':'ms','actual_time_jitter_max_ms':0.00098419189453125,
           'features':'per-node original shell thickness; globals thickness_scale,velocity_x,rwall_origin_y; no coupon BC masks',
           'targets':'TC7 original stress upper/lower and EPSP upper/lower; AR/TF3 positions only; no total strain ground-truth family',
           'frozen_manifest':'work/expanded-ml/multifield7-cache/manifest.json','frozen_manifest_sha256':sha(ROOT/'work/expanded-ml/multifield7-cache/manifest.json'),
           'data_shape':[13675,10,7],'split_counts':[112,12,7]},
 'phone_full':{'source_rad_sha256':sha(phone),'source_card':'/INIVEL/TRA/1','initial_velocity':pv,'velocity_unit':'mm/s',
               'source_group':1,'group_skew':0,'group_nodes':257099,'exported_support_nodes':76065,'all_exported_nodes_covered':True,
               't0_ANIM_velocity_available':False,'velocity_basis':'source card, not future FEM finite difference',
               'dt':2e-6,'time_unit':'s','nominal_horizon_s':20e-6,
               'temporal_labels':'All six canonical fields linearly interpolated in physical units to nominal2..20us within actual saved-frame brackets; source metadata records every bracket/weight, no extrapolation.',
               'geometry':'Fixed original RAD reference coordinates; wall normal rx varies, wall origin fixed',
               'features':'Source-derived reference mean incident corner-cell volume cube root (mm); no coupon fixed/grip masks. This scalar does not encode14 TYPE2 ties,wall contact,17 parts or material identities.',
               'scope':'All exported corner nodes of17 parts, not all257099 original solver nodes/degrees of freedom',
               'targets':'TC6 uXYZmm+GPSnodalVMMPa+native-totalstrainFrobenius+EPSP; AR/TF3 positions only',
               'data_freeze_status':'producer manifest-full-assembly-corners.json and dataset_READY_evidence.json now published; this source-only audit makes no GPU execution claim',
               'expected_manifest':'outputs/expanded-research/benchmarks/drop_test/doe/canonical/manifest-full-assembly-corners.json',
               'data_shape':[76065,10,6],'expected_split_counts':[5,2,2]},
 'new_driver_plan':{'coordinate_transform':'single train-geometry length scale and translation; kinematics normalized with same length scale, unlike arbitrary per-axis scale',
                    'loss_transform':'position error multiplied by geometry_length/train_displacement_common_std; each extra scalar uses the same train-only std as static models',
                    'kinematic_vector_statistics':'train-only streaming moments, source v0 vector, nominal dataset-specific dt; no full train stack',
                    'working_set':'mmap labels, one case transferred to GPU at once; stream training statistics and retain test only for final inference',
                    'strict_probes':'one real case forward/backward and one epoch before any long training; explicit root GPU allocation required',
                    'rough_not_measured_minutes_for_30_epochs':{'bumper_TC7':[45,65],'bumper_AR3':[60,100],'bumper_TF3':[60,100],
                                                               'phone_TC6':[15,30],'phone_AR3':[15,30],'phone_TF3':[15,30]},
                    'estimate_basis':'coupon3 actual275.33s plus earlier real13,675-node staticGeo step0.13~0.17s, node/case scaling; temporal memory/throughput must be measured',
                    'paid_resources_authorized':False}}
(OUT/'new-domains-source-and-resource-plan.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'status':'source_audit_pass','bumper_nodes':len(nodes),'phone_exported_nodes':len(pid),'new_domain_gpu_execution':False}))
