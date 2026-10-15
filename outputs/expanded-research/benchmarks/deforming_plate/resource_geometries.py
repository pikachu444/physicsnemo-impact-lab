import sys,pathlib,json
sys.path.insert(0,'work/expanded-plate')
from plate_data import load_trajectory,world_edges,ROOT
p=pathlib.Path('outputs/expanded-research/benchmarks/deforming_plate')
rows=[]
for path in sorted((ROOT/'data').glob('train_*.npz')):
 r=load_trajectory(path)
 for t in [0,79,159,239,319,398]:
  w=world_edges(r['world_pos'][t],r['mesh_edges']).shape[1]
  rows.append({'trajectory':r['name'],'frame':t,'nodes':len(r['mesh_pos']),'mesh_edges':r['mesh_edges'].shape[1],'world_edges':w,'total_edges':r['mesh_edges'].shape[1]+w})
rows.sort(key=lambda r:r['total_edges'],reverse=True)
(p/'training-resource-geometries.json').write_text(json.dumps({'scope':'Training data only, six sampled loading states per trajectory, no fresh test access','top10':rows[:10],'samples':len(rows)},indent=2),encoding='utf8')
print(rows[:3])
