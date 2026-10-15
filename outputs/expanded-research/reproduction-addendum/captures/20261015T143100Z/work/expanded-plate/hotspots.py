import pathlib,numpy as np,json
p=pathlib.Path('outputs/expanded-research/benchmarks/deforming_plate');rows=[]
for label in ['pilot8','noise0_pilot8']:
 for i in range(2):
  name=f'test_{i:04d}'
  with np.load(p/label/f'{name}_evaluation.npz') as a:
   ids=np.flatnonzero(a['free_mask']);truth=a['truth_stress'][:,ids,0];pred=a['prediction_stress'][:,ids,0]
   t,j=np.unravel_index(np.argmax(truth),truth.shape)
   rows.append({'run':label,'development_trajectory':name,'source_loading_step':int(t+1),'array_node_index_not_solver_id':int(ids[j]),'FEM_von_mises_at_peak':float(truth[t,j]),'AI_von_mises_at_same_point_and_step':float(pred[t,j]),'same_point_relative_error':float(abs(pred[t,j]-truth[t,j])/truth[t,j]),'units':'Unconfirmed dataset stress units','predicted_maximum_over_development_trajectory':float(pred.max())})
(p/'development-stress-hotspots.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print(json.dumps(rows,indent=2))
