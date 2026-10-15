import json,pathlib,numpy as np
p=pathlib.Path('outputs/expanded-research/benchmarks/deforming_plate');a=p/'pilot8';b=p/'pilot8_reuse';r={}
for i in range(2):
 name=f'test_{i:04d}'
 with np.load(a/f'{name}_evaluation.npz') as x,np.load(b/f'{name}_evaluation.npz') as y:
  r[name]={k:float(np.max(np.abs(x[k][:len(y[k])].astype(float)-y[k]))) for k in ['prediction_positions','prediction_stress']}
  assert r[name]['prediction_positions'] < 1e-6
  assert r[name]['prediction_stress'] < 1
r['scope']='New Python process and newly instantiated model, saved checkpoint loaded; first20ARsteps andteacherforcing in2heldouttrajectories; not a repeated 399-step accuracy run.'
(p/'checkpoint-reuse-check.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
