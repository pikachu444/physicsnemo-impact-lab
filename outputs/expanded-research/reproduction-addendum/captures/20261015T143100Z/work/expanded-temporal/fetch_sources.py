"""Fetch unmodified pinned official temporal examples; stdlib only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, urllib.request, datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'work/expanded-temporal/sources'
PIN='11685edb3ec1fb6f25665e96cf7a8c4d11a23f16'
files=[]
for example in ('crash','drop_test'):
    files += [f'examples/structural_mechanics/{example}/{name}' for name in ('rollout.py','datapipe.py','train.py')]
    files += [f'examples/structural_mechanics/{example}/conf/model/geotransolver_{name}.yaml' for name in ('time_conditional','autoregressive_rollout_training','one_step_rollout')]
def fetch(path):
    url=f'https://raw.githubusercontent.com/NVIDIA/physicsnemo/{PIN}/{path}'
    data=urllib.request.urlopen(url,timeout=60).read()
    target=OUT/Path(path).relative_to('examples/structural_mechanics')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    return {'path':path,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'local':str(target.relative_to(ROOT))}
with ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(fetch,files))
report={'pin':PIN,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records}
dest=ROOT/'outputs/expanded-research/benchmarks/temporal_strategies/source-manifest.json'
dest.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'downloaded':len(records),'bytes':sum(r['bytes'] for r in records),'manifest':str(dest)}))
