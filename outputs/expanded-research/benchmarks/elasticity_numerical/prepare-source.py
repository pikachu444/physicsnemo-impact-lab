"""Download only the previously inventoried, pinned official fp_ddm sources.

Standard library only. Never imports Torch, PhysicsNeMo, NumPy, TF or JAX.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
PIN='11685edb3ec1fb6f25665e96cf7a8c4d11a23f16'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=ROOT/'work/expanded-elasticity/source')
    a=p.parse_args()
    inventory=json.loads((ROOT/'outputs/expanded-research/source-matrix/source-blob-manifest.json').read_text())
    assert inventory['commit']==PIN
    selected=[x for x in inventory['files'] if x['path'].startswith('examples/tcad/fp_ddm/')]
    assert len(selected)==19 and sum(x['bytes'] for x in selected)<250000
    def obtain(row):
        path=a.destination/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/NVIDIA/physicsnemo/{PIN}/'+row['path']
        data=path.read_bytes() if path.exists() else urlopen(url,timeout=40).read()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert len(data)==row['bytes'] and blob==row['git_blob_sha'],row['path']
        if not path.exists():path.write_bytes(data)
        return {'path':row['path'],'url':url,'bytes':len(data),'git_blob_sha':blob,'sha256':hashlib.sha256(data).hexdigest()}
    with ThreadPoolExecutor(max_workers=2) as pool:files=list(pool.map(obtain,selected))
    record={'status':'pinned_official_source_verified','repository':'NVIDIA/physicsnemo','commit':PIN,
      'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'files':files,
      'execution_status':'static source download only; no framework imports, solver execution or package installation'}
    (OUT/'source-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'}))
if __name__=='__main__':main()
