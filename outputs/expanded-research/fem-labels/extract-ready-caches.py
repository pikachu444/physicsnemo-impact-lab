"""Consume SHA-verified per-run ANIM downloads sequentially with low RAM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT/'work/expanded-fem-labels/source-family')
    p.add_argument('--cache-root',type=Path,default=ROOT/'work/expanded-fem-labels/cache')
    p.add_argument('--expected-runs',type=int,default=131)
    p.add_argument('--max-seconds',type=float,default=7200)
    p.add_argument('--once',action='store_true')
    a=p.parse_args();start=time.perf_counter();complete=set();failed=set()
    extractor=Path(__file__).with_name('extract-anim-fields.py')
    extractor_hash=hashlib.sha256(extractor.read_bytes()).hexdigest()
    a.cache_root.mkdir(parents=True,exist_ok=True)
    while len(complete)<a.expected_runs and time.perf_counter()-start<a.max_seconds:
        progressed=False
        for receipt in sorted(a.source_root.glob('RAW_DATA/*/Run*/source-manifest.json')):
            name=receipt.parent.name
            if name in complete or name in failed: continue
            source=json.loads(receipt.read_text())
            if source.get('status')!='all_sources_size_and_sha256_verified': continue
            output=a.cache_root/name;manifest=output/'manifest.json'
            if manifest.exists():
                cached=json.loads(manifest.read_text())
                required=['nodal_targets.npy','node_id.npy','coords.npy','thickness.npy','time_ms.npy','shell_stress_local_inplane.npy','shell_epsp.npy','shell_animation_status.npy']
                recorded={x['name']:(x['bytes'],x['sha256']) for x in cached.get('source_files',[])}
                expected={Path(x['path']).name:(x['bytes'],x['sha256']) for x in source.get('files',[])}
                arrays_ok=all((output/x).is_file() for x in required)
                if arrays_ok:
                    arrays_ok=np.load(output/'nodal_targets.npy',mmap_mode='r').shape==(13675,11,7) and np.load(output/'shell_animation_status.npy',mmap_mode='r').shape==(11,13626)
                if cached.get('status')=='complete' and cached.get('run')==name and cached.get('nodal_target_shape')==[13675,11,7] and cached.get('reader_backend')=='direct' and cached.get('schema_version')==2 and arrays_ok and recorded==expected and cached.get('extractor_sha256')==extractor_hash and cached.get('output_files') and cached.get('commit_policy'):
                    complete.add(name);continue
            output.mkdir(parents=True,exist_ok=True)
            with (output/'extract.stdout.log').open('w') as out,(output/'extract.stderr.log').open('w') as err:
                result=subprocess.run([sys.executable,str(extractor),'--anim-dir',str(receipt.parent),'--run-name',name,'--output-dir',str(output)],stdout=out,stderr=err)
            if result.returncode: failed.add(name)
            else: complete.add(name)
            progressed=True
            print(json.dumps({'last_run':name,'exit_code':result.returncode,'complete_cases':len(complete),'failed_cases':sorted(failed),'elapsed_seconds':time.perf_counter()-start}),flush=True)
            (a.cache_root/'extraction-progress.json').write_text(json.dumps({'complete':sorted(complete),'failed':sorted(failed),'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')
        if a.once: break
        if not progressed: time.sleep(2)
    if len(complete)!=a.expected_runs:
        print(json.dumps({'incomplete':True,'complete_cases':len(complete),'expected_cases':a.expected_runs,'failed':sorted(failed)}),flush=True)
        return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
