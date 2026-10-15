"""Copy execution logs for review, redacting only workspace path and hostname."""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'work/expanded-fem-labels/run-total-strain'
records=[]
for name in ('Bumper_Beam_AP_meshed_0000.out','Bumper_Beam_AP_meshed_0001.out','starter.stdout.log','starter.stderr.log','engine.stdout.log','engine.stderr.log'):
    raw=(SOURCE/name).read_bytes();text=raw.decode(errors='replace')
    text=text.replace(str(ROOT),'[WORKSPACE]').replace(str(ROOT).replace('\\','/'),'[WORKSPACE]')
    text=re.sub(r'(HOSTNAME\s+CPU TYPE, FREQUENCY AND MEMORY\s*\r?\n\s*)\S+',r'\1[HOST]',text)
    destination=OUT/('total-strain-'+name)
    destination.write_bytes(text.encode('utf-8'))
    records.append({'source_name':name,'source_sha256':hashlib.sha256(raw).hexdigest(),'public_name':destination.name,
       'public_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'redactions':['workspace absolute path','hostname']})
(OUT/'solver-log-provenance.json').write_text(json.dumps({'records':records,'numeric_solver_content_changed':False},indent=2)+'\n')
print('Preserved six solver logs with documented privacy redaction.')
