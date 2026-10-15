import urllib.request,json,pathlib,hashlib
p=pathlib.Path('work/expanded-plate'); o=pathlib.Path('outputs/expanded-research/benchmarks/deforming_plate'); rev='11685edb3ec1fb6f25665e96cf7a8c4d11a23f16'; base=f'https://raw.githubusercontent.com/NVIDIA/physicsnemo/{rev}/examples/structural_mechanics/deforming_plate/'
for f in ['preprocessor.py','inference.py','conf/config.yaml','requirements.txt','README.md']:
 try:
  b=urllib.request.urlopen(base+f,timeout=30).read(); q=p/'official'/f;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 except Exception as e:print(f,str(e))
headers={}
for f in ['meta.json','train.tfrecord','valid.tfrecord','test.tfrecord']:
 url='https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/'+f
 try:
  r=urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30);headers[f]={'url':url,'status':r.status,'headers':dict(r.headers)}
  if f=='meta.json':(p/f).write_bytes(urllib.request.urlopen(url,timeout=30).read())
 except Exception as e:headers[f]={'error':str(e)}
(o/'data-head.json').write_text(json.dumps(headers,indent=2),encoding='utf8');print(json.dumps(headers,indent=2))
