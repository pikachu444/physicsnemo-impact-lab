import urllib.request,pathlib
base='https://raw.githubusercontent.com/NVIDIA/physicsnemo/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/deforming_plate/'
for f in ['deforming_plate_dataset.py','helpers.py']:
 b=urllib.request.urlopen(base+f,timeout=30).read();pathlib.Path('work/expanded-plate/official',f).write_bytes(b)
