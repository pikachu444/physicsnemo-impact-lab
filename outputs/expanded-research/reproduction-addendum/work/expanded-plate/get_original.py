import urllib.request,pathlib
u='https://raw.githubusercontent.com/google-deepmind/deepmind-research/master/meshgraphnets/deform_model.py'
p=pathlib.Path('work/expanded-plate/official/deepmind_deform_model.py');p.write_bytes(urllib.request.urlopen(u,timeout=30).read())
print(p.read_text())
