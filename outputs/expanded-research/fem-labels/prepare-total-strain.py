"""Preserve the Run224 physical input and append only post-processing outputs."""
from pathlib import Path
import argparse
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'work/bumper-fem-rerun'
RUN = ROOT / 'work/expanded-fem-labels/run-total-strain'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir',type=Path,default=SOURCE)
parser.add_argument('--run-dir',type=Path,default=RUN)
parser.add_argument('--report-dir',type=Path,default=OUT)
args=parser.parse_args()
SOURCE=args.source_dir;RUN=args.run_dir;OUT=args.report_dir
OUT.mkdir(parents=True,exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)
stem = 'Bumper_Beam_AP_meshed'
starter = (SOURCE / (stem + '_0000.rad')).read_bytes()
engine = (SOURCE / (stem + '_0001.rad')).read_bytes()
newline = b'\r\n' if b'\r\n' in engine else b'\n'
addition = newline + b'/ANIM/SHELL/TENS/STRAIN/ALL' + newline + b'/ANIM/SHELL/VONM' + newline
new_engine = engine + addition
assert new_engine[:len(engine)] == engine
for name, content in [(stem + '_0000.rad', starter), (stem + '_0001.rad', new_engine)]:
    destination = RUN / name
    if destination.exists() and destination.read_bytes() != content:
        raise RuntimeError('Refusing to replace an existing different prepared input: ' + name)
    destination.write_bytes(content)
result = {
    'run': 'Run224',
    'dataset_revision': '78f1302fba4d75926c6fdd22d01d5133efc14742',
    'solver_commit': 'cb760c206e346eff12161a2268b134ef9d48a4ef',
    'starter_sha256': hashlib.sha256(starter).hexdigest(),
    'original_engine_sha256': hashlib.sha256(engine).hexdigest(),
    'prepared_engine_sha256': hashlib.sha256(new_engine).hexdigest(),
    'starter_byte_identical': (RUN / (stem + '_0000.rad')).read_bytes() == starter,
    'original_engine_prefix_byte_identical': new_engine[:len(engine)] == engine,
    'only_added_output_cards': ['/ANIM/SHELL/TENS/STRAIN/ALL', '/ANIM/SHELL/VONM'],
    'changed_material_contact_mesh_boundary_initial_or_time_step_cards': False,
    'istrain_property_modified': False,
    'output_semantics_caution': 'Raw total-strain output is requested separately from EPSP. Successful nonzero strain output and paired kinematic/energy comparison will be verified after actual execution.'
}
(OUT / 'input-output-only-diff.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
