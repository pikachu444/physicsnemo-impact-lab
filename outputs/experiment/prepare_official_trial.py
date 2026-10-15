"""Add a one-line VTP reader compatibility copy and reduced config.

The downloaded upstream train.py and vtp_reader.py remain unchanged.
"""
import argparse
from pathlib import Path
import shutil

parser=argparse.ArgumentParser()
parser.add_argument("--download-root",type=Path,required=True)
args=parser.parse_args()
example=args.download_root/"original_crash"
source=(example/"vtp_reader.py").read_text(encoding="utf-8")
old=r'displacement_t0\.[0-9]{3}$'
new=r'displacement_t[0-9]+\.[0-9]+$'
assert source.count(old)==1,"Pinned upstream reader no longer matches expected patch"
(example/"vtp_reader_compat.py").write_text(source.replace(old,new),encoding="utf-8")
shutil.copyfile(Path(__file__).with_name("fem_cpu_trial.yaml"),example/"conf/fem_cpu_trial.yaml")
print("Prepared compatibility reader and reduced config; original upstream files unchanged.")
