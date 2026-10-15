"""Replay the pinned public download manifest without redistributing FEM data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

parser=argparse.ArgumentParser()
parser.add_argument("--manifest",type=Path,default=Path(__file__).with_name("source_manifest.json"))
parser.add_argument("--destination",type=Path,required=True)
args=parser.parse_args()
manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
def fetch(item):
    relative=Path(item["path"].replace("\\", "/"))
    assert not relative.is_absolute() and ".." not in relative.parts
    destination=args.destination/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        response=requests.get(item["url"],timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
    content=destination.read_bytes()
    assert len(content)==item["bytes"] and hashlib.sha256(content).hexdigest()==item["sha256"],str(destination)
    return item["bytes"]
with ThreadPoolExecutor(max_workers=6) as executor:
    sizes=list(executor.map(fetch,manifest["files"]))
print(f"Verified {len(sizes)} files; {sum(sizes)} bytes")
