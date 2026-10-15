"""Record runtime/package/binary identifiers without publishing solver binaries."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import psutil


def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def capture(solver,out):
    binaries={}
    for name in ["starter_win64.exe","engine_win64.exe","anim_to_vtk_win64.exe","th_to_csv_win64.exe"]:
        path=solver/"exec"/name
        binaries[name]={"bytes":path.stat().st_size,"sha256":sha(path)}
    sources={path.name:{"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(Path(__file__).parent.glob("*.py"))}
    manifest={"captured_utc":datetime.now(timezone.utc).isoformat(),"python_version":sys.version,
              "platform":platform.platform(),"processor":platform.processor(),"logical_cpus":os.cpu_count(),
              "physical_memory_bytes":psutil.virtual_memory().total,
              "package_versions":{name:version(name) for name in ["numpy","scipy","vtk","psutil","matplotlib"]},
              "binaries":binaries,"current_reproduction_source":sources,
              "scope":"Existing local Windows runtime and solver used; no paid resource provisioned. Per-process FEM wall/RSS logs are separate.",
              "source_provenance_limit":"Current reproduction scripts hashed after original pilot execution. Initial incremental pilot scripts were not snapshotted at process launch; prepared solver input hashes, solver binaries, native outputs and independent label reconstruction preserve the executed FEM/label evidence. Future run_design.py executions snapshot sources before launching.",
              "solver_header":"OpenCourant Starter/Engine Windows64 double precision; bundled release binaries, exact binary hashes above. Original distribution license remains with solver."}
    out.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"environment_manifest":str(out),"sha256":sha(out),"binaries":binaries},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--solver",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    capture(args.solver.resolve(),args.out.resolve())
