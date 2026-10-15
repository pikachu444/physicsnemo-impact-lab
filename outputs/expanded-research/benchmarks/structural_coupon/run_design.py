"""Reproduce a frozen case list serially in a fresh local work directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(work, solver, design_path, summaries, selected):
    design = json.loads(design_path.read_text(encoding="utf-8"))
    anchors = [{"case":"pilot_base", "strength_scale":1., "velocity_scale":1.},
               {"case":"corner_s090_v075", "strength_scale":.9, "velocity_scale":.75},
               {"case":"corner_s090_v125", "strength_scale":.9, "velocity_scale":1.25},
               {"case":"corner_s110_v075", "strength_scale":1.1, "velocity_scale":.75},
               {"case":"corner_s110_v125", "strength_scale":1.1, "velocity_scale":1.25}]
    runs = anchors + design["holdout_runs"]
    assert {r["case"] for r in runs} == set(sum(design["split"].values(), []))
    if selected:
        assert set(selected) <= {r["case"] for r in runs}
        runs = [r for r in runs if r["case"] in selected]
    for entry in runs:
        assert not (work/"runs"/entry["case"]).exists(), "Choose a fresh work directory or only unexecuted case IDs"
        assert not (summaries/entry["case"]).exists(), "Choose fresh summary output paths"
    work.mkdir(parents=True, exist_ok=True)
    snapshot = work/"reproduction-source-snapshot"
    assert not snapshot.exists(), "Use a fresh source snapshot for each orchestration"
    snapshot.mkdir()
    source_files = ["prepare_coupon.py", "run_coupon.py", "extract_coupon.py", "extract_nodal_displacement.py", "build_nodal_cache.py", "run_design.py"]
    source_hashes = {}
    for name in source_files:
        source = Path(__file__).with_name(name)
        shutil.copyfile(source, snapshot/name)
        source_hashes[name] = hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
    (snapshot/"manifest.json").write_text(json.dumps({"files":source_hashes, "design_sha256":hashlib.sha256(design_path.read_bytes()).hexdigest()},indent=2), encoding="utf-8")
    for entry in runs:
        name = entry["case"]
        case = work/"runs"/name
        started = time.perf_counter()
        commands = [[sys.executable,"-X","utf8",str(snapshot/"prepare_coupon.py"),"--work",str(work),"--case",name,
                     "--strength-scale",str(entry["strength_scale"]),"--velocity-scale",str(entry["velocity_scale"])],
                    [sys.executable,"-X","utf8",str(snapshot/"run_coupon.py"),"--case",str(case),"--solver",str(solver)],
                    [sys.executable,"-X","utf8",str(snapshot/"extract_coupon.py"),"--case",str(case),"--solver",str(solver),"--out",str(summaries/name),"--no-plot"],
                    [sys.executable,"-X","utf8",str(snapshot/"extract_nodal_displacement.py"),"--case",str(case)]]
        print(json.dumps({"event":"starting", "case":name}), flush=True)
        with (work/(name+".orchestration.log")).open("wb") as log:
            for command in commands:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = json.loads((summaries/name/"pilot-summary.json").read_text())
        print(json.dumps({"event":"completed", "case":name, "wall_seconds":time.perf_counter()-started,
                          "energy":summary["energy"]}), flush=True)
        assert summary["energy"]["max_kinetic_over_internal_after1ms"] < .01, "Kinetic energy guard exceeded; inspect before continuing"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    run(args.work.resolve(),args.solver.resolve(),args.design.resolve(),args.summaries.resolve(),args.cases)
