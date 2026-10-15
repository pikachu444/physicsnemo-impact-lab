"""Validate all frozen arrays and publish an atomic ready marker for consumers."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(cache, expected_cases):
    assert not (cache/"READY.json").exists(), "A ready cache is immutable"
    manifest = cache/"manifest.json"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    ids = sum(meta["split"].values(), [])
    assert len(ids) == len(set(ids)) == len(meta["runs"]) == expected_cases
    assert set(ids) == set(meta["runs"])
    xyz = np.load(cache/meta["coordinates"])
    edges = np.load(cache/meta["edge_index"])
    assert xyz.shape == (919,3) and edges.shape[0] == 2 and edges.min() >= 0 and edges.max() < 919
    assert len(np.unique(edges)) == 919 and np.isfinite(xyz).all()
    assert np.array_equal(np.load(cache/"node_ids.npy"), np.unique(np.load(cache/"node_ids.npy")))
    cases = []
    for case_id in ids:
        entry = meta["runs"][case_id]
        path = cache/entry["target"]
        target = np.load(path, mmap_mode="r")
        assert target.shape == (919,10,7) and target.dtype == np.float32 and np.isfinite(target).all()
        assert (target[...,3:5] >= 0).all()
        assert digest(path) == entry["target_sha256"]
        assert len(entry["actual_time_values_ms"]) == 10 and entry["max_time_jitter_ms"] < .001
        cases.append({"run":case_id,"manifest_sha256":digest(manifest),
                      "files":[{"name":entry["target"],"bytes":path.stat().st_size,"sha256":digest(path)}]})
    files = [{"name":p.relative_to(cache).as_posix(), "bytes":p.stat().st_size, "sha256":digest(p)}
             for p in sorted(cache.rglob("*")) if p.is_file() and p.name not in ("READY.json","READY.tmp")]
    ready = {"status":"ready","writer_exit_code":0,"expected_cases":expected_cases,
             "completed_utc":datetime.now(timezone.utc).isoformat(),"manifest_sha256":digest(manifest),
             "shape":[919,10,7],"split_counts":{k:len(v) for k,v in meta["split"].items()},"files":files,"cases":cases}
    temp = cache/"READY.tmp"
    temp.write_text(json.dumps(ready,indent=2),encoding="utf-8")
    temp.replace(cache/"READY.json")
    print(json.dumps({"ready":str(cache/"READY.json"),"manifest_sha256":ready["manifest_sha256"],"files":len(files),"cases":expected_cases},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--expected-cases",type=int,default=13)
    args = parser.parse_args()
    finalize(args.cache.resolve(),args.expected_cases)
