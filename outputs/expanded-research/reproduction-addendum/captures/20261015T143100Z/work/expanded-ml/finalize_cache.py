"""Commit a flat READY inventory after an independently observed producer exit0."""
import argparse
import json
from pathlib import Path
from benchmark_data import Cache, file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--producer-exit-record",type=Path,required=True)
    args=parser.parse_args()
    assert json.loads(args.producer_exit_record.read_text())["exit_code"] == 0
    ready_path=args.manifest.with_name("READY.json")
    assert not ready_path.exists(), "Frozen cache already has READY; do not mutate it"
    cache=Cache(args.manifest)
    for case in cache.meta["runs"]:
        assert cache.target(case).shape == cache.shape
        cache.inputs(case)
    files=[]
    for path in sorted(args.manifest.parent.rglob("*")):
        if path.is_file():
            files.append({"name":path.relative_to(args.manifest.parent).as_posix(),"bytes":path.stat().st_size,"sha256":file_hash(path)})
    ready={"status":"ready","writer_exit_code":0,"manifest_sha256":file_hash(args.manifest),
           "producer_exit_record_sha256":file_hash(args.producer_exit_record),"expected_cases":len(cache.meta["runs"]),
           "shape":list(cache.shape),"split_counts":{k:len(v) for k,v in cache.split.items()},"files":files}
    temporary=ready_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ready,indent=2),encoding="utf-8")
    temporary.replace(ready_path)
    print(json.dumps({"status":"ready","manifest_sha256":ready["manifest_sha256"],"ready_sha256":file_hash(ready_path),"files":len(files)}),flush=True)


if __name__=="__main__":
    main()
