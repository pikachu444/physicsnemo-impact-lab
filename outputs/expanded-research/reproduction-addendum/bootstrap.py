"""Verify and restore research sources into a fresh work/ tree; stdlib only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

PACKAGE = Path(__file__).resolve().parent
ALLOWED_SUFFIXES = {".py", ".ps1", ".sh", ".yaml", ".yml", ".json", ".txt", ".md"}
FORBIDDEN_PARTS = {"venv", ".venv", "__pycache__", ".git", "deps", "data", "raw_data", "checkpoints", "weights"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(value, prefix=None):
    if not isinstance(value,str) or not value or "\\" in value or ":" in value:
        raise ValueError("Invalid relative path in manifest")
    parts=value.split("/")
    path=PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ValueError("Absolute or traversing manifest path rejected")
    if any(part.lower() in FORBIDDEN_PARTS or part.lower().startswith("cache") for part in parts[:-1]):
        raise ValueError("Environment/cache/data path rejected")
    if prefix and parts[0]!=prefix:
        raise ValueError("Destination must stay under work/")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Non-source file extension rejected")
    return Path(*parts)


def is_link(path):
    return path.is_symlink() or (getattr(path,"is_junction",lambda:False)())


def contained_path(root,relative):
    root=root.resolve()
    path=root/relative
    if not path.resolve(strict=False).is_relative_to(root):
        raise ValueError("Resolved path escapes the selected root")
    current=path
    while current!=root:
        if is_link(current):
            raise ValueError("Symbolic links and junctions are not restored through")
        current=current.parent
    return path


def verify():
    manifest=json.loads((PACKAGE/"source-manifest.json").read_text(encoding="utf-8"))
    records={}
    for record in manifest["files"]:
        relative=safe_relative(record["packaged_path"])
        safe_relative(record["destination"],prefix="work")
        if relative.parts[0] not in ("work","recorded"):
            raise ValueError("Unexpected packaged-source root")
        source=contained_path(PACKAGE,relative)
        data=source.read_bytes()
        if len(data)!=record["bytes"] or digest(data)!=record["sha256"]:
            raise ValueError("Packaged source bytes/hash mismatch: "+record["packaged_path"])
        if record["packaged_path"] in records:
            raise ValueError("Duplicate manifest source")
        records[record["packaged_path"]]=record
    groups=[manifest["default_files"]]+[p["overrides"] for p in manifest["profiles"].values()]
    for keys in groups:
        destinations=[records[key]["destination"] for key in keys]
        if len(destinations)!=len(set(destinations)):
            raise ValueError("Conflicting destination within a manifest selection")
    return manifest,records


def restore(workspace,profile,dry_run=False):
    manifest,records=verify()
    chosen={records[key]["destination"]:records[key] for key in manifest["default_files"]}
    if profile!="current":
        if profile not in manifest["profiles"]:
            raise ValueError("Unknown profile; use --list-profiles")
        for key in manifest["profiles"][profile]["overrides"]:
            record=records[key]
            chosen[record["destination"]]=record
    workspace=workspace.resolve()
    plan=[]
    for destination,record in sorted(chosen.items()):
        target=contained_path(workspace,safe_relative(destination,prefix="work"))
        if target.exists() or target.is_symlink():
            raise FileExistsError("Refusing to overwrite an existing path: "+destination)
        plan.append((target,record))
    if not dry_run:
        for target,record in plan:
            target.parent.mkdir(parents=True,exist_ok=True)
            contained_path(workspace,safe_relative(record["destination"],prefix="work"))
            data=(PACKAGE/record["packaged_path"]).read_bytes()
            if digest(data)!=record["sha256"]:
                raise ValueError("Source changed after preflight")
            with target.open("xb") as stream:
                stream.write(data)
            if digest(target.read_bytes())!=record["sha256"]:
                raise ValueError("Restored source hash mismatch")
    return {"status":"dry_run" if dry_run else "restored", "profile":profile,"source_files":len(plan),
            "workspace":str(workspace),"bytes":sum(record["bytes"] for _,record in plan),
            "environment_installed":False,"data_downloaded":False,"training_or_inference_executed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,help="New clone/research workspace; every destination must be absent")
    parser.add_argument("--profile",default="current",help="current or a recorded run version; choose before first restore")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--list-profiles",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    try:
        if args.verify_only or args.list_profiles:
            manifest,records=verify()
            result={"status":"verified","files":len(records),"bytes":sum(r["bytes"] for r in records.values())}
            if args.list_profiles:
                result["profiles"]={"current":"Captured stable current sources; not a replacement for historical run versions",
                                    **{key:value["description"] for key,value in manifest["profiles"].items()}}
        else:
            if args.workspace is None:
                parser.error("--workspace is required for a restore")
            result=restore(args.workspace,args.profile,args.dry_run)
        print(json.dumps(result,indent=2))
    except (OSError,ValueError,KeyError) as error:
        print(json.dumps({"status":"refused","reason":str(error)}),file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
