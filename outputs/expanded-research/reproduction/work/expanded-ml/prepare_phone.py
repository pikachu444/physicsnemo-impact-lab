"""Adapt frozen phone canonical manifests without modifying their FEM fields."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from benchmark_data import file_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-manifest",type=Path,required=True)
    parser.add_argument("--source-ready",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    assert not (args.out/"manifest.json").exists(),"Choose a fresh conversion directory"
    source=args.source_manifest.parent
    marker_hash=file_hash(args.source_ready)
    marker=json.loads(args.source_ready.read_text(encoding="utf-8"))
    original=json.loads(args.source_manifest.read_text(encoding="utf-8"))
    assert marker["ready"] and marker["cases"]==9 and marker["all_support_checks_passed"]
    for relative,row in marker["file_hashes"].items():
        path=source/relative
        assert path.stat().st_size==row["bytes"] and file_hash(path)==row["sha256"],path
    assert original["ready"] and len(original["runs"])==9
    assert original["field_names"][:3]==["u_x","u_y","u_z"]
    assert original["static_feature_name"]=="reference_mean_incident_corner_cell_volume_cuberoot"
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"runs").mkdir(exist_ok=True)
    (args.out/"source_metadata").mkdir(exist_ok=True)
    for key,destination in [("coordinates","coordinates.npy"),("edge_index","edge_index.npy"),("static_feature","static_feature.npy")]:
        shutil.copyfile(source/original[key],args.out/destination)
    xyz=np.load(args.out/"coordinates.npy",mmap_mode="r")
    edge=np.load(args.out/"edge_index.npy",mmap_mode="r")
    feature=np.load(args.out/"static_feature.npy",mmap_mode="r")
    n=original["shape_per_run"][0]
    assert xyz.shape==(n,3) and feature.shape==(n,1) and edge.shape[0]==2
    assert edge.min()>=0 and edge.max()<n and np.isfinite(feature).all() and feature.min()>0
    assert np.ptp(feature)>0,"The supplied static field must be a real length, not a constant placeholder"
    split={"train":[],"validation":[],"test":[]}
    runs={}
    for row in original["runs"]:
        case=row["run_id"]
        group={"val":"validation"}.get(row["split"],row["split"])
        split[group].append(case)
        target=source/row["target"]
        y=np.load(target,mmap_mode="r")
        assert list(y.shape)==original["shape_per_run"] and y.dtype==np.float32 and np.isfinite(y).all()
        destination=f"runs/{case}.npy"
        shutil.copyfile(target,args.out/destination)
        assert file_hash(args.out/destination)==file_hash(target),"Labels changed during schema-only adaptation"
        shutil.copyfile(source/row["metadata"],args.out/"source_metadata"/f"{case}.json")
        timing=json.loads((source/row["metadata"]).read_text(encoding="utf-8"))
        runs[case]={"target":destination,"thickness":"static_feature.npy","globals":row["globals"],
                    "source_target_sha256":file_hash(target),"nominal_target_time_values":original["time_values"],
                    "actual_raw_saved_time_values":[frame["actual_time_s"] for frame in timing["raw_frame_records"]],
                    "target_time_brackets":timing["time_brackets"],"target_resample_method":timing["resample_method"]}
    assert {k:len(v) for k,v in split.items()}=={"train":5,"validation":2,"test":2}
    fields=["displacement_x","displacement_y","displacement_z"]+original["field_names"][3:]
    manifest={**{k:v for k,v in original.items() if k not in ["runs","ready","split_counts"]},
              "schema":"bumper_multifield_cache_v1","coordinates":"coordinates.npy","edge_index":"edge_index.npy",
              "static_feature":"static_feature.npy","static_feature_name":original["static_feature_name"],
              "runs":runs,"split":split,"field_names":fields,"original_field_names":original["field_names"],
              "field_alias_note":"Only u_x/y/z names were mapped to displacement_x/y/z. Numeric labels and units are bitwise unchanged.",
              "legacy_loader_key_note":"Per-run key thickness points to the declared reference mesh-length feature; it is not shell thickness.",
              "label_source":"Frozen public phone OpenRadioss20us canonical fields; original RAD reference geometry and nominal-frame interpolation",
              "source_ready_sha256":marker_hash,"source_manifest_sha256":file_hash(args.source_manifest),
              "source_ready_name":args.source_ready.name,
              "time_note":"Nominal 2..20us labels linearly interpolated within actual saved-frame brackets; no extrapolation. Canonical raw12frames remain separate.",
              "input_scope_note":"Fixed assembly geometry; e_scale_common and rwall_rx_deg change constitutive stiffness and wall normal, not initial node coordinates."}
    for key in ("source_case_manifest","geometry_contract","full_graph_contract"):
        if original.get(key):
            path=source/original[key]
            destination=f"source_metadata/{path.name}"
            shutil.copyfile(path,args.out/destination)
            manifest[key]=destination
    assert file_hash(args.source_ready)==marker_hash,"Original source READY changed during adaptation"
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    shutil.copyfile(args.source_manifest,args.out/"source_manifest.json")
    shutil.copyfile(args.source_ready,args.out/"source_READY_evidence.json")
    print(json.dumps({"status":"schema_only_phone_cache_conversion_completed","support":original["support"],
                      "shape":original["shape_per_run"],"source_ready_sha256":marker_hash,
                      "manifest_sha256":file_hash(args.out/"manifest.json"),"static_feature_range_mm":[float(feature.min()),float(feature.max())]}),flush=True)


if __name__=="__main__":
    main()
