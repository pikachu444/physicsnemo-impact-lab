"""Add provenance errata without rewriting any frozen cache or model artifact."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


root = Path(__file__).resolve().parents[2]
destination = root / "outputs/expanded-research/benchmarks/drop_test/common-cache-provenance-clarification.json"
assert not destination.exists(), "Additive errata evidence must not overwrite previous evidence"
environment_path = root / "outputs/expanded-research/benchmarks/drop_test/environment.json"
environment = json.loads(environment_path.read_text(encoding="utf-8"))
source_path = root / "outputs/expanded-research/benchmarks/drop_test/source-provenance.json"
source = json.loads(source_path.read_text(encoding="utf-8"))
rows = []
for support in ("full", "rear"):
    cache = root / f"work/expanded-ml/phone-{support}-cache-v2"
    manifest_path = cache / "manifest.json"
    before = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_proofs = []
    for case, record in manifest["runs"].items():
        actual_hash = sha256(cache / record["target"])
        assert actual_hash == record["source_target_sha256"], case
        target_proofs.append({"case": case, "common_target_sha256": actual_hash,
                              "canonical_target_sha256": record["source_target_sha256"],
                              "byte_identity_verified": True})
    rows.append({
        "support": support,
        "frozen_manifest": str(manifest_path.relative_to(root)),
        "frozen_manifest_sha256": before,
        "frozen_ready_sha256": sha256(cache / "READY.json"),
        "normalization_sha256": sha256(root / f"work/expanded-ml/phone-{support}-normalization.json"),
        "training_mean_field_sha256": sha256(root / f"work/expanded-ml/phone-{support}-training-mean-field.npy"),
        "original_label_source_text": manifest["label_source"],
        "target_proofs": target_proofs,
        "manifest_unchanged_after_verification": sha256(manifest_path) == before,
    })
report = {
    "status": "additive_provenance_clarification; no numerical artifact modified",
    "checked_date": "2026-10-15",
    "correction": {
        "issue": "The frozen common V2 label_source used 'public phone OpenRadioss20us canonical fields', which can be read as a ready-made public OpenRadioss result dataset.",
        "correct_description": "Nine new local OpenCourant FEM executions of the public phone deck obtained from an independently documented public mirror; labels are the resulting first-20-us canonical fields, interpolated from actual saved frame times onto nominal 2..20 us target times without extrapolation. These are not pre-existing public OpenRadioss FEM results.",
        "unchanged": "V2 manifests, READY inventories, label arrays, normalized statistics, mean fields, and any completed model checkpoint remain frozen. This document corrects the interpretation of text provenance only.",
    },
    "solver": {
        "product": "OpenCourant Starter and Engine Win64 Extended Single Precision",
        "threads": 2,
        "domains": 1,
        "binary_sha256": environment["solver_binary_sha256"],
        "binary_provenance_limit": "The actual executed binaries are pinned by SHA256. No independently verified equivalence to a particular source-code build commit is claimed.",
        "evidence": ["outputs/expanded-research/benchmarks/drop_test/environment.json",
                     "outputs/expanded-research/benchmarks/drop_test/20us/execution-metrics.json",
                     "outputs/expanded-research/benchmarks/drop_test/doe/*/execution-metrics.json"],
    },
    "public_input_deck": {
        "mirror_repository": source["mirror_repository"],
        "mirror_commit": source["mirror_commit"],
        "official_byte_identity_verified": source["starter_official_byte_identity_verified"],
        "source_provenance_file_sha256": sha256(source_path),
        "license_in_deck": source["model_license_in_file"],
    },
    "static_feature_clarification": {
        "name": "reference_mean_incident_corner_cell_volume_cuberoot",
        "unit": "mm",
        "meaning": "Cube root of the mean reference volume of incident corner-node cells; a local reference mesh-length scalar.",
        "legacy_internal_key": "thickness",
        "legacy_key_is_not_physical_definition": True,
        "new_nn_metadata": "run_benchmark.py explicitly substitutes cache.static_feature_name into model_source.input_information and the metrics input description.",
    },
    "frozen_cache_evidence": rows,
}
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"written": str(destination), "sha256": sha256(destination),
                  "canonical_target_identity_checks": sum(len(r["target_proofs"]) for r in rows)}))
