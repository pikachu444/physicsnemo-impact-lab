"""Freeze a validation-only plate pilot from JSON metadata; stdlib only.

No NPZ, original TFRecord, tensor, framework, or final-test trajectory is opened.
The output is a preregistered execution protocol, not an execution result.
An existing protocol is never overwritten.
"""
import datetime
import hashlib
import json
import math
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "outputs/expanded-research/benchmarks/deforming_plate"
OUTPUT = DIRECTORY / "official32-validation-protocol.json"
SOURCE_PIN = "11685edb3ec1fb6f25665e96cf7a8c4d11a23f16"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_json(name):
    path = DIRECTORY / name
    content = path.read_bytes()
    return json.loads(content.decode("utf-8")), {
        "path": path.relative_to(ROOT).as_posix(), "sha256": digest(content)
    }


def build_protocol():
    meta, meta_source = read_json("meta.json")
    manifest, manifest_source = read_json("data-manifest.json")
    contract, contract_source = read_json("final-split-contract.json")
    training = contract["training"]
    validation = contract["validation"]
    final_test = contract["fresh_final_test"]
    assert training == [f"train_{i:04d}" for i in range(32)]
    assert validation == [f"valid_{i:04d}" for i in range(6)]
    assert final_test == [f"test_{i:04d}" for i in range(2, 8)]
    assert contract["fresh_final_test_accuracy_evaluated"] is False
    assert contract["seed"] == 20261015
    assert not (set(training) & set(validation) or set(training + validation) & set(final_test))
    assert manifest["meta_sha256"] == meta_source["sha256"]
    assert meta["simulator"] == "comsol" and meta["trajectory_length"] == 400
    assert meta["features"]["stress"]["shape"] == [400, -1, 1]
    assert manifest["raw_strain_available"] is False

    # Only train/validation metadata are selected. No final-test hash is copied.
    records = {}
    for record in manifest["records"]:
        if record["split"] not in ("train", "valid"):
            continue
        name = f"{record['split']}_{record['index']:04d}"
        if name in records:
            raise ValueError(f"duplicate manifest metadata: {name}")
        records[name] = record
    dataset_hashes = {}
    validation_metadata = []
    for name in training + validation:
        record = records[name]
        expected_hash = contract["hashes"][name]
        assert isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        assert record["npz_sha256"] == expected_hash
        assert record["crc32c_verified"] is True
        assert record["shapes"]["world_pos"][0] == 400
        assert record["shapes"]["stress"] == [400, record["shapes"]["world_pos"][1], 1]
        dataset_hashes[name] = expected_hash
        if name in validation:
            maximum = record["stress_max"]
            assert not isinstance(maximum, bool) and isinstance(maximum, (int, float)) and math.isfinite(maximum)
            validation_metadata.append({
                "trajectory": name, "nodes": record["shapes"]["world_pos"][1],
                "frames": record["shapes"]["world_pos"][0],
                "source_scalar_stress_min": record["stress_min"],
                "source_scalar_stress_max": maximum,
                "npz_sha256_from_existing_contract": expected_hash,
            })
    assert len(dataset_hashes) == 38 and len(validation_metadata) == 6
    max_node = max(validation_metadata, key=lambda item: item["nodes"])
    max_stress = max(validation_metadata, key=lambda item: item["source_scalar_stress_max"])
    assert (max_node["trajectory"], max_node["nodes"]) == ("valid_0000", 1645)
    assert (max_stress["trajectory"], max_stress["source_scalar_stress_max"]) == ("valid_0005", 3531090.75)
    assert sum(item["nodes"] == max_node["nodes"] for item in validation_metadata) == 1
    assert sum(item["source_scalar_stress_max"] == max_stress["source_scalar_stress_max"] for item in validation_metadata) == 1

    protocol = {
        "schema_version": 1,
        "protocol_id": "deforming_plate_official128_15_train32_anchor8_epoch5_validation_only",
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "frozen_pre_execution_protocol_not_training_result",
        "training": training,
        "validation": validation,
        "dataset_npz_sha256": dataset_hashes,
        "dataset_hash_provenance": "Copied from the existing split contract and cross-checked against JSON manifest; no NPZ opened or rehashed by this script.",
        "selection_epochs": [1, 3, 5],
        "selection_trajectories": ["valid_0000", "valid_0005"],
        "required_options": {
            "hidden": 128, "processor": 15, "checkpoint_segments": 0,
            "epochs": 5, "frames_per_trajectory": 8, "noise": 0.003,
            "seed": 20261015, "lr": 0.0001, "lr_decay_per_update": 0.9999917,
            "rollout_steps": 399, "train_count": 32, "valid_count": 6,
            "validation_only": True, "checkpoint_selection": "rollout",
            "evaluation_mode": "ar_only", "selection_epochs": "1,3,5",
            "selection_trajectories": "valid_0000,valid_0005",
            "evaluation_label": "exploratory",
        },
        "diagnostic_selection_basis": {
            "scope": "All six validation metadata records; never final-test accuracy.",
            "maximum_nodes": {"trajectory": max_node["trajectory"], "nodes": max_node["nodes"], "unique_maximum": True},
            "maximum_source_scalar_stress": {"trajectory": max_stress["trajectory"], "value": max_stress["source_scalar_stress_max"], "unique_maximum": True},
            "validation_metadata": validation_metadata,
            "validation_metadata_canonical_sha256": digest(canonical_bytes(validation_metadata)),
            "stress_units": "Dataset scalar stress units; not independently confirmed Pa or MPa.",
        },
        "selection_rule": {
            "lexicographic_order": ["failure_flag", "any_frame_predicted_inversion_or_nonfinite_geometry_flag", "maximum_relative_l2_over_two_cases_and_displacement_scalar_stress", "epoch"],
            "completed_candidate_requirements": "Both fixed 399-step AR diagnostics must complete, displacement/stress relative L2 must be finite and defined, and prediction/reference inversion counts must cover all 399 frames.",
            "geometry_scope": "Plate tetrahedra excluding actuator-body cells; any predicted inverted tetrahedron in any frame flags the candidate. Reference inversions are separately recorded.",
            "tie_break": "Earlier epoch wins exact ties after the three ranking keys.",
            "if_every_completed_candidate_has_inversions": "Use the same predeclared worst-case/worst-field relative-L2 minimax among those candidates; preserve inversion counts and do not change the rule after observing results.",
            "zero_truth_norm": "Assign relative L2 zero only when truth norm, component RMSE, and maximum absolute error are all exactly zero. Otherwise mark relative L2 undefined, fail that candidate, and preserve absolute errors separately.",
            "failed_candidate_serialization": "Public JSON uses null with an explicit reason; infinity is permitted only in internal comparison keys, never serialized.",
            "baseline_diagnostics": "Report zero-motion displacement and train-mean constant-stress baselines separately; they do not change the frozen ranking key.",
            "negative_scalar_stress": "Preserve raw predicted negative scalar-stress count and minimum; do not clip predictions.",
            "qualification": "Checkpoint selection diagnostics are not an accuracy qualification or a final-test result.",
        },
        "recipe_limits": {
            "original_paper_recipe_reproduced": False,
            "scope": "A bounded validation-only pilot of the official 128-width/15-block architecture, not the original paper's complete training recipe.",
            "differences": [
                "Only the fixed first 32 training trajectories, not the full training corpus.",
                "Eight deterministic training frame anchors per trajectory and five epochs, not full-recipe sampling or convergence.",
                "Analytic noise-normalization treatment with fresh training noise, not a claim to reproduce the original framework's online statistics/noise implementation exactly.",
                "Normalization is computed from training data only; validation/final-test data do not enter normalization.",
                "Local PhysicsNeMo/PyTorch with torch_geometric graph backend, not the original paper's hardware/framework/distributed execution.",
            ],
            "strain_label_caveat": "Reference and predicted Green-Lagrange strain are kinematically derived from source FEM/AI nodal arrays; no original COMSOL strain tensor label is available.",
        },
        "fresh_final_test": final_test,
        "fresh_final_test_policy": {
            "content_accessed_by_protocol_script": False,
            "npz_hashes_read_or_recomputed_by_protocol_script": False,
            "npz_hashes_copied_into_protocol": False,
            "evaluation_permitted_for_this_pilot": False,
            "note": "Only six IDs are copied from the authorized JSON contract. No final-test NPZ is opened, hashed, or used for selection.",
        },
        "provenance": {
            "nvidia_reference_source_commit": SOURCE_PIN,
            "metadata_inputs": [meta_source, manifest_source, contract_source],
            "creator_script": {"path": pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix(),
                               "sha256": digest(pathlib.Path(__file__).read_bytes())},
            "input_access_scope": "Three JSON metadata files and this script only. No array or trajectory file is accessed.",
        },
    }
    protocol["hash_metadata"] = {
        "algorithm": "SHA256",
        "canonical_payload_sha256": digest(canonical_bytes(protocol)),
        "canonicalization": "UTF-8 json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False), excluding hash_metadata",
    }
    return protocol


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {OUTPUT}")
    protocol = build_protocol()
    encoded = (json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with OUTPUT.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"status": "frozen_metadata_only", "path": OUTPUT.relative_to(ROOT).as_posix(),
                      "json_sha256": digest(encoded), "canonical_payload_sha256": protocol["hash_metadata"]["canonical_payload_sha256"],
                      "training": len(protocol["training"]), "validation": len(protocol["validation"]),
                      "final_test_npz_accessed": False}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
