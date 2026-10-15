"""Metadata-only preparation. No networking, FEM, ML imports, or training runner.

Run with Python 3.10+ and -S. A fixture result is never an acquired-data result.
"""
from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sys

DATASET = "luminary-shift/SHIFT-Crash"
REVISION = "6faa776f7e363ce39d43eb8dc8b0d9c7d0c83049"
GEOMETRY = ("front_rails", "cabin_length", "car_width", "roof_height")
MATERIAL = ("rail_thickness_mm", "rail_yield_strength_mpa")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"yaris_x[0-9]{5}\Z")
SETTINGS = ("transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet")


class PreparationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise PreparationError(message)


def canonical(value):
    """Exact decimal strings: no implicit tolerance, rounding, or float grouping."""
    need(isinstance(value, str), "Physical parameters must be decimal strings")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise PreparationError("Invalid decimal parameter") from exc
    need(number.is_finite(), "Non-finite parameter")
    if number == 0:
        return "0"
    fixed = format(number, "f")
    return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_protocol(protocol):
    need(protocol.get("dataset", {}).get("id") == DATASET, "Wrong dataset")
    need(protocol["dataset"].get("revision") == REVISION, "Dataset revision is not pinned")
    algorithms = protocol.get("algorithms", [])
    need(tuple(a.get("setting") for a in algorithms) == SETTINGS, "Six settings are required in the documented order")
    need(len({a["family"] for a in algorithms}) == 5, "Six settings must be counted as five families")
    need(protocol.get("execution", {}).get("training_runner_implemented") is False, "Preparation does not provide a training runner")
    for key in ("case_count", "fit_budget_seconds", "hardware", "train_geometry_count", "validation_geometry_count", "test_geometry_count"):
        need(protocol["execution"].get(key) is None, "Unmeasured execution choices must remain null: " + key)
    need(protocol["coordinates"].get("per_case_bbox_normalization") is False, "Per-case geometry normalization is prohibited")
    need(protocol["coordinates"].get("clipping") is False, "Coordinate clipping would alter the shape")
    return {"status": "PREPARATION_SCHEMA_VALID", "five_families_six_settings": True,
            "training_executed": False, "accuracy_validated": False}


def validate_manifest(manifest, *, allow_fixture=False, require_metadata=False):
    need(manifest.get("schema_version") == 1, "Unsupported manifest schema")
    need(manifest.get("dataset_id") == DATASET and manifest.get("revision") == REVISION, "Wrong dataset/revision")
    origin = manifest.get("record_origin")
    need(origin in ("actual_metadata", "synthetic_fixture"), "Unknown metadata origin")
    need(origin != "synthetic_fixture" or allow_fixture, "Fixture manifest requires --allow-fixture; it is not acquired data")
    access = manifest.get("access", {})
    if require_metadata:
        need(access.get("approved") is True, "Access approval has not been recorded")
        need(access.get("params_acquired") is True, "Protected params have not been acquired")
        need(access.get("parameter_mapping_verified") is True, "Canonical/raw parameter mapping is unverified")
        need(access.get("condition_mapping_verified") is True, "Identical impact-condition mapping is unverified")
    cases = manifest.get("cases")
    need(isinstance(cases, list), "cases must be a list")
    ids = set()
    for case in cases:
        cid = case.get("case_id", "")
        need(CASE_ID.fullmatch(cid) is not None and cid not in ids, "Invalid or duplicate case ID")
        ids.add(cid)
        for key in GEOMETRY:
            canonical(case.get("geometry_mm", {}).get(key))
        for key in MATERIAL:
            need(Decimal(canonical(case.get("material", {}).get(key))) > 0, "Material parameters must be positive")
        condition = case.get("condition", {})
        need(set(condition) == {"impact_speed_kmh", "barrier_id", "duration_ms"}, "Explicit impact condition is required")
        need(Decimal(canonical(condition["impact_speed_kmh"])) > 0 and Decimal(canonical(condition["duration_ms"])) > 0, "Invalid impact condition")
        need(isinstance(condition["barrier_id"], str) and bool(condition["barrier_id"]), "Missing barrier ID")
        files = case.get("files", [])
        need(isinstance(files, list) and bool(files), "Explicit allowlisted file inventory required")
        roles = set()
        for file in files:
            role = file.get("role")
            need(role in ("params", "vtp") and role not in roles, "Invalid/duplicate file role")
            roles.add(role)
            expected = cid + "/" + {"params": "params.json", "vtp": "merged_surfaces.vtp"}[role]
            need(file.get("path") == expected, "Non-allowlisted file path")
            need(type(file.get("bytes")) is int and file["bytes"] > 0, "Verified positive file byte size required")
            need(type(file.get("acquired")) is bool, "Acquired flag must be a boolean")
            if file["acquired"]:
                need(HEX64.fullmatch(file.get("sha256", "")) is not None, "Acquired file needs SHA-256")
            else:
                need(file.get("sha256") is None, "Unacquired file cannot claim a measured SHA-256")
        if require_metadata:
            params = next((f for f in files if f["role"] == "params"), None)
            need(params is not None and params["acquired"], "Each selected record needs acquired, hashed params")
    if require_metadata:
        need(bool(cases), "No acquired case metadata: split refused")
    return {"status": "MANIFEST_SCHEMA_VALID", "case_metadata_records": len(cases),
            "record_origin": origin, "training_ready": False}


def geometry_key(case):
    return digest({key: canonical(case["geometry_mm"][key]) for key in GEOMETRY})


def material_key(case):
    return tuple(canonical(case["material"][key]) for key in MATERIAL)


def condition_key(case):
    c = case["condition"]
    return canonical(c["impact_speed_kmh"]), c["barrier_id"], canonical(c["duration_ms"])


def candidate_split(manifest, thickness, yield_strength, counts, seed, *, allow_fixture=False):
    validate_manifest(manifest, allow_fixture=allow_fixture, require_metadata=True)
    need(len(counts) == 3 and all(type(n) is int and n > 0 for n in counts), "Choose explicit positive train/validation/test geometry counts after metadata review")
    target = (canonical(thickness), canonical(yield_strength))
    selected = [c for c in manifest["cases"] if material_key(c) == target]
    need(bool(selected), "No exactly matched material/thickness stratum was found")
    need(len({condition_key(c) for c in selected}) == 1, "Selected stratum mixes impact conditions; geometry-only claim refused")
    groups = {}
    for case in selected:
        groups.setdefault(geometry_key(case), []).append(case["case_id"])
    need(len(groups) >= sum(counts), "Insufficient distinct geometries at the fixed material/thickness; no DOE size is inferred")
    ordered = sorted(groups, key=lambda g: (digest({"seed": seed, "geometry": g}), g))
    folds = {}
    start = 0
    for name, count in zip(("train", "validation", "test"), counts):
        for key in ordered[start:start + count]:
            folds[key] = name
        start += count
    selected_cases = {name: [] for name in ("train", "validation", "test")}
    all_nuisance_variants = {name: [] for name in selected_cases}
    for case in manifest["cases"]:
        fold = folds.get(geometry_key(case))
        if fold:
            all_nuisance_variants[fold].append(case["case_id"])
            if material_key(case) == target:
                selected_cases[fold].append(case["case_id"])
    for values in (*selected_cases.values(), *all_nuisance_variants.values()):
        values.sort()
    return {"status": "METADATA_CANDIDATE_SPLIT_ONLY", "record_origin": manifest["record_origin"],
            "source_manifest_digest": digest(manifest), "dataset_id": DATASET, "revision": REVISION,
            "geometry_parameters_mm": list(GEOMETRY), "fixed_material": dict(zip(MATERIAL, target)),
            "fixed_condition": selected[0]["condition"], "seed": seed,
            "geometry_counts": dict(zip(selected_cases, counts)), "geometry_assignments": folds,
            "selected_cases_by_split": selected_cases,
            "all_manifest_nuisance_variants_by_fold": all_nuisance_variants,
            "unused_geometry_groups_in_fixed_stratum": len(groups) - sum(counts),
            "grouping_rule": "All known nuisance variants of each selected initial geometry have one fold; only fixed-stratum cases enter the primary comparison",
            "training_ready": False, "fields_inspected": False, "training_executed": False,
            "claim_limit": "A metadata split does not establish geometry extrapolation, verified FEM labels, remesh convergence, or prediction accuracy"}


def download_plan(manifest, case_ids, roles, max_bytes):
    """Return a capped allowlist; deliberately contains no download implementation."""
    validate_manifest(manifest, require_metadata=True)
    need(manifest["record_origin"] == "actual_metadata", "Fixture download plans are prohibited")
    need(type(max_bytes) is int and max_bytes > 0, "An explicit positive byte cap is required")
    need(len(case_ids) == len(set(case_ids)) and len(roles) == len(set(roles)), "Duplicate case/role selection")
    need(set(roles) <= {"params", "vtp"} and bool(roles), "Explicit params/vtp roles required")
    cases = {c["case_id"]: c for c in manifest["cases"]}
    need(set(case_ids) <= set(cases) and bool(case_ids), "Unknown/empty case allowlist")
    files = []
    for cid in sorted(case_ids):
        inventory = {f["role"]: f for f in cases[cid]["files"]}
        need(set(roles) <= set(inventory), "Requested file size/path is not verified in the inventory")
        for role in sorted(roles):
            file = copy.deepcopy(inventory[role])
            file["url"] = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{file['path']}"
            files.append(file)
    total = sum(f["bytes"] for f in files)
    need(total <= max_bytes, "Allowlist exceeds the explicit download byte cap")
    return {"status": "DOWNLOAD_PLAN_ONLY_NO_NETWORK", "revision": REVISION, "files": files,
            "planned_source_bytes": total, "max_download_bytes": max_bytes,
            "downloader_implemented": False, "authentication": "Approved user-managed Hugging Face authentication; never browser-cookie/token extraction",
            "additional_storage_gate": "Measure free disk, staging/extraction/cache multiplier and reserved headroom before any actual transfer"}


def fixture_manifest():
    manifest = {"schema_version": 1, "record_origin": "synthetic_fixture", "dataset_id": DATASET,
                "revision": REVISION, "access": {"approved": True, "params_acquired": True,
                "parameter_mapping_verified": True, "condition_mapping_verified": True}, "cases": []}
    # Fabricated small records test grouping only. These are not SHIFT metadata.
    for i, (shape, thickness) in enumerate(((0, "1.2"), (0, "1.5"), (10, "1.2"), (20, "1.2"), (30, "1.2")), 1):
        cid = f"yaris_x{i:05d}"
        manifest["cases"].append({"case_id": cid,
            "geometry_mm": dict(zip(GEOMETRY, (str(shape), "0", "0", "0"))),
            "material": {"rail_thickness_mm": thickness, "rail_yield_strength_mpa": "300"},
            "condition": {"impact_speed_kmh": "56", "barrier_id": "FIXTURE_RIGID_WALL", "duration_ms": "120"},
            "files": [{"role": "params", "path": cid + "/params.json", "bytes": 283, "acquired": True, "sha256": "f" * 64},
                      {"role": "vtp", "path": cid + "/merged_surfaces.vtp", "bytes": 1000, "acquired": False, "sha256": None}]})
    return manifest


def selftest(protocol):
    checks = []
    validate_protocol(protocol)
    fixture = fixture_manifest()
    precise_a = copy.deepcopy(fixture["cases"][0])
    precise_b = copy.deepcopy(precise_a)
    precise_a["geometry_mm"]["front_rails"] = "1.1234567890123456789012345678901"
    precise_b["geometry_mm"]["front_rails"] = "1.1234567890123456789012345678902"
    need(geometry_key(precise_a) != geometry_key(precise_b), "More-than-28-digit geometry parameters were rounded")
    precise_b["geometry_mm"]["front_rails"] = precise_a["geometry_mm"]["front_rails"] + "000"
    need(geometry_key(precise_a) == geometry_key(precise_b), "Fractional trailing zeros changed the geometry group")
    checks.append("more-than-28-digit geometry values stay distinct while fractional trailing-zero variants match")
    result = candidate_split(fixture, "1.20", "300.0", (2, 1, 1), "fixture-seed", allow_fixture=True)
    need(result == candidate_split(fixture, "1.2", "300", (2, 1, 1), "fixture-seed", allow_fixture=True), "Determinism/canonical grouping failed")
    checks.append("exact decimal matching and deterministic geometry split")
    reverse = copy.deepcopy(fixture)
    reverse["cases"].reverse()
    again = candidate_split(reverse, "1.2", "300", (2, 1, 1), "fixture-seed", allow_fixture=True)
    need(result["geometry_assignments"] == again["geometry_assignments"] and result["selected_cases_by_split"] == again["selected_cases_by_split"], "Input order affected split")
    checks.append("case input order does not affect assignments")
    lookup = {cid: name for name, ids in result["all_manifest_nuisance_variants_by_fold"].items() for cid in ids}
    need(lookup["yaris_x00001"] == lookup["yaris_x00002"], "Same shape leaked across folds")
    need("yaris_x00002" not in sum(result["selected_cases_by_split"].values(), []), "Nuisance material entered geometry-only fit")
    checks.append("same shape across nuisance materials stays in one fold and outside fixed-stratum fit")

    def rejected(name, function):
        try:
            function()
        except PreparationError:
            checks.append(name)
            return
        raise PreparationError("Expected rejection did not happen: " + name)

    rejected("fixtures require explicit flag", lambda: candidate_split(fixture, "1.2", "300", (2, 1, 1), "s"))
    wrong = copy.deepcopy(fixture); wrong["revision"] = "main"
    rejected("unpinned revision rejected", lambda: validate_manifest(wrong, allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["access"]["approved"] = False
    rejected("missing access approval rejected", lambda: candidate_split(wrong, "1.2", "300", (2, 1, 1), "s", allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["access"]["parameter_mapping_verified"] = False
    rejected("unverified raw/canonical mapping rejected", lambda: candidate_split(wrong, "1.2", "300", (2, 1, 1), "s", allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["cases"][0]["files"][0]["path"] = "../params.json"
    rejected("non-allowlisted/traversal file path rejected", lambda: validate_manifest(wrong, allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["cases"][1]["case_id"] = wrong["cases"][0]["case_id"]
    rejected("duplicate case IDs rejected", lambda: validate_manifest(wrong, allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["cases"][0]["geometry_mm"]["car_width"] = "NaN"
    rejected("nonfinite geometry rejected", lambda: validate_manifest(wrong, allow_fixture=True))
    wrong = copy.deepcopy(fixture); wrong["cases"][0]["condition"]["impact_speed_kmh"] = "60"
    rejected("mixed impact conditions rejected", lambda: candidate_split(wrong, "1.2", "300", (2, 1, 1), "s", allow_fixture=True))
    rejected("insufficient matched shape groups rejected", lambda: candidate_split(fixture, "1.5", "300", (1, 1, 1), "s", allow_fixture=True))
    rejected("fixture download plan rejected", lambda: download_plan(fixture, ["yaris_x00001"], ["vtp"], 2000))
    # This copy remains synthetic: only the guard plumbing is tested, no request is made.
    planner_fixture = copy.deepcopy(fixture); planner_fixture["record_origin"] = "actual_metadata"
    rejected("explicit byte cap enforced without network", lambda: download_plan(planner_fixture, ["yaris_x00001"], ["vtp"], 999))
    plan = download_plan(planner_fixture, ["yaris_x00001"], ["vtp"], 1000)
    need(REVISION in plan["files"][0]["url"] and plan["planned_source_bytes"] == 1000, "Pinned allowlist plan failed")
    checks.append("allowlist plan uses exact pinned revision, exact file paths and measured-size contract")
    return {"status": "STDLIB_PREPARATION_SELFTEST_PASS", "record_origin": "SYNTHETIC_FIXTURES_ONLY",
            "checks_passed": len(checks), "checks": checks, "fixture_split": result,
            "network_requests": 0, "protected_data_acquired": False, "ml_imports": False,
            "fem_runs": 0, "training_runs": 0, "accuracy_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--manifest")
    validate.add_argument("--allow-fixture", action="store_true")
    preflight = sub.add_parser("preflight"); preflight.add_argument("--manifest", required=True)
    split = sub.add_parser("split"); split.add_argument("--manifest", required=True)
    split.add_argument("--thickness-mm", required=True); split.add_argument("--yield-mpa", required=True)
    split.add_argument("--counts", nargs=3, type=int, required=True, metavar=("TRAIN", "VALIDATION", "TEST"))
    split.add_argument("--seed", required=True); split.add_argument("--allow-fixture", action="store_true")
    split.add_argument("--out", required=True)
    plan = sub.add_parser("download-plan"); plan.add_argument("--manifest", required=True)
    plan.add_argument("--case-id", action="append", required=True); plan.add_argument("--role", action="append", required=True)
    plan.add_argument("--max-bytes", type=int, required=True); plan.add_argument("--out", required=True)
    test = sub.add_parser("selftest"); test.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        protocol = read_json(args.protocol)
        result = validate_protocol(protocol)
        if args.command == "validate" and args.manifest:
            result["manifest"] = validate_manifest(read_json(args.manifest), allow_fixture=args.allow_fixture)
        elif args.command == "preflight":
            manifest = read_json(args.manifest)
            validate_manifest(manifest)
            pending = [name for name, done in manifest["access"].items() if done is not True]
            if not manifest["cases"]:
                pending.append("actual case metadata")
            result = {"status": "PREPARED_NOT_EXECUTABLE", "metadata_gates_pending": pending,
                "implementation_gates_pending": protocol["implementation_gates"],
                "training_ready": False, "training_executed": False}
        elif args.command == "split":
            result = candidate_split(read_json(args.manifest), args.thickness_mm, args.yield_mpa, args.counts, args.seed, allow_fixture=args.allow_fixture)
            write_json(args.out, result)
        elif args.command == "download-plan":
            result = download_plan(read_json(args.manifest), args.case_id, args.role, args.max_bytes)
            write_json(args.out, result)
        elif args.command == "selftest":
            result = selftest(protocol)
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if args.command == "preflight" else 0
    except (PreparationError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc), "training_executed": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
