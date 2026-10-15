"""Capture only small research sources/configuration; never import experiment code."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/expanded-research/reproduction"
AREA = ROOT / "outputs/expanded-research/reproduction-addendum"
LIMIT = 2 * 1024 * 1024
SOURCE_DIRS = ("expanded-batch", "expanded-drop", "expanded-plate", "expanded-temporal", "expanded-evaluation-audit")
ML_SOURCES = ("benchmark_data.py", "model_adapters.py", "budget_protocol.py", "budgeted_runtime.py", "prepare_budgeted_plan.py", "train_budgeted.py", "evaluate_budgeted.py", "run_budgeted_suite.py", "summarize_budgeted.py", "estimate_b1_training_cost.py", "prepare_phone.py", "prepare_multifield.py", "finalize_cache.py", "field_diagnostics.py", "clarify_phone_provenance.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def small(path):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError("Link source refused: " + str(path.relative_to(ROOT)))
    if path.stat().st_size > LIMIT:
        raise ValueError("Oversize source/config refused: " + str(path.relative_to(ROOT)))
    data = path.read_bytes()
    if len(data) > LIMIT:
        raise ValueError("Source grew beyond limit")
    return data


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def redact(value):
    if isinstance(value, str):
        # Preserve path meaning while removing the workstation account and root.
        value = re.sub(re.escape(str(ROOT)), "${WORKSPACE}", value, flags=re.I)
        value = re.sub(re.escape(ROOT.as_posix()), "${WORKSPACE}", value, flags=re.I)
        value = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+", "${USER_HOME}", value, flags=re.I)
        return value
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {redact(key): redact(item) for key, item in value.items()}
    return value


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def capture(output, as_of):
    if not output.resolve().is_relative_to(AREA.resolve()) or output.exists():
        raise ValueError("Choose a new, absent directory within reproduction-addendum")
    datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    files, defaults, profiles, observation = [], [], {}, []
    pending = []

    def add(source, destination, profile=None, role="current_source", expected=None):
        raw = small(source)
        if expected and sha(raw) != expected:
            raise ValueError("Frozen source hash mismatch: " + relative(source))
        if source.suffix == ".py":
            ast.parse(raw, filename=relative(source))
        data = encoded(redact(json.loads(raw))) if source.suffix == ".json" else raw
        if re.search(rb"[A-Za-z]:[\\/]Users[\\/](?!\$\{)[^\\/\s\"']+", data, flags=re.I):
            raise ValueError("Unredacted local account path: " + relative(source))
        if source.suffix != ".json" and data != raw:
            raise ValueError("Executed source bytes must be immutable")
        packaged = ("recorded/" + profile + "/" if profile else "") + destination
        if any(record["packaged_path"] == packaged for record in files):
            raise ValueError("Duplicate packaged path: " + packaged)
        record = {"packaged_path": packaged, "destination": destination,
                  "bytes": len(data), "sha256": sha(data), "source_origin": relative(source),
                  "source_bytes": len(raw), "source_sha256": sha(raw), "role": role,
                  "captured_as_of_utc": as_of, "json_token_redacted": data != raw if source.suffix == ".json" else False}
        files.append(record)
        pending.append((packaged, data, source, sha(raw)))
        if profile is None:
            defaults.append(packaged)
        return packaged

    def evidence(source, group, profile=None, expected=None):
        return add(source, "work/expanded-coordination/addendum-evidence/" + group + "/" + source.name,
                   profile, "configuration_or_provenance_json", expected)

    for name in ML_SOURCES:
        add(ROOT / "work/expanded-ml" / name, "work/expanded-ml/" + name)
    for directory in SOURCE_DIRS:
        for source in sorted((ROOT / "work" / directory).iterdir()):
            if source.is_file() and source.suffix in (".py", ".ps1"):
                add(source, relative(source))
    for source in sorted((ROOT / "work/expanded-temporal/sources").rglob("*")):
        if source.is_file() and source.suffix in (".py", ".yaml", ".yml"):
            add(source, relative(source), role="pinned_official_temporal_dependency")
    for name in ("phone-full-normalization.json", "bumper7-normalization.json", "budgeted-suite-plan.json"):
        add(ROOT / "work/expanded-ml" / name, "work/expanded-ml/" + name, role="current_configuration_json")

    config_paths = (
        "outputs/expanded-research/full-support-static-training-protocol.json",
        "outputs/expanded-research/benchmarks/deforming_plate/official32-validation-protocol.json",
        "outputs/expanded-research/benchmarks/deforming_plate/official32-completion-source-lock-v2.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/budget-v2-execution-plan.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/phone_full/budget_v2_complete_epoch/prepared-contract.json",
        "outputs/expanded-research/benchmarks/temporal_strategies/bumper/budget_v2_complete_epoch/prepared-contract.json",
        "outputs/expanded-research/benchmarks/batching/batch-recommendations.json",
        "outputs/expanded-research/benchmarks/deforming_plate/final-split-contract.json",
        "outputs/expanded-research/benchmarks/drop_test/500us/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume02/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume03/frozen-inputs-and-guards.json",
        "outputs/expanded-research/benchmarks/drop_test/500us-resume03/source-control-and-restart-audit.json",
    )
    for value in config_paths:
        source = ROOT / value
        if source.exists():
            evidence(source, "current-" + source.parent.relative_to(ROOT).as_posix().replace("/", "__"))

    for domain, folder in (("phone", "drop_test/models-full-c6"), ("bumper", "bumper/models-c7")):
        parent = ROOT / "outputs/expanded-research/benchmarks" / folder
        for run in sorted(parent.iterdir()) if parent.exists() else ():
            if not run.is_dir():
                continue
            train_path, test_path = run / "training_report.json", run / "test_report.json"
            if not (train_path.exists() and test_path.exists()):
                observation.append({"scope": domain + "/" + run.name, "status": "not_completed_as_of_capture", "path": relative(run)})
                continue
            train, test = json.loads(small(train_path)), json.loads(small(test_path))
            if test.get("status") != "completed" or test.get("test_performance_passes") != 1:
                observation.append({"scope": domain + "/" + run.name, "status": "test_not_completed_once", "path": relative(run)})
                continue
            profile = "fullsupport-" + domain + "-" + run.name
            overrides, source_checks = [], []
            if train["source_sha256"] != test["source_sha256"]:
                raise ValueError("Training/final source maps differ: " + relative(run))
            for name, expected in sorted(test["source_sha256"].items()):
                source = run / "source" / name
                destination = "work/" + ("expanded-batch/" if name == "batch_adapters.py" else "expanded-ml/") + name
                overrides.append(add(source, destination, profile, "exact_executed_source_snapshot", expected))
                source_checks.append({"name": name, "sha256": expected})
            for name in ("normalization.json", "prospective-training-protocol.json", "checkpoint-frozen.json", "train-command.json", "validation-command.json", "test-command.json", "training_report.json", "validation_fresh_process.json", "test_report.json", "data-provenance-clarification.json"):
                source = run / name
                if source.exists():
                    expected = test.get("frozen_evidence_sha256", {}).get(str(source))
                    overrides.append(evidence(source, profile, profile, expected))
            command = json.loads(small(run / "train-command.json"))["command"]
            cache_path = Path(command[command.index("--cache") + 1])
            overrides.append(evidence(cache_path, profile + "-cache", profile, command[command.index("--expected-manifest-sha256") + 1]))
            profiles[profile] = {"description": "Completed single final-test run; exact captured executed sources and token-redacted settings: " + domain + "/" + run.name,
                                 "status": "completed_actual_training_and_single_final_test", "result_origin": relative(run),
                                 "overrides": overrides, "executed_source_sha256": source_checks,
                                 "training_report_source_sha256": sha(small(train_path)), "test_report_source_sha256": sha(small(test_path)),
                                 "quality_gate_passed": test.get("quality_gates", {}).get("gate_passed"),
                                 "accuracy_or_convergence_claimed": False, "arrays_and_checkpoint_included": False}

    # These are source preservation groups, never completed experiment profiles.
    for scope, path in (("plate_completion", "outputs/expanded-research/benchmarks/deforming_plate/official32-completion-source-lock-v2.json"),
                        ("temporal_v2", "outputs/expanded-research/benchmarks/temporal_strategies/budget-v2-execution-plan.json")):
        source = ROOT / path
        value = json.loads(small(source))
        observation.append({"scope": scope, "status": value.get("status"), "evidence_origin": path,
                            "evidence_source_sha256": sha(small(source)), "completed_profile_created": False})
    final_summary = ROOT / "outputs/expanded-research/benchmarks/drop_test/500us-final/summary.json"
    observation.append({"scope": "phone_FEM_0.5ms", "status": "final_summary_present_review_separately" if final_summary.exists() else "no_final_summary_as_of_capture",
                        "completed_profile_created": False})

    destinations = [next(record["destination"] for record in files if record["packaged_path"] == key) for key in defaults]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Conflicting default destinations")
    for profile in profiles.values():
        targets = [next(record["destination"] for record in files if record["packaged_path"] == key) for key in profile["overrides"]]
        if len(targets) != len(set(targets)):
            raise ValueError("Conflicting profile destinations")
    manifest = {"schema_version": 1, "created_utc": as_of, "capture_scope": "Small stdlib source/config preservation addendum; no experiment execution",
                "base_bundle_unchanged": {"manifest_path": "outputs/expanded-research/reproduction/source-manifest.json", "manifest_sha256": sha(small(BASE / "source-manifest.json"))},
                "files": sorted(files, key=lambda row: row["packaged_path"]), "default_files": sorted(defaults),
                "profiles": profiles, "observations_as_of_capture": observation,
                "redaction": {"json_strings_and_keys": "Local workspace root -> ${WORKSPACE}; remaining Windows user homes -> ${USER_HOME}",
                              "executed_sources": "Exact bytes, never redacted; refused if account paths present", "hashes": "sha256 is packaged/public bytes; source_sha256 is original bytes"}}
    output.mkdir(parents=True)
    try:
        for name, data, source, expected in pending:
            if sha(small(source)) != expected:
                raise ValueError("Source changed during capture: " + relative(source))
            write_new(output / name, data)
        original_bootstrap = small(BASE / "bootstrap.py")
        marker = b"    return manifest,records\n"
        conflict_check = b'''    groups=[manifest["default_files"]]+[p["overrides"] for p in manifest["profiles"].values()]
    for keys in groups:
        destinations=[records[key]["destination"] for key in keys]
        if len(destinations)!=len(set(destinations)):
            raise ValueError("Conflicting destination within a manifest selection")
    return manifest,records
'''
        if original_bootstrap.count(marker) != 1:
            raise ValueError("Original restoration helper changed; review before reusing")
        write_new(output / "bootstrap.py", original_bootstrap.replace(marker, conflict_check))
        write_new(output / "source-manifest.json", encoded(manifest))
        write_new(output / "README.md", README.encode("utf-8"))
        write_new(output / "capture-tool.py", Path(__file__).read_bytes())
        write_new(output / "selfcheck-tool.py", small(ROOT / "work/expanded-coordination/addendum_selfcheck.py"))
        write_new(output / "licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt", small(BASE / "licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt"))
        index = []
        for path in sorted(output.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                index.append({"path": path.relative_to(output).as_posix(), "bytes": len(data), "sha256": sha(data)})
        write_new(output / "package-manifest.json", encoded({"schema_version": 1, "as_of_utc": as_of, "files": index,
                  "bootstrap_source_origin": "outputs/expanded-research/reproduction/bootstrap.py", "bootstrap_original_sha256": sha(original_bootstrap),
                  "bootstrap_change": "Added explicit duplicate destination rejection in each default/profile selection; original restoration safety logic retained",
                  "capture_tool_source_origin": relative(Path(__file__)), "manifest_self_hash_excluded": True,
                  "selfcheck_tool_source_origin": "work/expanded-coordination/addendum_selfcheck.py",
                  "license_source_origin": "outputs/expanded-research/reproduction/licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt",
                  "acceptance_report": "Generated separately by addendum_selfcheck.py after actual verification"}))
    except Exception:
        # Keep the partial capture for diagnosis; a later run must choose a new path.
        raise
    return {"status": "captured", "as_of_utc": as_of, "files": len(files), "profiles": sorted(profiles), "bytes": sum(row["bytes"] for row in files),
            "output": output.relative_to(ROOT).as_posix(), "experiment_code_executed": False}


README = """# 추가 실행 소스 복원

원본 `../reproduction/`(169 source files / 22 historical profiles)을 수정하지 않고 후속 full-support 정적 학습, 배치 검사, Phone 0.5ms FEM 제어·분석, Plate completion, Temporal V2, 독립 수치 감사의 작은 실행 소스·설정을 보존했다. `source-manifest.json`의 `created_utc`가 관측 시점이다. `profiles`에 있는 완료 실행만 실제 학습과 최종 시험 완료로 분류한다. Plate·Temporal·FEM은 `observations_as_of_capture`를 읽는다. 완료 profile도 정확도·수렴 성공을 뜻하지 않는다.

## 별도 빈 workspace에서 복원

Python 표준 라이브러리만 필요하다. 이 폴더의 `bootstrap.py`는 원본 번들의 안전 복원 코드를 재사용한 독립 실행본이며, 같은 선택 내 복원 목적지가 충돌하면 명시적으로 거부하는 검사만 추가했다. 원본/public bootstrap hash와 변경 내용은 `package-manifest.json`에 기록했다.

```powershell
python outputs/expanded-research/reproduction-addendum/bootstrap.py --verify-only
python outputs/expanded-research/reproduction-addendum/bootstrap.py --list-profiles
python outputs/expanded-research/reproduction-addendum/bootstrap.py --workspace <EMPTY_DIRECTORY> --profile fullsupport-phone-transolver --dry-run
python outputs/expanded-research/reproduction-addendum/bootstrap.py --workspace <EMPTY_DIRECTORY> --profile fullsupport-phone-transolver
```

하나의 빈 폴더에 하나의 profile을 먼저 선택한다. `current`는 관측 시점의 소스 모음이며 완료 실험의 동결 실행본을 대신하지 않는다. 원본 번들과 같은 대상에 연속 복원하면 중복 파일 때문에 거부될 수 있으므로 각각 별도의 빈 폴더를 쓴다. 두 묶음을 합치는 자동 덮어쓰기 기능은 없다. 대상 파일은 내용이 같아도 덮어쓰지 않으며 경로 이탈·symbolic link·junction·환경/데이터 경로를 차단한다.

실행된 `.py`와 `.ps1`은 원본 바이트 그대로다. 모든 복사 경로, 원 위치, 대상 위치, 원본 SHA256, 공개 SHA256, profile 관계는 manifest에 기록한다. JSON 설정의 계정 경로는 `${WORKSPACE}`와 `${USER_HOME}` token으로 가렸으므로 설정은 기록용 template이다. 새 장비에서 해당 token과 로컬 환경/데이터 경로를 명시적으로 바인딩하고, 영향을 받는 설정 hash를 새 실행용으로 재고정해야 한다. 과거의 hash를 변경된 설정에 붙이면 안 된다. 동결 완료 profile은 당시 `source/` snapshot의 hash를 최종 `test_report.json`의 source hash와 맞춘다. `checkpoint-frozen.json`의 model-config/normalization/contract hash는 canonical JSON 객체 hash이며 파일 byte hash와 다를 수 있다. 공식 NVIDIA wrapper의 저작권 header와 `licenses/` Apache-2.0 고지는 유지했다. 코드 라이선스를 FEM 모델·데이터에 적용하지 않는다.

## 보존 범위와 제한

데이터·raw FEM deck·예측 배열·학습 checkpoint·solver binary·가상환경·패키지·비밀값은 포함하지 않는다. `work/`의 원래 상대 구조로 복원하지만 framework 설치·upstream 코드·데이터·solver·재시작 checkpoint는 각 연구 안내에 따라 별도로 확보해야 한다. 이 패키지는 신규 환경 설치부터 학습까지의 독립 재실행을 입증하지 않는다. GPU·framework import·데이터 로드·FEM·학습·추론은 packer/복원 검사에서 실행하지 않는다. FEM 제어 소스가 있어도 실제 0.5ms 종료나 중간 재시작 경계의 bitwise 일치를 뜻하지 않는다.

`bootstrap-acceptance.json`은 source bytes/hash, 모든 Python AST, 빈 임시 폴더 복원, overwrite 거부와 경로 검사를 실제 수행한 결과다. 검증 시점 이후 새 결과는 이 capture에 자동으로 추가되지 않는다.

## 새 결과를 보존할 때

원 workspace에서 `work/expanded-coordination/addendum_pack.py --as-of <UTC_ISO_TIME> --output outputs/expanded-research/reproduction-addendum/captures/<NEW_NAME>`으로 **없는 새 경로**를 선택한다. 같은 입력·같은 `--as-of`는 같은 manifest/바이트를 만든다. 기존 capture와 실험 source는 덮어쓰지 않는다. 완료된 full-support `test_report.json`만 새 profile로 검출하고, snapshot hash가 맞지 않으면 거부한다. 생성 후 `addendum_selfcheck.py --package <NEW_CAPTURE>`를 실행한다. 공개 저장소에서는 보존된 `capture-tool.py`가 원 capture 코드를 설명하며 원 workspace의 입력 없이는 refresh할 수 없다.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="Explicit ISO UTC capture time; held fixed for deterministic refresh")
    parser.add_argument("--output", type=Path, default=AREA)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(capture(output, args.as_of), indent=2))


if __name__ == "__main__":
    main()
