# 실행 소스 복원

공개 결과를 만드는 데 사용한 실행 코드·helper·설정 파일을 원래 `work/` 구조로 복원하는 묶음이다. 데이터, 가상환경, 설치된 패키지, solver, raw FEM deck, 학습 가중치와 큰 예측 배열은 포함하지 않는다. 이 묶음의 검증은 **소스 보존·복원 검증**이며 새 clone에서 환경 설치부터 전체 학습까지 재실행했다는 뜻이 아니다.

## 새 clone에서 사용

공개 저장소의 `outputs/` 트리를 유지한 새 clone 루트에서 실행한다. Python 표준 라이브러리만 사용한다.

```powershell
python outputs/expanded-research/reproduction/bootstrap.py --verify-only
python outputs/expanded-research/reproduction/bootstrap.py --workspace . --dry-run
python outputs/expanded-research/reproduction/bootstrap.py --workspace .
```

`reproduction/work/expanded-ml/...`은 새 clone의 `work/expanded-ml/...`로 복사된다. `Path(__file__).parents`에 의존하는 기존 코드가 원래 깊이와 형제 파일을 찾도록 하는 절차다. **기존 파일은 내용이 같아도 덮어쓰지 않는다.** 원래 연구 workspace에서 실행하면 기존 `work/` 파일 때문에 거부되는 것이 정상이다. 출력된 `restored`는 코드 복원 성공만 뜻한다. 이후 각 연구 README의 데이터 재획득·FEM 생성·환경 구성 절차를 따른다.

## 기록된 실행 버전 선택

현재 helper와 과거 실행 helper가 다른 경우를 확인했다. 특히 초기 변위 전용 실행, 후속 다물리량 실행, POD, plate의 버전을 섞으면 같은 실행을 재현하지 못한다. 과거 버전은 `recorded/`에 별도로 보존했고, 원래 파일명으로 복원한다.

```powershell
python outputs/expanded-research/reproduction/bootstrap.py --list-profiles
# 아직 work 소스를 복원하지 않은 별도 새 clone에서 한 가지를 선택한다.
python outputs/expanded-research/reproduction/bootstrap.py --workspace . --profile pod-coupon13
```

Profile은 기본 소스 묶음 중 필요한 파일을 해당 실행의 동결본으로 선택한다. 먼저 `current`를 복원한 뒤 같은 clone에 다른 profile을 덮어쓰는 기능은 제공하지 않는다. 다른 실행 버전에는 별도 clone을 사용한다. 신경망 profile은 `run_benchmark.py`, `benchmark_data.py`, `model_adapters.py` 세 형제 파일을 함께 복원한다. POD 초기 curated3 실행은 `expanded-ml` 쪽 helper를 찾는 코드까지 반영했다.

Profile 존재는 모델의 정확도 성공을 뜻하지 않는다. Probe, 학습 실행, 실패 결과의 판정은 각 모델 결과 JSON과 독립 검토 기록을 읽어야 한다. 실행 옵션·분할·정규화·정답 데이터의 버전도 해당 결과와 맞춰야 한다.

## 보존 범위와 확인 결과

[source-manifest.json](source-manifest.json)은 각 파일의 원 위치·복원 위치·bytes·SHA256, 공식 commit·Git blob SHA, profile 대응을 기록한다. 공식 crash/drop/plate 예제는 pin `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`의 파일과 대조했다. 빠져 있던 utils·reader·Hydra 설정, plate의 원 training·download 코드를 함께 보완했다. 공식 NVIDIA 파일의 저작권 header와 [Apache-2.0 고지](licenses/NVIDIA-PhysicsNeMo-APACHE-2.0.txt)를 유지하며, 이 코드 라이선스를 FEM 모델·데이터에 적용하지 않는다.

실제 확인한 항목은 모든 소스의 bytes/hash, Python 문법의 정적 파싱, 새 폴더에 원래 구조로 복원, 기록된 POD 버전의 복원, 기존 파일 덮어쓰기 거부, 경로 이탈·환경/데이터 경로 차단이다. [복원 검사 결과](bootstrap-acceptance.json), [정적 의존성 감사](dependency-audit.json)에 범위를 기록했다. Framework import, 추가 데이터 다운로드, FEM, GPU, 모델 학습은 이 패키지 통합 검사에서 실행하지 않았다.

## 남아 있는 재현 전제

- 환경 설치 스크립트는 기존 공통 Python 패키지와 앞선 GPU 가상환경을 참조한다. 가상환경을 복사한 묶음이 아니므로 [기록된 환경 안내](../benchmarks/bumper/environment/README.md)에 따라 호환 환경을 별도로 구성해야 한다. 완전히 독립적인 신규 설치는 아직 검증하지 않았다.
- 배열과 가중치를 제외했으므로 과거 저장 모델의 즉시 추론 배포본이 아니다. POD·RBF의 저장 predictor도 별도 sidecar 배열이 필요하다. 재획득/생성 후 재학습하거나 로컬 보존본을 명시적으로 사용한다.
- Temporal의 `summarize_execution.py`는 과거 probe JSON의 절대 설치 경로를 읽는 출처 감사 코드다. 다른 장비에서 그 과거 경로를 그대로 재사용할 수 없다. 현재 학습 driver 복원과 이 과거 출처 감사의 이식성은 구분한다. 새 환경의 실제 모델 source 위치와 hash를 새 실행 기록으로 확인해야 한다.
- 공식 crash의 기본 `conf/config.yaml`은 해당 pin에 없다. 공식 training을 직접 호출하려면 README의 실제 recipe에 맞는 `--config-name bumper_geotransolver_oneshot` 같은 명시적 설정을 선택해야 한다. 소스와 config 보존은 그 예제의 데이터·호환성·정확도 검증 완료와 다르다.
- Plate 보조 분석과 일부 취득기는 clone 루트를 현재 작업 폴더로 가정한다. `work/` 소스를 `reproduction/` 안에서 직접 실행하지 않는다.

기존 수치 결과와 초기 공개 증거 파일은 이 통합 작업에서 수정하지 않았다. Git commit·push와 공개 사본 생성은 상위 작업에서 수행한다.
