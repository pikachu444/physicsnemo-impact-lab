# 구조 예제 실행 범위 감사

기준 시각: **2026-10-15T06:48:04.020404+00:00**. 현재 저장된 완료 JSON을 읽은 시점의 snapshot이다. 후속 실행은 자동으로 이 문서에 포함되지 않는다. Python 표준 라이브러리만 썼고 배열 열람·배열 hash·ML import·GPU query·학습·FEM을 하지 않았다. 새 파일은 이 coverage-audit 폴더에만 만들었다.

## 유한 범위와 집계

고정 공식 pin `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`의 현행 행렬은 **20행(기성 신경망 recipe8, 추가 설정11, 수치 baseline1)**이며 archived Sym3행은 별도다. 범퍼/BIW/시간전략을 나눈 설정 행 수로서 서로 다른20알고리즘이 아니다. “모든 알고리즘 완료”라는 결론은 이 근거에서 나오지 않는다. [원 행렬](../source-matrix/official-execution-matrix.json), [원 조사와 실제 실행 교차표](../source-matrix/executed-evidence-crosswalk-2026-10-15.md)

| 집계 단위 | 완료 수 | 중복 방지 해석 |
|---|---:|---|
| Coupon 정적 neural 학습 artifact | 6 | class5개; GALE_FA는 GeoTransolver 변형 |
| Coupon 시간전략 학습 artifact | 3 | 같은 GeoTransolver에 TC/AR/TF; 새 backbone0 |
| Plate 축소 학습 artifact | 2 | 같은 HybridMeshGraphNet의 noise/정규화 민감도; 정확도 실패 |
| 현재 full-support main 학습 / test report | 2 / 2 | 아래12개 상태행에서 완료 metadata만 셈 |
| 이 한정 ledger의 neural 학습 artifact | 13 | reload·probe·evaluation은 학습 수에 더하지 않음 |
| 학습된 named implementation class | 6 | Transolver/FLARE/GeoTransolver/MGN/FIG/HybridMGN; 독립 알고리즘6개라는 뜻이 아님 |
| 이 보고서의 큰 backbone 묶음 | 5 | HybridMGN을 MGN 묶음에 포함한 명시적 분류 |
| 최종 ROM predictor fit | 12 | 3방법×4support, 실제 물리 사례군은 coupon/bumper/phone3개 |
| 최종 ROM train POD basis fit 그룹 | 4 | support마다 공유 decomposition; RBF/Ridge·rank·reload로 재집계하지 않음 |
| 과거 curated3 predictor fit | 3(별도) | 같은 bumper의 변위-only 단계; 새 알고리즘0 |
| B1 / B2·B5 probe | 12 / 24시도 | 본학습 수에서 제외; batch22pass/1OOM/1OSguard |
| 공식 Q1 numerical solve | 2 | 독립13검사 통과; AI학습0 |

Seed가 다르면 반복 run이고 새 알고리즘이 아니다. 현재 coupon·Plate·확인된 main은 단일 seed20261015이며, 추가 coupon seed20261016은 제안된 FEM 설계일 뿐 실행한 학습이 아니다. POD+RBF/POD+Ridge는 같은 train basis를 공유하고, phone rank4의 POD+RBF와 directRBF는 양수 mode를 모두 써 여기서는 거의 같은 predictor다. Phone rear도 같은9FEM조건의 부분 support다. 집계는 범위가 다른 수치를 하나의 “완료 알고리즘” 합계로 더하지 않는다. [Coupon 비교 규약](../benchmarks/structural_coupon/static-six-model-comparison.json), [ROM 기록](../benchmarks/rom/README.md), [세부 ledger](coverage-audit-2026-10-15.json)

## 예제별 설치·검산·학습·FEM평가·fresh reload·감사

| 예제/범위 | 설치/기능 역전파 | 실제 학습 | FEM 정답 평가 | fresh reload | 독립 감사 |
|---|---|---|---|---|---|
| Coupon919nodes C7 정적6설정 | 완료 | 6완료 | test4조건 완료 | 6개 full test 완료 | 수치기록 검산 완료; 정확도 한계 별도 |
| Coupon TC7 / AR3 / TF3 | 완료; AR7/TF7은 실제 차원 실패 | 각30epoch | test4조건 자유추론 완료 | 세 모델 차이0 | 완료; TC응력/AR·TF rollout 정확도 한계 |
| Bumper13,675nodes C7 정적6설정 | B1·배치 기능 완료 | main완료는 아래행만 | ROM과 neural 분리 | main아래행만 | 단계별 기록 |
| Phone76,065corner C6 정적6설정 | B1·배치 기능 완료 | main완료는 아래행만 | main아래행만 | 별도 test process | Transolver pass; FLARE 수치감사 보류 상태 보존 |
| ROM coupon/bumper/phonefull/rear | CPU regression 실행, neural backward 해당없음 | predictor12fit | 완료 | 12완료 | 완료, 물리 정확도 한계 별도 |
| Plate 축소48×3 두 설정 | 완료 | 두 fit 완료 | 399step 완료, 유용정확도 실패 | pilot20step만 별도process; noise0 별도증거 없음 | 수치·kinematics 완료 |
| Plate 공식128×15 | Adam5회 자원검사 | 본학습 완료증거 없음 | 없음 | 없음 | 자원·준비 정적감사만 |
| TCAD Q1 elasticity | CPU2문제 solve; neural 아님 | 해당없음 | 독립 해석·평형·에너지13검사 | neural reload 해당없음 | 완료, full2×2DDM 제외 |
| Sinter / BIW / archived Sym | recipe/class 존재와 runtime 설치 구분 | 해당 데이터 학습 미완료 | 미완료 | 미완료 | source audit만 |
| 구조 FNO / full FNO-DDM | 공식 elasticity는 numerical path | 구조 FNO 학습 없음 | fullDDM 미실행 | 없음 | Q1 component만 검산 |

위 완료는 고정 형상·작은 조건축·지정된 target/support의 완료다. CUDA3Adam 기능검사를 학습 정확도로, 수치감사 통과를 제품 유용정확도로, Plate 좌표 유도 GL strain을 원 COMSOL strain 출력 검증으로 바꾸지 않는다. 원본 GPU 환경의 설치2.2.2/Torch2.10과 원 조사 Torch2.13/CPU Q1 환경도 구분한다. [독립 검토](../independent-review.md), [Plate](../benchmarks/deforming_plate/README.md), [시간전략](../benchmarks/temporal_strategies/execution-summary.json), [Q1](../benchmarks/elasticity_numerical/execution-review.md), [배치](../benchmarks/batching/merged-summary.json)

### 진행 중 main 작업의 metadata snapshot

| 도메인/설정 | 학습 완료 | FEM test 완료 | reload 근거 | 독립 감사 상태 |
|---|---|---|---|---|
| phone/static/transolver | True | True | completed_separate_test_process_selected_checkpoint_load; test_pass_is_not_an_extra_fit | passed |
| phone/static/flare | True | True | completed_separate_test_process_selected_checkpoint_load; test_pass_is_not_an_extra_fit | metadata_passed_metrics_deferred_memory |
| phone/static/geotransolver | False | False | not_yet_proven | pending_actual_completed_test |
| phone/static/geoflare | False | False | not_yet_proven | pending_actual_completed_test |
| phone/static/meshgraphnet | False | False | not_yet_proven | pending_actual_completed_test |
| phone/static/figconvunet | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/transolver | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/flare | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/geotransolver | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/geoflare | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/meshgraphnet | False | False | not_yet_proven | pending_actual_completed_test |
| bumper/static/figconvunet | False | False | not_yet_proven | pending_actual_completed_test |

`False`는 이 조사에서 완료 metadata를 읽지 못했다는 뜻이다. 준비/디렉터리 존재를 성공으로 셈하지 않는다. FLARE의 `metadata_passed_metrics_deferred_memory`는 test 실행은 끝났으나 독립 metric 재계산이 메모리 때문에 보류됐다는 뜻이다. main audit도 이 snapshot 이후 갱신될 수 있다. [현재 독립 검토](../full-support-evaluation-review.json); 각 실행 파일·읽은 시각·소형 metadata SHA는 JSON의 `evidence_files_at_read`에 있다.

## 미실행·제한 범위와 현실적인 해결 경로

아래 비용 판단은 새 GPU나 유료 계산이 꼭 필요한지에 대한 소스 기반 가능성 판단이며, 추가 작업을 실행한 결과나 자원 보장은 아니다. 저장된 pin 코드와 원논문 조사가 충분해 최신 외부 가용성 주장을 새로 만들지 않았다.

### Sinter / VFGN

**미완료:** 해당 소결 데이터의 학습·정답 평가 근거가 없다. 고정 recipe는 train/test TFRecord·metadata·checkpoint의 로컬 경로를 가리키지만, 확인한 자료에서는 검증된 공개 다운로드를 확보하지 못했다. 공개 데이터가 어디에도 없다는 단정은 아니다. 모델은 온도·좌표 이력으로 가속도와 다음 위치를 예측하며 응력·변형률 head가 없다. 저장된 source audit에는 target rank와 scheduler API 호환 문제도 있다.

**새 GPU 없이 해결할 수 있는 범위(추론):** 기존 coupon/범퍼/phone 자료는 온도 의존 소결 이력을 대신하지 못한다. 작은 합성 그래프의 CPU 기능 검사는 별도 구현 후 가능할 수 있지만 소결 학습·물리 정확도를 채우지 않는다. 실제 소결 검증에는 적절한 온도·좌표 시계열 정답이 여전히 필요하다.

근거: [공식 recipe](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/additive_manufacturing/sintering_physics/README.md#L123-L171), [모델과 입력 검증](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/models/vfgn/graph_network_modules.py#L1078-L1157), [원논문의 데이터 설명](https://arxiv.org/html/2404.11753v1#S3.SS1), [저장된 source audit](../source-matrix/sinter-sym-notes.md).

### BIW crash

**미완료:** GeoTransolver/GALE_FA one-shot 설정은 있으나, 확인한 자료에서 원 BIW FEM deck와150조건 정답의 검증된 공개 다운로드를 확보하지 못했다. 다른 구조에서 같은 class가 동작한 사실은 BIW 데이터·형상·하중 검증이 아니다.

**새 GPU 없이 해결할 수 있는 범위(추론):** 현재 공개 구조 예제만으로 원 BIW 과제를 완료했다고 할 수 없다. 공개 thin-shell 사례를 대체 사용하면 새 benchmark이며 별도 입력·정답 adapter가 필요하다. 새 GPU 구매가 데이터 부재를 해결하지는 않는다.

근거: [공식 BIW 설정](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/conf/crash_geotransolver_oneshot.yaml), [소스 조사에서 연결한 원논문](https://arxiv.org/abs/2510.15201), [공개자료 확인 범위](../source-matrix/official-execution-matrix.json).

### Archived Sym bracket / panel / plane displacement

**미완료:** 별도 archived 저장소의 구 Solver/Domain/PDE API이며 current core 설치만으로 실행 환경이 충족되지 않는다. 기본6×512 network·100만/200만 step의 로컬 자원 측정도 없다. Bracket의 상용 solver 참조 supplemental 자료 가용성은 확인하지 않았고 pretrained checkpoint와도 다르다.

**새 GPU 없이 해결할 수 있는 범위(추론):** 작은 평면 해석영역의 CPU PINN/VPINN과 해석해·Q1 참조 검증은 가능한 개발 경로다. 구 환경을 분리해 설치하거나 current API로 이식하고 실제 학습·검증을 해야 한다. 원 bracket/panel 전체 recipe나 동적 충격·소성 재현을 뜻하지 않는다.

근거: [공식 migration](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/v2.0-MIGRATION-GUIDE.md#L206-L227), [archived bracket](https://github.com/NVIDIA/physicsnemo-sym/blob/0c5f18c31d03d46ec0cc3ef4ccbf761a956dd38c/examples/bracket/bracket.py), [plane displacement VPINN](https://github.com/NVIDIA/physicsnemo-sym/blob/0c5f18c31d03d46ec0cc3ef4ccbf761a956dd38c/examples/plane_displacement/plane_displacement.py), [구체적 API·자원 조사](../source-matrix/sinter-sym-notes.md).

### 구조 FNO와 full FNO-DDM

**미완료:** 이 pin의 elasticity 경로는 Q1 FEM/BiCGSTAB·분할해법이며 elasticity FNO를 생성하지 않는다. Thermal/FNO 코드의 간접 import를 구조 신경망 학습으로 셀 수 없다. 완료한 것은 Q1 component의 CPU2문제이지 전체2×2 DDM이 아니다.

**새 GPU 없이 해결할 수 있는 범위(추론):** 자체 제조해를 쓰는 numerical DDM에는 외부 데이터가 필수는 아니므로, 의존성·RAM을 확인한 뒤 제한된 CPU 실행이 가능해 보인다. Q1으로 새 라벨을 만들고 작은 CPU FNO를 학습하는 일도 새 연구로 설계할 수 있으나, 추가 데이터·architecture·학습 구현이 필요하며 현재 완료 항목은 아니다.

근거: [원 Q1 solver](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L43-L47), [자체 제조해·분할 예제](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py#L92-L190), [공식 README](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/README.md#L121-L141), [실제 실행과 한계](../benchmarks/elasticity_numerical/execution-review.md).

### 공식 크기 Plate와 신규 Phone/Bumper 시간전략

**미완료:** Plate128×15는 실제 Adam5회 자원검사까지 완료했고, official32 protocol·정적감사는 본학습 완료가 아니다. 새 domain의 prepared contract도 실제 학습·평가가 아니다.

**새 GPU 없이 해결할 수 있는 범위(추론):** 공개 Plate 결과와 로컬 FEM 라벨이 있어 데이터 부재보다 host RAM·실측 실행시간·동결 규약의 실제 수행이 남았다. 기존 GPU에 공식 크기 probe는 들어갔지만 수렴·정확도는 미확인이다. CPU 또는 현재 보유 GPU의 순차 실행은 새 하드웨어 비용 없이 검토할 수 있으나 시간·구현 비용이 없다는 뜻은 아니다.

근거: [공식 HybridMeshGraphNet 생성](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/deforming_plate/train.py#L136-L145), [원 MeshGraphNets 논문](https://arxiv.org/abs/2010.03409), [기존128 probe](../benchmarks/deforming_plate/probe128_default/probe.json), [새 실행 전 감사](../benchmarks/deforming_plate/review/official32-preexecution-audit.json).

## 보존과 한계

[기계 판독 감사](coverage-audit-2026-10-15.json)에 원 공식20행과 archived3행, neural/ROM artifact ledger, 완료 판단 규칙, metadata 읽기 시각·해시·근거 경로를 기록했다. 행렬의 recipe 수와 actual fit 수는 분모가 달라 완료율로 나누지 않는다. 공식 full recipe를 변경 없이 재현했다고 주장하지 않으며, 기존 독립 수치검산 결과를 인용했을 뿐 이번 감사에서 재계산하지 않았다. 전체 README·source·suite·cache·Git·Page는 수정하지 않았다.


## 후속 계획 변경 메모 — 완료 수에는 0건 추가

총괄의 직접 통지에 따라, Phone/Bumper 본학습은 각각 TC full C6/C7와 **새 V3 native scalar response-head AR/TF**의 세 전략, 합계6개 run을 구현하는 계획으로 구체화됐다. 원 공식 AR/TF의 C3-only 제한, 기존 shape 실패와 coupon C3 학습 근거는 그대로 보존한다. 향후 계획을 C3-only 본학습 예정으로 읽으면 안 된다.

V3는 앞3개 acceleration만 geometry 적분에 쓰고 나머지 native scalar response를 별도 readout한다. Phone의 VM/strain Frobenius/EPSP, Bumper의 상·하층 VM/EPSP를 직접 다루는 연구 확장이며 **수정 없는 공식 recipe 재현이 아니다**. TF는 train에서만 GT geometry를 사용하고 eval target 값을 쓰지 않으며, 동일 frozen data·norm·internal validation·checkpoint freeze·final test1회 규약을 따를 계획이다.

이 메모 수신 시점에 GPU 실행은 시작하지 않았다는 통지이며, 이 감사의 as-of 완료 ledger나 counts에는 한 건도 더하지 않았다. 세부 계획 수신시각은 JSON의 `planned_temporal_extension_addendum`에 있다. 이후 실제 완료 여부는 새 실행 metadata로만 갱신한다.
