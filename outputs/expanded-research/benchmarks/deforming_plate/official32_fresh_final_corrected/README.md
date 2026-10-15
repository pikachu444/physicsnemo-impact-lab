# DeformingPlate 확대 학습·최종 시험 종료 기록

**실제 학습과 검증 6건, 새 시험 6건의 399단계 자동회귀는 완료했다. 정확도와 기하학 안정성은 확보하지 못했다.** 실행 완료를 해석 정확도 성공으로 취급하지 않는다.

## 고정한 실험

- 설치된 공식 PhysicsNeMo HybridMeshGraphNet: hidden 128, processor 15, activation checkpoint segments 0, 파라미터 3,605,636개.
- 원 DeepMind DeformingPlate 독립 궤적 train 32건 × 고정 anchor 8개 × 5 epoch, 총 1,280 Adam update. Noise 0.003, 초기 학습률 0.0001, update당 감소율 0.9999917.
- Train-only 정규화. Anchor는 `0,56,113,170,227,284,341,398`이며 각각 다음 단계를 학습했다.
- Epoch 1·3·5에 미리 고른 `valid_0000`, `valid_0005`를 각각 AR399로 평가했다. 실패 → 예측 요소 역전 여부 → 두 사례의 변위·응력 상대 L2 중 최댓값 → 앞선 epoch 순으로 선택했다.
- 원 시험 `test_0000`, `test_0001`은 앞선 개발 평가로 분리했다. 최종 시험은 고정 모델로 `test_0002..0007`에 각각 한 번만 수행했다. 최종 시험에는 teacher forcing을 사용하지 않았다. 알려진 actuator/clamp 경계 위치는 원 공식 추론 계약대로 처방했다.
- 이는 제한된 데이터·anchor·epoch와 해석적 noise 정규화를 쓰는 확대 실험이며, 원 논문의 전체 학습 recipe 재현이나 수렴 완료가 아니다.

## 모델 선택과 검증

| 후보 epoch | 사전 선택 key `(실패, 역전, 최악 상대 L2)` |
|---:|---|
| 1 | `(0,1,6.2183798503)` |
| 3 | `(0,1,6.1242159178)` |
| **5** | **`(0,0,4.0884963368)`** |

선택한 epoch 5 모델의 SHA256은 `c29e4ecd258553622ecee13fc054a58854950e667b3a838610ab52f799339429`다. 두 선택 진단에서 예측 요소 역전은 없었지만, 전체 validation에서는 다른 사례의 역전과 큰 오차가 나타났다. 선택 진단 2건의 결과를 전체 사례의 적합성으로 확대하지 않는다. 마지막 후보까지 최악 오차가 개선되어 수렴도 확인되지 않았다.

| 합산 지표 | Validation 6 | 최종 시험 6 |
|---|---:|---:|
| 자유 절점 변위 상대 L2 | 3.669985 | 2.412175 |
| 자유 절점 변위 성분 RMSE, 원 좌표 단위 | 0.0788845 | 0.0711330 |
| 영변위 기준 상대 L2 | 1.000000 | 1.000000 |
| 자유 절점 원 scalar stress 상대 L2 | 0.697753 | 0.682359 |
| 원 scalar stress RMSE, 원 응력 단위 | 23,959.8 | 29,905.0 |
| 학습 평균 상수 응력 기준 상대 L2 | 0.716115 | 0.707319 |
| 좌표 유도 GL strain 상대 L2 | 505.776 | 1293.454 |
| 좌표 유도 GL strain 성분 RMSE, 무차원 | 3.55702 | 11.3451 |

변위는 두 split 모두 영변위 기준보다 나빴다. 응력의 합산 오차는 학습 평균 상수 기준보다 조금 낮지만, 큰 변위·변형률 오차와 요소 역전이 남아 있어 공학 해석용 모델로 적합하다고 판정하지 않는다.

## 한 번 수행한 최종 시험의 사례별 결과

변위·응력은 399개 예측 단계 전체, GL strain은 고정된 21개 표본 단계에서 평가했다. 아래 역전 수는 역전 요소가 하나라도 있는 **단계 수 / 399**이며, 요소 개수 자체와 구분한다.

| 원 사례 | 변위 상대 L2 | 응력 상대 L2 | 유도 GL 상대 L2 | 유도 GL 성분 RMSE | 예측 역전 단계 | 원본 역전 단계 |
|---|---:|---:|---:|---:|---:|---:|
| test_0002 | 8.96174 | 0.717598 | 9374.649 | 22.9761 | 385 | 0 |
| test_0003 | 2.16639 | 0.682944 | 57.4380 | 0.311840 | 73 | 0 |
| test_0004 | 2.39485 | 0.663348 | 9.76844 | 0.158562 | 0 | 0 |
| test_0005 | 2.01097 | 0.647729 | 32.3339 | 0.392155 | 24 | 0 |
| test_0006 | 54.6968 | 0.887478 | 31.4262 | 0.154817 | 119 | 0 |
| test_0007 | 1.85775 | 0.676663 | 20.6621 | 0.186072 | 0 | 0 |

최종 시험 6건 모두 399단계를 계산했으며 실행 예외·NaN·OOM으로 분류된 실패는 없었다. 6건 모두 음수 예측 scalar stress 개수는 0이었다. 큰 상대 오차는 절대 RMSE와 함께 읽어야 한다. 원본과 예측의 매 단계 역전 수·최소 det(F)·원 그래프 간선 수는 사례별 JSON에 각각 보존했다.

## 응력·변형률의 해석 범위

공개 원 자료의 nodal von Mises scalar stress를 평가했다. 물리 단위는 독립 확인되지 않아 Pa/MPa로 임의 환산하지 않았다. 변위 역시 원 데이터 좌표 단위를 유지했다.

원 COMSOL strain tensor 출력은 공개 자료에서 확보하지 못했다. 여기의 Green–Lagrange strain은 실제 tetrahedron mesh와 FEM/AI 절점 좌표로부터 유도한 값이다. Actuator가 포함된 요소를 제외하고 tensor shear 성분 순서 `xx,yy,zz,xy,yz,xz`를 썼으며, 21개 표본 단계의 요소·성분을 비가중 합산했다. **원 solver-native strain 정답 또는 399단계 전체의 strain 검증으로 세지 않는다.** Validation `valid_0005`의 원 FEM 좌표에서도 74단계에 요소 역전이 관측되어 예측 역전과 별도 보존했다.

원 COMSOL deck은 확보되지 않았다. 이 공개 source benchmark를 OpenCourant 원 모델로 대체 재현했다고 주장하지 않는다.

## 시간·메모리와 실제 실행 조건

| 항목 | 학습 + 전체 validation | 고정 모델 최종 시험 |
|---|---:|---:|
| PID | 48016 | 33220 |
| 시작 UTC | 08:46:15.917903 | 09:31:50.708570 |
| 완료 receipt UTC | 09:24:55.893447 | 09:36:42.078322 |
| Import·gate·I/O 포함 wall seconds | 2319.9749 | 291.3698 |
| OS peak working set, bytes | 1,732,485,120 | 1,329,582,080 |
| 표본 host available 최저, bytes | 214,568,960 | 193,302,528 |
| Torch tensor allocation peak, bytes | 3,187,859,968 | 195,064,320 |
| Torch allocator reserved peak, bytes | 3,953,131,520 | 274,726,912 |

학습 기록의 `train_seconds=1613.3270`은 학습·epoch별 teacher 진단·선정 AR·checkpoint I/O를 포함하고, 마지막 전체 validation AR은 제외한다. 순수 optimizer 연산 시간으로 부르지 않는다. 원 판 protocol에는 숫자 RAM minimum gate가 없었다. 낮은 host available 최저값을 실행 무오류와 구분해 보존한다. 표본 간격은 0.2초이며 OS high-water와 표본 peak를 혼동하지 않는다. Windows commit/pagefile 값은 물리 RSS가 아니다.

RTX 3060 Laptop GPU에서 실행했으며 PhysicsNeMo 2.2.2, Torch 2.10.0+cu126, PyG 2.8.1, torch-scatter 2.1.2+pt210cu126을 사용했다. 매 단계 완전 반경 접촉 그래프를 다시 구성했고 임의 neighbor cap으로 줄이지 않았다. 실제 wall에는 머신 상태가 포함된다. Root의 light stdlib/AST/Git 작업 및 교정 window의 CPU-only ANTLR build와 시간 중첩 가능성이 있으므로 단독 통제 환경이나 slowdown의 단일 원인을 주장하지 않는다. 사용자 앱 종료·전력 설정 변경·새 유료 자원 집행은 하지 않았다.

## 사전 gate 실패와 최소 교정

원 final driver PID39388은 09:25:47 UTC에 0.0643초 만에 사전 assertion으로 중단됐다. Runtime config에 정규화되어 저장된 선택 epoch/사례 **목록**을 원 명령의 **문자열**과 직접 비교한 오류였다. 실패는 원본 `official32_fresh_final/`에 traceback과 함께 보존했다. `fresh_test_content_access_started=false`, 접근 marker 부재 및 해당 assertion의 preimport 위치가 그 실패에서 시험 배열·모델 추론을 쓰지 않았다는 근거다.

원 source 9개, v2 lock, 학습 checkpoint·정규화·protocol은 바꾸지 않았다. 새 `run_frozen_final_corrected.py`에서 두 비교만 고정 protocol 목록으로 정규화했다. 잘못된 epoch/순서/추가·누락/시험 사례 치환 등을 거부하는 15개 stdlib mock과 독립 review를 통과한 뒤 새 output에서 실제 최종 시험을 한 번 수행했다. 모델·과학 설정·선정 결과·학습량 변경은 없었다.

## 보존 자료와 closure 범위

- `final-attempt.json`, `metrics.json`: 원 실행 완료·자원 기록과 전체/사례별 지표.
- `pre-evaluation-lock.json`, `preexecution_correction_lock.json`: 실제 시험 접근 전에 고정한 모델·정규화·source·분할·교정 근거.
- `test_0002..0007_metrics.json`, `test_0002..0007_evaluation.npz`: 각각의 원 예측/정답 배열, 기하학·응력·유도 strain 지표. Raw prediction arrays를 덮어쓰지 않았다.
- `executed_*.py`: 원 source 9개와 실제 교정 driver snapshot.
- `closure-provenance.json`: source/split/config/norm/checkpoint/marker/metrics hash 검사, 사례 JSON과 합산 지표의 stdlib 수치 일치 검사, 저장 prediction NPZ SHA256.
- `gpu-release.json`: 중단 후 재개한 **13:25:03.673 UTC**에 원 session48506 exit0 회수, PID33220 부재, NVIDIA compute-app 목록 비어 있음을 확인한 근거. 09:36 완료 receipt와 이 늦은 관측 시각을 구분한다. GPU 소유권을 root에 반환했다.

최종 metrics SHA256은 `04eaa2a4c140b4da658da5b6c3a78a6bc948713b2429a95d95042cd4308eb8a8`이다. 원 v2 source lock은 `5992ae6deed73d0b6fc4476de35db6b8483d070400597cf9f0a350de5d7af640`, 교정 lock은 `84d27a69606a245ac5a78957b9bbca010daf05497aa0d80a05f266623c3afb70`이다.

이번 closure는 stdlib만 사용했고 추가 학습·신경망 추론·marker 삭제·시험 재시도는 0회다. 저장 NPZ는 SHA256만 확인했으며 지표를 배열에서 독립 재계산하지 않았다. 새 field PNG·hotspot 재읽기·raw-array 수치 감사도 아직 완료하지 않았으므로 완료한 것처럼 세지 않는다.
