# PhysicsNeMo Impact Lab

공개 FEM 데이터와 직접 실행한 OpenCourant 해석으로 NVIDIA PhysicsNeMo의 변위·응력·변형률 예측을 비교하는 연구 저장소다.

**학습·예측·FEM 대비 오차 검산까지 실제로 수행했다. 현재 결과는 제품 낙하·충격 FEM을 대체할 정확도를 확보하지 못했다.** 고정 형상에서는 회귀·축약 모델이 유리한 경우가 있었고, 형상·메시가 달라지는 판재 시험에서는 큰 오차와 예측 요소 역전이 발생했다.

→ **[결과 요약과 알고리즘 비교 읽기](docs/RESULTS_KO.md)** — 무엇을 실행했고, 어떤 물리량을 얼마나 맞췄는지 먼저 확인한다.

## 어디까지 확인했나

| 대상 | 형상·메시 범위 | 실제 학습·평가 | 평가 물리량 | 현재 판단 |
|---|---|---|---|---|
| DP600 인장 시편 | 고정 형상, 13조건 | 정적 신경망 6설정 + 시간 전략 3설정, 회귀·축약 비교 | 변위·응력·총변형률·EPSP | 작은 조건 범위의 보간 결과 |
| 휴대폰 17부품 조립체 | 고정 형상, 9조건 | 정적 신경망 6설정, 회귀·축약 비교 | 변위·응력·총변형률·EPSP | 초기 20µs만 평가; 신경망은 평균장 기준선보다 열세 |
| 범퍼 충돌 | 고정 형상, 131조건 | 정적 신경망 6설정, 회귀·축약 비교 | 변위·shell 상하면 응력·EPSP | 변위보다 응력·소성변형률 오차가 큼 |
| DeformingPlate | 형상·메시가 달라지는 공개 궤적 | HybridMeshGraphNet 계열 3설정 | 변위·응력·좌표 유도 변형률 | 공식 크기 최종 시험: 변위 **241.217%**, 응력 **68.236%**, 예측 역전 **4/6조건** |
| Q1 탄성·수치 DDM | 별도 수치 FEM 문제 | CPU 해법 실행·검산 | 변위·미소변형률·응력 | 수치 해법 확인; AI 학습 결과와 구분 |

후속 신경망 학습·평가 산출물은 **24개**이며 알고리즘 24종을 뜻하지 않는다. 휴대폰·범퍼의 후속 시간 전개 6설정은 본학습 완료 0개다. **동일 형상의 메시 민감도 시험은 수행하지 않았다.** 판재의 형상 변화 평가를 다른 모든 알고리즘의 형상 일반화 검증으로 확대하지 않는다.

새 복잡 형상 캠페인의 **[문제 구성과 실행 전 설정](outputs/expanded-research/geometry-next/README.md)**을 준비했다. SHIFT-Crash 차량의 형상 변경, 5계열·6설정과 회귀·축약 기준선, 형상별 분할, AI 입력 해상도와 FEM 재메시 시험을 구분했다. 데이터 접근 승인과 새 로더 구현은 남아 있으며 **새 모델 학습·정확도 결과는 없다.**

## 대표 비교 그림

[![고정 형상의 휴대폰·범퍼에서 동일 평가 조건으로 비교한 실제 오차](outputs/expanded-research/revisions/v2/full-support-algorithm-comparison.png)](outputs/expanded-research/revisions/v2/full-support-algorithm-comparison.png)

작을수록 낮은 오차다. 열마다 로그 축과 물리량이 다르다. 변위에서 평균 병진을 제거한 오차도 함께 본다. [수치·조건 설명](docs/RESULTS_KO.md#휴대폰과-범퍼의-정적-알고리즘-비교) · [판재의 FEM·예측·오차 그림](outputs/expanded-research/benchmarks/deforming_plate/review/official32-first-fresh-field-comparison.png)

## 실행 환경과 비용

실제 학습은 로컬 **RTX 3060 Laptop 6GiB**에서 수행했다. 휴대폰 MeshGraphNet의 큰 배치는 VRAM 부족, FIGConvUNet의 큰 배치는 host RAM 부족으로 실패해 더 작은 배치로 학습했다. 휴대폰·범퍼 정적 12실행의 학습 wall 합계는 **128.235분**이다. 별도 Codex Cloud 검사에서는 NVIDIA GPU가 노출되지 않았다. **추가 유료 GPU 지출은 0**이며 기존 장비·전기·구독 비용을 측정한 값은 아니다. [시간·메모리·미완료 범위](docs/RESULTS_KO.md#시간자원비용)

## 상세 자료로 이동

- [결과 요약](docs/RESULTS_KO.md): 비교표, 물리량 정의, 실패와 적용 한계
- [후속 실험 근거 부록](outputs/expanded-research/README.md): 실행별 지표·그림·독립 검산·상세 이력
- [초기 변위 전용 실험](docs/experiment-results.md): 전체 범퍼 메시 Transolver 변위 L2 6.12%; 후속 다물리량 실험과 별도
- [설치·소스·재현 명령](outputs/experiment/README.md) · [원 FEM 재실행](outputs/bumper-fem-rerun/README.md) · [공식 소스와 실제 실행의 교차표](outputs/expanded-research/source-matrix/executed-evidence-crosswalk-2026-10-15.md)
- [개편 전 README 원문](docs/archive/README-before-20261016.md): 기존 상세 안내 보존

이 저장소의 수치는 **우리의 제한된 재현 실험**이다. NVIDIA·원논문의 학습 규모나 저자 성능을 재현했다고 주장하지 않는다. 원 데이터·타사 실행 파일은 출처에서 별도로 받으며, 코드·데이터·deck의 권리 조건은 각 실행 문서에 기록했다.
