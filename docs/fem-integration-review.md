# 공개 FEM·낙하 AI 통합 검토

조사 기준일: 2026-10-15. **확인 사실**, **문헌 주장**, **기술 제안**, **직접 실행**을 구분한다.

## 판단

접촉·소성·대변형을 포함한 구조 surrogate의 공개 근거는 있다. 그러나 특정 형상의 몇 개 두께/속도 조건에서 변형장을 근사한 결과가 TV·세탁기의 모든 조립체/낙하방향/파손모드 예측 능력을 뜻하지 않는다. 현 단계에서 현실적인 제품 경로는 전체 조립체의 제한된 입력 범위에서 중요한 설계 지표를 빠르게 예측하는 접근과, 불확실성이 큰 접촉/파손 부위는 FEM 또는 별도 서브모델로 계산하는 접근을 함께 비교하는 것이다. 전체제품 가능성을 처음부터 배제할 이유도, 기존 공개 예제로 전체제품 수준이 검증되었다고 볼 근거도 없다.

## OpenCourant와 OpenRadioss의 현재 상태 — 확인 사실

[OpenCourant 공식 사이트](https://opencourant.org/)는 2026-10-01 OpenRadioss 프로젝트 중단과 저장소 삭제 후 공개 코드를 계승하는 독립 커뮤니티 프로젝트라고 설명한다. OpenCourant를 오기나 다른 solver로 처리하면 안 된다. 이 연구가 확인한 실행 배포는 [2026-10-11 release](https://github.com/OpenCourant/OpenCourant/releases/tag/latest-20261011)다. Windows/Linux x86-64 및 Linux arm64가 있으며 MiniQA 통과가 릴리스 고지에 포함된다. 이것은 배포자의 CI 주장이고 아래 실제 실행과 구분한다.

[현재 repository](https://github.com/OpenCourant/OpenCourant)는 `.rad`, `.k/.key`, `.inp` 변환 경로와 ANIM→VTK, TH→CSV 도구를 제공한다. [실행 문서](https://github.com/OpenCourant/OpenCourant/blob/main/INSTALL.md)의 Starter→Engine 두 단계, OpenMP/MPI 환경 구성이 실제 설치 경로다. 예전 OpenRadioss Confluence 예제 페이지는 조사 시점 `This space is locked`여서 다운로드 원본으로 의존하지 않았다. 영구 재현을 위해 upstream commit와 asset SHA256를 함께 보존해야 한다.

## 바로 활용할 수 있는 공개 모델/데이터

| 후보 | 실제 공개 상태 | 적합한 검증 | 지금 남은 확인 |
|---|---|---|---|
| OpenCourant MiniQA BOX1BL | 원본 deck+기준값+Windows 실행기 확보, **직접 실행 성공** | solver 설치, 반복 실행, 변형 mesh·history 변환, FEM 데이터 연결의 첫 회귀시험 | 학습 데이터 다양성 없음. 실제 제품 물성/큰 접촉/파손 검증과 다름 |
| [OpenRadioss2PhysicsNeMo](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNeMo) bumper | 원본 `.rad`, Python 데이터 생성, ANIM→d3plot pipeline 공개 | 접촉/소성 shell 부품, 두께·pole 위치 변화와 공식 crash reader 연결 | 구형 OpenRadioss 다운로드 링크 교체; 실제 solver/변환기/훈련 실행 검증 |
| [BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/main) | 131 runs 표방, 파일 tree 공개. 전체15.8GB, curated VTP540MB | 직접 FEM 재계산 없이 reader/학습/정량평가 가능 | 전체 다운로드/무결성은 아직 이 조사에서 검증하지 않음. public validation 외 별도 holdout 필요 |
| [Plasticity-MeshGraphNets](https://github.com/mikelunizar/Plasticity-MeshGraphNets) | 코드·학습/평가 명령·dataset Drive 링크 공개 | 작은 판의 접촉+소성, 응력텐서 및 변형 예측 비교 | 고속 동적 충격이 아니라 **quasi-static pseudo-time** 문제. Drive 실제 획득/재학습 미검증 |
| [CarCrashNet](https://github.com/Mohamedelrefaie/CarCrashNet) | 논문·README·그림 공개, **데이터와 loader는 peer review 후 공개 예정** 명시 | full-vehicle와 부품 hierarchy의 연구 근거 | 즉시 실행 가능한 공개 corpus로 계산하면 안 됨 |
| 상용 Radioss phone-drop tutorial | [공식 tutorial](https://2021.help.altair.com/2021.2/hwsolvers/rad/topics/solvers/rad/cell_phone_drop_test_sl_r.htm)은 SimLab/Radioss license 필요 명시 | 전자기기 assembly의 모델링 항목 참고 | 공개 실행 가능 원본/허가를 따로 확인해야 하며 공개 학습 데이터와 다름 |

공개 TV/세탁기 전체 조립체에 대해 재분배 가능한 FEM 입력+여러 조건의 학습 corpus+평가 기준을 한꺼번에 제공하는 primary resource는 이번 검색에서 확인하지 못했다. 이는 존재하지 않는다는 증명이 아니다.

### Bumper pipeline 소스 검토

[실제 generator 코드](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNeMo/blob/main/radioss_generate_d3plot_dataset.py)는 shell property IDs 2,7의 두께와 pole Y-offset을 변경한다. 독립적인 수백 가지 geometry가 공개되었다는 뜻이 아니다. `run_radioss()`는 Windows `.bat`를 호출하고 random seed/solver 반환코드 검사를 제공하지 않아, Linux/Cloud에서 README만 따라 실행하면 완전 재현된다고 주장할 수 없다. 연구용 wrapper에서는 seed, 실행 상태, stdout/stderr, 변경 카드·단위·최종 해석 시각을 기록해야 한다.

[원본 Engine deck](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNeMo/blob/main/Bumper_Beam_AP_meshed_0001.rad)은 stress/plastic strain/displacement/contact output을 요청하지만 animation interval이 10(입력 시간 단위)이다. 변형 학습용 snapshot 수와 가속도/접촉력 peak의 시간 해상도는 별도로 평가해야 한다. 원본 deck의 **CC BY-NC 4.0** 고지를 repository의 MIT 고지가 대체하지 않는다. 개인 비상업 검증과 회사 상용 재사용 조건을 구분한다.

### 신규 full-vehicle 문헌을 해석하는 방법

CarCrashNet 저자는 14,742 bumper simulations+825 vehicle simulations, 6.65TB corpus, part hierarchy를 이용한 CrashSolver를 제안한다. 공개 README의 vehicle displacement RMSE는 약22–62mm 범위여서, 이런 자동차 benchmark 숫자를 가전제품 허용공차로 그대로 수용할 수 없다. 이 수치는 저자 보고이고 재현 결과가 아니다. 데이터 release가 지연된 상태는 현재 즉시 재현 가능한 후보와 구분해야 한다. [논문](https://arxiv.org/abs/2605.07098), [공개 상태 고지](https://github.com/Mohamedelrefaie/CarCrashNet#dataset-availability).

## 알고리즘을 고정하기 전 비교할 실험 — 기술 제안

| 접근 | 유리한 목표 | 주요 위험/추가 조건 |
|---|---|---|
| Gaussian process·kernel·tree/boosting·MLP의 scalar surrogate | peak acceleration, force, residual deformation, plastic work, critical clearance 등의 빠른 설계 순위 | 공간장/새로운 파손모드 없음. 작은 데이터에서 강한 baseline으로 필요 |
| POD/SVD+latent regression, 비선형 autoencoder ROM | 동일 topology, 유사 모드에서 전체 변위장 복원 | 접촉 전환·국소좌굴·비가역 경로·파손으로 필요한 latent 차원 급증 가능 |
| Mesh GNN autoregressive | unstructured mesh, local connectivity, history-dependent evolution | rollout error 누적, 긴 전달거리, mesh/접촉 graph 비용; teacher-forcing loss만으로 검증 불가 |
| time-conditional GNN/Transformer/neural operator | 초기 형상·조건·시간→변형; 시간별 병렬 추론 | trajectory path consistency/충돌 제약이 자동 보장되지 않음. time interpolation과 새 조건 generalization을 분리 |
| multiscale graph·part hierarchy·global attention | 큰 조립체, 부품 간 하중 경로 | hierarchy/interface 설계 및 remeshing 대응 필요. 큰 raw graph를 무조건 단일 GPU에 넣는 방식과 비교 |
| FEM+학습 국소물성/접촉/오차보정, coarse-FEM+fine residual | solver의 보존식·접촉처리를 유지하면서 특정 병목 가속 | 안정성/접선/열역학/외삽 검증 필요. 독립 예측보다 개발 난도가 높을 수 있음 |
| PINN/physics-informed hybrid | 데이터 부족 보완 및 conservation regularization | 불연속 접촉·소성 내부변수·파손에서 optimizer 난도. 데이터가 필요 없는 낙하 solver로 간주하지 않음 |

문헌 예시는 [MeshGraphNets](https://arxiv.org/abs/2010.03409), [multihierarchical crash ROM](https://arxiv.org/html/2402.09234v2), [Neural Stress Fields](https://arxiv.org/abs/2310.17790)다. 후자는 neural reduced state와 physics를 결합해 elastoplasticity/fracture를 다루지만 상용 가전 FEM deck의 바로 교체 가능한 모델은 아니다. Multihierarchical의 [현재 공개 PyTorch 구현](https://github.com/jkneifl/Multi-hierarchical-Surrogates)은 README가 논문 원코드와 다른 재구현이라고 명시하므로 원논문 숫자의 재현 구현이라고 단정하면 안 된다.

정해진 모델·DOE 수량을 선결하지 않는다. 먼저 공개 데이터의 실제 변동 범위와 VRAM/epoch/오차를 측정한 뒤 scalar baseline, field baseline, 시간모델의 의미 있는 비교를 정한다. 데이터 생성 비용도 FEM 실패율·저장/후처리까지 실측해야 한다.

## 가전제품에서 우선 예측할 결과 — 기술 제안

TV는 잔류 휨/패널-프레임 간극/체결부 상대변위와 국소 가속도, 세탁기는 cabinet/tub/frame/transport bolt 등 부품 간 상대변위·충격전달·변형·간섭을 목표 후보로 삼을 수 있다. 이는 제품 요구조건이 제공되기 전의 연구 후보이지 확정 acceptance criteria가 아니다. 재질별 한계, 표시부 균열·glass 파손·clip 이탈 같은 mode를 제품 요구에 맞춰 정의해야 한다.

최종 변위만 예측하면 transient clearance와 peak acceleration을 놓칠 수 있다. 평균 von Mises stress가 낮아도 hotspot·국부 strain·용접/나사 파손 위험을 놓칠 수 있다. 데이터에 파손이 없으면 파손확률을 학습했다고 말할 수 없다. 원시 stress/strain tensor의 좌표계·shell integration point·상하표면을 명확히 하고, node 평균값과 element 값의 혼합을 피한다.

## 전체 조립체와 부품 접근을 가르는 조건 — 기술적 추론

전체 모델의 topology, connector, material families, 접촉쌍, 낙하 자세가 제한되고 충분한 FEM trajectories가 있으면 전체의 scalar/주요 변형 예측을 검토할 수 있다. 형상/조립/접촉 모드가 크게 바뀌거나 국소 요소 삭제·유리/복합재 파손이 핵심이면 전체를 하나의 부드러운 변형 회귀모델로 근사하는 위험이 크다.

부품/서브어셈블리 접근은 계산 자원을 낮추지만 경계조건이 실제 전체 조립체를 대표해야 한다. 전체 FEM 또는 coarse assembly 모델에서 전달력·변위/속도 history를 추출하고, 국소 부품을 surrogate 또는 fine FEM으로 계산하는 연결이 필요하다. 단독 고정단 부품 학습을 실제 조립체에 바로 연결하면 interface stiffness와 energy transfer가 달라진다. 최종단계에서는 FEM 복귀 기준과 적용 범위 밖 입력 탐지가 필요하다.

## 평가에서 반드시 분리할 항목 — 기술 제안

- **분할:** 서로 다른 FEM run을 train/validation/test로 나눈다. 같은 trajectory의 인접 프레임이 양쪽에 들어가는 누출을 막는다. 두께/속도 범위 보간, 범위 외 조건, 새 형상·mesh·접촉 모드 시험을 따로 한다.
- **공간:** displacement/velocity RMSE·MAE를 물리 단위로, part/critical-region별 p95 및 최대오차와 함께 보고한다. 원좌표의 상대 L2는 물체 위치가 커서 좋은 수치가 나올 수 있으므로 displacement-relative와 혼동하지 않는다.
- **시간:** one-step와 완전 autoregressive rollout을 구분한다. peak 크기·발생시각, contact onset, impulse, pulse duration, residual state를 비교하고 출력 간격을 명시한다. 낮은 주기 snapshot을 수치 미분한 acceleration peak는 별도로 검증한다.
- **물리:** 에너지/운동량 balance, nonpenetration, contact force direction, plastic strain 비감소(해당 material 변수 정의에서), failure/erosion consistency를 확인한다. 원 FEM의 hourglass/contact energy·mass scaling도 품질표에 기록한다.
- **속도:** FEM만의 engine time, 전처리/후처리, dataset 생성, training walltime/VRAM, cold/warm inference 및 I/O를 각각 측정한다. 추론 가속만으로 전체 연구비용이 줄었다고 말하지 않는다.
- **불확실성:** 학습 seed, held-out run별 오차분포, 비정상 FEM case 처리, 실험 또는 검증된 FEM 기준과의 차이를 기록한다. 기준 FEM 자체의 모델오차와 AI 근사오차를 분리한다.

## 상용 Radioss/HyperMesh로 확장하는 데이터 경로

제안 경로는 HyperMesh의 재료/접촉/connector 모델링→승인된 Radioss `.rad`→Starter/Engine 배치 실행→원본 ANIM/TH/H3D와 해시 보존→명시적인 node/element/part ID 대응→VTK/HDF5/Parquet 등 학습 schema→학습/추론→HyperView/ParaView에서 비교다. [상용 Radioss 파일 문서](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/faq_rad_radioss_files_r.htm)는 Starter/Engine/ANIM/TH/H3D의 역할과 버전 호환 방향을 설명한다. 현재 solver의 새 keyword가 공개 계승 solver에서 모두 지원된다는 보장은 없으므로 같은 deck의 수치 회귀를 별도 실시한다.

단위계·좌표계·part/material/property ID·shell thickness·element type·연결/접촉/경계·initial velocity/drop orientation·시간축·failure flag·원본 version을 dataset manifest에 넣어야 한다. geometry만 d3plot/VTK로 옮기면 이런 조건이 모두 자동 전달되지 않는다. 회사 실제 데이터는 이번 연구에 제공되지 않았으며 이 검토에서 회사 환경 접속·모델 실행·업로드를 수행하지 않았다.

## 직접 실행 결과

[FEM 실행 기록](fem-experiment/README.md), [측정값](fem-experiment/execution-metrics.json), [기준값 검증](fem-experiment/validation-summary.json), [결과 그림](fem-experiment/fem-results.png)에 원본·스크립트·로그를 묶었다. 350MB 공개 Windows package로 작은 FEM case를 약4.2s의 solver 자체 보고 시간에 실행했고, 원본 기준값을 표시 정밀도로 재현했다. 11개 VTK와 100개 time-history sample을 생성했다. **이는 실제 FEM 실행 성과이며 AI 예측 정확도 또는 Codex Cloud/GPU 실행 결과가 아니다.**

추가 비용을 지출하거나 유료 자원을 생성하지 않았다. 이 작은 회귀모델의 시간/메모리를 전체 가전제품·대규모 학습의 자원 추정치로 외삽하면 안 된다.
