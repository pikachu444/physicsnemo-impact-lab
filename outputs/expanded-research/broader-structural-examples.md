# 휴대폰·범퍼 외 구조 예제와 연구

확인일2026-10-15. 기존20행 실행 교차표는 선정한 예제·recipe·설정의 조사 범위이며 PhysicsNeMo의 모든 구조 연구나 가능한 학습 조합의 전체 목록이 아니다. 아래 추가 조사도 소스 확인·데이터 취득·실제 실행을 구분한다. 새 학습·FEM 실행은 이 조사에서 하지 않았다.

## 이미 실제 실행한 다른 구조 문제

| 문제 | 실제 수행 | 해석할 때의 범위 |
|---|---|---|
| OpenCourant DP600 인장 시편 | 13조건 FEM, 신경망6설정과 회귀/축약 fit·별도 평가·저장 모델 재추론 | native 응력·총변형률·EPSP 비교. 고정 형상·작은 조건 공간이며 접촉 충돌 검증이 아님 |
| 공식 DeformingPlate | 공개400시점 궤적으로 축소 HybridMeshGraphNet2설정 학습·399단계 AR 평가 | 비선형 판 변형. 현재 두 설정은 정확도 실패. 공식 크기128/15 모델의 후속 실행은 별도 진행 상태로 기록 |
| 공식 Q1 선형 탄성 | CPU 실제 solve와 독립13검사, 후속 전체DDM64반복 수렴·독립25검사 통과 | 수치 FEM 기준선이며 AI 학습이 아님. 첫 시도11반복 뒤 메모리 가드 중단과 후속 정상 완료를 함께 보존 |
| BOX1BL crash section | 앞선 단계에서 공개 MiniQA의 실제 Starter/Engine 실행 성공 | 작은 구조 충돌 회귀시험. 응력·변형률 surrogate 학습 완료가 아님 |

실제 지표·실패는 [연구 결과](README.md), [판재](benchmarks/deforming_plate/README.md), [탄성](benchmarks/elasticity_numerical/execution-report.md), [기존 BOX1BL 기록](https://github.com/pikachu444/physicsnemo-impact-lab/blob/main/docs/fem-integration-review.md)에 있다. BOX1BL을 이번 추가 조사에서 새로 실행한 것으로 중복 집계하지 않는다.

## 추가 확인한 공식 예제와 외부 구조 연구

| 예제·연구 | 실제 물리·학습 접근 | 공개성과 이번 실행 상태 |
|---|---|---|
| 공식 TopoDiff | 하중·경계조건에 따른2D 구조 topology 생성. Diffusion에 compliance regressor·부유 재료 classifier guidance 결합 | 공식 pin에 학습3entry와 inference·Dropbox 링크 존재. 낙하 시간응답·응력/변형률 surrogate와 목적이 다르며 이번에는 미실행. [공식 소스](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/generative/topodiff/README.md) |
| Carbon LatticeGraphNet | Neo-Hookean elastomer lattice의 큰 압축·좌굴·IPC 접촉. Skeletal MGN에서 tetrahedral MGN으로 이어지는 두 단계, 변위와 stress invariant | NVIDIA 설명과 원논문에서 Modulus/PhysicsNeMo 사용 확인. 학습 코드·원 데이터 취득 경로는 확인하지 못했으며 미실행. [NVIDIA 소개](https://developer.nvidia.com/blog/using-graph-neural-networks-for-additive-manufacturing/), [원논문](https://arxiv.org/abs/2402.01045) |
| UVA MFGNN_PDE | 노치·구멍 강판의 정적 Von Mises 응력. Coarse→fine 계층 및 curriculum 학습 | 저자 코드가 `modulus.models.meshgraphnet`을 import함을 확인. ANSYS MAPDL 생성 스크립트 안내는 있으나 완성 pickle 데이터 취득은 미확인. 구 API·상용 solver 환경을 확인해야 하며 이번에는 미실행. [저장소](https://github.com/MehdiTaghizadehUVa/MFGNN_PDE), [모델](https://github.com/MehdiTaghizadehUVa/MFGNN_PDE/blob/main/MFGNN_H.py), [논문](https://doi.org/10.1111/mice.13312) |

TopoDiff의 [데이터 링크](https://www.dropbox.com/scl/fo/57pktjf5mj7lzcw5jncct/AKD-ENBYSoZT_9c77jRGWPk?rlkey=wb0z0acdbmayu2k0djgmiukyy&dl=0)는 계정 인증 없이 HEAD200·GET200의237,369byte 공개 landing HTML과 제목을 확인했다. Archive 파일명·크기·README·라이선스는 이 응답에서 확인되지 않았다. 링크 응답 성공을 데이터 파일 취득·내용 검증·학습 성공으로 바꾸지 않는다.

접촉·불확실성 관련 공개 논문도 있다. [접촉 간선을 재구성하는 hybrid crash 연구](https://arxiv.org/html/2605.11784)는 비공개 데이터 조건을 명시한다. [범퍼 MC Dropout/Ensemble 비교](https://arxiv.org/abs/2607.18294)와 [PhysicsNeMo CAE 불확실성 연구](https://arxiv.org/html/2609.25430)는 향후 앙상블·불확실성 평가의 비교 자료다. 논문 확인을 구현·데이터 취득·실제 실험으로 세지 않는다. 이 조사만으로 calibration이나 파손 확률 정확도를 주장하지 않는다.

## OpenCourant의 추가 소형 접촉 후보

| 공개 회귀시험 | 확인한 입력 | 공개 기준 결과와 남은 확인 |
|---|---|---|
| small_boule | 149절점·113shell·2part. PLAS_JOHNS, TYPE7접촉2개, rigid wall·imposed velocity. Solid 구 모델이 아님 | ANIM A001/A008 두 frame, T01·ABF·out 공개. 명시 stress/strain/EPSP 출력 카드 없음. Binary 실제 필드·단위·소성 발생량은 미검증. [공식 트리](https://github.com/OpenCourant/OpenCourant/tree/main/qa-tests/miniqa/INTERF/INT_7/small_boule) |
| TYPE25 hexa8 | 409절점·120brick+180shell·4part. PLAS_JOHNS·FAIL/JOHNSON·마찰0.2, 제어변위 접촉5ms | T01·out만 공개하며 ANIM/H3D 없음. 자유 낙하가 아니고 파손 카드 존재가 실제 파손 발생 증거도 아님. [공식 트리](https://github.com/OpenCourant/OpenCourant/tree/main/qa-tests/miniqa/INTERF/INT_25/hexa8) |

두 후보 모두 native RAD의 Starter→Engine 경로다. small_boule에는 동반 include 입력도 필요하다. 응력·총변형률 AI 데이터로 사용하려면 출력 카드·유효 적분점·좌표계·층·단위부터 검증하고 새 조건별 FEM을 생성해야 한다. 공개 회귀시험의 존재는 물리 실험 검증이나 학습 가능한 충분한 DOE의 존재를 의미하지 않는다. 이번 조사에서는 solver나 추가 fit을 실행하지 않았다.

## 같은 모델 계열 안의 미시험 기능

실제 학습한 여섯 정적 설정은 Transolver·FLARE·GeoTransolver GALE·GALE_FA·MeshGraphNet·FIGConvUNet이다. 설치2.2.2 GeoTransolver에는 `plus=False`, `concrete_dropout=False`가 있지만 이번 비교에서 활성화하지 않았다. 설치본에는 GALE/GALE_FA만 있고 GALE_FPP는 없다. 해당 설치 소스 SHA256은`1355b0f37d02875762a2d32a0660676803fe8f5fae6f1e4fb4bb7f451efd3cdc`다. Framework import 없이 AST·텍스트로 확인했다.

반면 pin의2.3개발 [GeoTransolver 소스](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/models/geotransolver/geotransolver.py)에는 GALE_FPP/FLARE++ 경로가 있다. 현재2.2.2의 실제 학습을 개발판의 모든 attention·dropout 조합 시험으로 확대하지 않는다. 데이터·모델·시간 전략·버전별 실행 교차표와 실제 완료 checkpoint 수를 구별해 읽어야 한다.
