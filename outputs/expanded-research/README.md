# 구조해석 다중 물리량 실행 검증

2026-10-15에 실제 생성·학습·추론한 증거를 모은 후속 연구 기록이다. **구조역학만** 다루며 유체 예제는 포함하지 않는다. 진행 중인 실험을 완료로 세지 않는다. 아래 수치의 원본은 개별 실행의 JSON·로그·독립 검산이다.

## 실제 정답 확보

| 구조 예제 | 실제 정답과 규모 | 응력·변형률의 의미 |
|---|---|---|
| 공개 범퍼 충돌 | 131해석, 1,441 ANIM 원본, 전체 변형체 13,675절점, 공통 11시점. 학습112/내부 검증12/평가7 | shell local 응력에서 계산한 upper/lower 면내 VM, upper/lower 등가소성변형률(EPSP). 면적 투영 nodal scalar와 원 cell tensor·상태를 별도 보존 |
| OpenCourant DP600 인장 시편 | 직접 생성한 13해석, shell808개·학습 절점919개·10시점. 학습5/검증4/평가4 | native VM, EPSP, native total-strain tensor의 면내 주값 major/minor. 총변형률을 변위 미분으로 대체하지 않음 |
| OpenCourant 휴대폰 초기 충격 | 실제 20µs baseline과 조건 변경 FEM을 생성 중 | native solid stress·total strain·EPSP 및 GPS nodal stress. 20µs는 초기 충격 구간이며 전체 낙하·최대 응답 완료가 아님 |
| 공개 DeformingPlate | 실제 400시점 궤적 다운로드·학습·자기회귀 실행 | 원본 nodal VM 응력. 원 COMSOL strain 출력은 없으며, 좌표에서 계산한 Green–Lagrange strain은 별도 유도 기준값 |

공개 curated 범퍼의 변위만으로 응력 학습이 됐다고 주장하지 않았다. RAW ANIM 3.799GB를 추가 확보하고, hash·형상·ID·실제 시점·물리량·112/12/7 분할을 검증한 2.921GB 라벨 캐시를 동결했다. 전체 cache 작성자의 전수 hash 검증과 독립 검토자의 메타데이터 전수·원 binary 표본 검산은 별개다. [범퍼 정답 추출](fem-labels/README.md), [시편 정답 생성](benchmarks/structural_coupon/README.md)

범퍼 Run224는 total-strain 출력 카드를 추가하여 OpenCourant FEM을 실제 재실행했다. Starter 10.72초 + Engine 502.69초이며, 출력 추가 전후 11시점의 변위·응력·EPSP 차이는 0이었다. 이 total-strain 정답은 **한 조건**이며 131조건의 total-strain 일반화 데이터로 취급하지 않는다.

## 확인된 예측 결과

인장 시편에서 세 회귀/축약 모델과 첫 PhysicsNeMo Transolver가 실제 fit·학습·별도 평가·체크포인트 재추론을 마쳤다. 다른 아키텍처의 비교는 개별 실행 결과가 완료될 때 추가한다. 단위·절점·시간·분할·train-only 통계를 맞춘 결과이며, 전부 고정 형상과 두 가지 조건 변수의 보간 시험이다.

| 시편, 평가4조건 | 변위 벡터 상대 L2 | VM 응력 상대 L2 / RMSE | EPSP 상대 L2 | 총변형률 주값 major / minor 상대 L2 |
|---|---:|---:|---:|---:|
| POD + quadratic ridge | 0.10381% | 0.13648% / 0.70462MPa | 0.40997% | 0.38762% / 0.38841% |
| Transolver, 1,000epoch | 1.91108% | 5.49081% / 28.3485MPa | 2.09416% | 2.05924% / 2.48257% |

POD는 내부 검증으로 rank2를 골랐으며 training의 중심화 응답 분산 99.99374%를 설명했다. 형상이 고정되고 조건이 두 개뿐인 응답은 저차원 회귀에 유리하다. 이 낮은 오차를 임의의 가전제품·새 메시·접촉·파손 일반화로 해석하지 않는다. [회귀/축약 실행](benchmarks/rom/coupon13/metrics.json)

Transolver는 최선 epoch가 예산 상한 1,000과 같아 수렴을 입증하지 못했다. 변위 전체 L2는 1.91%지만 작은 횡변위 y의 상대 오차는 55.17%였다. 원래 평면 문제의 u_z는 약 1e-13mm인데 예측 z RMSE는 0.002512mm였다. 예측 응력 36,760개 중 3,417개가 음수(최소 -57.88MPa), EPSP는 6,753개가 음수였다. 전역 평균 오차만으로 물리적으로 타당한 모델이라고 판정하지 않는다. POD의 EPSP 음수는 약 1e-9 수준이며 크기를 함께 구분한다. 원 예측을 clipping하지 않았다.

범퍼의 다중 물리량 POD+RBF도 실제 학습·내부 검증 선택·평가를 마쳤다. 변위 L2는 3.0320%지만 VM upper/lower L2는 15.059%/15.572%(RMSE 43.357/45.201MPa), EPSP는 51.849%/55.022%였다. EPSP 예측의 약 22%가 음수였다. **변위가 잘 맞는다는 이유로 소성변형·파손 위험도 맞는다고 판단할 수 없다.** 이 사례의 strain은 EPSP이며 total strain과 다르다.

DeformingPlate의 공식 HybridMeshGraphNet 축소 구조를 두 설정으로 실제 학습했다. 각각 12epoch·1,920업데이트이며 399단계 자기회귀를 평가했다. noise 제거와 loss 정규화 변경을 묶은 설정은 drift를 줄였지만 내부 검증 응력을 악화했고, 두 설정 모두 정확도 실패였다. 원 COMSOL strain이 없는 이 예제의 유도 strain 비교를 native FEM strain 검증이라고 부르지 않는다. [판 실행 기록](benchmarks/deforming_plate/README.md)

## 실행 범위와 읽는 기준

공식 source pin은 `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`이다. [공식 실행 행렬](source-matrix/README.md)은 현행 20개 설정 조합과 archived Sym 3개를 분리한다. 20개 모델을 학습했다는 뜻이 아니다. 소스 확인, 설치, GPU 역전파, 실제 학습, 별도 FEM 평가, 독립 수치 검산, 물리 실험 검증을 각각 구분한다. [공통 실험 규약](experiment-contract.json)

실제 실행 장소는 사용자 PC의 RTX3060 Laptop 6GiB다. Codex Cloud는 별도 컨테이너에서 CPU·RAM·네트워크·GPU 장치를 검사했지만 NVIDIA GPU가 노출되지 않았다. 이 로컬 GPU 결과를 Cloud GPU 실행으로 기록하지 않는다. 유료 GPU나 추가 결제는 사용하지 않았다.

PhysicsNeMo2.2.2와 Torch2.10.0+cu126, 호환 공식 torch_scatter wheel, PyG를 새 전용 환경에 설치했다. 실제 Transolver·FLARE·GeoTransolver·GeoFLARE 방식(GALE_FA)·MeshGraphNet·FIGConvUNet의 여섯 아키텍처는 13,675절점에서 CUDA 역전파와 가중치 갱신을 통과했다. 이것은 각각의 학습 정확도 완료와 구분한다. 기존 실행 환경과 앞선 변위 전용 연구는 보존했다.

## 보존과 재현

원본 URL·commit·license·입력/결과 hash, solver/framework 버전, 분할, 정규화, 실제 실행 소스, 실패 로그, 시간·메모리, 물리 단위 오차와 그림을 기록한다. 로컬에는 체크포인트와 예측·FEM 원 배열도 보존한다. 이번 후속 공개 묶음은 원본 deck·대용량 데이터·타사 실행 파일·모델 가중치를 제외하고 재획득/생성 코드와 증거를 제공한다. 공개 경로를 지운 사본은 `publication-byte-manifest.json`에 원본/공개본 hash를 모두 기록해 구별한다.

공개 FEM deck의 CC BY-NC4.0, solver AGPL, NVIDIA 코드 Apache-2.0, 라이선스 미확인 데이터는 서로 다른 권리 범위다. 회사 모델·데이터는 사용하지 않았다. 개인 공개 연구의 재현 경로를 회사 상용 데이터 사용 허가로 해석하지 않는다.
