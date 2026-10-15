# PhysicsNeMo 구조·충격 예제 독립 소스 검토

확인일: 2026-10-15 (Asia/Seoul). 공식 저장소의 `main` 소스, 원 저자 논문, 데이터 제공자 파일 트리를 직접 열어 확인했다. 아래의 **확인**은 문서·코드 열람으로 확인했다는 뜻이며, 이 문서 작성자가 학습·FEM 해석을 실행했다는 뜻은 아니다. `main`은 변동 가능하며 이번 소스 검토에서는 커밋 SHA 고정에 성공하지 못했다. 실행 결과와 환경 검증은 별도 실행 기록을 우선한다.

**후속 실행으로 확인한 사항:** 별도 실행에서는 소스 commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`과 데이터 revision을 고정했다. 범퍼 curated132파일을 실제 읽었으며 변위와 두께만 존재하고 응력·소성변형률은 없다. 원 `.rad`에서 CC BY-NC4.0 고지를 확인했다. 전체13,675 변형체 절점의 GPU 학습·평가까지 수행했다. 아래 최초 조사 당시의 “미확인”과 구별해 [실제 실행 결과](experiment-results.md)를 우선한다.

## 판단에 직접 영향을 주는 발견

현재 공식 `examples/structural_mechanics/`에는 `crash`, `deforming_plate`, `drop_test`, `openradioss_dataset_gen`이 있다. 따라서 PhysicsNeMo의 구조 예제를 준정적 판 문제 하나로 한정해서 평가하면 현재 기능을 놓친다. 특히 **OpenRadioss 휴대폰 낙하 → VTU → GeoTransolver → 변위·Von Mises 응력 비교** 경로가 공식 코드로 존재한다. [공식 폴더 트리](https://github.com/NVIDIA/physicsnemo/tree/main/examples/structural_mechanics)

다만 코드 공개, 학습 데이터 공개, 원 FEM 입력 공개, 검증 완료는 각각 별개다. 가장 빨리 실험할 데이터는 커뮤니티가 제공한 **범퍼 충돌 curated VTP 540 MB**다. 휴대폰 예제의 전체 DOE 결과 다운로드는 이번 확인에서 찾지 못했다. [범퍼 데이터 파일 트리](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/main/CURATED_DATA_VTP)

## 공식 예제 비교

| 예제·실제 코드 경로 | 물리와 학습 목표 | 원본·학습 데이터 상태 | 실행 가능성 판단 |
|---|---|---|---|
| `examples/structural_mechanics/drop_test/` | solid tet/hex 휴대폰 낙하. 초기 좌표·재료와 벽 방향 파라미터로 전체 시간의 위치/변위와 nodal Von Mises stress 예측 | 코드 공개. 입력 `.rad`는 별도 확보 필요. 완성 DOE 데이터 다운로드 미확인 | 가전 낙하와 가장 직접 연결. 먼저 baseline FEM·변환·필드 정합성 검증 필요 |
| `examples/structural_mechanics/crash/` | 범퍼·crushcan·차체 BIW의 과도 변형. 현재 GeoTransolver, Transolver, MeshGraphNet, FIGConvUNet 및 GeoFLARE 구성 포함 | 범퍼 공개 커뮤니티 데이터 있음. 논문 BIW 150회 데이터의 공개 다운로드는 미확인 | 이미 생성된 범퍼 데이터로 학습·추론·오차 측정이 가능할 유력 후보 |
| `examples/structural_mechanics/deforming_plate/` | actuator에 의한 hyperelastic plate 준정적 변형. autoregressive MeshGraphNet | DeepMind train/valid/test TFRecord 공개. 원 COMSOL 생성 모델/스크립트의 완전 공개는 미확인 | 작은 구조 데이터 파이프·GNN 학습 검증에 유용. 낙하 동역학 입증으로 간주할 수 없음 |
| `examples/additive_manufacturing/sintering_physics/` | 소결 시 25–50% 큰 수축·중력 처짐. Virtual Foundry GraphNet | 예제는 sample test·checkpoint 사용 절차 제공. 전체 원 FEM 데이터 생성 재현성 미확인 | 큰 변형·형상 일반화 참고. 충돌·소성 파손의 직접 대체 예제는 아님 |
| 구 `physicsnemo-sym/examples/{bracket,fuselage_panel,plane_displacement}` | 선형 탄성 평형, PINN/VPINN, 변위·응력 | 과거 예제 코드 공개 | 소규모 PDE/경계조건 검증용. 현재 본체 API와 버전 호환 확인 필요 |

위 표의 세부 근거는 아래 각 소스에 연결한다.

## 휴대폰 낙하: 실제 파일과 재현 경로

**확인:** `drop_test/conf/drop_test_geotransolver_oneshot.yaml`은 `num_time_steps: 100`, train 20, validation 4, `dynamic_targets: [Von_Mises]`, `log_transform_targets: true`, global feature 8개를 설정한다. 5종 재료 E scale과 rigid wall 회전 3축이다. 출력 396은 `(100−1) × (위치 3 + 응력 1)`이다. 이 수치는 제공 구성의 기본값이며 본 연구의 적정 DOE·데이터 수를 의미하지 않는다. [실험 YAML](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/conf/drop_test_geotransolver_oneshot.yaml)

**확인:** backbone은 `rollout.GeoTransolverOneShot`; 기본 slice 128, `use_te: false`, local geometry feature 사용, checkpointing 옵션이 있다. 실험 YAML에서 층 수를 5로 덮어쓴다. 따라서 Transformer Engine은 해당 구성의 필수 조건이 아니다. [모델 YAML](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/conf/model/geotransolver_one_shot.yaml)

**확인:** `train.py`는 MSE loss, Adam 또는 Muon, cosine LR, CUDA autocast/GradScaler, DDP, batch 1을 사용한다. GPU reserved memory와 batch/epoch 시간을 로그에 남기며 위치와 응력 MSE를 나누어 검증한다. 이 MSE는 정규화된 학습 공간의 수치이므로 MPa·mm 오차로 바로 해석할 수 없다. [학습 코드](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/train.py)

**확인:** `datapipe.py`는 응력에 `log1p(clamp(min=0))`를 적용할 수 있고 training split의 평균·표준편차를 저장·재사용한다. 위치 궤적·목표 시계열을 메모리에 모으며, 부족한 dynamic field 시점은 마지막 값으로 채운다. **제안:** 원 데이터 시점 수 검사와 padding 발생 기록을 필수로 남겨야 한다. 패딩을 실제 FEM 시점으로 오인하면 후반부 오차가 왜곡된다. [데이터 코드](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/datapipe.py)

**확인:** `vtu_reader.py`는 `UnstructuredGrid`를 요구하고, `displacement_tNNNN` 또는 소수점 시간 이름에서 절대 위치를 복원한다. 입력 메시에 응력·속도·가속도 같은 필드가 있으면 함께 읽는다. **제안:** element 응력의 node averaging 여부·부품 경계 평활화·단위를 원 FEM 결과와 대조해야 한다. [VTU reader](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/vtu_reader.py)

공식 흐름은 `.rad` 확보 → `openradioss_dataset_gen/drop_test/generate_dataset.py`로 조건 변경 → `run_simulations.py`로 starter/engine → `anim_to_vtk` → Curator → train/val/test 분리 → `train.py` → `inference.py`다. 추론은 exact/predicted/difference field를 담은 comparison VTU를 출력한다. [낙하 README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/drop_test/README.md)

**확인:** DOE generator의 polymer는 `/MAT/PLAS_TAB`, battery/glass/PCB/composites는 `/MAT/ELAST`로 취급된다. E만 ±20% 변경하고 벽을 회전하는 기본 DOE다. **판단:** 유리 파손, 체결부 파괴, 고무 점탄성, 배터리 내부 손상 등을 자동으로 학습·검증하는 예제라고 확대 해석할 수 없다. [낙하 DOE 실제 소스](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/openradioss_dataset_gen/drop_test/generate_dataset.py)

## 범퍼 충돌: 즉시 활용 가능한 공개 데이터와 모델 비교

**확인:** `crash/conf/bumper_geotransolver_oneshot.yaml`은 51시점, train 121·validation 5, xyz와 `effective_plastic_strain`, `stress_vm`를 목표로 한다. global은 velocity, thickness scale, wall y offset. `out_dim: 250 = 50 × 5`; 기본 optimizer는 Muon이다. [범퍼 YAML](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/conf/bumper_geotransolver_oneshot.yaml)

**확인:** 공통 training default는 10,000 epochs, validation/checkpoint 10epoch마다, AMP true, worker 8이다. Muon은 PyTorch ≥2.9 조건으로 표시된다. **제안:** 설치가 되는지 확인하는 첫 run에 10,000epoch를 그대로 사용할 이유는 없다. 먼저 I/O·한 step·checkpoint roundtrip·소수 epoch를 검증하고 convergence와 오차를 따로 평가한다. [학습 기본 YAML](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/conf/training/default.yaml)

**확인:** 현재 crash datapipe는 위치·일반 feature를 정규화하지만 dynamic target은 원 수치를 위치 목표와 연결한다. README TODO에도 dynamic target·global normalization을 명시한다. 응력과 변위를 함께 학습할 때 scale 지배를 확인해야 한다. drop_test의 응력 정규화 코드를 crash에도 이미 있다고 가정하면 안 된다. [범퍼 datapipe](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/datapipe.py)

**확인:** 공식 crash README에는 one-shot, time-conditional, AR rollout, teacher forcing 및 후처리 `compute_l2_error.py`, `compute_probe_kinematics.py`, 단면 비교 도구가 있다. 공개 벤치마크 수치는 저자 결과이며 이번 재현 결과가 아니다. 위치 L2가 작아도 가속도·접촉력·파손 오차가 작다고 결론내릴 수 없다. [충돌 README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/crash/README.md)

**확인:** Hugging Face `AIRBORNEPANDA/BumperBeamCrashExample`은 전체 파일 트리 15.8 GB, RAW_DATA 15.3 GB, CURATED_DATA_VTP 540 MB로 표시된다. VTP 아래 TRAINING_DATA, VALIDATION_DATA, GLOBAL_FEATURES.json이 있다. 데이터 card는 shell thickness와 pole 위치를 바꾼 OpenRadioss 데이터를 설명한다. **미확인:** 데이터 자체의 명시 라이선스, 모든 VTP의 stress/plastic strain 필드, 정확한 파일별 샘플 수와 시점 수. 모델 학습 전 실제 파일을 열어야 한다. [전체](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/main), [원본](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/main/RAW_DATA), [curated](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/main/CURATED_DATA_VTP), [data card](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/blob/main/README.md)

커뮤니티 `HoussemMouradi/OpenRadioss2PhysicsNeMo`는 Windows/Linux/Colab notebook, `.rad` 입력, DOE runner와 global feature extractor를 공개하고 MIT license를 표시한다. 이것은 재생성 경로의 유용한 교차 자료다. 저장소 MIT 표시는 별도 HF 데이터 전체의 라이선스 확인을 대체하지 않는다. [공개 구현](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNeMo)

## 데이터 생성·Curator의 현재 재현 함정

**확인:** NVIDIA generator README의 기본은 범퍼 135회, 휴대폰 192회다. `.rad` deck은 예제에 포함되지 않으며 범퍼 약 1.8 MB·휴대폰 약 42 MB라 설명한다. OpenRadioss Linux GFortran executable과 `vortex-radioss`를 전제로 한다. `/INCLUDE`·`/SUBMODEL`을 지원하지 않고 주석·고정 열 위치에 의존하는 deck mutation이다. 임의 회사 모델을 그대로 넣어 범용 DOE로 쓰기는 어렵다. [생성 README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/openradioss_dataset_gen/README.md)

**확인:** 위 PhysicsNeMo README들이 연결하는 Curator `examples/structural_mechanics/crash`·`drop_test` raw README는 이번 확인에서 404였다. 현재 Curator 트리는 **`examples/cae/crashsim`**, **`examples/cae/drop`**이다. 같은 날의 `main`끼리도 안내가 어긋난다. [현재 Curator examples](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples)

현재 crashsim은 `python main.py --input ... --output ...` 형태이며 d3plot → filter → compressed Zarr다. 예전 `physicsnemo-curator-etl` Hydra 예시를 그대로 적용하기 전에 호환 여부를 확인해야 한다. 벽 노드 제거는 displacement variation threshold를 사용한다. **제안:** 실제로 움직이지 않는 구조 부품까지 제거되는지 part ID로 대조한다. [현재 crashsim README](https://github.com/NVIDIA/physicsnemo-curator/blob/main/examples/cae/crashsim/README.md)

현재 drop ETL은 OpenRadioss의 시점별 VTK → VTU이며 `Von_Mises_tNNNN`를 지원한다. README 안 `cd examples/cae/drop_etl` 표기는 실제 폴더 `drop`과도 다르다. 실행 시 실제 파일 경로와 entry point를 사용해야 한다. [현재 drop ETL](https://github.com/NVIDIA/physicsnemo-curator/tree/main/examples/cae/drop)

## DeformingPlate: 무엇을 검증할 수 있는가

**확인:** 원 논문은 hyperelastic plate와 kinematic actuator의 **quasi-static** 문제이며, 위치 갱신과 nodal Von Mises stress를 학습한다. FEM은 COMSOL. contact/world-space edge와 adaptive meshing을 다루지만 그것이 전체 가전제품의 explicit impact·failure 입증은 아니다. [원 논문, §4](https://arxiv.org/html/2010.03409v3)

**확인:** NVIDIA README의 데이터는 train/val/test 1000/100/100, 400step, 평균 1271node. hidden128, message passing15, GPU당 batch1, **8×H100·25epoch·20시간** 저자 실행 조건을 기재한다. 이는 최소 GPU 요구량이 아니다. [공식 README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/deforming_plate/README.md)

**확인:** 현재 YAML은 30epoch·학습시점200·4 outputs·8 edge features라 README 조건과 다르다. 정확한 재현은 코드·설정을 pin하고 사용할 조건을 명시해야 한다. [실제 config](https://github.com/NVIDIA/physicsnemo/blob/main/examples/structural_mechanics/deforming_plate/conf/config.yaml)

DeepMind download script의 원 URL은 `https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/` 아래 `meta.json`, `train.tfrecord`, `valid.tfrecord`, `test.tfrecord`다. loader/model/data 공개는 확인했으나 다운로드 총 용량·데이터 별도 라이선스·COMSOL 생성 deck은 이번 검토에서 확정하지 않았다. 코드의 Apache-2.0와 원 데이터 권리를 구분한다. [다운로드 소스](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh), [원 구현 설명](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)

## 관련 연구: 실제 범위와 공개 상태

| 자료 | 확인한 내용 | 연구에 주는 의미 |
|---|---|---|
| [Automotive Crash Dynamics Modeling Accelerated with Machine Learning, 2510.15201v3](https://arxiv.org/abs/2510.15201) | NVIDIA/GM 저자. PhysicsNeMo MGN·Transolver, 150 LS-DYNA BIW runs, 200개 이상 부품, 여러 시간 모델 비교 | 조립체 추론의 가능성 근거. 정확도는 FEM과 동일 수준에 도달하지 않았다고 저자도 명시 |
| [GeoTransolver, 2512.20399v3](https://arxiv.org/abs/2512.20399) | multiscale ball query geometry/global context + physics slice attention. crash benchmark 포함, PhysicsNeMo 공개 구현 | geometry·재료·하중 조건화를 적극 비교할 후보 |
| [Industrial Crash Dynamics / FLARE, 2605.27758v1](https://arxiv.org/abs/2605.27758) | Notre Dame/NVIDIA/GM. low-rank FLARE 결합과 여러 시간 전략, 범퍼·차체 비교 | one-shot/GeoFLARE 유력 후보지만 연구별 순위가 사용자 낙하 문제에 보장되는 것은 아님 |
| [CarCrashNet / CrashSolver](https://github.com/Mohamedelrefaie/CarCrashNet) | MIT/TRI. part hierarchy + global component transformer + interface passing 제안. 15,567회·6.65TB 소개 | **Dataset Availability는 peer review 완료 후 공개 예정**. 현재 repo는 설명/assets이며 즉시 학습 데이터로 취급 불가. PhysicsNeMo 직접 활용 여부도 미확정 |
| [Virtual Foundry GraphNet, 2404.11753](https://arxiv.org/abs/2404.11753) 및 [공식 기여 코드](https://github.com/NVIDIA/physicsnemo/tree/main/examples/additive_manufacturing/sintering_physics) | HP Labs 기여의 소결 변형 GNN. sample·checkpoint·TFRecord 변환 절차 | 외부 기여 구조 변형 사례. 큰 변형만으로 낙하 검증을 대체하지 않음 |

2026 crash 논문은 약 400,000node·380,000element BIW를 설명하고 FEM 120ms 물리시간당 HPC에서 1시간 이상이라고 기재한다. 데이터는 특정 충돌·부품 두께 변화 범위다. 공개 본문에서 GPU 모델·VRAM 절대값을 확인하지 못했으므로 표의 epoch 시간을 개인 GPU 시간으로 환산하지 않는다. [논문 §5–6](https://arxiv.org/html/2605.27758v1)

## 자원·적용 범위에 대한 판단

**계산으로 얻는 하한:** float32 목표만 `4 × Nnode × Ntime × Nchannels` byte다. 예를 들어 400,000node·100시점·xyz+stress4channel이면 한 trajectory 목표만 640 MB(십진)다. 20개를 모두 RAM에 유지하면 12.8 GB이며 원 좌표·사본·VTP parsing·gradient·optimizer·activation은 추가다. 이는 실제 측정이 아닌 저장량 산술이며 GPU 최소 사양 제안이 아니다.

**기술적 판단:** 전체 조립체가 원리적으로 불가능하다고 볼 근거는 없다. 차체 연구는 큰 조립체 surrogate를 입증한다. 다만 동일 제품군·연결 topology·낙하 방향·재료·접촉 범위를 벗어난 zero-shot 일반 제품 낙하 solver의 공개 검증은 찾지 못했다. 가전 전체를 목표로 하되 첫 단계로 재현 가능한 범퍼/휴대폰을 비교하고, 검증된 envelope에서 부품→서브어셈블리→전체로 확대하는 것이 타당하다.

**실용 출력 제안:** 우선 residual deformation, 주요 부품 변위/간극, strain·stress hotspot, 주요 센서 가속도, 접촉력 pulse, 흡수에너지와 failure risk를 각각 평가한다. 공간 평균 위치 L2 외에 peak 오차·peak 시간·impulse·부품별 상위 분위수·에너지 imbalance를 기록한다. 파손 판단은 손상 law·failure label·false negative 검증이 있어야 한다.

**알고리즘 비교 제안:** MGN은 mesh/connectivity bias, GeoTransolver/GeoFLARE는 전역 결합·geometry conditioning, POD/autoencoder ROM+회귀는 작은 고정 topology에서 baseline, scalar tree/MLP는 설계 지표 baseline으로 비교할 가치가 있다. 접촉·파손을 모두 순수 PINN으로 대체하기보다 FEM 라벨 surrogate, coarse FEM correction, 부품별 ROM/AI+접촉 solver 결합을 별도 가설로 실험한다. 이 순위는 실행 전 확정하지 않는다.

## 버전·라이선스 상태

PhysicsNeMo에서 직접 열어 본 코드·YAML은 Apache-2.0 SPDX 표기를 갖는다. 데이터·외부 deck의 라이선스를 자동으로 포함시키지 않는다. `physicsnemo-sym`은 2026-05-29 archive 되었고 본체 `nvidia-physicsnemo[sym]`으로 통합됐다. built-in PDE class 제거 등 breaking changes가 있으므로 25.08 탄성 문서와 현재 본체를 섞지 않는다. [Sym 현재 공지](https://github.com/NVIDIA/physicsnemo-sym), [과거 선형탄성 문서](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/foundational/linear_elasticity.html)

## 이번 독립 검토의 실행 사실

- 웹 도구로 위 공식 코드·논문·파일 트리를 읽었다.
- 로컬 PowerShell의 GitHub API 조회는 `알려진 호스트가 없습니다` DNS 오류로 실패했다. 이 실패는 Codex Cloud 자원 검증 결과가 아니다.
- 이 문서 작성자가 PhysicsNeMo 설치·FEM 실행·학습·추론을 수행했다고 주장하지 않는다. 다른 실행 담당자의 로그와 결과로 별도 판단해야 한다.
