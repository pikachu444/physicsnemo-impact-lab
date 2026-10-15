# 구조 예제별 공식 실행 행렬

확인일: **2026-10-15**. GitHub API로 확인한 PhysicsNeMo main은 `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`이며, 앞선 실험에 사용한 pin과 같다. 커밋 시각은 2026-10-14 21:35:45 UTC다. [공식 커밋](https://github.com/NVIDIA/physicsnemo/commit/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16)

이 문서는 **소스에 존재하는 조합과 실제 실행 결과를 구분하는 기준표**다. 여기서 기성 설정은 데이터를 주면 사용할 recipe가 있다는 뜻이며, 데이터 공개·설치 성공·학습 성공·정확도 검증을 의미하지 않는다. 실제 실행 증거는 각 실행 담당의 로그와 결과를 연결해야 한다. 알고리즘 전체와 데이터 전체의 임의 교차조합을 요구 목록으로 만들지 않았다.

- [기계 판독용 실행 행렬](official-execution-matrix.json): 현행 20행, 별도 archived Sym 3행. 현행은 기성 실험/recipe 8행, 추가 모델 설정 11행, 수치 FEM baseline 1행이다. 범퍼/차체 설정을 별도 행으로 셌으므로 서로 다른 모델 20개라는 뜻은 아니다.
- [소스 blob 목록](source-blob-manifest.json): 관련 99개 파일의 pin·Git blob SHA. 기존 로컬 crash 원본은 모두 pin과 일치했고, 호환성 수정본은 원본 검증에 섞지 않았다.
- [낙하·판 상세 노트](drop-plate-notes.md), [소결·선형탄성 상세 노트](sinter-sym-notes.md), [범퍼 원본 파일 목록](bumper-raw-file-inventory.json).

## 실행할 모델과 시간 전략

| 예제 | 공식 조합 | 제공 수준 | 직접 예측하는 목표 |
|---|---|---|---|
| crash / bumper | GeoTransolver one-shot | 기성 실험 | 위치 xyz + 등가소성변형률 + Von Mises 응력, 50단계 |
| crash / bumper | GeoTransolver time-conditional | 기성 실험 | 같은 5채널을 시각별 독립 예측 |
| crash / bumper | GeoFLARE one-shot | 기성 실험 | 같은 5채널, GeoTransolver의 `attention_type=GALE_FA` |
| crash / BIW | GeoTransolver one-shot, GeoFLARE one-shot | 기성 실험 2개 | 위치 xyz, 25단계 |
| crash / 공통 | Transolver, MeshGraphNet, FIGConvUNet one-shot | 모델 설정·wrapper, 별도 실험 설정 필요 | 위치 + 설정한 추가 target 채널 가능 |
| crash / 공통 | GeoTransolver AR rollout training, one-step teacher forcing | 모델 설정·wrapper, 기성 실험 없음 | 현재 구현은 가속도3 → 위치3 전용 |
| drop_test | GeoTransolver one-shot | 유일한 기성 낙하 실험 | 위치 xyz + `Von_Mises`, 99단계 |
| drop_test | Transolver, MeshGraphNet, FIGConvUNet one-shot; GeoTransolver time-conditional | 설정·wrapper 있으나 README가 낙하 데이터 미검증 명시 | 추가 target 채널 코드 존재, 실제 데이터·차원 설정 검증 필요 |
| drop_test | GeoTransolver AR rollout training, one-step teacher forcing | 설정·wrapper 있으나 낙하 데이터 미검증 | 현재 위치3 전용. 응력4채널을 단순 연결할 수 없음 |
| deforming_plate | **HybridMeshGraphNet**, 한 단계 학습·위치 AR rollout | 기성 recipe | 다음 위치 증가량3 + `stress` scalar1 |
| sintering_physics | **VFGNLearnedSimulator**, 가속도 학습·적분 rollout | 기성 recipe, 배포 데이터 URL 미확보 | 정규화 가속도 → 다음 좌표. 직접 응력·변형률 head 없음 |
| tcad/fp_ddm elasticity | Q1 FEM·BiCGSTAB·domain decomposition | 수치 baseline | 변위와 그로부터 계산한 소변형률·plane-stress·Von Mises. **구조 FNO 아님** |

근거: [crash 설정·시간전략](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/README.md#L343-L362), [crash wrapper](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/rollout.py), [drop 대체 설정 경고](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/drop_test/README.md#L206-L220), [판 모델 생성](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/deforming_plate/train.py#L136-L145), [소결 모델](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/models/vfgn/graph_network_modules.py#L1078-L1132), [수치 탄성 baseline](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/README.md#L121-L141).

## 데이터와 응력·변형률의 차이

**범퍼:** 공식 설정은 응력과 등가소성변형률을 이미 포함하지만, 앞서 실제 다운로드·학습한 공개 curated VTP에는 변위와 thickness만 있다. 반면 공개 RAW tree에는 d3plot family와 ANIM A001–A011이 있다. 확보한 engine의 `/ANIM/SHELL/TENS/STRESS/ALL`, `/ANIM/SHELL/EPSP/ALL` 카드도 확인했다. 이는 응력·변형률 추출 경로의 근거이며, 이 소스 조사자가 실제 binary 값을 읽었다는 주장은 아니다. 추출·매핑·학습은 FEM/ML 실행 담당이 수행한다. [공개 RAW](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/78f1302fba4d75926c6fdd22d01d5133efc14742/RAW_DATA), [범퍼 목표 설정](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/conf/bumper_geotransolver_oneshot.yaml)

범퍼 deck의 kg/mm/ms 단위에서 응력 차원은 GPa다. reader가 변환했는지 확인한 뒤 MPa 환산을 해야 한다. EPSP는 무차원 **등가소성변형률**이며 전체 변형률 텐서가 아니다. Cell stress를 nodal target으로 옮길 때 shell layer·적분점·평균화·node ID 규칙을 기록해야 한다. deck 내부 CC BY-NC 4.0과 커뮤니티 코드의 MIT 표시는 별개다.

**휴대폰:** 기본 직접 출력은 Von Mises 응력이며 strain은 없다. 완성 DOE 결과 다운로드는 공식 recipe에서 찾지 못했다. 별도 담당자가 42,378,237 byte starter를 공개 mirror에서 실제 회수했으며 내부 CC BY-NC 4.0을 확인했다. 원 FEM deck 확보와 192-case DOE 결과 확보는 구분한다. mirror engine은 수정본이므로 공식 원본과 같다고 쓰지 않는다. 출처와 SHA는 JSON에 기록했다. [공식 generator의 원본 미포함 설명](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/openradioss_dataset_gen/README.md#L101-L111)

**판:** 공개 TFRecord에 위치와 scalar `stress`가 있지만, loader만으로 stress invariant/단위를 확정할 수 없다. NVIDIA 기본 추론은 네 번째 응력 출력을 저장하지 않아 실제 응력 비교에는 수정이 필요하다. 기본 `disp_mag`도 초기 위치를 뺀 변위가 아니라 위치 norm을 표시하므로 비교 영상을 그대로 해석하면 안 된다. 공개 원 COMSOL input deck은 확인하지 못했다. [판 target](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/deforming_plate/deforming_plate_dataset.py#L144-L200), [추론](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/deforming_plate/inference.py#L133-L205)

**소결:** 공식 README가 샘플·checkpoint를 언급하지만 실제 URL은 찾지 못했다. VFGN은 좌표/온도로 가속도를 학습하므로 응력·변형률 예측 완료 항목으로 대체할 수 없다. 구조 충격 목표와 공개 데이터 재현성을 고려하면 후순위다. [소결 README](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/additive_manufacturing/sintering_physics/README.md#L123-L171)

## 현재 환경에서 실행할 때 필요한 수정

현 설치는 PhysicsNeMo 2.2.2 / Torch 2.13.0+cu126 / Windows / RTX 3060 Laptop 6 GB다. 앞선 Transolver CUDA 학습 성공은 다른 모든 모델의 CUDA·메모리 적합성을 증명하지 않는다. 6 GB 최대 규모와 실행시간은 모델별 실측 칸에 기록해야 한다.

- **Geo/GeoFLARE:** 설치 2.2.2에도 `GALE_FA` 인자는 있다. 하지만 main 설정의 checkpointing 관련 세 인자는 설치 생성자에 없어 별도 호환 설정에서 제거해야 한다. Transformer Engine은 기본 `use_te=false`다. geometry neighborhood kernel과 GALE_FA의 실제 forward는 따로 확인한다.
- **범퍼 입력 차원:** 기본 MGN node4·FIG in4는 좌표3+thickness1인 BIW용이다. 범퍼 globals3을 node에 concat하면 **MGN node6, FIG in6, Transolver functional6**이 된다. Geo는 local3/global3, time-conditional은 local4/global3이다. 휴대폰의 concat형 모델은 globals8이므로 node11이 된다. [공통 concat](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/rollout.py#L61-L73)
- **출력 정규화:** crash는 위치만 정규화하고 응력·EPSP 및 globals를 원 수치로 전달한다. drop은 비음수 target에 log1p를 적용하므로 signed strain tensor에 그대로 쓰면 음수가 사라진다. 다물리량 비교에는 train-only 채널별 통계와 개별 물리 단위 지표가 필요하다. [crash TODO](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/README.md#L708-L711), [drop target 변환](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/drop_test/datapipe.py#L311-L340)
- **시간 적분:** AR·one-step 기본 `dt=5e-3`, scalar `initial_vel=9.22`는 실제 데이터 시간·단위·방향에 맞춰 검토해야 한다. scalar가 xyz 모두의 backstep에 들어간다. 그 기본값을 kg/mm/ms 범퍼에 곧바로 적용해 물리적 AR 검증이라고 부르면 안 된다. [AR 구현](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/rollout.py#L182-L257)
- **의존성:** MGN은 PyG/적절한 graph backend, FIGConv는 kNN/grid backend, 판은 tfrecord·PyG·torch_scatter와 neighborhood 연산이 필요하다. 소결은 TensorFlow CPU reader와 torch_scatter가 실제 필요하고 DGL/Apex는 현재 경로의 필수가 아니다. 정확한 요구 파일은 JSON과 상세 노트에 연결했다.

Archived Sym의 bracket/panel/plane-displacement는 별도 SHA `0c5f18c31d03d46ec0cc3ef4ccbf761a956dd38c`의 구 API다. 선형 정적 탄성 PINN/VPINN의 비교 자료로 남기되 현행 설치에 바로 실행되는 충격 예제라고 분류하지 않는다. 현행 TCAD 수치 탄성은 향후 구조 정답 데이터 생성 참고로 사용할 수 있다. **전체 연구에 FEM 실행 금지 제약은 없다.** [현행 Sym migration](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/v2.0-MIGRATION-GUIDE.md#L206-L227)
