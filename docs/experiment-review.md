# 공개 FEM–PhysicsNeMo 실험 독립 검토

검토일: 2026-10-15. 검토 방식: 실행 담당자와 별개로 소스·로그·결과 파일을 읽었다. 학습 코드를 변경하거나 재실행하지 않았다. 검토 대상은 `work/physicsnemo-experiment/run_fem_trial.py`이며 공식 예제 `train.py`를 그대로 실행한 재현과 구별한다.

## 현재 판정

**판정: 공개 범퍼 전체 deformable mesh 13,675node의 GPU 60epoch 학습·held-out 평가와 수치 정합성 확인 완료. 앞선 1,024node 실행도 검증했다. source hash, NPZ 수치, split, baseline, 그림을 독립 검증했으며, CPU 중단 실행·공식 entry point 2epoch 호환 실행과 구별한다.**

현재 설계는 공개 OpenRadioss FEM의 고정 topology·공통 시간구간에서 실제 PhysicsNeMo Transolver의 학습·held-out 추론을 검증하는 작은 architecture의 실험이다. 전체 공식 recipe, 백만 node급 조립체의 성능, 실제 가전 낙하, 응력·소성·파손 정확도의 입증으로 확대할 수 없다.

CPU 축소 실험의 로그는 41,286 parameter, 1,024/13,676 node, 112/12/7 split, source t=0..100을 확인한다. 1epoch 27.74초, 5epoch 20.78초와 loss 감소가 보이지만 held-out 결과와 최종 checkpoint를 만들기 전 중단됐다. 이 로그는 실제 gradient 학습의 증거이며 정확도 검증 완료의 증거는 아니다.

## 공식 entry point의 별도 호환 실행

`official-run-console.log`는 원 `train.py` + 작은 Transolver 구성 + training 2run/validation 1run으로 2epoch 완료와 두 checkpoint 저장을 확인한다. validation normalized MSE는 epoch1 0.738358, epoch2 0.720199다. 2epoch로 정확도를 판단하지 않는다.

원본 `vtp_reader.py`는 시간 이름 regex가 `displacement_t0.xxx`만 읽어 이 데이터에서 1시점만 반환했다. selfcheck 로그는 원본 shape `(1,13676,3)`와 호환 복사본 `(11,13676,3)`을 비교한다. `Compare-Object`로 복사본의 변경이 해당 regex 한 줄뿐임을 확인했다. 원본 파일을 보존한 수정이며, 완전 무수정 원본 실행이라고 표현해서는 안 된다. 사용자 config는 AMP off·Adam·worker 0 및 작은 architecture를 사용한다.

## 데이터 분리와 누설

1. `TRAINING_DATA` 내 run을 seed 20261015로 섞어 내부 validation 12개와 나머지 training으로 나눈다. 원 제공자의 `VALIDATION_DATA` 7개를 최종 held-out test로 재정의한다. 별도 read-only metadata 계산으로 112/12/7개와 세 집합 간 교집합이 모두 비어 있음을 확인했다. 시간별 프레임을 같은 run에서 train/test로 나누는 누설은 보이지 않는다.
2. 입력 feature·출력 target·좌표의 평균/표준편차는 `x_train`, `y_train`, `xyz_train`에서만 계산한다. test 입력은 training 통계로 변환한다.
3. Transolver checkpoint는 내부 validation MSE로만 선택한다. RBF의 gamma·alpha도 내부 validation으로만 고른다. test 성능을 선택 조건으로 사용하는 코드는 보이지 않는다.
4. test 라벨은 학습 전에 메모리로 읽지만 loss·checkpoint·hyperparameter 선택에는 사용하지 않는다. 이 사실은 그 자체로 누설은 아니다. 코드 첫머리의 “untouched”는 “선택에 사용하지 않음”이라는 뜻으로 이해해야 한다.
5. 노드는 첫 training-folder 파일의 thickness>0 마스크에서 균등한 인덱스 간격으로 1,024개를 뽑는다. 최적 오차를 찾기 위한 test-label 기반 node 선택은 없다. 다만 공간·부품별 균등 표본이나 hotspot 보존 표본은 아니므로 전체 mesh나 응력 집중부의 대표성을 입증하지 않는다.

별도 공식 CPU 2epoch 호환 실행은 Run224를 validation으로 이미 평가했다. 이 수치/라벨을 GPU Transolver 학습·checkpoint 선택이나 RBF 튜닝에 쓰는 코드는 없다. 따라서 “GPU 모델의 학습·선정에서 제외한 7개 run”은 맞지만, “이번 작업의 어디에서도 미리 보지 않은 완전 blind test”라고 표현해서는 안 된다. 공개 benchmark를 이용한 탐색 실험이다.

## 정렬·시간구간 검토

첫 실행 로그에는 `current_times/shape/current_fields` 결합 assertion 실패가 있었다. 이후 소스에는 다음 검사가 추가됐다.

- 모든 run이 기준 시점의 공통 prefix를 갖는지 검사하고 그 길이로 displacement를 자른다.
- 선택한 deformable node의 초기 xyz가 기준 run과 `allclose(atol=1e-4)`인지 확인한다.
- 선택 노드의 thickness가 양수인지 확인한다.

이는 고정 mesh의 mean-field 및 RBF comparator 정합성에 필요한 개선이다. reference xyz 일치는 대응 노드에 대한 근거이나, connectivity·part ID 동일성을 별도로 검사하지는 않는다. 입력의 구조 geometry 변경과 메시 변경을 검증하는 실험으로 주장할 수 없다. `time_values_as_stored`와 실제 사용 horizon을 최종 결과에 명시해야 한다.

## baseline과 오차 정의

- `displacement_relative_l2`는 원 displacement를 분모로 사용하는 `||u_pred−u_FEM||₂ / ||u_FEM||₂`다. 절대 위치 크기로 분모를 부풀리지 않는다. t=0은 목표에서 제외한다.
- zero-displacement baseline은 분모가 0이 아니면 정확히 1이 되어야 한다. 결과 JSON에서 확인할 항목이다.
- `training_mean_field`는 training runs의 각 node·각 time·각 component 평균이다. test field를 평균내지 않는다. 고정 mesh에서 적절한 강한 baseline이다.
- RBF KernelRidge는 training의 global parameter에서 같은 고정 node/time field를 회귀한다. training global 통계로 정규화하고, training 평균장을 제거한 잔차를 학습한다. test 라벨을 사용하지 않는다.
- Transolver는 정규화 MSE, RBF는 원 단위 MSE로 내부 validation 선택을 한다. RBF는 12개 hyperparameter 조합을 탐색하지만 Transolver architecture는 단일 작은 구성이다. 실용적인 comparator로 유효하나, 동일 튜닝 예산의 알고리즘 순위 실험으로 볼 수 없다.
- 전체 relative L2는 큰 displacement run·시점에 가중되는 값이며 `per_run_relative_l2`의 산술평균과 다르다. 단일 숫자 외에 run/time별 오차를 함께 보고해야 한다.
- 구현에는 0 분모 guard가 없다. 현재 충돌 데이터의 목표 norm이 양수라면 문제가 드러나지 않지만, near-zero/zero field를 포함하는 향후 데이터에는 정의가 필요하다.

## 실제로 변하는 파라미터

`GLOBAL_FEATURES.json`의 131개 entry를 read-only 집계했다. 두께 scale은 1.21–2.00, wall Y는 −480.26–491.83, velocity_x는 **모든 entry에서 −5.0**이다. 동일 3변수 조합의 중복은 발견하지 못했다.

따라서 이 데이터로 실험하는 것은 주로 두께·pole offset 변화다. velocity feature가 코드에 들어 있다는 이유로 속도 일반화가 검증되었다고 말할 수 없다. 범퍼 기하·재료 law·접촉 topology의 일반화도 검증 대상이 아니다. 이 커뮤니티 데이터의 범위는 공식 135-run default DOE와 다르다.

독립 metadata 재계산에서 training 두께/offset 범위는 각각 [1.21, 2.00], [−480.26, 491.83], test 범위는 [1.24, 1.71], [−415.52, 298.95]였다. 7개 test 모두 training의 각 변수 min/max 안에 있다. 이는 축별 bounding box 내 보간 평가이며, convex hull 포함 여부나 다른 충돌 모드 coverage까지 증명하지 않는다.

## 시간·메모리·단위 보고의 한계

- `training_seconds`는 학습 epoch loop 시간이며 다운로드·패키지 설치·데이터 load를 제외한다. `load_seconds`는 별도다.
- CPU inference는 warm-up 뒤 이미 메모리에 올라간 7개 run의 model forward와 결과 역정규화를 잰다. FEM 실행과 비교한 전체 workflow speedup으로 보고할 수 없다.
- `peak_sampled_rss_bytes`는 시작과 epoch 말의 process RSS 표본 최대값이다. optimizer/forward 중 순간 peak나 kernel baseline/plotting 단계 peak를 포함하지 않을 수 있다. “측정된 표본 RSS 최대”로 표시해야 한다.
- VTP 자체 unit metadata는 없으나, 뒤이어 확보된 `Run100_0000.rad`의 `/BEGIN`에서 kg/mm/ms, `Run100_0001.rad`에서 100 ms·10 ms 출력 간격을 직접 확인했다. 동일 원본 변환이라는 provenance 아래 displacement RMSE는 mm, 시간은 ms로 해석할 근거가 있다. 응력은 현재 예측하지 않으며 임의로 MPa라고 붙이지 않는다.
- pip-check에는 환경 내 dependency conflict가 기록되어 있다. 이 실행 성공이 환경 전체가 dependency-clean하다는 뜻은 아니다.

## 시각화 검토

GPU 산출물 `fem_vs_prediction.png`와 `learning_and_error.png`를 직접 열어 확인했다. FEM/Transolver magnitude는 공통 0–약250 mm 색범위, error는 별도 약0–29 mm 범위로 읽힌다. 글자·단위·legend가 겹치거나 잘린 문제는 보이지 않는다. 첫 그림은 Run224, t=100 ms의 **초기 좌표 위에 변위 크기를 색으로 나타낸 point scatter**다. 변형된 mesh를 겹쳐 그린 형상 비교 그림은 아니다. 학습/내부 validation 곡선과 세 baseline의 시간별 held-out 오차를 구분한다.

후속 `deformed_geometry_comparison.png`도 직접 확인했다. 이 그림은 실제 `reference+displacement` 위치에서 FEM·Transolver·RBF를 나란히 놓고 회색 초기 좌표를 덧그린 1배율 XY projection이다. 공통 colorbar와 1,024 sampled nodes 표기가 명확하다. 면/요소 연결을 그린 전체 FEM mesh 그림과는 구별한다.

## 출처·재현성 보존

source manifest에 공식 crash 예제 commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`, dataset commit `78f1302fba4d75926c6fdd22d01d5133efc14742`, 파일별 SHA256이 있다. 실제 학습 library는 설치 package version과 그 실행 기록으로 구분한다. 별도로 저장한 최신 model 소스가 installed package에서 실행된 것과 같다고 가정하지 않는다.

CPU 학습 중 disk의 runner가 GPU/device/checkpoint 함수까지 추가하는 형태로 바뀌었다. 실행 중 프로세스는 시작 당시 코드를 사용한다. 최종 GPU 실행은 `gpu1024/executed_source.py`와 JSON의 `execution_source_sha256`가 일치함을 독립 확인해 재현 대응이 보존됐다. CPU 부분 실행과 GPU 완성 실행을 혼동하지 않아야 한다.

확보된 원 `.rad` 두 파일 header에 Altair copyright와 **CC BY-NC 4.0**을 직접 확인했다. 이는 NVIDIA 코드의 Apache-2.0 및 커뮤니티 runner의 MIT와 별개다. HF card의 데이터 별도 라이선스는 미명시 상태이나 원 model의 비상업 조건을 무시하고 전체가 무제한 상용 데이터라고 취급해서는 안 된다. 회사의 자체 모델·데이터로 같은 기술 경로를 쓰는 문제와 이 benchmark 자산의 이용 조건을 구별한다. [CC BY-NC 4.0 공식 조건](https://creativecommons.org/licenses/by-nc/4.0/)

## GPU 1,024node 최종 수치 독립 검증

실제 package `nvidia-physicsnemo 2.2.2`, PyTorch `2.13.0+cu126`, device `cuda`. train/val/test는 112/12/7. 1,024개 sampled node에서 t=10..100 ms의 10시점×xyz를 한 번에 출력한다. history 60행, 내부 validation 최저 epoch=27이 JSON의 best epoch와 일치한다.

이번 학습은 FEM displacement 라벨에 대한 MSE 지도학습이다. 접촉·소성이 포함된 reference의 변위를 학습하지만, PDE residual·에너지 보존·접촉 비관통 constraint loss를 별도로 적용하지 않는다.

| 예측 | displacement relative L2 | xyz component RMSE (mm) |
|---|---:|---:|
| PhysicsNeMo Transolver | 0.0637568283 | 5.36853486 |
| Training mean field | 0.2058786710 | 17.33566196 |
| RBF KernelRidge field | **0.0314411061** | **2.64744465** |
| Zero displacement | 1.0 | 같은 FEM target norm에 따른 기준 |

`heldout_predictions.npz`에서 별도로 float64 norm/RMSE를 계산했다. relative L2의 JSON 차이는 모든 모델에서 4×10⁻¹¹ 이하, 모든 prediction·truth는 finite다. zero baseline은 정확히 1이고 분모는 양수다. split 교집합 세 쌍은 모두 공집합이다. NPZ shape는 truth/prediction `[7,1024,30]`, 좌표 `[7,1024,3]`, node index `[1024]`다.

kernel checkpoint의 `X_fit`, dual weights, gamma=1.0, alpha=0.01, training global 통계로 RBF를 별도 재계산했다. NPZ prediction과 최대 절대 차이는 0.000672 mm로, float32 pairwise distance 계산 방식 차이 수준이다. checkpoint의 training 평균장과 NPZ mean baseline 차이는 정확히 0이다.

따라서 **이 작은 고정 mesh·2변수 연구에서는 RBF가 작은 Transolver보다 오차가 작았다**고 보고해야 한다. Transolver의 학습·추론 성공과 최고의 알고리즘 선정은 구별한다. 6.38% relative L2를 “정확도 93.62%”라고 변환하지 않는다.

비교 모델의 저장 크기도 다르다. 이 run의 NN checkpoint는 약193 kB, kernel checkpoint는 약12.97 MB다. RBF의 각 node/time 출력은 고정 mesh에 연결된 다중 출력 회귀이므로, 이 결과만으로 geometry가 달라지는 surrogate의 우열을 결정할 수 없다.

7개 test의 Transolver relative L2는 0.04757–0.07857 범위다. test가 7개이고 모두 축별 training 범위 안이라는 사실 때문에 새로운 속도·재료·topology·낙하방향·제품에 대한 일반화 결론을 낼 수 없다.

평균 오차의 한계를 보기 위해 7run×1,024node×10시점의 **변위 벡터 오차 크기**를 독립 계산했다.

| 예측 | median vector error (mm) | P95 (mm) | P99 (mm) | 최대 (mm) |
|---|---:|---:|---:|---:|
| Transolver | 6.350 | 18.106 | 24.855 | 46.037 |
| RBF | 1.466 | 10.969 | 15.412 | 36.454 |
| Training mean | 18.584 | 58.289 | 72.684 | 110.943 |

Transolver 최대값은 Run227, 100 ms, 원 VTP의 zero-based point index 5854에서 발생한다. 이 index는 Radioss node ID라고 단정하지 않는다. 같은 점 FEM displacement magnitude는 약269.34 mm, 예측은 약223.41 mm다. P95·최대는 **sampled point/time 집합** 기준이며 전체 mesh·연속 시간·누락한 hotspot의 상한이 아니다. 앞의 5.37 mm RMSE는 xyz component 기준이고, 벡터 오차 RMS는 9.30 mm로 정의가 다르다.

### 추가 물리 진단

후처리 진단으로 각 run·시점에서 노드 평균 변위를 빼 보았다. 평균 병진 성분은 원 displacement squared norm의 약 **78.23%**를 차지한다. 평균 제거 후 상대 L2는 Transolver 0.10062, RBF 0.04886, training mean 0.33333이다. 이는 회전까지 제거한 완전한 rigid alignment가 아니며, 별도 진단 수치다. 원 변위 6.38%가 잔류 찌그러짐·소성 변형·응력 6.38%를 뜻하지 않는다는 점을 보여준다.

### 자원 수치의 해석

training loop 152.72초, data load 35.76초, warmed forward 7run 합계 0.04364초(6.23 ms/run), kernel 선택·fit 1.47초다. neural training 시간에는 내부 validation과 checkpoint 출력이 포함된다. CUDA peak allocated는 44,537,344 byte(약42.47 MiB), sampled process RSS는 1,468,145,664 byte(약1.37 GiB). Torch allocator 수치는 CUDA context·다른 GPU 프로세스·전체 system 메모리와 다르다. 이 결과를 전체 FEM 해석 대비 end-to-end speedup 또는 전체 가전 최소 VRAM으로 쓰지 않는다.

**1,024node 검증 결론:** 실제 GPU·공개 FEM·실제 PhysicsNeMo library에서 학습과 held-out displacement 평가가 완료됐으며 기록된 수치는 독립 재계산과 일치한다. 위 절은 `outputs/experiment/gpu1024` 범위다. 이후 full mesh probe와 전체 node 실학습은 아래에 별도 기록한다.

## 후속 full mesh probe: 소스·기록 검토

`outputs/experiment/full_mesh/executed_probe.py`와 `full_mesh_probe.json`을 읽었다. probe는 원 checkpoint를 reload해 Run224의 기존 1,024node prediction과 비교하고, 1,024/4,096/13,675 node에서 warm-up 뒤 forward/backward/Adam update를 3번 측정한다. 이후 원 checkpoint를 다시 restore하고 `eval()`로 바꿔 7개 held-out run 전체 deformable node를 추론한다. 따라서 자원 step probe의 임시 weight update가 최종 full mesh 추론 결과에 섞이지 않도록 되어 있다.

- checkpoint reload 결과: Run224, 최대 절대 차이 0.0 mm, assert 통과.
- 13,675 node의 실제 학습 step 3회 평균: 23.77 ms, CUDA peak allocated 118,536,192 byte(약113.05 MiB).
- **1,024node로 학습한 모델의 13,675node 추론** 보고값: relative L2=0.0637559573, component RMSE=5.37588778 mm, train mean baseline L2=0.2055027147.
- 7run의 warmed synchronized inference는 각 39.28–44.40 ms로 보고한다.

이는 전체 deformable mesh **추론**과 단일 run의 실제 update **자원 probe**다. 전체 mesh로 60epoch 학습했다는 뜻은 아니다. 41k parameter의 작은 architecture 측정값을 공식 GeoTransolver나 백만 node 가전 조립체의 최소 VRAM으로 일반화하지 않는다.

이 최초 full mesh 산출물에는 prediction/truth array가 없어 source·formula·checkpoint roundtrip은 검토했지만 위 전체-node L2/RMSE를 NPZ에서 독립 재계산하지는 않았다. 추가 array/aggregate가 제공되면 별도 검증한다. 시간 출력은 10 ms 간격이므로 그보다 짧은 접촉 pulse·국부 가속도 peak의 정확도도 이 결과로 입증되지 않는다.

## 최종 전체 13,675node 실학습 검증

`outputs/experiment/fulltraining`은 위 probe와 별개로 **전체 deformable node에서 실제로 60epoch 학습**한 run이다. 원본 13,676point 중 thickness>0인 13,675개를 모두 사용했다. 1,024node run과 seed·112/12/7 split·41,286 parameter의 model config가 같고, dense RBF fitting만 호스트 RAM 압박 때문에 건너뛰었다. 새 실행 source snapshot과 JSON hash가 일치한다.

독립 확인 결과는 다음과 같다.

- NPZ truth/prediction shape `[7,13675,30]`, 모든 수치 finite, zero baseline=1.
- history 60행, 내부 validation 최저 epoch 27이 저장된 best epoch와 일치.
- NPZ float64 재계산 Transolver relative L2=**0.06120344835685**, component RMSE=**5.16066080709 mm**. JSON과 L2 차이 1.3×10⁻¹² 이하.
- training mean field relative L2=0.20550271457624, component RMSE=17.3279420251 mm. JSON과 L2 차이 6.6×10⁻¹² 이하.
- 이전 GPU run과 split 목록 및 model config 동일.

| 전체-node Transolver 변위 벡터 오차 | 값 |
|---|---:|
| median | 5.8854 mm |
| P95 | 17.5929 mm |
| P99 | 24.1506 mm |
| 최대 | 46.8028 mm |

최대값은 Run227, 100 ms, 원 VTP zero-based point index 5853에서 발생했다. 이 집계는 7×13,675×10개 point/time의 벡터 오차이며 xyz component RMSE와 정의가 다르다.

평균 병진 성분은 전체 displacement squared norm의 **78.3176%**다. 각 시점 노드 평균을 제거한 후의 진단 relative L2는 **0.0976487568**(9.7649%), component RMSE는 3.83398 mm다. 회전까지 제거한 rigid alignment는 아니며, 응력·소성·잔류 영구변형 오차로 해석하지 않는다.

RBF는 전체 mesh fitting을 하지 않았으므로 전체-node NN 6.1203%와 sampled RBF 3.1441%를 동일 support의 직접 비교라고 표시해서는 안 된다. 공정한 point 집합 비교를 위해 fulltraining 결과에서 이전 1,024point를 추출해 독립 계산했다. 두 파일 FEM truth의 차이는 0이었다. **같은 1,024point에서 전체-node 학습 NN relative L2=0.0616159606 / RMSE=5.18827 mm, 기존 RBF=0.0314411061 / 2.64744 mm**다. RBF가 이 고정 mesh·저차원 보간 실험에서 우세하다는 결론은 유지된다.

전체-node training loop는 **260.53초**, data load 39.97초, warmed forward는 7run 합계 0.08253초(11.79 ms/run)다. CUDA peak allocated는 **383,345,152 byte(약365.59 MiB)**, sampled process RSS는 **1,836,900,352 byte(약1.71 GiB)**다. 전체 system peak나 CUDA context 포함 총 VRAM 값으로 표시하지 않는다. 주변 프로세스를 포함한 호스트 RAM 사용률 97% 기록은 실험 고유의 최소 RAM 요구량과 구별한다.

전체-node `fem_vs_prediction.png`와 `learning_and_error.png`를 직접 열어 확인했다. 범례·축·단위·공통 FEM/NN color range가 정상이며, RBF를 실행하지 않은 run의 error plot에 RBF curve를 넣지 않았다. 기본 field 그림은 초기 좌표 위 displacement 색맵이다.

후속 `fulltraining/deformed_geometry_comparison.png`도 직접 확인했다. Run224, 100 ms의 `reference+displacement`를 1배율 XY projection으로 비교하고 회색 원 형상을 함께 나타낸다. FEM/NN의 공통 색범위와 13,675 evaluated nodes가 명시돼 있다.

### 저장 checkpoint의 독립 추론 재사용

`inference_reuse/executed_inference.py`는 새 model instance에 저장 checkpoint를 불러와 `eval()`·inference mode로 추론한다. FEM target은 예측 후 선택적 오차 계산 단계에서만 읽고 fitting을 하지 않는다. `inference_result.npz`를 독립 확인해 다음을 검증했다.

- 새 Run224 전체-node prediction과 학습 종료 때 저장 prediction의 최대 차이 **0.0 mm**.
- 새로 읽은 FEM truth와 원 평가 truth의 최대 차이 **0.0 mm**.
- 재계산 Run224 relative L2=0.06442129962394, component RMSE=4.85940432576 mm로 inference report와 일치.
- mesh contract의 checkpoint SHA256·node index SHA256·reference coordinate SHA256이 실제 저장 자산과 모두 일치.
- `inference_wrong_mesh_rejection.log`에서 노드 좌표 순서가 다른 입력을 geometry/node-order assertion으로 거부한 결과를 확인.

따라서 저장된 NN artifact를 새 모델로 불러와 고정 geometry·node ordering 조건에서 재사용하는 경로까지 실제 검증됐다. mesh hash 일치는 물리 parameter coverage나 새 제품에 대한 정확도를 보장하지 않는다.

**최종 판정:** 실제 공개 FEM displacement 데이터와 PhysicsNeMo library를 이용한 전체 13,675 deformable node GPU 학습·추론·7run 평가가 완료됐고, 저장 수치는 독립 재계산과 일치한다. 코드/데이터 누설이나 오차 분모의 오류는 발견하지 못했다. 작은 고정 topology 범퍼의 보간 실험으로서 유효하며, 가전 조립체의 충돌·낙하·파손 surrogate가 검증됐다는 주장은 뒷받침하지 않는다.

## 추가 용량 민감도 실험: capacity1024

`outputs/experiment/capacity1024`는 기존 두 run을 보존한 별도 탐색 실험이다. 층2→4, hidden48→96, slice8→16으로 바꾸고 head4를 유지했다. 실제 parameter는 293,582개다. 동일 seed·split·1,024point·시간구간·optimizer·60epoch 예산을 사용했다. 사전 resource probe는 실제 FP32 forward/backward/Adam step 10회 평균 40.41 ms, peak CUDA allocation 44,851,200 byte, 10% overhead를 더한 예상 training 298.73초를 기록한다. 이 예상과 실제 학습 시간을 구분한다.

실행 source를 읽어 split 생성·training-only 정규화·내부 validation 최저 checkpoint 선택이 유지됨을 확인했다. test 라벨·test 점수를 선택 조건으로 사용하는 분기는 없다. 기존 평가를 이미 확인한 뒤 수행한 공개 benchmark의 용량 민감도 분석이며, 독립 blind test나 보편적 방법 순위 시험은 아니다. 단일 seed이고 network 초기화도 바뀌므로 이 한 run만으로 용량의 인과 효과를 확정하지 않는다.

### 독립 수치 검산

- 실제 PhysicsNeMo2.2.2·CUDA. 실행 source SHA256 일치, history60행, 내부 validation 최저 epoch60과 JSON best epoch 일치.
- 기존 `gpu1024`와 split 목록 동일·세 split 교집합0. node index와 time 배열 동일, 좌표·FEM truth·training mean field의 최대 차이 모두 **0.0**.
- NPZ shape `[7,1024,30]`, prediction/truth finite, zero baseline=1.
- 독립 relative L2=**0.05476745658969**, component RMSE=**4.61160016769 mm**. JSON과 L2 차이 1.8×10⁻¹¹ 이하.
- 실제 60epoch training291.08초, warmed forward9.70 ms/run, peak CUDA allocated65,503,232 byte, sampled RSS1,471,422,464 byte.

동일 7run×1,024point×10시점의 비교는 다음과 같다. RBF는 기존의 내부 validation 선택 결과를 재사용했으며 새 fitting·test 기반 조정이 없다.

| 방법 | 변위 relative L2 | component RMSE (mm) | vector P95 (mm) | vector 최대 (mm) |
|---|---:|---:|---:|---:|
| 기존 작은 NN | 0.06375683 | 5.36853 | 18.1059 | 46.0368 |
| 추가 용량 NN | 0.05476746 | 4.61160 | 14.5701 | **112.9247** |
| 기존 RBF | **0.03144111** | **2.64744** | **10.9694** | **36.4540** |

큰 NN의 median vector error는4.6337 mm, P99는22.1284 mm다. 평균·P95는 좋아졌지만 최대 오차는 악화됐다. 최악값은 Run227, 90 ms, 원 VTP zero-based point index10171에서 발생한다. 그 점의 FEM displacement는 `[10.1536,28.0312,−27.6430] mm`, 큰 NN은 `[−98.1015,2.1202,−8.6311] mm`다. 같은 점·시점의 작은 NN 벡터 오차38.5501 mm, RBF32.7578 mm보다 크다. 71,680개 point/time 중138개에서 큰 NN의 벡터 오차가50 mm를 넘었다.

평균 병진을 제거한 진단 relative L2는 큰 NN0.10753394·component RMSE4.2250 mm다. 작은 NN0.10061726보다 높다. 따라서 평균 displacement 오차 감소를 국부 변형 정확도의 전반적 개선으로 해석할 수 없다. 이 진단은 회전 제거·영구 변형·응력 오차 평가와는 다르다.

best epoch가60epoch 예산의 마지막이고 validation MSE도 epoch55의0.20063에서60의0.18809로 낮아졌다. **수렴 완료를 입증하지 못했으며**, 추가 학습의 개선량·과적합 가능성은 이번 결과로 확정할 수 없다. test 점수에 따른 추가 hyperparameter 재조정을 수행한 것으로 기록하지 않는다. RBF 우세는 이 고정 mesh·2변수·현재 계산 예산에서 관측한 결과로 한정한다.

## 공개 문서의 scope·오차 정의 검토

`outputs/physicsnemo-impact-lab/README.md`와 `docs/experiment-results.md`의 현재 본문을 read-only 확인했다. 다음 표현은 수치와 소스에 맞는다.

- 전체13,675node 실학습과 동일1,024point baseline 비교를 별도 표·문단으로 구별한다.
- displacement 상대 L2의 분모는 기준 displacement norm이며 원 좌표가 포함된 position norm을 쓰지 않았다고 명시한다.
- 공식 crash postprocessing의 position-relative L2와 직접 비교하지 않는다고 명시한다. 실제 보존된 `post_processing/compute_l2_error.py`의 `relative_position_error`는 `mesh.points`의 absolute position을 분모로 쓰는 것을 확인했다.
- RBF는 고정 node 정렬의 global 조건→field 회귀로 설명하고, 새로운 geometry/topology·속도·가전 파손 검증과 구별한다.
- Run224의 앞선 CPU 기능 검사 사용 사실을 공개해 완전 blind test로 포장하지 않는다.

정의 차이의 독립 산술 예로, 동일한 fulltraining prediction을 displacement norm으로 평가하면6.1203%이고 `reference+FEM displacement`의 position norm으로 나누면1.4319%다. 같은 prediction도 분모에 따라 숫자가 달라지므로 benchmark의 norm·support·aggregation을 맞춰야 한다.

Capacity 결과를 공개 문서에 추가할 때는 평균·P95 개선과 최대 오차 악화, 마지막 epoch 선택·수렴 미확증, 고정 geometry의 탐색 실험이라는 범위를 함께 유지해야 한다. 본 검토자는 공개 문서를 수정하지 않았다.
