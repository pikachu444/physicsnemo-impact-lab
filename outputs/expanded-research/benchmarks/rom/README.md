# POD / ROM 회귀 실험

실제 FEM 범퍼·인장·휴대폰 데이터셋으로 POD+RBF, POD+Ridge, 직접 RBF를 모두 CPU에서 학습·검증·시험했다. 이들은 PhysicsNeMo 신경망과 비교하는 수치 축약·회귀 baseline이다. 최종 모델을 고른 뒤 시험 결과를 보고 재튜닝하지 않았다. 아래 오차는 **학습에 쓰지 않은 고정 시험 사례**의 결과다.

## 최종 실행: 새 인장 FEM 13조건, 7채널

919개 물리 shell 연결 절점·10시점·변위XYZ/응력/EPSP/총변형률 principal major·minor를 사용했다. 중심·네 모서리5조건 train, 내부4조건 validation, 다른 내부4조건 test다. 재료 yield/UTS 배율과 속도 배율의 두 변수, 고정 형상, 매우 작은 초기 데이터셋이다. 낮은 오차를 가전제품 낙하·접촉·파손으로 일반화할 수 없다.

| 방법 | 검증으로 고른 설정 | VM L2 / RMSE | EPSP L2 / RMSE | 총변형률 major / minor L2 |
|---|---|---|---|---|
| POD+RBF | rank2, gamma0.1, alpha0.1 | 0.2310% / 1.1925 MPa | 0.4753% / 1.4593e-4 | 0.4489% / 0.4536% |
| POD+Ridge | rank2, degree2, alpha0.01 | 0.1365% / 0.7046 MPa | 0.4100% / 1.2586e-4 | 0.3876% / 0.3884% |
| 직접 RBF | gamma0.1, alpha0.1 | 0.2291% / 1.1828 MPa | 0.4370% / 1.3415e-4 | 0.4124% / 0.4191% |

POD rank2는 중심화·물리장 정규화 후 train 변동의99.99374%를 보존한다. 물리적인 변형에너지 보존율이 아니다. POD 두 방법 합계 실행4.69초·최대105.17MB, 직접 RBF0.82초·최대46.03MB였다. 세 체크포인트 모두 새 프로세스 재추론과 저장 예측의 최대 차이0을 확인했다.

- [POD 실측·모든 물리장 지표](coupon13/metrics.json), [직접 RBF](coupon13-direct-rbf/metrics.json), [별도 수식·support·peak 진단](coupon13/frozen_support_peak_diagnostics.json).
- FEM/예측/절대오차 그림: [POD+RBF](coupon13/pod_rbf/stress_strain_comparison.png), [POD+Ridge](coupon13/pod_ridge/stress_strain_comparison.png), [직접 RBF](coupon13-direct-rbf/stress_strain_comparison.png). 같은 첫 시험조건·마지막10ms·고정 reference 형상이며 FEM/예측 색 범위는 공유하고 오차에는 독립 범위를 썼다.

## 최종 실행: 공개 범퍼 131조건, 상하층 포함 7채널

전체13,675절점·10개 비영시점, 112 train / 12 validation / 7 test를 사용했다. 변위XYZ와 shell 상하층 in-plane VM, 상하층 EPSP가 목표다. 원본131조건은 **총변형률을 출력하지 않았으므로 이 표의 변형률은 등가소성변형률 EPSP**다.

| 방법 | 검증으로 고른 설정 | 변위 L2 | VM 상/하 L2 | VM 상/하 RMSE | EPSP 상/하 L2 |
|---|---|---:|---|---|---|
| POD+RBF | rank27, gamma3, alpha0.1 | 3.0320% | 15.06% / 15.57% | 43.36 / 45.20 MPa | 51.85% / 55.02% |
| POD+Ridge | rank109, degree2, alpha1e-4 | 6.6234% | 28.17% / 28.56% | 81.10 / 82.91 MPa | 68.31% / 70.28% |
| 직접 RBF | gamma3, alpha0.1 | 3.0530% | 15.45% / 15.96% | 44.49 / 46.33 MPa | 52.75% / 56.30% |

원 배열·metrics의 응력 단위는GPa이고 표와 그림에서 **×1000하여 MPa**로 표시했다. 상하층은 바깥쪽 integration point다. VM은 local in-plane tensor의 invariant, EPSP는 solver scalar를 초기 shell 면적으로 가중하여 절점에 투영한 값이다. 원 element/integration-point 최대값이나 전체 두께 평균 solver VONM과 같지 않다.

POD+RBF는 변위가 약3%여도 EPSP는 약52~55%였다. EPSP 음수는 상층208,644/957,250, 하층215,197/957,250개이며 최소−0.06594/−0.06765다. 값을 clipping하지 않았다. 원래 EPSP가 비영인 support만 계산해도 L2는49.62%/52.53%다. 원 정답 peak 위치에서 가장 크게 놓친 투영 절점 EPSP 크기는0.1992/0.3694였다. 현재 모델은 이 범퍼의 소성·파손 판단을 대체할 정확도로 검증되지 않았다.

원 shell status0은Run100/130/139/144에서 각각 최대1/2/1/1개 관찰됐다. 생산자 정책은 이들을 제거하지 않는 고정 초기 topology projection이다. 시험7조건에는 status0이 관찰되지 않아 erosion/damage 정확도 검증으로 사용할 수 없다. 이 정의를 [동결 진단](bumper7/frozen_support_peak_diagnostics.json)에 보존했다.

POD 실행59.21초·최대279.80MB, 직접 RBF34.81초·최대253.17MB였다. 파일 hash 검사·준비·입출력·그림을 포함한 시간이며 모델 forward만의 지연시간과 혼동하지 않는다. 세 체크포인트 모두 새 프로세스에서 예측이 정확히 재현됐다.

- [POD 실측](bumper7/metrics.json), [직접 RBF](bumper7-direct-rbf/metrics.json), [층·support·peak·음수 진단](bumper7/frozen_support_peak_diagnostics.json), [라벨 정의](bumper7/input_manifest.json).
- FEM/예측/절대오차 그림: [POD+RBF](bumper7/pod_rbf/stress_strain_comparison.png), [POD+Ridge](bumper7/pod_ridge/stress_strain_comparison.png), [직접 RBF](bumper7-direct-rbf/stress_strain_comparison.png). Reference XY를 보기 좋게 Y수평/X수직으로 표시했으며 변형 형상 사진이 아니다.

## 최종 실행: 새 휴대폰 FEM 9조건, 전체 조립체와 후면 커버

공개 RAD 원 deck으로 **이번 연구에서 OpenCourant를 실제 실행해 생성한** 9조건의 초기 20µs 응답을 사용했다. 기존 공개 정답9개를 받은 것이 아니다. 원 starter SHA256은 `435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2`다. 동결 wrapper의 OpenRadioss/public 문구는 [공통 provenance 정정](../drop_test/common-cache-provenance-clarification.json)과 [ROM 정의 메모](phone9-metadata-clarification.json)로 명확히 했으며 cache/hash/예측은 수정하지 않았다.

전체17부품의 **선형화 코너 절점76,065개**와 그중 part2 후면 커버19,933개를 각각 학습했다. 원 RAD257,099절점 중 TETRA10 midside181,034개는 출력 support에서 제외됐다. 전체 원 FEM 모든 절점을 사용한 모델로 표현하면 안 된다. 같은5 train/2 validation/2 test, 공통 Young modulus 배율0.8~1.2와 벽 법선 Rx −1~1도의 두 입력이다. 원5재료를 독립으로 바꾸거나 새로운 조립체·접촉 topology를 검증한 실험이 아니다.

| support / 방법 | 변위 L2 | 중심이동 제거 변위 L2 | GPS VM L2 | 총변형률 Frob L2 | EPSP L2 |
|---|---:|---:|---:|---:|---:|
| 전체 POD+RBF | 0.02620% | 1.3560% | 2.6163% | 2.1109% | 30.9711% |
| 전체 POD+Ridge | 0.02847% | 1.4741% | 2.2495% | 2.1358% | 25.0761% |
| 전체 직접 RBF | 0.02620% | 1.3560% | 2.6163% | 2.1109% | 30.9711% |
| 후면 POD+RBF | 0.05083% | 1.3571% | 2.5452% | 2.1109% | 30.9711% |
| 후면 POD+Ridge | 0.05565% | 1.4861% | 2.2556% | 2.1359% | 25.0761% |
| 후면 직접 RBF | 0.05083% | 1.3571% | 2.5452% | 2.1109% | 30.9711% |

선정은 각 domain의 train-only 공유 정규화와 validation MSE만 사용했다. 두 POD 방법은 rank4를 선택했고5개 train snapshot의 양수4모드를 전부 보존했다. 이 경우 POD+RBF와 직접 RBF는 같은 train field span을 사용하여 수치적으로 거의 같으므로 서로 다른 학습법의 우열 증거로 세지 않는다. RBF는 gamma0.1/alpha1e-4, Ridge는 degree2/alpha1e-4를 골랐다. 전체 POD93.57초·최대404.19MB/직접 RBF10.40초·363.90MB, 후면 POD20.17초·136.73MB/직접 RBF1.34초·124.43MB였다. 세 방법×두 support의 체크포인트 모두 새 프로세스 추론과 최대 차이0이었다.

응력은 native nodal GPS tensor 6성분으로 계산한 VM(MPa)다. 총변형률은 valid integration point 처리 후 초기 선형화 solid 체적으로 global tensor를 절점 투영하고 Frobenius norm을 계산했다. EPSP는 별도 native cell scalar의 체적 가중 절점 투영이다. 원 element/IP 극값이나 총변형률과 EPSP를 혼동하지 않는다. 목표10시점은 실제 raw12시점에서 명목2~20µs로 선형 보간했으며 외삽하지 않았다. 후속100µs 단일 FEM은 이 학습 배열에 포함하지 않았다.

전체 시험의 EPSP는1,521,300개 절점·시점 값 중27개만 비영이며 전부 후면 커버에 있다. 첫 testcase는 EPSP 정답이 전부0이므로 그 사례의 상대 L2는 정의되지 않는다. 두 사례를 합친 표의 분모는 다른 testcase의 소수 소성값에서 온다. 0영역이 많아 EPSP의 P95 절대오차0 또는 매우 작은 RMSE를 성공 근거로 쓰면 안 된다. 음수 예측48개(RBF)/39개(Ridge)를 그대로 보존했다. 큰 병진 성분을 포함하는 변위 L2와 중심이동 제거 shape 지표의 차이도 표에 함께 제시했다.

- 전체 [POD 지표](phone9-full/metrics.json), [직접 RBF](phone9-full-direct-rbf/metrics.json), [별도 수식·support·peak·EPSP history 검산](phone9-full/frozen_support_peak_diagnostics.json).
- 후면 [POD 지표](phone9-rear/metrics.json), [직접 RBF](phone9-rear-direct-rbf/metrics.json), [동일 진단](phone9-rear/frozen_support_peak_diagnostics.json).
- 전체 FEM/예측/오차: [첫 조건](phone9-full/pod_rbf/stress_strain_comparison.png), [둘째 조건](phone9-full/pod_rbf/stress_strain_comparison_case1.png). 같은 파일명이 Ridge/direct/후면 폴더에도 있다. 참조 XY 스칼라 지도이며 변형된3D 형상 사진이 아니다. 부품 겹침으로 내부 국소값이 가려질 수 있다.
- 국소 소성 비교는 [전체 EPSP 확대](phone9-full/pod_rbf/epsp_truth_support_zoom.png)와 [후면 EPSP 확대](phone9-rear/pod_rbf/epsp_truth_support_zoom.png)를 본다. 고정 test 두 조건 모두를 표시하며, 두 정답의 정확한 비영 영역 합집합 주변을 같은 범위로 확대한 사후 시각화다. XY 겹침에는 절대값이 큰 점을 나중에 그려 국소값을 드러냈고 정의를 각 그림의 JSON에 기록했다.

## EPSP 누적 시간 이력의 추가 진단

정답과 raw 예측에서 `np.diff(time)`의 모든 음수를 세었으며 임의 허용오차·성공기준·clipping을 두지 않았다. 네 support의 FEM 정답에는 감소가 없었다. Phone full/rear는 RBF41회·최소Δ약−2.2910e-4, Ridge33회·−1.8669e-4였다. 범퍼 POD+RBF는 상/하층202,554/207,753회 감소했고 최소Δ−0.02400/−0.02481이었다. 이 모델들은 누적성 제약을 만족하도록 학습된 모델이 아니다.

Coupon의 감소는27/29/75회, 최소Δ약−6~8e-9로 훨씬 작았다. 이를 임의로0 취급하지 않고 저장 float32 endpoint의 ULP 대비 감소 비율과 사례별 감소 크기 분위수를 [요약 JSON](epsp-history-summary.json) 및 각 support 진단에 보존했다. 총변형률 Frobenius나 signed principal minor는 unloading 때 감소할 수 있으므로 이 누적성 요구를 적용하지 않는다. 이전 coupon/범퍼 진단은 `frozen_support_peak_diagnostics.before-epsp-history.json`으로 보존했고 예측·학습·선정은 변경하지 않았다.

## 공통 재현·선정 규칙

원 writer exit0과 READY를 확인하고 모든 파일 hash를 검사한 뒤 학습했다. 공유된 train-only 정규화·평균장을 다시 계산해 exact equality로 대조했다. 변위XYZ는 회전불변 공통 scale `sqrt((var_x+var_y+var_z)/3)`, scalar는 각 train std를 썼다. 원 라벨의 잡음 축이나 음수를0으로 치환하지 않았다. BLAS1 thread를 실제 확인했고 GPU는 쓰지 않았다.

POD는 채널 scale로 나눈 train snapshot에서 train 평균장을 뺀 뒤 block Gram/SVD로 구했다. 보존 분산90/95/99/99.9/99.99%에 대응하는 고유 후보 차수와 RBF의 gamma0.1/0.3/1/3·alpha1e-4/0.01/0.1, Ridge의 degree1/2·같은alpha를 validation MSE로 선택했다. 이 유한 grid의 선택 결과이며 전역 최적 설정이라고 주장하지 않는다. 시험 자료는 모델 선정에 쓰지 않았다.

[재사용 소스와 실행한 라이브러리 버전](source/)에 학습·체크포인트 추론·후처리 코드를 모았다. 각 실행 폴더의 `executed_*.py`, `normalization.json`, `validation_candidates.json` 또는 `selection.json`, `run.log`, `metrics.json`이 실제 실행 증거다. POD dual 기저는 로컬 train snapshot에 의존하므로 작은 NPZ 하나만으로 독립 배포되는 ROM은 아니다. 예측·기저·snapshot 배열은 원 FEM 유래이며 라이선스 검토 전 공개 재배포하지 않는다.

실제 휴대폰 full 실행 명령의 핵심 인자는 다음과 같다. 저장된 원 실행 폴더를 덮어쓰지 않도록 `--out`과 POD `--work`에는 새 경로를 준다. `python`은 기록된 버전의 NumPy·Matplotlib·psutil·threadpoolctl 환경을 가리켜야 하며, 원자료와 공통 cache를 먼저 복구해야 한다.

```powershell
python outputs/expanded-research/benchmarks/rom/source/run_pod_rom.py --cache work/expanded-ml/phone-full-cache-v2/manifest.json --work work/expanded-rom/reproduce-phone-full --out outputs/expanded-research/benchmarks/rom/reproduce-phone-full --ready-file work/expanded-ml/phone-full-cache-v2/READY.json --expected-manifest-sha256 aa0848d5da1ce3ab103a03bcad6ae356896869357d5eec00cde82bb8d1ef1dca --normalization work/expanded-ml/phone-full-normalization.json --mean-field work/expanded-ml/phone-full-training-mean-field.npy
python outputs/expanded-research/benchmarks/rom/source/run_direct_rbf.py --cache work/expanded-ml/phone-full-cache-v2/manifest.json --out outputs/expanded-research/benchmarks/rom/reproduce-phone-full-direct --mode train --ready-file work/expanded-ml/phone-full-cache-v2/READY.json --expected-manifest-sha256 aa0848d5da1ce3ab103a03bcad6ae356896869357d5eec00cde82bb8d1ef1dca --normalization work/expanded-ml/phone-full-normalization.json --mean-field work/expanded-ml/phone-full-training-mean-field.npy
```

후면은 `phone-rear-cache-v2`, `phone-rear-normalization.json`, `phone-rear-training-mean-field.npy`와 manifest SHA256 `fb2e0dd2eb7b15766ea6250c24812acf4678c135bdaee0b57227871a75d7fb48`를 사용한다. POD 재추론은 같은 `--cache`와 `--reuse <실행폴더>/pod_rbf` 또는 `pod_ridge`, 새 `--out`을 준다. 직접 RBF 재추론은 `--mode infer --checkpoint <실행폴더>/checkpoint.json --reference <실행폴더>/predictions.npy`와 새 `--out`을 준다. 실제 fresh reload 기록은 각 `inference_reuse` 폴더에 있다.

## 이전 개발 실행: 공개 범퍼 변위 3채널

[실측 결과](curated3/metrics.json), [실행 로그](curated3/run.log), [별도 수식 검산](curated3/author_verification.json), [차수 선택 그림](curated3/rank_selection.png), [물리 단위 오차](curated3/field_rmse.png).

기존 공개 FEM 131사례의 전체 13,675노드·10개 비영시점을 사용했다. 분할은 공통 112 train / 12 validation / 7 test이고, 모든 채널 통계·평균장·POD 기저는 train에서만 계산했다. 차수와 회귀 설정은 validation의 채널 표준화 MSE로 선택했다. GPU는 사용하지 않았고 BLAS 1 thread를 실제 확인했다.

이 개발 실행은 변위축별 표준편차를 썼다. 이후 최종7채널·coupon 실행은 승인된 공통 규칙에 따라 XYZ에 `sqrt((var_x+var_y+var_z)/3)`를 함께 적용하고 scalar는 각각 train 표준편차를 쓴다. 기존 개발 결과를 바꾸지 않으며 서로 다른 정규화 단계의 validation loss를 직접 비교하지 않는다.

| 방법 | 검증으로 선택한 설정 | 시험 변위 L2 | 성분 RMSE | 벡터 오차 P95 / 최대 |
|---|---|---:|---:|---:|
| POD + Gaussian kernel ridge | rank44, gamma3, alpha0.1 | 3.0996% | 2.6136 mm | 11.3395 / 41.0158 mm |
| POD + polynomial ridge | rank44, degree1, alpha0.1 | 10.0560% | 8.4792 mm | 29.9941 / 82.2878 mm |
| 직접 RBF (다른 담당자의 동일 조건 실행) | gamma3, alpha0.1 | 3.0530% | 2.5743 mm | 11.3660 / 35.8970 mm |

POD+RBF의 축별 상대 L2는 x1.7164%, y13.8051%, z65.6818%이고 z RMSE는2.1854 mm다. 큰 x방향 병진이 전체 L2의 분모를 지배하므로 전체3.10%만으로 작은 변형까지 정확하다고 판단할 수 없다. 중심 이동을 제거한 형상 상대 L2는4.7595%다.

후보 차수는 train에서 중심화하고 채널 표준화한 snapshot의 누적 분산90/95/99/99.9/99.99%에 대응하는11/22/44/71/96이었다. 선택한44차원의 보존 비율은99.0695%다. 여기서 분산 보존율은 **물리적인 변형에너지 또는 운동에너지 보존율이 아니다.** 선택 차수에서 validation의 이상적인 POD 투영 MSE는0.006659, 실제 RBF 예측은0.117645다. 축약 오차 외에 입력조건에서 계수로 가는 회귀 오차도 크다.

전체 준비·학습·선택·시험·그림 생성은28.92초, 최대 프로세스 RSS는230.82 MB였다. 큰 snapshot은 디스크 mmap으로 저장하고 작은 Gram 행렬만 메모리에 누적했다. 이 시간에는 파일 입출력이 포함된다. 두 방법 모두 별도 프로세스에서 체크포인트를 읽어 시험7사례를 재추론했으며 저장 예측과 최대 차이는0이었다.

모든 변위를 한 번에 출력하는 고정 메시 surrogate다. 입력 회귀는 전역조건을 사용하며 공간·시간 응답은 POD 기저에 들어 있다. 이 dual 기저 체크포인트는 로컬 train snapshot 파일에도 의존하므로, 현재 결과를 독립적인 작은 파일 하나로 배포 가능한 ROM이라고 표현하지 않는다. [직접 RBF와의 대조](curated3/direct_rbf_comparison.json)에서 지원점·분할 계약과 정답 배열의 bitwise 일치를 확인했고 평균장 중심화도 같다. 통계 계산법 개선에 따른 표준편차 차이는1.4e-11 mm 미만이다. POD는 validation 점수가 조금 낮고 test 변위 L2는 조금 높았으므로 어느 방법의 일반적인 우월성도 주장하지 않는다.

최종 응력·등가소성변형률7채널 결과와 새 구조 coupon 결과는 별도 폴더로 기록한다. 이 3채널 실행을 응력·변형률 학습 성공으로 간주하지 않는다. 최종 원본 writer의 종료와 READY/hash 고정이 끝난 뒤에만 추가 학습을 시작한다.

예측 NPZ·POD snapshot·평균장에는 원 FEM에서 유래한 배열이 포함된다. 내부CC BY-NC4.0 표시가 있는 공개 모델의 로컬 실험 증거로 보존하며, 공개 저장소에는 코드·통계·로그·그림과 원자료 경로를 우선 보존한다.
