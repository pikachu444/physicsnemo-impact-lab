# 공개 범퍼 FEM 데이터의 PhysicsNeMo 실행과 재현

OpenRadioss 범퍼 충돌 데이터로 PhysicsNeMo 2.2.2의 Transolver를 실제 학습하고, 별도 7개 해석 결과와 비교했다. NVIDIA 원본 crash 학습 드라이버도 별도 소규모 구성으로 전처리·학습·검증·체크포인트 저장까지 실행했다.

이 자료의 적용 범위는 같은 기준 메시에서 두께와 강체 폴 위치를 바꾸는 문제다. 초기 속도는 전 데이터에서 −5 mm/ms로 일정하다. 세탁기·TV 조립체의 임의 낙하 방향, 새 형상, 파손이나 응력 예측 성능을 검증한 결과는 아니다.

## 직접 측정한 결과

| 실험 | 학습 노드 | 모델 | 학습 시간 | 별도 7개 해석의 변위 relative L2 | 변위 성분 RMSE |
|---|---:|---|---:|---:|---:|
| `gpu1024` | 1,024 | Transolver 2층, 폭48, slice8 | 152.72초 | 6.3757% | 5.3685 mm |
| `fulltraining` | 13,675 | 같은 구조 | 260.53초 | 6.1203% | 5.1607 mm |
| `capacity1024` | 1,024 | Transolver 4층, 폭96, slice16 | 291.08초 | 5.4767% | 4.6116 mm |
| `gpu1024`의 RBF 회귀 | 1,024 | 전역 설계변수→정렬된 변위장 | 탐색·학습 합계1.47초 | 3.1441% | 2.6474 mm |

공통 분할은 학습112개, 내부 검증12개, 최종 평가7개다. 정규화와 평균장 기준선은 학습 데이터만 사용했다. 처음 두 신경망 실험은 60 epoch 중 내부 검증값이 가장 작았던27 epoch의 가중치를 평가했다. 큰 모델의 내부 검증 최선은 마지막60 epoch였다. 따라서 큰 모델이 충분히 수렴했다거나 알고리즘의 보편적 순위를 확인했다고 해석하면 안 된다. RBF 비교는 고정 메시의 저차원 설계변수 보간 결과이며, 다른 형상이나 메시로의 확장을 검증하지 않는다.

큰 모델은 평균 오차를 줄였지만 국부 오차를 모두 개선하지 않았다. 1,024노드의 벡터 오차95백분위는18.11→14.57mm로 낮아졌으나 최대값은46.04→112.92mm로 커졌다. 최악 위치는Run227의90ms, VTP의0부터 시작하는 point index10,171이다. 각 시점의 평균 병진을 제거한 변형장 relative L2도10.0617→10.7534%로 악화됐다. 가전제품의 국부 간극·좌굴·파손 판단에는 평균 변위 오차보다 이런 국부 지표와 필요한 물리량을 함께 검증해야 한다.

변위 relative L2의 분모는 FEM **변위**의 제곱합이다. 공식 예제의 절대 위치를 분모로 하는 position-relative L2와 직접 비교하면 안 된다. RMSE는 x·y·z 성분까지 평균한 **성분 RMSE**이며, 노드 벡터 RMSE는 이 값의 √3배다. 모든 노드는 균등 가중했다. 병진 이동이 포함되므로 이 수치만으로 국부 좌굴·파손 정확도를 판정하면 안 된다. 자세한 정의는 `metric_definitions.json`에 있다.

실행 장비는 RTX3060 Laptop 6GiB, PyTorch2.13.0+cu126, CUDA12.6이다. 전체 노드 실험의 PyTorch GPU tensor allocation 최대값은383.35MB, epoch 경계에서 관측한 프로세스 RSS 최대값은1.837GB였다. 전자는 CUDA context·driver·다른 프로그램의 GPU 메모리를 제외하고, 후자는 실제 최대 RSS의 하한이다. 시스템 여유 RAM은397~818MB까지 낮아졌지만 전체 학습은 완료됐다.

GPU warm forward만 잰 7회 평균은1,024노드6.23ms, 전체 노드11.79ms였다. 새 프로세스에서 전체 checkpoint를 다시 불러온 Run224의 단일 forward 측정은37.97ms였다. 데이터 읽기·모델 import·전송·파일 저장 시간은 이 수치에 포함되지 않는다. 동일 범퍼의 실제 FEM 재실행과 연결 비교는 [별도 기록](../bumper-fem-rerun/README.md)에 있다. FEM 실행과 NN forward의 측정 범위가 다르므로 이 두 시간만 나누어 전체 업무 속도 향상 배수를 주장하지 않는다.

## 입력과 버전

- [공개 OpenRadioss 범퍼 데이터](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample), dataset commit `78f1302fba4d75926c6fdd22d01d5133efc14742`.
- [NVIDIA crash 예제](https://github.com/NVIDIA/physicsnemo/tree/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash), source commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`.
- 실제 모델 패키지는 `nvidia-physicsnemo==2.2.2`다. 최신 main 예제와 배포 패키지의 API 차이를 구분해 기록했다.
- 원 VTP는13,676노드이며 양의 두께를 갖는13,675노드를 변형체 평가 대상으로 선택했다. 시간은0,10,…,100ms다. 학습 Run194와Run203의 추가110ms는 공통 시간 구간에 맞춰 제외했다.
- 원본 Run100 starter의 `/BEGIN`은kg/mm/ms를 선언한다. VTP 자체에는 단위 메타데이터가 없다. Curated VTP에는 두께와 변위만 있으며, 응력·소성변형률·힘을 학습한 것으로 해석하면 안 된다.

`source_manifest.json`은 다운로드 URL·고정 commit·크기·SHA256을 담는다. 입력덱 헤더는 Altair CC BY-NC4.0을 명시한다. 원본 FEM 데이터는 이 코드 묶음에 재배포하지 않으며 출판자에게서 직접 내려받는다. 회사의 자체 모델·데이터로 확장하는 작업과 이 공개 모델의 비상업 조건을 구분해야 한다.

## 환경 구성

아래 PowerShell 명령은 이 README와 helper 파일이 있는 디렉터리에서 실행한다. `requirements-direct.txt`가 재구성용 버전 목록이며, `environment_freeze.txt`는 실제 Windows 환경의 증거다. GPU ML 실행에 필요한 전이 의존성95개는 `requirements-lock-windows-cu126.txt`에 따로 고정했고, 설치된 각 버전이 의존 조건을 만족하는지 확인했다. 이 목록에는 별도 CPU 원본 드라이버의 선택적 로깅 도구는 포함하지 않았다. Windows 로컬 wheel 경로가 들어갈 수 있는 freeze 파일이나 Windows lock을 Linux 설치 입력으로 사용하지 않는다. Linux에서의 설치·실행은 아직 검증하지 않았다.

```powershell
New-Item -ItemType Directory -Force local-run/tmp | Out-Null
$env:TEMP = (Resolve-Path local-run/tmp).Path
$env:TMP = $env:TEMP
py -3.13 -m venv local-run/venv
$experimentPython = Join-Path (Get-Location).Path 'local-run/venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:TORCHDYNAMO_DISABLE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $experimentPython -X utf8 -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
& $experimentPython -X utf8 -m pip install -r requirements-direct.txt
& $experimentPython -X utf8 download_from_manifest.py --destination local-run/public
$dataRoot = (Resolve-Path local-run/public/data/CURATED_DATA_VTP).Path.Replace('\','/')
```

실제 실행에서는 프로젝트 전용 venv가 기존 numpy·VTK 등 일부 패키지를 상속했다. `pip-check.log`의3개 충돌은 상속된 questionary/google-genai/mixpanel 관련 항목이다. PhysicsNeMo 모델 생성과 GPU 학습은 직접 실행해 확인했다. 새 빈 venv에서 위 명령 전체를 한 번에 실행하는 절차 자체는 별도의 clean-install 검증을 거치지 않았다.

Codex 로컬 sandbox의 기본 임시 폴더에서는 ensurepip 쓰기가 거부돼 TEMP/TMP를 프로젝트 하위로 바꾼 뒤 venv를 만들었다. Windows 기본 cp949 인코딩에서는 PyTorch 템플릿 import가 실패해 UTF-8 설정을 적용했다. Warp1.18은 설치 드라이버보다 높은 CUDA13을 요구하는 경고를 출력했다. 이번 Transolver는 `use_te=false`의 PyTorch 경로로 실제 CUDA 학습·추론에 성공했고 Warp GPU 연산을 검증한 것은 아니다.

## 저장한 모델로 추론

완료된 `transolver_cuda_checkpoint.pt`와 같은 폴더의 `mesh_contract.json`을 함께 둔다. helper는 원 checkpoint SHA, 노드 선택, 기준 좌표의 hash를 검사한다. 두 노드의 좌표 순서를 바꾼 음성 시험은 예상한 오류로 거부됐다. 범위를 벗어난 전역 설계변수는 결과 JSON에 표시한다.

```powershell
& $experimentPython -X utf8 infer_checkpoint.py --checkpoint fulltraining/transolver_cuda_checkpoint.pt --vtp "$dataRoot/VALIDATION_DATA/Run224.vtp" --globals "$dataRoot/GLOBAL_FEATURES.json" --device cuda --out local-run/inference
```

`inference_report.json`과 `inference_result.npz`를 생성한다. 공개 FEM 참조 변위가 있으면 수치 오차도 계산한다. 전체 checkpoint의 새 모델 로드 시험은 원 실험의 예측과 최대 차이0.0mm로 일치했다. 로컬에 재계산한 `heldout_predictions.npz`가 있으면 `--reference-predictions <경로>`로 같은 검사를 반복할 수 있다.

## 학습 재실행

각 실험의 `executed_source.py`가 실행 당시의 소스 snapshot이다. `metrics.json`에 그 SHA256이 있고 분할·정규화·모델 크기·시점·오차·시간·메모리 정의가 함께 남아 있다. 다른 출력 폴더를 사용하면 보존된 가중치를 덮어쓰지 않는다.

```powershell
& $experimentPython -X utf8 -u gpu1024/executed_source.py --device cuda --epochs 60 --nodes 1024 --data $dataRoot --out local-run/retrain1024
& $experimentPython -X utf8 -u fulltraining/executed_source.py --device cuda --epochs 60 --nodes 13675 --skip-kernel --data $dataRoot --out local-run/retrain-full
& $experimentPython -X utf8 -u capacity1024/executed_source.py --device cuda --epochs 60 --nodes 1024 --skip-kernel --layers 4 --hidden 96 --slices 16 --heads 4 --data $dataRoot --out local-run/retrain-capacity
```

전체 노드 학습에서는 큰 dense RBF 계수 배열을 다시 만들지 않도록 kernel fitting을 생략했다. 전체 노드 표의 RBF 결과는 미실행이며,1,024노드 결과를 전체 해상도 결과로 바꿔 쓰지 않는다. 새로 학습한 checkpoint의 추론 계약은 다음과 같이 생성한다.

```powershell
& $experimentPython -X utf8 make_mesh_contract.py --checkpoint local-run/retrain-full/transolver_cuda_checkpoint.pt --reference-vtp "$dataRoot/TRAINING_DATA/Run100.vtp" --globals "$dataRoot/GLOBAL_FEATURES.json"
```

추가 구조 크기 비교는 `capacity1024`에 별도 보존한다. 이 실험은4층·폭96·slice16·293,582 parameters를 사용하며 나머지 분할·seed·1,024노드·60epoch 상한은 같다. 사전 실제 step 계측은40.41ms, GPU tensor allocation44.85MB였고,112×60 steps에10% 여유를 더한298.73초 추정이 실행 기준을 통과했다. 실제 전체 학습은291.08초, GPU tensor allocation 최대65.50MB였다. 1,024노드의 기존 RBF 결과와 비교할 수 있으며 RBF를 다시 학습하지 않았다. 단일 seed의 탐색 실험으로, 최종 알고리즘 선정을 의미하지 않는다.

## 원본 예제 드라이버 실행 확인

`prepare_official_trial.py --download-root local-run/public`은 원본 `train.py`와 `vtp_reader.py`를 그대로 두고, reader 호환 복사본과 작은 YAML 설정을 추가한다. 원 reader는 `displacement_t0.xxx`만 먼저 고르기 때문에 실제ms 필드 중t0만 읽었다. 동일 Run100의 원 reader 출력은 `(1,13676,3)`, 수정 복사본은 `(11,13676,3)`임을 assert로 확인했다.

CPU 전용 PyTorch 환경에서 원 `train.py --config-name=fem_cpu_trial`에 학습·검증 경로를 지정한 실행은 exit0으로 끝났다. Run100/101 학습,Run224 검증,전체13,676노드,2epoch의 경로 확인이며 예측 정확도 입증용 실험이 아니다. `official-run-console.log`, `official_cpu_compat/.hydra`, 체크포인트와 `reader-selfcheck.log`가 실행 근거다.

```powershell
py -3.13 -m venv local-run/cpu-venv
$cpuExperimentPython = Join-Path (Get-Location).Path 'local-run/cpu-venv/Scripts/python.exe'
& $cpuExperimentPython -X utf8 -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
& $cpuExperimentPython -X utf8 -m pip install -r requirements-direct.txt
& $cpuExperimentPython -X utf8 prepare_official_trial.py --download-root local-run/public
$cpuTrialOutput = Join-Path (Get-Location).Path 'local-run/official-cpu'
& $cpuExperimentPython -X utf8 local-run/public/original_crash/train.py --config-name=fem_cpu_trial "training.raw_data_dir=$dataRoot/TRAINING_DATA" "training.raw_data_dir_validation=$dataRoot/VALIDATION_DATA" "hydra.run.dir=$($cpuTrialOutput.Replace('\','/'))"
```

## 공개 복사에서 제외할 파일

원본 VTP·입력덱·FEM 결과는 `work/`에 유지한다. `heldout_predictions.npz`, `kernel_checkpoint.npz`, `inference_result.npz`는 원본 좌표·변위를 포함하거나 직접 재구성하므로 로컬에만 둔다. venv·다운로드 wheel·cache와 큰pip debug 원본도 공개 복사에서 제외한다. 코드·manifest·명시적 버전 목록·compact 설치 로그·실행 로그·CSV·수치 JSON·그림·NN checkpoint·mesh contract는 별도로 보존할 수 있다. 데이터 유래 가중치·그림의 출처와 비상업 조건 표시는 유지한다.
