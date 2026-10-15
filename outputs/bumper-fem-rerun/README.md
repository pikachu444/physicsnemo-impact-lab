# Run224 동일 범퍼 FEM 재실행·NN 연결 검증

검증일: 2026-10-15, 로컬 Windows. 공개 데이터의 **heldout Run224와 동일한 원본 입력**을 OpenCourant로 실제 재실행하고, 원 Dataset 및 이미 학습한 PhysicsNeMo 모델의 저장 예측값과 비교했다. 자동차 범퍼/강체 pole 충격 예제이며 가전제품 또는 새로운 형상에 대한 검증으로 확대하지 않는다.

## 원본과 실행 조건

- Dataset: [AIRBORNEPANDA/BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/tree/78f1302fba4d75926c6fdd22d01d5133efc14742/RAW_DATA/VALIDATION_DATA/Run224), revision `78f1302fba4d75926c6fdd22d01d5133efc14742`.
- 실제 경로는 `RAW_DATA/VALIDATION_DATA/Run224`. Starter/Engine 입력 2개와 원본 로그 2개만 추가 다운로드했다. URL·bytes·SHA256는 `source-manifest.json`에 보존했다.
- Solver: [OpenCourant latest-20261011](https://github.com/OpenCourant/OpenCourant/releases/tag/latest-20261011), commit `cb760c206e346eff12161a2268b134ef9d48a4ef`, `starter_win64_sp.exe` / `engine_win64_sp.exe`, SMP 1 domain, **2 threads**.
- 입력 물리·솔버 카드·고유 파일명을 변경하지 않았다. 실제 실행 입력의 SHA256가 고정 원본과 일치한다.
- 원본 FEM 13,832 nodes, 13,598 quad shells + 28 triangular shells + 78 springs. Dataset VTP는 13,676 points이며 NN은 positive-thickness 13,675점을 사용한다. 원본 mesh와 VTP의 숫자·순서를 같다고 가정하지 않았다.
- `/BEGIN` input/work 양쪽 kg/mm/ms, steel density 7.85e-6 kg/mm³와 E=210 GPa가 일관된다. 변위 mm, 시간 ms, 응력 GPa, 에너지 J를 사용한다.
- Run224 global features: shell parameter 1.52, velocity_x -5 mm/ms(= -5 m/s), rigid pole origin_y -415.52 mm. 모델 종료 요청 100 ms.

## 실제 실행·자원

| 단계 | 종료 코드 | 이번 프로세스 벽시계 시간 | 표본 최대 working set |
|---|---:|---:|---:|
| Starter | 0 | 4.3997 s | 220,282,880 B (210.1 MiB) |
| Engine | 0 | 742.5954 s | 93,376,512 B (89.1 MiB) |
| 합계 | | **746.9951 s (12분 27초)** | 순차 실행 |

Engine `NORMAL TERMINATION`, total cycles 100001(마지막 index100000)을 확인했다. 솔버 내부 보고 Engine 시간은742.29s, memory89MB다. 위 표는 약100ms 간격 프로세스 관측으로 GPU VRAM이 아니다. 다운로드·변환·NN 검증 시간은 포함하지 않는다. 후처리까지 포함한 local raw/converted case 파일은 약171.1MB이며 큰 원본 결과는 `work/bumper-fem-rerun/run-extended`에 보존했다.

원본 로그는 i7-11850H / OpenRadioss extended single precision /16threads에서 Starter5.72s+Engine326.99s=332.71s를 보고한다. 이번도 extended single precision이지만 solver build·CPU·threads가 다르고 다른 학습 작업이 병행되었다. 이 두 시간으로 solver 성능 우위를 주장하지 않는다. NN forward-only 시간과 FEM의 100,000 integration steps 및 다양한 출력 생성 시간도 구분해야 한다.

초기600s 제한 attempt1은 ETA 관측 후 203.1343s, 모델 시간19.6ms에서 **제 소유 Engine만 수동 중단**했다. 승인된1200s 제한으로 별도 폴더에서 attempt2를 처음부터 완주했다. `attempt-1-metrics.json`/`attempt-1-note.json`에 분리했으며 완주 시간에 합산하지 않았다. 유료 자원을 만들거나 다른 작업을 종료하지 않았다.

## 최종 FEM 전역 출력

| 출력 | 공개 원본 | 새 FEM |
|---|---:|---:|
| 마지막 출력 시간(ms, 출력 정밀도) |100.0|100.0|
| internal energy(J) |6037|6037|
| translational kinetic energy(J) |215.2|215.2|
| rotational kinetic energy(J) |0.1671|0.1605|
| energy error(%) |-0.4|-0.4|
| total mass(kg) |517.0|517.0|
| added mass(kg) |0.1191|0.1196|
| reported mass error fraction |0.0002304|0.0002315|
| 위 fraction의 백분율(%) |0.02304|0.02315|

같은 숫자가 나타난 항목은 **출력 정밀도에서의 일치**다. bitwise 동일성을 주장하지 않는다. 두 Starter 모두 warning343: interface13 initial penetrations638개/212nodes를 보고하며, 오류0·경고1이다. 이를 원본 그대로 기록했다. warning 수만으로 무효를 판정하거나 접촉/물성을 임의로 수정하지 않았다. 모델의 실험적 타당성과 AI의 기준 FEM 근사 정확도는 별도 문제다.

## 원 Dataset과 새 FEM의 변위 비교

공식 `anim_to_vtk`로11개 ANIM을 변환했고 `th_to_csv`도 종료 코드0으로 실행했다. 초기 RAD 좌표와 VTP 좌표를0.001mm 이내에서 유일하게 연결한 뒤, 각 실제 VTK의 `NODE_ID`와 `Displacement`를 사용했다. 최종 좌표에 대한 최근접 검색을 사용하지 않았다.

- 비교13,675점, 비유한점0, missing ID0, 제외VTP index13675(두께0) 1점.
- 제외점은 RAD 최근접거리2.2867mm이고 실제 ANIM에는 중복 NODE_ID0 시각화점들이 있다. 고유한 원본 ID로 억지 연결하지 않았다.
- 수용한 RAD↔VTP 초기 매핑 최대거리0.0000330463mm. ANIM 초기좌표↔VTP 최대차이0.004275mm도 별도 기록했다.
- 대상은 명목100ms이며 실제 최종 ANIM `FIELD TIME`은 **100.001ms**다. 차이0.001ms를 기록했고 시간 보간은 수행하지 않았다.
- 최종 변위 **vector RMSE0.0125181mm**, component RMSE0.00722734mm, p95 vector error0.0126495mm, max0.113596mm, displacement relative L2=0.0000886155(약0.00886%).

`fem-to-fem-run224-comparison.json`/`.png`에 정의·프레임 시간·ID 상태·해시를 보존했다. Vector RMSE는 노드별 3D 오차 norm의 제곱평균제곱근, component RMSE는 xyz 성분 전체의 제곱평균제곱근이므로 서로 다른 값이다.

## 이미 학습한 NN과 새 FEM의 비교

`outputs/experiment/fulltraining/heldout_predictions.npz`의 저장 예측을 재사용했다. 추가 학습이나 NN 추론을 실행했다고 주장하지 않는다. Run224..230 순서, node index contract 해시, Run224 좌표와 VTP의 float32 bitwise 일치, 저장 truth와 원 VTP의 node/time/component 순서까지 assert했다. 실제10개 프레임의 고유 NODE_ID를 다시 정렬했고 모든13675점의10시점 변위가 유한함을 확인했다.

| 10개 시점 전체(10..100ms) | component RMSE(mm) | vector RMSE(mm) | displacement relative L2 |
|---|---:|---:|---:|
| NN vs 원 Dataset FEM |4.859404|8.416735|0.0644213|
| **NN vs 새 FEM** |**4.857321**|**8.413127**|**0.0643917 (6.4392%)**|
| 새 FEM vs 원 Dataset FEM |0.00529681|0.00917435|0.0000702200|

최종100ms만의 NN vs 새 FEM component RMSE는7.666989mm, vector RMSE13.279614mm, relative L2=0.0940009(9.4001%)다. 위 전체 시점 값과 혼용하지 않는다. NN 오차와 FEM 반복/변환 오차를 서로 다른 표·축으로 표시했다. 이 결과는 이 heldout1개 조건과 변위에 대한 추가 연결 검증이며 stress/가속도/파손 정확도 검증이 아니다.

`nn-vs-rerun-run224.json`/`.png`에는 예측 파일/원 VTP/입력/프레임 해시, 모든 순서·시간 assertion과 오차 정의를 보존했다. 이 NN 연결 계산은 source truth를 float32로 읽기 때문에 순수 FEM 비교의 float64 VTP 값과 최하위 자리의 차이가 있다.

## 재현·보존

아래 명령은 저장소 루트에서 실행한다. 먼저 [학습 환경 구성](../experiment/README.md#환경-구성)의 가상환경을 준비한다. 아래 `$analysisPython`은 그 절차에서 생성하는 Python이며, 전역 Python이나 존재하지 않는 기존 연구 venv를 가정하지 않는다. OpenCourant 패키지와 입력·curated 자료는 고정 manifest에서 받아 해시를 확인한다.

```powershell
$analysisPython = (Resolve-Path 'outputs/experiment/local-run/venv/Scripts/python.exe').Path
& $analysisPython -X utf8 outputs/experiment/download_from_manifest.py --destination work/physicsnemo-experiment
./outputs/fem-experiment/download-fem.ps1
./outputs/bumper-fem-rerun/download-bumper.ps1
./outputs/bumper-fem-rerun/run-bumper.ps1
./outputs/bumper-fem-rerun/postprocess-bumper.ps1
& $analysisPython -X utf8 ./outputs/bumper-fem-rerun/summarize-bumper.py
& $analysisPython -X utf8 ./outputs/bumper-fem-rerun/compare-bumper.py
```

변위 비교에는 NumPy/SciPy/VTK/Matplotlib 및 기존 curated Run224가 필요하고 NN 비교에는 위 저장 prediction/metrics/mesh contract가 필요하다. 분석은 단일 CPU thread로 수행하며 Torch/GPU를 사용하지 않는다. 재실행 시 기존 `work/.../run-extended` 원본 결과를 먼저 별도로 보존해야 한다.

NN prediction NPZ는 공개 저장소에서 제외했고, NN checkpoint와 contract는 `outputs/experiment/fulltraining`에 보존한다. 이 비교 스크립트는 전체 평가 예측 NPZ를 요구하므로, 새 clone에서는 아래처럼 기록된 결과를 덮어쓰지 않는 별도 폴더에 학습을 재실행한 뒤 contract를 만들 수 있다. 아래 명령은 GPU 학습을 **다시 수행할 때의 재현 절차**이며 이번 NN 연결 검증에서 추가 학습을 실행한 것은 아니다. 저장 checkpoint만으로 한 케이스를 추론하는 더 짧은 경로는 [추론 안내](../experiment/README.md#저장한-모델로-추론)에 있다.

```powershell
$env:PYTHONUTF8 = '1'
$env:TORCHDYNAMO_DISABLE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $analysisPython -X utf8 outputs/experiment/fulltraining/executed_source.py --data work/physicsnemo-experiment/data/CURATED_DATA_VTP --out work/reproduced-fulltraining --nodes 13675 --epochs 60 --device cuda --skip-kernel
& $analysisPython -X utf8 outputs/experiment/make_mesh_contract.py --checkpoint work/reproduced-fulltraining/transolver_cuda_checkpoint.pt --reference-vtp work/physicsnemo-experiment/data/CURATED_DATA_VTP/TRAINING_DATA/Run100.vtp --globals work/physicsnemo-experiment/data/CURATED_DATA_VTP/GLOBAL_FEATURES.json
& $analysisPython -X utf8 outputs/bumper-fem-rerun/compare-nn-bumper.py --training-dir work/reproduced-fulltraining --output-dir work/reproduced-bumper-comparison
```

`--training-dir`에서 `metrics.json`·`mesh_contract.json`·`heldout_predictions.npz`를 읽는다. 기본값은 기존 `outputs/experiment/fulltraining`이며 `--output-dir` 기본값은 이 보고서 폴더다. 재현 시 위처럼 새 출력 폴더를 지정하면 이번 기록을 보존한다. 비교 전에 같은 Run224 FEM 재실행 및 VTK 변환 단계도 완료해야 한다.

독립 감사자가 원본/새 입력 해시, 전역 출력과 mapping/시간/오차를 별도 재계산해 일치함을 확인했다. NN 비교도 별도 NumPy/VTK 집계로 같은 값을 재현했다. 감사에서 발견된 Engine 로그 메타데이터 참조와 최근접거리 메타데이터 집계는 수정·재실행했다. 수치 예측오차에는 영향이 없었다.

## 공개 배포 범위·라이선스

FEM 입력은 Copyright(C)2022 Altair Engineering Inc., **[CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/)**다. [원본 입력 URL](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample/blob/78f1302fba4d75926c6fdd22d01d5133efc14742/RAW_DATA/VALIDATION_DATA/Run224/Bumper_Beam_AP_meshed_0000.rad)과 고지를 유지한다. 이 공개 패키지는 해당 비상업 연구 모델에서 나온 수치와 그림임을 명시하며 원본/파생 geometry를 일괄 MIT 또는 상업 허용 데이터로 재표시하지 않는다. Solver는 AGPL-3.0이고 배포 package의 외부 runtime에는 별도 고지가 있다.

이 출력 폴더에는 solver EXE/DLL/ZIP, 원본 RAD, restart, ANIM/VTK 등 대용량 원시 결과나 NN checkpoint를 복사하지 않았다. 텍스트 로그·수치 JSON·PNG·재현 스크립트만 포함한다. 공개 로그의 사용자 경로/컴퓨터 이름은 가렸고 원본은 `work/`에 보존했다. `.gitignore`로 실행기/원시 결과/캐시의 우발적 포함을 막는다. 생성 수치 결과에 대한 별도 upstream 라이선스는 확인하지 못했으므로 원 모델의 출처·비상업 조건을 보존해 표시한다.
