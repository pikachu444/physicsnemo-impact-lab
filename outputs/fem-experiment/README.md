# OpenCourant 실제 FEM 실행 기록

검증일: 2026-10-15 (Asia/Seoul). 실행 위치는 이 대화의 **로컬 Windows PC**다. Codex Cloud 실행 결과로 표시하면 안 된다.

공개 OpenCourant MiniQA의 BOX1BL 원본 입력으로 Starter와 Engine을 실제 실행했다. 두 프로세스 모두 종료 코드 0, Engine `NORMAL TERMINATION`을 확인했다. 이는 작은 FEM 예제 실행 및 결과 변환 검증이다. AI 학습, AI 정확도 검증, 가전제품 낙하해석 검증에 해당하지 않는다.

## 고정한 원본

- Solver release: [latest-20261011](https://github.com/OpenCourant/OpenCourant/releases/tag/latest-20261011)
- Commit: `cb760c206e346eff12161a2268b134ef9d48a4ef`
- [Windows package](https://github.com/OpenCourant/OpenCourant/releases/download/latest-20261011/OpenCourant_win64.zip): 350,476,183 bytes
- Package SHA256: `6D0EA8B43B68AA12F0A2787B339384A554978CBB65A5D3DBAF1E40598F02E7E6`
- [원본 입력 디렉터리](https://github.com/OpenCourant/OpenCourant/tree/cb760c206e346eff12161a2268b134ef9d48a4ef/qa-tests/miniqa/CRASH/crashsection/box1_42/data)
- 입력 SHA256: `validation-summary.json`의 `input_sha256`. 실행 입력의 물리/솔버 설정은 수정하지 않았다.

입력은 Johnson-Cook 소성 shell box, -8(모델 속도 단위) 초기속도, 강체 평면, 강체/질량, section history를 포함하는 오래된 회귀시험 모델이다. 실제 제품 모델도, 제품별 물성에 맞춘 낙하시험 모델도 아니다. `/BEGIN`에는 input/work 양쪽 `kg m s`가 선언되었지만 rho=7.8e-6, E=205, yield=.25라는 재료 수치가 실제 steel SI 값과 일관되지 않는다. 따라서 모든 geometry/time/energy 그래프는 **모델 단위**로 표시하며 SI 변환이나 재료 보정이 검증되었다고 주장하지 않는다. VTK의 168 points와 268 cells에는 강체/시각화용 엔티티가 포함된다. 이를 원본 FEM shell 수로 보고하면 안 된다.

## 실제 측정

| 단계 | 종료 코드 | 프로세스 벽시계 시간 | 표본 관측 최대 working set |
|---|---:|---:|---:|
| Starter | 0 | 4.383 s | 172,793,856 B (164.8 MiB) |
| Engine | 0 | 1.856 s | 50,266,112 B (47.9 MiB) |

2 OpenMP threads, 단일 SMP process를 사용했다. 메모리는 약 50 ms 간격 프로세스 관측값이며 GPU VRAM이 아니다. 솔버 내부 보고 시간은 Starter 3.18 s + Engine 1.02 s = 4.20 s, Engine memory 47 MB다. 패키지 다운로드/압축해제/전처리 시간은 이 표에 포함하지 않았다. `execution-metrics.json`에 원측정값을 보존했다.

## 기준값과의 비교

| 물리량/상태 | upstream ref.extract | 이번 실행 출력 |
|---|---:|---:|
| 마지막 cycle index | 1102 | 1102 |
| time | 0.1001 | 0.1001 |
| timestep | 8.502e-5 | 8.502e-5 |
| controlling element ID | 12 | 12 |
| internal energy | 34.97 | 34.97 |
| translational kinetic energy | 7646 | 7646 |
| rotational kinetic energy | 0.006788 | 0.006788 |

표시 정밀도에서 일치함을 독립 파서로 확인했다. 이 비교는 고정한 회귀시험 기준과의 일치이며, 물리 실험과의 검증 또는 전체 정밀도 bitwise 일치를 뜻하지 않는다. 솔버 요약의 total cycles 1103은 마지막 cycle index 1102와 구분했다. Energy error는 출력 정밀도에서 -0.0%, added mass는 0이다.

입력 경고 두 개는 삭제하거나 조용히 변경하지 않았다: `/IOFLAG`의 구형 unsupported field(100214), strain-rate filtering 미활성(1220). 정상종료/회귀 일치와 공학적으로 보정된 모델이라는 판단은 별개다.

## 저장한 결과

- `fem-results.png`: 실제 FEM energy histories(kinetic/internal/plastic/hourglass)와 초기/최종 shell mesh 비교.
- `run/BOX1BLT01.csv`: 실제 time-history 100 samples. 마지막 sample의 모델 시간은 0.09903467로, 최종 Engine 출력 시각과 다르다.
- `run/BOX1BLA001.vtk` … `BOX1BLA011.vtk`: gzip ANIM 11개를 패키지의 공식 변환기로 변환했다. 기본 ANIM에는 변형된 geometry가 있으나 이 예제는 stress/strain field 학습용 출력을 요청하지 않는다.
- `run/*.out`, `run/*.stdout.log`, `run/*.stderr.log`: solver 및 converter 로그. 공개본의 컴퓨터 이름과 사용자 경로만 `<HOST>`/`<WORKSPACE>`로 치환했다. 수치/물리/경고는 변경하지 않았다. 완전한 원본은 로컬 `work/opencourant-experiment/raw-results`에 보존했다. 공개 준비 스크립트는 timing JSON의 SHA256 앞16자를 실행별 archive key로 사용하므로 새로운 재실행은 별도 폴더에 보관한다.
- `validation-summary.json`: 기준 비교, 입력 해시, 출력 규모 및 경고.

## 재현

Windows PowerShell, Python 3 + NumPy + Matplotlib가 필요하다. 스크립트 위치를 repository의 `outputs/fem-experiment` 구조로 유지하고 순서대로 실행한다.

그림·파서는 로컬 Python 3.13, NumPy 2.5.2, Matplotlib 3.10.1로 실행했다.

```powershell
./outputs/fem-experiment/download-fem.ps1
./outputs/fem-experiment/run-fem.ps1
./outputs/fem-experiment/postprocess-fem.ps1
python ./outputs/fem-experiment/analyze_fem.py
./outputs/fem-experiment/prepare-publication.ps1
```

다운로드는 공개 패키지 약 350 MB와 입력 41 KB 정도다. 실행 파일/라이브러리/원본 zip은 `work/`에만 둔다. 전역 환경변수/서비스/라이선스 서버를 변경하지 않는다. 실행 스크립트의 환경변수는 해당 PowerShell 프로세스와 자식 프로세스에서만 설정된다. 재실행은 새로운 timing으로 `execution-metrics.json`을 갱신한다.

## 라이선스·출처

OpenCourant solver는 [AGPL-3.0](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/LICENSE.md)이며, 패키지 내부의 Intel runtime에는 별도 고지가 있다. 이 출력 폴더에는 solver `.exe`, `.dll`, `.zip`을 넣지 않았다.

FEM 입력 `BOX1BL_0000.rad`와 `BOX1BL_0001.rad` 상단은 **Copyright (C) 2026 SISW Siemens Industry Software Inc.; CC BY-NC 4.0**를 명시한다. 원본은 위 고정 URL에서 받으며 입력 고지를 보존한다. [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)은 출처/라이선스/변경 표시와 비상업 사용 조건을 명시한다. solver 코드의 AGPL 또는 다른 데이터 파이프라인의 MIT를 이 모델의 라이선스 대신 적용하지 않는다.

생성된 순수 수치 출력에 대한 별도 upstream 결과 라이선스는 확인하지 못했다. 이 패키지는 **위 CC BY-NC 모델을 이용한 비상업 연구 시연과 그 파생 geometry/그림**임을 표시한다. 출처를 유지하고 입력·파생 그림을 무조건적인 MIT/상업 허용 데이터로 재표시하지 않는다. 상용 제품 연구 확장에는 별도 사용 허가나 자체 생성/허용 모델이 필요하다. 원본 입력과 binary 결과는 `.gitignore`에서 제외했으므로 공개 저장소에는 필요한 텍스트 결과와 그래프만 선택해 넣을 수 있다.
