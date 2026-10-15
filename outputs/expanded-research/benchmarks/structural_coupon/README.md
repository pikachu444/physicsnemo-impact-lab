# 구조 인장 coupon: native FEM 응력·총변형률·소성변형률

공개 RD-E1101 DP600 LAW2 shell 인장 모델을 OpenCourant로 실제 계산하는 독립 구조 benchmark다. 원래 40 ms 예제의 앞 10 ms를 사용하며, 초기 0 ms를 제외한 10개 animation frame을 보존한다. 전체 제품 낙하나 파손을 검증하는 사례는 아니다. 모델·단위·field 정의와 라이선스 근거는 [field-conventions.md](field-conventions.md)에 있다.

최종 실제 실행 상태와 수치는 [dataset-summary.json](dataset-summary.json), 조건별 표는 [dataset-summary.csv](dataset-summary.csv)를 따른다. 학습 결과는 모델 실행 담당자가 `models/`에 별도로 기록한다. FEM 정답 생성과 AI 정확도 검증을 구분한다.

실제로 완료한 조건은 training5 + validation4 + test4 = **13개**다. 모든 starter/engine이 정상 종료했다. Solver stage wall time 합계는765.54초, 개별 engine은32.81–72.17초였다. Process RSS를0.05초 간격으로 샘플링한 최대값은 starter173.20MB, engine52.91MB이며 전체 시스템 메모리와 다르다. 1ms 이후 KE/IE의 전체 최대값은0.00121936(약0.122%), animation의 nominal time 대비 편차는 최대0.0002ms였다. 이 값은 실행 안정성 관측이며 mesh·시간간격 수렴성 증명을 대신하지 않는다.

## 정답의 두 표현

- **원본 cell C8:** 808 physical shell element에서 native local 응력 xx/yy/xy(MPa), native local total strain xx/yy/xy, native IP3 EPSP, 별도 native shell von Mises scalar(MPa). Total strain을 변위에서 미분해 대체하지 않았다. Layer3은 두께 integration point3/5이다.
- **공동 학습 node C7:** source physical node919개에서 raw ANIM global displacement xyz(mm), native shell VM의 면적 projection(MPa), native IP3 EPSP의 면적 projection, native IP3 in-plane total-strain tensor의 principal major/minor를 먼저 계산한 뒤 각각 면적 projection한 값이다. 뒤 네 채널은 solver의 raw nodal tensor output이 아니다.

Local tensor 성분을 이웃 절점에서 직접 평균하지 않는다. Scalar projection은 incident element의 초기 면적/절점수를 가중치로 사용한다. Lumped nodal area를 통한 적분 보존을 float64에서 검사하고 float32 target을 저장한다. 이 projection은 element 경계와 국부 극값을 평활화하므로 원 cell C8도 보존한다. 2D converter가 채운 zz=0을 물리적인 3D strain label로 사용하지 않는다.

Source node는 920개이며, shell 연결에 쓰이는 node919개만 공동 target에 포함한다. 나머지990은 RBODY main node다. 실제 shell 연결의 adjacency를 사용하고 rigid-body 표시용 cell91개를 학습용 shell에서 제외한다.

Base의 첫 animation `A001`은 실제0ms이며 물리 node919개에서 displacement와 velocity가 모두0이었다. 원 velocity curve는0~1e30ms에 일정1이고, imposed X velocity는 t0 이후 배율×1mm/ms로 적용된다. 원본에 없는 ramp를 추가하지 않았다.

## 조건과 분할

Training anchor는 원조건과 네 corner, 총5개다. Iflag1 engineering yield와 UTS의 동시 배율0.9–1.1, 원 velocity curve 값의 배율0.75–1.25를 사용한다. E, 밀도, ν, engineering strain at UTS, source mesh와 BC 종류는 고정이다. 이 범위는 알고리즘 민감도 연구 가정이며 실제 DP600 산포나 제품 허용 공차가 아니다.

Solver가 실제 산출한 corner 계수는 A=0.27/0.33GPa, B=0.6484657767505/0.7925692826951GPa, n=0.1980449212604였다. 이번 입력 배율에서는 결과적으로 A/B가 같은 배율을 따르고 n이 유지됐다. 입력을 임의의 A/B 직접 수정으로 바꾼 것은 아니다.

별도의 내부8조건은 scrambled Sobol seed20261015로 먼저 고정했다. 네 조건은 validation, 네 조건은 test이며 training anchor와 중복되지 않는다. [frozen-initial13-design.json](frozen-initial13-design.json)에 실제 수치와 생성기 SHA가 있다. Test는 training 수량·설정·epoch 선택에 사용하지 않는다. 향후 training 증분의 최종 수량은 초기13조건의 생성으로 확정된 것이 아니다. 이전에 제안됐던 임시3/1/1 분할은 이 최종 데이터 설계에 사용하지 않았다.

공통 cache는 `work/expanded-structural-coupon/cache-initial13-nodal7/manifest.json`, 원 cell cache는 `cache-initial13-cell8/manifest.json`이다. Manifest는 nominal time1–10 ms와 case별 actual time/jitter, 물리 단위, input global 값, source/prepared input·target SHA, solver가 산출한 LAW2 A/B/n을 함께 기록한다. 시간 보간은 하지 않았다. `READY.json`은 array·manifest hash 전수 확인 후 소비자에게 인계하는 완료 표시다.

Planar 예제의 raw u_z는 roundoff 수준이다. 이 성분의 상대 오차는 유용하지 않을 수 있으므로 절대 RMSE와 물리 크기를 함께 읽어야 한다. 공통 학습 helper에는 training-only XYZ 공통 vector scale인 sqrt(trace(training displacement covariance)/3)을 구현했다. 개별 AI 실행의 실제 적용은 해당 모델의 frozen normalization 파일에서 확인한다.

추가 training8개는 [proposed-training-increment8.json](proposed-training-increment8.json)에 별도 seed20261016 Sobol 순서와 중복 검사만 준비했다. **이 추가 FEM은 실행하지 않았다.** 현재 training5개와 고정 validation4/test4를 바꾸지 않았으며, 추가 생성 여부는 초기 validation 학습곡선을 보고 결정한다.

## 새 폴더에서 재현

원본 ZIP와 `.rad` 파일은 재배포 허가를 확인하지 못해 `work/`에만 둔다. 아래 downloader가 공식 URL과 고정 SHA를 확인한다. Solver 바이너리도 이 묶음에 포함하지 않으며 기존 OpenCourant 설치를 지정한다. Python 의존성은 `numpy`, `scipy`, `vtk`, `psutil`, `matplotlib`이다. 실제 기록은 Windows Python3.13.5, NumPy2.5.2, SciPy1.18.0, VTK9.4.2, psutil7.2.2로 생성했다. 별도 환경의 신규 설치 재현까지 이 FEM 결과 생성에 포함했다고 주장하지 않는다.

작업 공간 루트의 PowerShell 예:

```powershell
$scripts = (Resolve-Path 'outputs/expanded-research/benchmarks/structural_coupon').Path
$solver = (Resolve-Path 'work/opencourant-experiment/solver/OpenCourant').Path
$freshWork = Join-Path (Get-Location) 'work/reproduced-structural-coupon'
$freshSummaries = Join-Path (Get-Location) 'outputs/reproduced-structural-coupon'
$design = Join-Path $scripts 'frozen-initial13-design.json'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -X utf8 "$scripts/run_design.py" --work $freshWork --solver $solver --design $design --summaries $freshSummaries
if ($LASTEXITCODE -ne 0) { throw 'FEM reproduction failed; inspect preserved logs' }
python -X utf8 "$scripts/build_cache.py" --work $freshWork --summaries $freshSummaries --design $design --out "$freshWork/cache-cell8"
if ($LASTEXITCODE -ne 0) { throw 'Cell cache failed' }
python -X utf8 "$scripts/build_nodal_cache.py" --cell-cache "$freshWork/cache-cell8" --starter "$freshWork/original/tensile_LAW2_0000.rad" --work $freshWork --out "$freshWork/cache-nodal7"
if ($LASTEXITCODE -ne 0) { throw 'Nodal cache failed' }
python -X utf8 "$scripts/finalize_cache.py" --cache "$freshWork/cache-nodal7" --expected-cases 13
if ($LASTEXITCODE -ne 0) { throw 'Ready validation failed' }
```

기존 case와 frozen cache를 덮어쓰지 않도록 새 경로를 지정한다. Solver는 1 thread씩 직렬 실행하며, 400,000,000 bytes의 process RSS cap과 stage timeout을 사용한다. `run_design.py`는 시작 전에 source를 snapshot하고 입력 작성→starter→engine→native field 추출→nodal displacement 추출을 기록한다. 현재 시간 측정에는 같은 host의 다른 연구 작업이 영향을 줄 수 있으므로 전용 장비에서의 성능 보장으로 해석하지 않는다.

## 해석 범위와 한계

이 사례는 작은 탄소성 shell의 동일 mesh·동일 하중 형태 안에서 조건을 변경한 benchmark다. 접촉, 접합부, 낙하 자세, 파단, mesh 변경을 포함하지 않는다. Local strain은 full nonlinear shell의 native accumulated total-strain output이며, 특정 Hencky/Green strain과 엄밀히 같다는 독립 증명은 하지 않았다. Force–extension은 T01 impulse의 수치 미분으로 얻은 진단값이며 direct native force label이나 element true stress의 대용품으로 사용하지 않는다.

기존 CPU 환경에서 solver wall time과 sampled process RSS를 실측했고 유료 자원은 생성하지 않았다. 전력과 기존 PC의 보유 비용은 측정하지 않았다. AI가 이 작은 데이터로 학습하더라도 전체 가전제품 조립체의 낙하·충격해석에 대한 정확도를 입증한 것은 아니다.
