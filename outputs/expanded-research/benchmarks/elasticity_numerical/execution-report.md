# 공식 수치 선형탄성 모듈 실제 실행 결과

공식 PhysicsNeMo current 소스의 `Elasticity2DSolver`를 수정 없이 불러와 **CPU에서 두 문제를 실제로 계산했고, 독립 해석식·평형·에너지 검사 13개를 모두 통과했다.** 이 결과는 Q1 유한요소와 행렬을 조립하지 않는 preconditioned BiCGSTAB 수치 모듈 검증이다. AI 학습 결과나 전체 domain decomposition 예제 실행 결과로 세지 않는다.

이 문서가 현재 실행 결과와 재현 안내다. 기존 `README.md`는 실행 전 준비 단계의 역사 기록으로 보존했다. 그 문서의 “아직 실행하지 않았다”와 GPU 가상환경 예시는 당시 상태를 나타내며, 아래의 실제 실행 환경과 구분해야 한다.

## 실제 실행과 자원

| 항목 | 실측 |
|---|---:|
| 외부 감독 프로세스의 전체 경과 시간 | 10.89775 s |
| Python 드라이버 내부 총 시간 | 9.96601 s |
| Torch·psutil 최초 import | 9.28601 s |
| 균일 변형률 문제 solve | 0.270859 s |
| 체적력·고정단 문제 solve | 0.321112 s |
| 감독자가 관측한 최대 working set | 206,229,504 B ≈ 196.68 MiB |
| 시작 직전 사용 가능한 RAM | 1,983,922,176 B |
| 실행 결과 | exit 0, 시간·메모리 cap 미발동 |

환경은 보존된 기본 Python 3.13.5, **Torch 2.13.0+cpu**, float64, CPU 1 thread다. `CUDA_VISIBLE_DEVICES=-1`, CUDA build 없음, `cuda_initialized=false`, 모든 solve tensor가 CPU에 있음을 실행 기록에서 확인했다. 추가 패키지 설치와 GPU 사용은 없었다. 감독 한도는 300초·working set 2,000,000,000 B였다. 9×9 절점, 변위 162자유도로 두 문제를 계산했다.

전체 시간에는 import와 프로세스 시작이 포함된다. 두 solve 합계는 **0.591971 s**다. 이 작은 수치 문제의 시간을 가전제품 Explicit FEM이나 AI 속도 향상 배수로 환산하지 않는다. 첫 실행 슬롯에서는 RAM 부족으로 Torch import 전에 두 번 연기했고, 실제 실행은 FIG 학습 종료 통보 후 진행했다. 별도로 배정된 plate GPU probe와 정확히 어느 순간 겹쳤는지는 측정하지 않았다. 원로그의 초기 슬롯 관련 문구와 이 실행 시점의 차이는 `execution-context.json`에 설명했다.

## 균일 변형률 문제의 독립 해석 검산

길이 `(Ly,Lx)=(1,2)`, E=120, ν=0.25인 plane-stress 문제다. 축과 tensor 순서는 `(y,x)`이며, 네 외곽 경계에 `uy=.001y+.0003x`, `ux=.0003y+.002x`를 지정하고 체적력은 0으로 두었다. 독립 기대값은 다음과 같다.

- 변형률 `[[.001,.0003],[.0003,.002]]`; shear는 tensor shear다.
- 응력 `[[.192,.0288],[.0288,.288]]`.
- 면적 2, 단위 두께당 변형에너지 `0.00078528`.
- 경계 응력 `σn`을 각 경계 길이에 따라 적분한 절점력. 모서리에서는 두 경계의 기여를 더했다.

공식 코드를 이용해 `K*u`를 만든 뒤 그것을 정답 RHS로 재사용하는 방식은 쓰지 않았다.

| 검산 항목 | 실제 오차 |
|---|---:|
| 변위 relative L2 | 2.32419×10⁻¹⁰ |
| 변형률 최대 절대오차 | 4.04699×10⁻¹² |
| 응력 최대 절대오차 | 5.18015×10⁻¹⁰ |
| 독립 경계 traction 절점력 최대 절대오차 | 1.20986×10⁻¹⁰ |
| 해석 에너지 대비 상대오차 | 1.38066×10⁻¹⁶ |
| 자유 자유도 힘 잔차 상대값 | 9.43714×10⁻¹⁰ |

81절점의 좌표·계산 변위·변형률·응력을 `cpu-execution/affine-patch-fields.csv`에 보존했다. 독립 검토자는 솔버를 다시 실행하지 않고 이 CSV를 별도의 해석식과 산술 대조해 변위·응력·변형률 오차를 재현했다.

## 체적력·고정단 문제의 검산 범위

`(fy,fx)=(-.01,0)`인 상수 체적력을 독립 입력하고 x=0의 두 변위 성분을 고정했다. 나머지 경계는 자연 경계다. 단위 두께당 전체 체적력의 독립 기댓값은 `[-.02,0]`, 전체 지지반력은 `[+.02,0]`이다.

| 검산 항목 | 실측 |
|---|---:|
| 전체 체적력 `(y,x)` | `[-.0200000000,0]` |
| 전체 지지반력 `(y,x)` | `[.0199999999529,1.01305×10⁻¹³]` |
| 자유 자유도 힘 잔차 상대값 | 5.12680×10⁻⁹ |
| 전체 힘 평형 잔차 상대값 | 2.14065×10⁻⁹ |
| 전체 모멘트 평형 잔차 상대값 | 1.13067×10⁻⁹ |
| `2U = uᵀ(F_body+R)` 상대오차 | 2.50510×10⁻⁹ |
| 단위 두께당 변형에너지 U | 1.08296562909×10⁻⁵ |
| 고정 경계 변위 오차 | 0 |

공식 residual `V⁻¹Ku-f`는 절점 dual area로 나눈 값이므로, 반력과 힘·에너지는 V를 곱해 계산했다. 지지 경계의 반력까지 0으로 요구하지 않는다. `uᵀF`는 최종 힘과 변위의 내적이며, 하중 증가 경로에서 적분한 일과 구분한다. 이 문제에서는 독립 해석 변위해에 대한 정확도나 메시 수렴을 검증했다고 주장하지 않는다.

## 원본·버전·단위 계약

원본은 [NVIDIA/physicsnemo 고정 commit의 elasticity.py](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py)이며, 소스가 선언하는 버전은 **2.3.0a0**다. 기존 GPU 연구 환경에 설치된 PhysicsNeMo **2.2.2**와 구분한다. 실제 CPU 실행에서는 설치된 PhysicsNeMo 패키지를 import하지 않고 고정 원본 파일을 그대로 load했다. 공식 수치 알고리즘을 다른 자체 FEM 구현으로 바꾸지 않았다.

| 검증 대상 | SHA256 |
|---|---|
| 실행한 원본 `elasticity.py` | `822d7d4a11b36411975b0aa08f7ed87ac90ca0e691555de13db5f4b3473147ee` |
| 실행한 `run-official-q1.py` | `526c7e4b962d9e06e5e655693daebbe0407894782996f7d0e0b9f57e6a31c4a2` |
| 19파일 원본 출처 manifest | `19471259c443d4f421213bbb2bb9d878634154a5671a36351e9182acb6b358e3` |

원본 19파일 141,239 B의 길이·Git blob SHA1·SHA256을 확인했다. 원본은 `work/expanded-elasticity/source/`에 보존했고, 공개 산출물에는 원본 설치 패키지·실행 바이너리·checkpoint를 넣지 않았다. 공식 코드의 Apache-2.0 고지를 유지한다.

공식 예제는 물리 단위계를 지정하지 않는다. 일관된 모델 길이 L와 응력 S로 해석해야 한다. `u[L]`, strain[-], E·σ[S], 체적력[S/L], 단위 두께당 에너지[S·L²]다. mm·MPa나 SI를 임의 배정하지 않았다.

전체 [공식 2×2 DDM 예제](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py)의 수렴이나 인터페이스 교환은 이번에 실행하지 않았다. 그 경로에서 thermal FNO 모듈을 전이 import하더라도 구조 FNO 생성·학습을 뜻하지 않는다. 이번 성공은 정적 선형탄성 수치 모듈에 한정되며 접촉·소성·동적 충격·파손·제품 조립체 검증은 별도다.

## 재현

저장된 실행 결과를 덮어쓰지 않도록 새 출력 디렉터리를 지정한다. 이미 Torch와 psutil을 제공하는 CPU Python을 사용하며, 이 작업에서 새 패키지를 설치하지 않았다. 저장소 루트의 PowerShell에서 실행한다.

```powershell
python outputs/expanded-research/benchmarks/elasticity_numerical/prepare-source.py
& outputs/expanded-research/benchmarks/elasticity_numerical/run-supervised.ps1 -PythonPath 'C:\Program Files\Python313\python.exe' -OutputDirectory 'work/expanded-elasticity/reproduced-cpu'
```

다른 환경에서는 `-PythonPath`만 해당 CPU Python 경로로 바꾼다. 감독 스크립트가 Torch import 전에 사용 가능한 RAM 1 GiB 이상을 확인하고, CPU 1 thread·GPU 비활성화·300초·2GB 제한을 적용한다. 실행 코드와 출력의 원래 해시를 결과 JSON과 비교할 수 있다. 최초 보존 결과는 `cpu-execution/execution.json`, `supervisor.json`, stdout/stderr 로그와 CSV다. 추가 검토는 `execution-review.md`, 정적 수식 검토는 `validation-contract-review.md`에 있다.
