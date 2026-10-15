# 공식 수치 선형탄성 예제: 실행 전 준비

**정적 소스 확보와 독립 검증 드라이버 준비를 완료했다. 아직 Torch import·탄성 계산·GPU 실행은 하지 않았다.** 이 행은 AI/FNO 학습이 아닌 공식 Q1 FEM/BiCGSTAB 수치 탄성 예제다. 실제 실행과 자원 측정은 root가 CPU 슬롯을 배정한 뒤 진행한다.

공식 `NVIDIA/physicsnemo@11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`에서 기존 99파일 inventory의 fp_ddm 19파일만 회수했다. 총 **141,239 B**, Git blob SHA1과 SHA256을 모두 확인했다. 원본은 `work/expanded-elasticity/source/examples/tcad/fp_ddm/`에 수정 없이 보존했다. 설치나 외부 데이터·checkpoint는 필요하지 않았다.

## 소스 버전·범위

고정 current source는 `__version__="2.3.0a0"`이고 설치 패키지는 **PhysicsNeMo 2.2.2**, Torch metadata는 **2.13.0+cu126**이다. 서로 같은 버전이라고 표시하지 않는다. 드라이버는 설치 패키지를 덮어쓰지 않고 고정 소스의 `Elasticity2DSolver`를 그대로 load한다. 이 단일 파일은 Torch만 import한다. [공식 solver](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py), [current 버전](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/physicsnemo/__init__.py).

공식 전체 `run_elasticity()`는 2×2 domain decomposition과 monolithic solve를 비교한다. Schwarz 코드가 thermal FNO 관련 모듈을 전이 import하지만 탄성 FNO를 생성하거나 훈련하지 않는다. 이번 첫 실행 슬롯은 독립 공식 solver 검증에 사용하며 **전체 DDM 기본 recipe 성공으로 확대하지 않는다.** API의 size4/overlap2/patch3은 소스의 최소 조건을 만족하는 작은 DDM 설정이지만 아직 실행하지 않았다. [공식 예제](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py).

전체 requirements의 `opensimplex`는 현 metadata에서 발견하지 못했다. 이는 thermal layout용 의존성이며 단일 `elasticity.py` 실행에는 필요하지 않다. 지금은 추가 설치하지 않았다. 실제 import 성공은 아직 검증하지 않았다. 상세 AST import·설치 metadata는 `static-source-report.json`에 기록했다.

## 검증 설정과 독립 참조

드라이버는 공식 계산 함수를 다른 FEM 구현으로 대체하지 않는다. 9×9 node, 길이 (Ly,Lx)=(1,2), float64, E=120, ν=0.25를 사용한다. 축 순서는 `(y,x)`이고 변위는 `[B,2,H,W]`, 텐서는 `[B,2,2,H,W]`다. 공식 소스가 물리 단위계를 지정하지 않아 MPa/GPa를 임의 배정하지 않는다. 일관된 길이 L, 응력 S 단위로 u[L], strain[−], E/σ[S], body force[S/L], 단위 두께당 에너지[S·L²]를 기록한다.

첫 문제는 외곽 네 변에 독립 해석 변위 `uy=.001y+.0003x`, `ux=.0003y+.002x`를 지정하고 체적력을 0으로 둔다. 기대값은 균일 strain `[[.001,.0003],[.0003,.002]]`, plane-stress `[[.192,.0288],[.0288,.288]]`, 단위 두께당 에너지 **0.00078528 S·L²**다. 해석 응력의 σn을 경계 길이로 적분한 nodal traction도 독립 비교한다. `K·u`로 RHS를 만들어 같은 구현을 왕복하는 검증으로 대신하지 않는다.

두 번째 문제는 **독립 지정 상수 체적력 (fy,fx)=(−.01,0)**과 x=0 clamp다. 여기서는 자유 DOF 평형, 지지반력+체적력 및 moment 합, `2U=uᵀ(F_body+R)` 에너지 항등식을 검사한다. 전체 체적력의 독립 기댓값은 (−.02,0), 전체 지지반력은 (+.02,0)이다. 이 식의 우변은 최종 하중과 변위의 내적이며 하중을 올리는 동안 적분한 일 자체로 부르지 않는다. 이 문제의 변위 정확도를 별도 해석해와 비교했다는 주장도 하지 않는다.

공식 residual은 nodal dual-area로 나눈 `V⁻¹Ku−f`다. Dirichlet 경계 residual은 반력이므로 전체 residual=0을 요구하지 않는다. 반력과 에너지는 V를 곱해 계산한다. 소스의 nodal strain 미분은 경계 one-sided 차분을 사용하므로 일반 quadratic 굽힘 문제에서 경계 응력의 기계정밀도 일치를 요구하지 않는다. affine patch는 이 편향을 피하면서 축·shear·구성식·평형을 검사한다. 독립 정적 검토는 `validation-contract-review.md`에 있다.

## 자원 예상과 실행 슬롯

9×9 float64 계산 배열과 scratch는 1MiB보다 훨씬 작다. **전체 RSS는 Torch import/runtime이 지배**하며 수백 MiB에서 약1.5GiB까지 가능하다는 추정만 있다. 두 solve와 import가 5분 안에 끝날 가능성은 높지만 시간·메모리는 아직 실측하지 않았다. CPU 1 thread를 명시하고 GPU를 조회·사용하지 않는 첫 실행으로 준비했다. root가 배정한 CPU 슬롯 이전에는 `--execute`를 사용하지 않는다.

```powershell
# 정적 준비: framework import 없음
python outputs/expanded-research/benchmarks/elasticity_numerical/prepare-source.py
./work/gpu-experiment/venv/Scripts/python.exe outputs/expanded-research/benchmarks/elasticity_numerical/static-source-report.py
./work/gpu-experiment/venv/Scripts/python.exe outputs/expanded-research/benchmarks/elasticity_numerical/run-official-q1.py

# root CPU 슬롯 배정 후에만 실행할 명령
./work/gpu-experiment/venv/Scripts/python.exe outputs/expanded-research/benchmarks/elasticity_numerical/run-official-q1.py --execute --device cpu
```

드라이버는 기존 `execution.json`을 덮어쓰지 않으며 새 재현은 `--out work/expanded-elasticity/reproduced`처럼 다른 폴더에 기록한다. 공식 소스는 Apache-2.0 고지를 유지한다. 연구 전체의 기존 완료 산출물·캐시·타인 코드는 변경하지 않았다.
