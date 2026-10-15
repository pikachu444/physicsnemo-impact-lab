# 공식 FP-DDM 선형탄성 수치 검증 계약 — 독립 정적 검토

검토일: 2026-10-15. 공식 core commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`.
이 검토는 소스 텍스트와 해석적 수식만 다뤘다. Torch/NumPy/PhysicsNeMo/TF/JAX import, GPU 검사, 설치, 수치 실행, 대체 solver 구현은 하지 않았다. 아래 기대값은 실행 결과가 아니다.

기존 [source-matrix 검토](../../source-matrix/sinter-sym-notes.md)와 99-file source-blob manifest를 재사용했고, 실제 읽기는 별도 실행 담당이 확보한 19-file local source와 대응 공식 파일에 한정했다. source checksum은 [실행 소스 manifest](source-manifest.json)에 보존되어 있다.

## 1. 소스가 정의하는 API·좌표·구성식

| 항목 | 공식 구현 계약 |
|---|---|
| 변위·체적력·지정변위 | `[B,2,H,W]`, 성분 순서는 **(u_y,u_x)** / **(f_y,f_x)**. component-wise Dirichlet mask는 같은 shape의 bool이다. |
| 좌표·간격 | 격자 행이 y, 열이 x. `spacing=(dy,dx)`이며 각각 양수, H/W≥2다. `meshgrid(y,x,indexing="ij")`를 사용한다. |
| 변형률·응력 | `[B,2,2,H,W]`, 두 tensor 축 모두 **(y,x)**. [0,0]=yy, [1,1]=xx, [0,1]=yx. |
| 미분 | spatial axis 0→y, axis 1→x. 내부는 중앙차분, 첫·마지막 노드는 1차 한쪽 차분. gradient의 [i,j]는 ∂u_i/∂q_j. |
| 변형률 정의 | 소변형률 `ε=(∇u+∇uᵀ)/2`. XY 성분은 tensor shear이며 engineering γyx=2εyx다. |
| 구성식 | `μ=E/[2(1+ν)]`, **plane-stress** `λ₂D=Eν/(1−ν²)`, `σ=2με+λ₂D tr(ε)I`. 3D/plane-strain Lamé 상수를 넣으면 안 된다. |
| von Mises | `sqrt(σyy²−σyyσxx+σxx²+3τyx²)`, shape `[B,H,W]`. |
| 재료 | scalar 또는 `[B,1,H,W]`로 broadcast할 수 있는 E,ν. Q1 평형에서는 노드 재료를 Gauss 점에 보간한다. 물리적으로 E>0, 통상 3D 등방성 재료 범위 −1<ν<0.5를 검증 fixture에서 고정한다. 소스 자체가 이 재료 범위를 사전 검사하지는 않는다. |

근거: [API·shape](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L36), [미분·strain](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L103), [plane-stress·von Mises](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L134), [공식 좌표 생성](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py#L97).

## 2. 균일 변형률 patch의 독립 해석 기준

직사각형에 `q=(y,x)`, `u=Aq+c`를 놓는다.

```text
u_y = a y + b x + c_y
u_x = c y + d x + c_x

εyy = a
εxx = d
εyx = εxy = (b+c)/2
σyy = E/(1−ν²) · (a+νd)
σxx = E/(1−ν²) · (d+νa)
τyx = E/[2(1+ν)] · (b+c)
f = −div σ = 0
```

이 affine field는 Q1에서 정확히 표현된다. 공식 `strain()`의 중앙·한쪽 차분도 affine 미분에 일치해야 하므로 **모서리와 외곽까지** 동일한 상수 strain/stress가 기대된다. 해상도 H≠W 또는 dy≠dx를 포함하면 축 순서·간격 오류를 놓칠 가능성이 줄어든다.

구체적인 작은 strain fixture 제안:

```text
E=120, ν=0.25 → μ=48, λ₂D=32
a=0.010, b=0.004, c=−0.002, d=−0.003, c_y=c_x=0

ε (y,x order) = [[0.010, 0.001], [0.001, −0.003]]
σ (y,x order) = [[1.184, 0.096], [0.096, −0.064]]
strain-energy density w = 0.5 ε:σ = 0.006112
von Mises² = 1.509376
```

위 숫자는 구성식에 직접 대입한 해석값이다. 단위는 사용자가 지정하는 일관된 E·길이 단위에 따른다. 예제에 실제 SI 단위가 부여되어 있다고 가정하지 않는다. 두께 변수는 없으므로 에너지는 **단위 면외 두께당** 값이다.

실행 계약:

1. 외곽 전체의 양쪽 변위를 affine 해석값으로 지정하고 내부를 free로 둔다. `body_force=0`.
2. `initial=None` 또는 경계만 채운 초기장을 사용해 실제 내부 solve를 시험한다. 정답 전체를 initial로 넣어 잔차만 검사하고 solver 검증으로 끝내지 않는다.
3. 반환 변위와 독립 affine 해석값, 공식 strain/stress/von-Mises와 위 식을 각각 비교한다.
4. BC 오차, 최대절대오차·상대 L2 오차, 유한값 여부, 실제 tolerance 조건을 기록한다. float64의 반올림 및 반복해법 허용오차와 물리 단위를 고려해 gate를 선언한다. 이 정적 검토가 아직 수치 gate 통과를 확인한 것은 아니다.
5. 이 시험은 선형·소변형·정적 plane-stress 수치 solver 검사이며 NN 학습·낙하·접촉·소성 검증이 아니다.

## 3. force 부호, boundary traction, 반력

공식 구현은 `-div(σ)=f`를 목표로 하지만, `_equilibrium()`이 반환하는 것은 단순한 nodal finite-difference divergence가 아니다. 2×2 Gauss 완전적분 Q1의 assembled internal nodal force를 노드 control volume으로 나눈 값이다.

```text
K u       = assembled internal nodal force
V_i       = dy·dx (interior), dy·dx/2 (edge), dy·dx/4 (corner)
Fhat(u)   = V⁻¹ K u
residual  = Fhat(u) − f
nodal body load = V f
```

근거: [Q1 quadrature 및 force assembly](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L205), [V 나눗셈·residual 부호](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L260).

**검사해야 할 것은 free DOF 잔차다.** 외곽 Dirichlet 노드에서 `residual=0`을 요구하면 비영 반력을 오류로 판정하게 된다. 전체격자 residual을 strong-form PDE 오차라고 부르지 않는다.

고정 DOF의 constraint force를 `R_i=V_i·residual_i`로 정의하면, 이 부호는 body를 유지하기 위해 가해지는 외력이며 continuum traction `t=σn`과 대응한다. interior/free 잔차가 충분히 작을 때 다음이 기대된다.

```text
Σ R + Σ V f = 0
Σ (x·R_y − y·R_x) + Σ V(x·f_y − y·f_x) = 0
```

unit rectangle의 면 법선도 성분 순서 (y,x)로 써야 한다.

| 면 | outward n=(n_y,n_x) | t=(t_y,t_x) |
|---|---|---|
| y=0 | (−1,0) | (−σyy,−σxy) |
| y=Ly | (+1,0) | (+σyy,+σxy) |
| x=0 | (0,−1) | (−σyx,−σxx) |
| x=Lx | (0,+1) | (+σyx,+σxx) |

공식 domain helper의 이름 `TOP`은 array 첫 행이다. 공식 좌표가 0→1로 증가하므로 첫 행은 **y=0**이다. 그림의 화면 방향과 실제 좌표의 outward normal을 혼동하지 않는다. [domain boundary slices](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/domain.py#L121).

이 solver의 `solve()`에는 traction 전용 인자가 없다. free 외곽은 weak form의 자연조건에 해당하며, nonzero traction fixture를 만들려면 대응 consistent nodal load와 volume scaling을 별도로 정의해야 한다. boundary 노드 body_force 값을 임의로 traction과 동일하게 넣으면 단위·크기가 달라진다.

## 4. 에너지 검증 계약

공식 operator와 동일한 FEM 에너지는

```text
U_h = 0.5 uᵀ K u
    = 0.5 Σ_i V_i u_i · Fhat_i(u)
```

이다. `0.5 Σ u·Fhat`로 V를 빼먹거나, 전체 최종 외력·변위 내적을 그대로 U로 쓰면 안 된다. 선형 loading path에서는 최종 `u·(Vf+R)=2U_h`이며 점진적으로 가한 외력의 work가 U_h다.

Affine patch에서는 `U_exact=w·Lx·Ly`다. 위 fixture의 unit square는 `U_exact=0.006112`이고, 모든 node에 같은 w가 있으므로 올바른 trapezoid/control-volume 가중합도 같은 값을 준다. 이 경우 `ΣV=Lx·Ly`를 별도 확인한다.

변형률 tensor 내적은 대칭 비대각 성분을 두 번 센다.

```text
ε:σ = εyy σyy + εxx σxx + 2 εyx τyx
```

재료가 양정이고 Dirichlet constraints가 강체모드를 제거했을 때, free perturbation v에 대해 `vᵀKv>0`가 기대된다. 강체운동에는 0이므로 모든 nonzero field의 에너지가 양수라고 요구하지 않는다.

**중요한 구분:** `stress()`는 nodal finite-difference strain을 사용하지만 `_equilibrium()`은 Gauss-point Q1 strain을 사용한다. 비균일 변형에 nodal `0.5ε:σ`를 단순 평균한 값을 FEM 에너지와 동일하다고 단정하지 않는다. 공식 energy API는 없으며, operator work 또는 독립 quadrature 정의를 명시해야 한다.

## 5. 공식 bending 제조해의 독립 기준

[공식 제조해](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py#L92)는 unit square에

```text
u_y=0.01 y²
u_x=−0.032 x y
E=120, ν=0.25
```

를 쓴다. 해석적으로

```text
εyy=0.02 y, εxx=−0.032 y, εyx=−0.016 x
σyy=1.536 y, σxx=−3.456 y, τyx=−1.536 x

(div σ)_y = ∂y(1.536 y) + ∂x(−1.536 x) = 0
(div σ)_x = ∂y(−1.536 x) + ∂x(−3.456 y) = 0

w=0.070656 y² + 0.024576 x²
U_continuum on unit square = 0.031744
```

이다. 위 식은 공식 소스에 적힌 −3.2 비율을 독립 확인한다.

하지만 `strain()`의 경계 차분 때문에 정확한 nodal 해석변위를 넣어도 y=0,1의 ∂u_y/∂y는 해석값에서 **±0.01·dy**만큼 다르다. 따라서 경계 stress가 해석식과 기계 정밀도로 일치해야 한다는 gate는 잘못이다. 내부 nodal strain/stress, 전체장 error, 경계 error를 분리하고 refinement 추세를 본다. Q1 strain은 cell 내부에서 quadratic u_y를 정확히 재현하지 못하므로 FEM 에너지와 위 continuum 에너지도 finite mesh에서 차이가 가능하다.

공식 example이 반환하는 `displacement_relative_error`와 `stress_relative_error`는 **DDM 대 monolithic** 상대오차다. 제조해 대 monolithic 오차는 따로 계산·보고해야 독립 정확도 검사가 된다. 같은 solver 두 경로의 일치만으로 구성식·축·평형을 독립 입증하지 않는다. [return error 계산](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py#L249).

## 6. solve tolerance·rigid mode·DDM 수렴

공식 `solve()`는 다음 양수 diagonal scale D를 쓴다.

```text
D_y = [(λ₂D+2μ)/dy² + μ/dx²]⁻¹
D_x = [μ/dy² + (λ₂D+2μ)/dx²]⁻¹
fixed = masked Dirichlet field, correction v=0 on fixed DOF
rhs = D [f−Fhat(fixed)] on free DOF
A v = D Fhat(v) on free DOF
```

최종 gate는 batch별 `||rhs−Av||₂ ≤ tolerance·max(||rhs||₂,1)`다. 이는 **전처리된 이산 선형 시스템 잔차**이며 unscaled physical residual < tolerance라는 뜻이 아니다. `max_iter` 종료 후 이 조건을 다시 검사하고 실패하면 `RuntimeError("elasticity solve did not converge")`를 낸다. [scale 및 threshold](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L313), [최종 검사](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity.py#L425).

공식 example은 float64, inner max_iter=1000, inner tolerance=1e−11을 고정한다. 외부 `run_elasticity(tolerance=...)`는 DDM interface stopping tolerance이며 inner solver tolerance를 바꾸지 않는다. 두 값을 실행 기록에 별도로 적는다. [설정](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/elasticity_example.py#L174).

2D 선형탄성의 강체모드는 두 평행이동과 작은 회전이다.

```text
u_y=c_y−ωx
u_x=c_x+ωy
ε=σ=0, K u=0
```

이들은 constitutive/operator null test로 유용하지만, 유일한 solve에는 constraints로 제거되어야 한다. 예를 들어 원점에서 두 성분 고정 + 다른 x축 점에서 u_y 고정이면 rotation도 제거된다. 공식 example처럼 외곽 전체를 고정하면 충분하다. solver는 mask의 강체모드 제거를 사전에 분석하지 않으며, f=0·initial=0이면 singular free-body 문제도 zero-residual 상태로 반환할 수 있으므로 “반환 성공=유일성 확인”으로 해석하지 않는다.

DDM `converged`는 exchange 전 interface displacement RMSE가 tolerance보다 **작은지**의 상태다. 최대 iteration 도달은 종료지만 converged가 아니다. DDM 결과의 global free-DOF equilibrium·energy와 monolithic 오차는 추가로 검사해야 한다. [interface metric](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/interfaces.py#L221), [EarlyStopping](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/tcad/fp_ddm/fpddm/schwarz.py#L119).

## 실행 담당에게 넘길 최소 판정표

| 검사 | 기대·기록 계약 |
|---|---|
| Source/API | pinned SHA·source manifest, dtype/device, H/W, (dy,dx), (y,x) 순서 보존 |
| Affine patch | 독립 상수 strain/stress/VM·U와 비교; 외곽 포함; body_force=0 |
| Solve | BC 만족, finite값, inner preconditioned residual gate 만족, raw free residual 함께 기록 |
| Equilibrium | free DOF 기준, 반력에는 V 곱함; 전체 force/moment balance |
| Energy | V-weighted operator work와 affine 해석 U; 단위 면외 두께 명시 |
| Rigid field | strain/stress/operator가 0 기대; singular solve 성공과 분리 |
| Bending | continuum 제조해 확인, 내부·경계 오차 분리, FEM/continuum 에너지 차이 구분 |
| DDM | interface stopping 상태와 실제 global residual/monolithic/제조해 오차 분리 |

이 문서는 실행 계약을 정의한다. 어떤 수치 gate도 실제 통과했다고 주장하지 않는다.

## 부록: 실행 전 driver 정적 검토

2026-10-15에 `run-official-q1.py`를 **텍스트로만** 독립 검토했다. 스크립트 실행, Torch/NumPy/PhysicsNeMo import 및 GPU 접근은 하지 않았다. 검토 대상 기본값은 size=9, CPU, float64이며 `--execute` 전에는 표준 라이브러리만 로딩하는 구조다. 실행 시간·메모리·수치 gate 통과는 아직 미검증이다.

**큰 수학적·형상·부호 문제를 발견하지 않았다.** 코드는 실행 시 공식 solver 원본 SHA256를 검사한 뒤 단일 class 파일을 importlib로 로딩하도록 되어 있으며, 별도 solver 대체 구현이나 `K*u`를 제조 RHS로 돌려 넣는 검사는 없다.

- 도메인은 `(Ly,Lx)=(1,2)`, `(dy,dx)=(1/(n−1),2/(n−1))`, meshgrid `ij`다. 변위·tensor 성분의 (y,x) 순서가 공식 API와 맞는다.
- affine 입력 `u_y=.001y+.0003x`, `u_x=.0003y+.002x`의 독립 기준 `ε=[[.001,.0003],[.0003,.002]]`, `σ=[[.192,.0288],[.0288,.288]]`는 E=120, ν=.25 plane-stress 식과 맞는다.
- 에너지 밀도는 `.5·(.001·.192+.002·.288+2·.0003·.0288)=.00039264`, 면적 2를 곱한 단위 두께 에너지는 **.00078528**이다. driver의 해석 기준이 맞는다.
- 경계 force 비교의 `sigma[:,normal_direction]`는 tensor 열에 outward-normal 부호를 곱한 `σn`이다. y 면에는 dx 가중치, x 면에는 dy 가중치를 곱하고, 선분 끝점을 1/2 가중하여 corner의 두 면 기여를 더한다. 해당 indexing/broadcast는 `(2,n)` force 배열과 일치한다.
- 반력은 고정 DOF에서 `V·(internal−body_force)`, free 잔차는 V를 곱한 force 기준이다. 에너지 `.5Σu·(V·internal)`와 `2U` 대 최종 외력·변위 내적 비교의 1/2와 부호가 맞는다.
- 별도 `f_y=−.01`, `f_x=0` 문제는 x=0의 양쪽 변위를 고정하므로 2D 강체모드가 제거된다. 입력 하중은 독립적으로 지정된다. 변위 정확도의 닫힌 해를 주장하지 않는 scope 문구도 적절하다.

body-force 문제에서 추가로 대조할 수 있는 독립 전역 기댓값은 `ΣV=2`, `ΣF_y=−.02`, `ΣR_y=+.02`, 양쪽 x 성분 합 0이다. 원점 기준 body-load moment `ΣV(xf_y−yf_x)=−.02`이므로 support moment는 +.02다. 현재 driver는 force balance를 검사하지만 moment balance와 양의 body strain energy를 별도 gate로 두지는 않는다. 이는 이번 정적 검토에서 실행을 막을 큰 오류로 판단하지 않았으며, 결과 해석에서 구분한다.

`external_work_body_plus_reactions` 필드는 **최종 외력과 변위의 내적**이므로 `2U`와 비교하는 값이다. 적분한 준정적 외력 work 자체를 U라고 설명할 때 이 필드 값을 그대로 인용하지 않는다. body energy/work 검사는 평형에 수반되는 virtual-work 일관성 검사이고, 독립적인 body displacement 정확도 검사는 아니다. 독립 에너지 기준은 affine 문제의 .00078528이 제공한다.

기본 9×9 fixture와 inner 1000 iteration 상한에서 정적으로 보이는 비합리적인 배열·출력 규모는 없다. 다만 아직 실행 전이므로 5분 이내 완주를 보증하지 않는다. `--size`를 크게 바꾸는 실행은 이 bounded 정적 검토 범위 밖이다. 기존 source·cache·driver는 이 검토에서 수정하지 않았다.

