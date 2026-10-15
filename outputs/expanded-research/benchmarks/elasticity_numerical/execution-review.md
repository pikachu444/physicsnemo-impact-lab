# 공식 Q1 탄성 CPU 실행 기록 — 독립 검토

검토일: 2026-10-15. 판정: **기록된 단일 CPU 실행과 13개 검증 gate는 일관되며, 제한된 선형탄성 수치 검증 성공이라는 표현이 타당하다.** PhysicsNeMo NN 학습·예측 검증 또는 공식 전체 DDM recipe 완료로 확대하면 안 된다.

이 검토자는 solver를 재실행하지 않았고 Torch/NumPy/PhysicsNeMo/TF/JAX를 import하지 않았다. 소스·로그·JSON·CSV를 읽고, 표준 라이브러리만으로 파일 해시와 저장 CSV의 해석 오차 산술 대조를 수행했다.

## 검토한 증거와 무결성

- [실행 결과](cpu-execution/execution.json), [외부 감시 기록](cpu-execution/supervisor.json), [stdout](cpu-execution/driver.stdout.log), [stderr](cpu-execution/driver.stderr.log), [아핀장 CSV](cpu-execution/affine-patch-fields.csv).
- [실행 driver](run-official-q1.py), [감시 실행기](run-supervised.ps1), [원본 manifest](source-manifest.json), [사전 정적 소스 보고서](static-source-report.json), [정적 검증 계약](validation-contract-review.md).
- 원본 source 19개, 총 141,239 B의 현재 bytes/SHA256가 manifest와 모두 일치했다. 공식 solver 파일 SHA256는 `822d7d4a11b36411975b0aa08f7ed87ac90ca0e691555de13db5f4b3473147ee`다.
- 현재 driver SHA256 `526c7e4b962d9e06e5e655693daebbe0407894782996f7d0e0b9f57e6a31c4a2`, manifest SHA256 `19471259c443d4f421213bbb2bb9d878634154a5671a36351e9182acb6b358e3`는 execution.json에 고정된 값과 일치했다.
- stdout의 pass 상태·13개 true gate는 JSON과 일치하며 stderr는 0 B다. 감시 기록의 exit code는 0, 시간·메모리 cap에 의한 종료는 false다.

위 source code와 driver의 현재 상태가 실행 기록과 연결됨을 확인했다. 독립 파일/CSV 감사의 상세 중간 증거는 `work/expanded-elasticity/review/record-audit.json`에 보존했다.

## 실행 환경·범위 구별

| 구분 | 확인된 기록 |
|---|---|
| 공식 소스 | NVIDIA/physicsnemo commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`, source metadata **2.3.0a0** |
| 이전 GPU 환경 관찰 | PhysicsNeMo **2.2.2**, Torch **2.13.0+cu126**는 사전 static report의 별도 환경 metadata |
| 실제 이번 프로세스 | 보존된 base Python, **Torch 2.13.0+cpu**, CPU 1 thread, float64 |
| PhysicsNeMo 패키지 | runtime distribution metadata **null**, `physicsnemo_imported=false`. 설치 패키지를 호출한 실행이라고 표현하지 않는다. |
| GPU | CPU-only Torch build, CUDA build version null, `CUDA_VISIBLE_DEVICES=-1`, `cuda_initialized=false`, solve tensor CPU 확인 |
| 문제 | 9×9 node, (Ly,Lx)=(1,2), E=120, ν=.25의 작은 선형·정적 plane-stress 두 문제 |

실제 실행은 공식 repository의 단일 `Elasticity2DSolver` 원본 class를 importlib로 읽은 것이다. 물리 단위계는 공식 코드가 지정하지 않아 길이 L·응력 S 기준이며 에너지는 단위 면외 두께당 S·L²다. 이전 설치 환경의 2.2.2와 이번 고정 source·base CPU runtime을 합쳐 하나의 재현 환경으로 부르면 안 된다.

감시 기록의 `GPU_used=false`는 실행기의 선언값이며 독립 GPU telemetry는 아니다. 하지만 CPU-only Torch build와 실제 runtime의 CUDA 미초기화·tensor CPU 기록이 이 프로세스의 CPU 계산을 함께 뒷받침한다.

## 수치 결과와 독립 확인

### 아핀 patch

독립 해석 기준은 `u_y=.001y+.0003x`, `u_x=.0003y+.002x`, tensor shear εyx=.0003, plane-stress σ=[[.192,.0288],[.0288,.288]], 단위 두께 에너지 .00078528이다. source의 (y,x), (dy,dx) 순서, `σn` nodal traction 적분, V가중 반력·에너지 정의가 정적 계약과 일치한다.

| 지표 | 기록값 |
|---|---:|
| 변위 상대 L2 | 2.3241896529×10⁻¹⁰ |
| 최대 strain 절대오차 | 4.0469927756×10⁻¹² |
| 최대 stress 절대오차 | 5.1801507528×10⁻¹⁰ |
| 해석 nodal traction 최대 절대오차 | 1.2098624880×10⁻¹⁰ |
| 해석 에너지 상대오차 | 1.3806568007×10⁻¹⁶ |
| V가중 free-force 잔차 상대값 | 9.4371444026×10⁻¹⁰ |
| 지정변위 최대오차 | 0 |

저장 CSV는 **81행·9개 y좌표·9개 x좌표**, 모든 값이 유한했다. 이 CSV에서 변위 해석식과 상수 strain/stress 기준을 별도 산술로 대조해 위 세 오차(변위 L2, strain/stress 최대오차)를 그대로 재현했다. 이 확인은 저장 기록 검산이며 FEM 재실행이 아니다.

에너지·traction gate의 계산 경로와 기록값은 검토했지만, reaction/internal-force 전체 배열은 CSV에 없으므로 이 감사에서 독립적으로 다시 조립하지 않았다. 아핀장의 해석 기준 자체는 별도 수식으로 확인했다.

### 독립 상수 체적력·left clamp

입력은 (fy,fx)=(−.01,0), x=0 양쪽 변위 고정이다. 단위 직사각형 면적 2에 대한 독립 총체적력은 (−.02,0), 기대 총반력은 (+.02,0)이다.

| 지표 | 기록값 |
|---|---:|
| 총체적력 y | −0.019999999999999997 |
| 총반력 y | +0.01999999995292146 |
| V가중 free-force 잔차 상대값 | 5.1268035479×10⁻⁹ |
| 전체 force balance 상대값 | 2.1406468830×10⁻⁹ |
| 전체 moment balance 상대값 | 1.1306687312×10⁻⁹ |
| `2U−uᵀ(F_body+R)` 상대값 | 2.5050998977×10⁻⁹ |
| 단위 두께 strain energy | 1.0829656291×10⁻⁵ > 0 |
| clamp 변위 최대오차 | 0 |

driver에 moment balance, 양의 에너지, 독립 총체적력 gate가 추가되어 있고 실제 실행 JSON에 모두 true로 남아 있음을 확인했다. `final_force_displacement_product_body_plus_reactions` 명칭과 2U 항등식 설명도 정적 검토 권고와 맞는다.

이 두 번째 문제의 최대 변위는 약 .00249418 L이지만 **해석해와 비교한 변위 정확도 결과가 아니다.** 여기서 에너지 항등식은 이산 평형·work 일관성 확인이며, 독립적인 전체 변위 정확도 검사를 대신하지 않는다.

총 13개 gate는 각각 driver에 선언된 기준을 만족한다. inner solver의 허용오차 1e−12는 preconditioned linear residual 기준이므로, 위 raw V가중 상대잔차가 약 5e−9라는 이유로 모순이라고 판단하지 않는다. 그 반대로 작은 patch에서의 통과를 다른 해상도·재료에 대한 정확도 보장으로 확대하지 않는다.

## 시간·메모리의 측정 범위

| 항목 | 시간·메모리 |
|---|---:|
| Torch/psutil import와 thread 설정 구간 | 9.2860109 s |
| affine solve 구간 | .2708589 s |
| body-force solve 구간 | .3211119 s |
| driver 내부 total | 9.9660099 s |
| 외부 supervisor process wall | **10.8977499 s** |
| 내부 process peak working set | 206,200,832 B |
| 외부 관측 peak working set | **206,229,504 B = 206.23 MB ≈196.68 MiB** |
| 실행 제한 | 300 s / 2,000,000,000 B |

내부 total은 driver 내부 import 시작점 이후, 외부 wall은 프로세스 시작·감시·종료까지다. 서로 다른 범위이므로 같은 “총시간” 하나로 합치거나 중복 합산하지 않는다. 이 기록의 9.286초는 새 프로세스 import 구간이며 파일시스템 cache가 차갑다는 cold-disk benchmark를 입증하지는 않는다. 단일 실행으로 통계적 성능이나 GPU·다른 solver 대비 배속을 주장하지 않는다.

## 문서 동기화·주장 한계

최종 확인 시 [execution-report.md](execution-report.md)가 현재 실행 결과·재현 진입점을 제공하고, README·static-source-report·memory-deferral은 **준비·연기 당시의 역사 기록**으로 명시적으로 구별했다. 최초 README의 “아직 실행 전”과 GPU venv 예시를 이번 실제 CPU 환경으로 읽지 않는다. 정적 검증 계약 역시 최초 검토 상태로 보존하고, 후속 gate 추가·실제 실행 정보는 이 새 실행 검토에만 기록한다.

[execution-context.json](execution-context.json)을 읽어 최초 FIG 작업 중 슬롯에서 available RAM 부족으로 Torch import 전에 두 번 연기했고, coordinator의 FIG 종료 통보 후 실제 CPU 실행으로 넘어갔다는 설명을 확인했다. 이는 coordination 기록에 근거한 순서 설명이며 이 감사에서 FIG·plate 프로세스를 관측한 것은 아니다. 별도 plate GPU probe와의 정확한 동시 실행 구간은 미계측이라고 명시되어 있다. 따라서 raw execution.json의 competing-workload 문구를 실제 FIG 동시 실행의 증거로 인용하지 않는다. 원 실행 로그·driver는 소급 수정하지 않았다.

타당한 결론은 **“고정 공식 소스의 Q1/BiCGSTAB 선형탄성 CPU 계산 두 문제를 실제 실행했고, 아핀 독립 해석값과 정적 평형·반력·에너지 검사를 통과했다”**다. 아래 범위는 이번 결과에서 주장하지 않는다.

- 전체 2×2 DDM 예제 실행·수렴
- 설치 PhysicsNeMo 2.2.2 package의 해당 example 재현
- AI 학습·추론 성능 또는 FEM 대비 AI 오차
- 동적 낙하·접촉·소성·파손·가전제품 검증

