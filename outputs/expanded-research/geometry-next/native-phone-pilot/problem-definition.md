# 휴대폰 조립체 형상 변경: Starter 실행 준비 문제

현재 상태는 **원본 해시 확인과 문제 구성 완료**다. 변경 deck 생성, 메시 검사, Starter, Engine, AI는 모두 **미실행(UNEXECUTED)**이다. 주 연구 경로는 [SHIFT 캠페인](../campaign/README.md)이며, 이 문서는 공개 휴대폰 조립체를 이용한 별도 native 실행 가능성 확인을 정의한다. 캠페인 문서의 작성·데이터 접근을 이 구성의 전제로 삼지 않는다.

## 대상과 변경

- 원본: `work/expanded-drop/source/Cell_Phone_Drop_0000.rad` (경로는 연구 workspace 기준).
- SHA256: `435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2`. 이번 구성에서 파일을 스트리밍하여 다시 확인했다.
- 기존 원본·native 증거 기준: 17 parts, 257,099 nodes, 130,218 solids = 109,600 TETRA10 + 20,618 BRICK. 단위는 Mg/mm/s. 원 모델 고지 CC BY-NC 4.0을 유지한다.
- 목적: 한 방향으로 1% 늘린 실제 조립체 형상이 기존 native Starter에서 읽히고 연결 조건을 유지하는지 확인한다. 학습 데이터셋이나 DOE를 정의하는 단계가 아니다.

원 `/RWALL/PLANE/1`의 두 점은 다음과 같다.

```text
M  = (-26.944114, -49.826285, 3.587149) mm
M1 = (-26.354080, -49.179045, 4.069787) mm
c = M
n = (M1 - M) / ||M1 - M||
e_z = (0, 0, 1)
t = (n × e_z) / ||n × e_z||
A = I + 0.01 t t^T
x_new = c + A (x_original - c)
```

고정 중심 `c`는 원 wall의 점 M으로 선택한다. 실제 n은 전역 z축과 평행하지 않으므로 이 접선 t의 정의가 가능하다. 위 수식은 구성 명세이며 아직 변환 좌표를 계산하거나 저장하지 않았다. 모든 257,099개 물리 노드에 동일한 변환을 적용해야 하며, 181,034개 TETRA10 midside 노드도 포함한다. NODE_ID, element ID, 순서 있는 연결, part/material 할당과 집합·surface의 ID 참조는 유지한다.

원 wall 좌표, 마찰, 초기 속도, 중력, 재료 및 나머지 물리 카드는 고정한다. 원 입력의 `/INIVEL/TRA`와 `/GRAV`는 전역 좌표계(skew 0)를 사용하며 별도 `/SKEW`, `/FRAME`, `/TRANSFORM` 또는 참조 형상 카드는 확인되지 않았다. 이 변경은 좌표계 변환이 아니라 고정 전역 좌표계에서의 실제 형상 변경이다. 밀도 고정으로 기하학적 체적과 질량은 이론상 1.01배가 된다. 실제 Starter 질량도 확인해야 한다.

`n^T A = n^T`, `det(A)=1.01`이므로 정확 산술에서는 원 wall의 법선 gap이 유지되고 `J_new=A J_original`이다. 기존 Jacobian의 부호를 보존한다는 성질은 기존 메시 품질이나 변형 후 동적 안정성을 보증하지 않는다. 파일 직렬화와 solver 정밀도를 적용한 뒤 모든 노드의 gap, 요소의 적절한 적분점 Jacobian 및 품질을 확인해야 한다. 원 BRICK의 퇴화 wedge 표현은 실제 정식화에 맞게 검사한다. wall 법선 방향 크기 보존을 휴대폰 자체 두께 보존으로 해석하지 않는다.

## 연결 및 native 판정

실제 14개 `/INTER/TYPE2` 카드는 모두 `Ignore=1`, `Spotflag=27`, `Dsearch=0`이다. `/DEFAULT/INTER/TYPE2`의 Ignore=3보다 개별 카드가 우선한다. 비등방 형상 변경은 최근접 main segment와 투영 좌표, 크기에 의존하는 탐색을 바꿀 수 있다. 원·변경 모델에서 다음을 비교한다.

1. 실제 유지·제외된 secondary NODE_ID와 연결된 main segment/NODE_ID.
2. 각 interface의 kinematic/penalty 전환 NODE_ID, 투영 경고의 NODE_ID·투영 좌표·거리.
3. 실제 rigid-wall 포함·제외 NODE_ID 및 초기 gap. 분석적 gap 불변만으로 solver의 최종 wall 집합 불변을 선언하지 않는다.
4. 새 오류, 연결 누락, formulation 변경 및 예상 질량 변화. 기존 경고는 기준과 비교하여 설명한다.

기준 `outputs/expanded-research/benchmarks/drop_test/starter-full-listing.out`에는 0 errors / 7 warnings가 있다. ID 1873은 interface 1의 30개 노드에 대한 투영 경고가 두 번 기록되어 있다. ID 1179는 interface 1의 146개와 interface 8의 57개 노드가 penalty로 전환되었음을 기록한다. ID 446은 rigid wall에서 1,175개 노드가 제외되었음을, ID 312는 3,525개 kinematic 조건의 충돌과 wall 제외 후 해소를 기록한다. ID 1606은 Engine 입력 부재로 `/DT1TET10`을 확인할 수 없다는 경고다.

기존 listing에는 경고 대상의 상세 ID가 있지만 **모든 정상 연결의 전체 대응표가 있는지는 확인하지 않았다**. 그 증거가 부족하면 같은 진단 설정으로 원본 Starter control을 먼저 확보해야 한다. 단순히 경고 개수가 같거나 새 경고가 없다는 이유로 연결이 같다고 판정하지 않는다.

## 재개할 실행 범위와 자원 조건

미래 작업 경로는 `work/geometry-campaign/phone-tangent-plus1pct/`로 예약하며 이번 구성에서는 만들지 않았다. 새 원본 사본·변경 deck은 그곳에 별도로 보존한다. frozen history와 기존 입력·결과를 덮어쓰지 않는다. 실행할 때는 다음 **Starter executable만** owned process runner로 호출한다.

```text
executable: work/opencourant-experiment/solver/OpenCourant/exec/starter_win64_sp.exe
arguments:  -i Cell_Phone_Drop_0000.rad -np 1 -nt 2
working directory: work/geometry-campaign/phone-tangent-plus1pct/
```

환경 변수와 DLL 경로 설정은 `outputs/expanded-research/benchmarks/drop_test/run-starter.ps1`을 참조한다. 이 기존 스크립트는 옛 출력 경로를 사용하므로 그대로 실행하지 않고, 환경 설정만 새 owned runner에 반영한다. `OMP_NUM_THREADS=2`, 1 domain, 기존 single-precision executable을 유지한다.

- 시작 조건: available physical RAM ≥ **1.25 GiB**, 10초 간격의 두 관측 모두 충족.
- 실행 중 floor: available RAM **512 MiB**. owned process family memory cap **1 GiB**, wall time cap **180 s**, threads **2**.
- 리드가 이번 구성에 전달한 available RAM 약 **0.7 GiB**는 시작 조건 미달이다. 이 문서 작성 과정에서 시스템 메모리를 새로 측정하지 않았다.
- 기존 Starter 실측은 9.83초, peak working set 561,160,192 B(약 535.16 MiB)다. 시작 조건은 이 실측과 floor에 여유를 둔 값이며 완료 시간이나 메모리 상한을 보장하지 않는다. 조건 미달이면 실행하지 않고, 실행 중 제한 초과 시 owned family를 종료·정리한다.

Starter 종료 성공, 출력 가독성, 수치적 타당성, 공학적 적합성을 각각 보고한다. Starter가 성공해도 Engine 응답·AI 정확도·실물 휴대폰 성능이 검증된 것이 아니다. 독립 FEM 재메시는 아직 생산 실행용으로 구성하지 않았다. AI 입력의 점 수·샘플 밀도 변경은 FEM 메시 수렴 시험과 구분한다. Engine horizon, DOE 및 샘플 수, 모델 학습은 이 구성에서 정하지 않았다.

## 근거

- 기존 증거: `outputs/expanded-research/benchmarks/drop_test/PHONE-FEM-AND-DATASET.md`, `starter-execution-metrics.json`, `starter-full-listing.out` 및 원 RAD 카드. 카드 위치: INIVEL 497324, GRAV 523043·548760·548765, 첫 TYPE2 548772, RWALL 629252.
- 실제 보관 producer: `work/expanded-drop/control-source-review/OpenCourant/starter/source/interfaces/inter3d1/i2tid3.F`의 누락 연결 처리(110줄), 투영 검사(132줄), 선택된 연결 출력(211줄).
- [Radioss 2022 /RWALL](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/rwall_starter_r.htm), [TYPE2](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/inter_type2_starter_r.htm), [TETRA10 이론](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/theory_element_library_solid_tegrahedron_elements_r.htm). 분석적 affine 성질은 이 요소의 등매개 표현에 적용한 수학적 추론이다.

기계 판독용 구성과 각 단계의 미실행 상태는 [problem-config.json](problem-config.json)에 기록했다.
