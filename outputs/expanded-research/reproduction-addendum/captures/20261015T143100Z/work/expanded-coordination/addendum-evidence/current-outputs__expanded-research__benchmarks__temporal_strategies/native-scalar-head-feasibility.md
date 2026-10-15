# AR/TF native scalar response-head 확장 검토

현재 상태: **소스만 확인한 실행 가능성 검토**. 새 확장 구현·프레임워크/NumPy import·데이터 배열 접근·GPU 실행은 하지 않았다. 기존 V2, 원 crash/drop wrapper, 원 C3 결과와 checkpoint는 보존한다.

사용자의 ‘알고리즘별 변위·응력·변형률’을 정적 여섯 아키텍처에 적용하면 예정된 공통 C6/C7 정적 학습이 해당 범위를 다룬다. 그러나 AR/teacher forcing까지 별도 비교 알고리즘으로 제시한다면 TC의 full-field 출력으로 AR/TF의 응력·변형률 예측을 충족했다고 주장할 수 없다. **원 C3 재현 실험은 유지하고, 두 새 도메인 × AR/TF의 4개 확장 학습을 별도 표에 추가하는 것이 유한하고 직접적인 보완**이다. 단순 shape/probe 성공으로 이 보완을 완료했다고 처리하면 안 된다.

## 출력과 물리량

| 도메인 | 위치/변위 | 직접 지도할 native scalar 응답 | 한계 |
|---|---|---|---|
| Phone full C6 | 3성분 | GPS nodal VM [MPa], projected native total-strain Frobenius, EPSP | VM은 scalar 한 개이며 stress tensor의 3성분이 아니다. GPS nodal VM과 raw cell hotspot은 서로 다르다. |
| Bumper C7 | 3성분 | upper/lower in-plane VM [GPa] 2개, upper/lower EPSP 2개 | 이 데이터에는 native total-strain 라벨이 없다. EPSP를 총변형률이라고 부르지 않는다. |

모든 라벨은 기존 immutable cache의 같은 field 순서와 단위를 사용한다. 변위 미분으로 만든 Green–Lagrange strain, strain tensor나 구성식으로 추정한 stress로 native 라벨을 대체하지 않는다. 새로운 head는 stress tensor를 복원하거나 재료 구성식을 식별하는 모델이 아니다.

## 코드상 가능한 최소 확장

설정의 `out_dim`만 늘리면 실패한다. 두 원 AR wrapper는 `outf` 전체를 3성분 acceleration 통계와 곱한다(crash/drop_test `rollout.py:247–250`). 원 TF는 C3 초기 좌표와 전체 `node_target`을 먼저 concatenate한다(crash `370–377`, drop_test `377–384`). 이미 기록된 원 C7 AR/TF 실패와 같은 차원 제약이다.

별도 wrapper/profile에서만 다음을 적용한다.

1. 원 GeoTransolver trunk와 source velocity·dt·3성분 velocity/acceleration 통계·gradient checkpointing을 유지한다. `functional_dim=4`, `geometry_dim=3`, 10 rollout steps도 유지한다. 신규 build에서만 `channels=data.c`, `out_dim=C6/C7`로 바꾼다.
2. 매 단계 head 출력을 `acceleration_normalized = raw[..., :3]`, `native_scalars_normalized = raw[..., 3:]`로 나눈다. acceleration 통계는 첫 3성분에만 적용한다.
3. 원 식으로 위치 3성분을 적분하고 `[position_next, native_scalars_normalized]`를 출력한다. 다음 단계 기하 상태에는 위치 3성분만 전달한다. native scalar는 입력이나 기하 적분으로 되돌리지 않는다.
4. TF의 GT geometry sequence는 **학습 중에만** `node_target[..., :3]`에서 만든다. 추론은 target 값 없이 원 좌표와 source initial velocity로 시작한다.

TF 첫 단계는 `P[-1], P[0]`에서 `P[1], S[1]`을 예측한다. 이후 학습의 단계 t는 과거 정답 위치 `P[t-1], P[t]`로 미래 `P[t+1], S[t+1]`을 예측한다. 같은 미래 위치 `P[t+1]` 또는 native scalar 정답을 forward input에 넣으면 안 된다. validation/final은 항상 자유 rollout이다.

`run_domain_temporal.py:104–137`의 sample/loss/physical은 이미 C>3 표현을 지원한다. 위치는 `coords + u/L`, position loss는 `L/train_u_std`로 환산하고, 나머지 scalar는 `(s-train_mean)/train_std`를 직접 예측한 뒤 동일 통계로 역변환한다. 원 C3/V2 함수는 고치지 않고 신규 build와 확장 wrapper만 별도 파일로 둔다.

**이 확장은 원 NVIDIA AR/TF의 다물리량 지원이라고 부를 수 없다.** 원 구조를 바탕으로 native scalar readout을 추가한 별도 adaptation이다. scalar 재료 이력이 별도 상태로 누적되지는 않으므로 보편적인 재료 이력 의존성이나 구성식 일관성을 보장하지 않는다. 음수 출력이나 EPSP 감소를 자르거나 보정하지 않고 그대로 평가한다.

## 추가 자원과 실행 예산

현재 실제 core의 output projection은 `LayerNorm(H) + Linear(H,out_dim)`이고 `include_local_features=False`이므로 H=48이다(core `geotransolver.py:503–507, 541–547`). 추가 파라미터는 `(C-3)*(H+1)`이다.

| 항목 | Phone C6 | Bumper C7 |
|---|---:|---:|
| 추가 affine 파라미터 | 147 | 196 |
| float32 target 또는 output 한 벌의 추가량 `4*N*10*(C-3)` | 9,127,800 bytes (8.70 MiB) | 2,188,000 bytes (2.09 MiB) |
| target와 output 두 벌만 합친 추가량 | 18,255,600 bytes | 4,376,000 bytes |
| head 추가 forward MAC/한 case의 10 step | 109,533,600 | 26,256,000 |

위 수치는 파라미터와 명시적 버퍼의 계산량이며 GPU peak 예측이 아니다. gradient·loss·stack/cat 임시 버퍼와 checkpoint 재계산의 추가 peak는 실제 3-update probe에서 측정해야 한다. trunk 폭·층·입력과 rollout 깊이는 늘지 않는다. 데이터 추가 생성이나 설치는 필요 없다.

권장 추가 실험은 **4개 fit, seed 하나, sweep 없음, 각 180 optimizer seconds**다. 따라서 추가 nominal optimizer cap 합계는 **720초(12분)**이며 기존 6개 원 실험의 1080초와 별개다. probe·setup·전체 validation·fresh reload·final을 더한 wall time은 아직 측정하지 않았다. 실행 계획상 20–30분 여유를 잡되 약속된 측정치로 제시하지 않고, 원 AR/TF 및 확장 probe의 warm step/validation 시간으로 즉시 갱신한다. 응답 head가 작다는 이유만으로 실행시간 증가율을 확정하지 않는다.

180초에서 한 complete case epoch의 단순 경계는 phone `180/5=36`초/update, bumper `180/112=1.6071`초/update다(일회 setup·validation 제외). bumper가 경계를 넘거나 whole epoch가 없으면 시험셋을 열지 않고 quality failure로 기록하고 root가 300초의 **별도 profile** 여부를 결정한다. 180초 fit을 덮어쓰거나 test 성능으로 예산을 바꾸지 않는다.

## 유한한 검증과 비교

- **Train-only adaptation probe:** 원 C3와 확장 모델의 shared trunk/acceleration 첫 3행을 동일하게 맞추면 scalar가 feedback되지 않는 한 position 출력이 일치해야 한다. 원 AR와 TF 양쪽에서 이 동등성, C6/C7 shape·finite, 실제 backward/Adam 갱신과 scalar head 행별 gradient/변경을 확인한다.
- **누출 점검:** eval에서 모든 target 값을 바꿔도 모든 출력이 동일해야 한다. TF training에서 scalar target 값만 바꿔도 forward 출력은 동일해야 한다. geometry teacher forcing은 과거 위치에만 한정한다. 검사에는 train case만 쓴다.
- **실제 학습:** probe 뒤 별도 4개 180초 fit을 끝낸다. 한 complete epoch와 전체 validation은 budget fit에서 확인하여 중복 pilot을 줄인다. 미완료라면 root에게 보고한다. loss는 TC와 같은 C6/C7 채널 평균이므로 C3 원 실험과 전체 MSE를 동일 목표로 순위 비교하지 않는다.
- **Validation 선택:** free rollout의 전체 validation cases × 10 time, train-only norm·split·seed 고정, complete epoch만 후보로 선정한다. native field별 RMSE/P95/max/hotspot, EPSP support·raw negative/history, displacement global/centroid-removed와 train-mean/zero/phone ballistic 기준값을 함께 남긴다. test로 threshold/weight/budget을 고르지 않는다.
- **고정 후 final:** wrapper/core/head-layout/config/norm/source/split/예측 해시를 freeze하고 새 프로세스 validation reload가 허용 오차를 만족해야 한다. 그 뒤 final test는 한 번만 자유 예측한다. 추가 오차 계산은 저장 NPZ만 쓴다.
- **표 구분:** (a) 원 NVIDIA wrapper 재현: TC full C6/C7, AR/TF C3; (b) 연구 확장: AR/TF native scalar response C6/C7. TC 원 실험은 다시 학습하지 않고 공통 full-field 참조로 사용한다. 변위 원 결과와 확장 결과의 변화도 보존한다.

검토 근거: frozen crash wrapper SHA `59d797f4fdf52473baac61d134923465b2620f344f5af3a2888f8b71bafa0afb`, drop wrapper `5b349a87aa6b5fcf994499d5848a655965fbec9ad51e1eeb384af56cfe548354`, prepared driver `955f25cf30a173c578eaa620cce6a0fae1e3d89bf072b0e21c5177e3854f3013`. 읽은 현 core SHA는 `1355b0f37d02875762a2d32a0660676803fe8f5fae6f1e4fb4bb7f451efd3cdc`다. 확장 구현/실행 시 실제 runtime core hash를 다시 기록해야 한다.
