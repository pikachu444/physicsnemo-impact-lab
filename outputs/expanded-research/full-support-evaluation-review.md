# Full-support 실행결과 독립 감사

갱신: 2026-10-15T07:47:39.138459+00:00

저장된 raw 예측과 FEM 배열을 CPU 1개로 독립 검산한다. 아직 완료되지 않은 실행은 pending이며 통과로 세지 않는다. passed는 실행 계약·저장 결과·독립 재계산의 일치 판정이다. 공학적 예측 품질은 동명 JSON의 실제 오차와 기준선 비교로 별도 평가한다.

|도메인 / 설정|감사 상태|업데이트 / 완료 epoch|선택 epoch|
|---|---|---:|---:|
|phone/transolver|passed|795 / 795|754|
|phone/flare|passed|247 / 247|247|
|phone/geotransolver|passed|682 / 682|682|
|phone/geoflare|passed|208 / 208|208|
|phone/meshgraphnet|passed|764 / 152|150|
|phone/figconvunet|passed|403 / 134|133|
|bumper/transolver|passed|4120 / 179|149|
|bumper/flare|passed|1488 / 64|58|
|bumper/geotransolver|pending_actual_completed_test|— / —|—|
|bumper/geoflare|pending_actual_completed_test|— / —|—|
|bumper/meshgraphnet|pending_actual_completed_test|— / —|—|
|bumper/figconvunet|pending_actual_completed_test|— / —|—|

phone/transolver 추가 기준선 감사: `passed`. 최초 필드 감사의 passed 기록과 백업을 보존한 별도 추가 감사다.

원 RAD 초기속도와 nominal 저장 시점으로 계산한 접촉 없는 균일 이동 기준선도 변위 벡터 지표에 한해 독립 검산했다.

- 순수 `v0×t`: relative L2 `0.019350611460141`.
- 중력 포함 `v0×t + 0.5×a×t²`: relative L2 `0.019351616593866`.

production의 정확한 중력 포함 지표 경로는 `phone_ballistic_reference.ballistic_physical_metrics.displacement_vector`이다. `gravity_only_physical_metrics.displacement_vector`는 초기속도를 제외한 중력만의 별도 기준선이다. 균일 이동 기준선은 접촉·변형·구속을 계산하지 않으며 scalar 기준선 지표와 그림은 별도 검산 범위에 포함하지 않는다.

무결성과 학습량 충족을 분리한다. 100 updates / 완료 epoch 2개 미달은 insufficient_budget이며 수렴·알고리즘 우열의 근거로 쓰지 않는다. 완료 epoch가 1개 이상이고 checkpoint가 고정됐으면 단 한 번의 test는 허용된다.

Phone full C6는 76,065 corner nodes, 10개 nominal 2..20us 시점, 5/2/2 split이다. 큰 강체 병진 비중 때문에 centroid 제거 오차를 함께 읽어야 한다. Bumper C7의 upper/lower VM 단위는 GPa(표시 MPa는 ×1000)이며 EPSP와 total strain을 혼동하지 않는다.

Raw 음수/비유한 값, EPSP 양수 support와 시간 감소는 보정 없이 센다. 분모가 0인 relative L2는 null이며 RMSE는 유지한다. 투영된 nodal head와 native cell hotspot은 같은 필드·support가 아니다.

검산 세부 수치와 해시는 동명의 JSON에 있다. GPU·Torch·PhysicsNeMo를 사용하지 않았고 예측이나 학습을 다시 실행하지 않았다. 미완료 결과의 정확도와 수렴은 확인하지 않았다.
