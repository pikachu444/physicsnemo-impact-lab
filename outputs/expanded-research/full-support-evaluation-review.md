# Full-support 실행결과 독립 감사

갱신: 2026-10-15T06:24:29.121057+00:00

저장된 raw 예측과 FEM 배열을 CPU 1개로 독립 검산한다. 아직 완료되지 않은 실행은 pending이며 통과로 세지 않는다.

|도메인 / 설정|감사 상태|업데이트 / 완료 epoch|선택 epoch|
|---|---|---:|---:|
|phone/transolver|passed|795 / 795|754|
|phone/flare|pending_actual_completed_test|— / —|—|
|phone/geotransolver|pending_actual_completed_test|— / —|—|
|phone/geoflare|pending_actual_completed_test|— / —|—|
|phone/meshgraphnet|pending_actual_completed_test|— / —|—|
|phone/figconvunet|pending_actual_completed_test|— / —|—|
|bumper/transolver|pending_actual_completed_test|— / —|—|
|bumper/flare|pending_actual_completed_test|— / —|—|
|bumper/geotransolver|pending_actual_completed_test|— / —|—|
|bumper/geoflare|pending_actual_completed_test|— / —|—|
|bumper/meshgraphnet|pending_actual_completed_test|— / —|—|
|bumper/figconvunet|pending_actual_completed_test|— / —|—|

무결성과 학습량 충족을 분리한다. 100 updates / 완료 epoch 2개 미달은 insufficient_budget이며 수렴·알고리즘 우열의 근거로 쓰지 않는다. 완료 epoch가 1개 이상이고 checkpoint가 고정됐으면 단 한 번의 test는 허용된다.

Phone full C6는 76,065 corner nodes, 10개 nominal 2..20us 시점, 5/2/2 split이다. 큰 강체 병진 비중 때문에 centroid 제거 오차를 함께 읽어야 한다. Bumper C7의 upper/lower VM 단위는 GPa(표시 MPa는 ×1000)이며 EPSP와 total strain을 혼동하지 않는다.

Raw 음수/비유한 값, EPSP 양수 support와 시간 감소는 보정 없이 센다. 분모가 0인 relative L2는 null이며 RMSE는 유지한다. 투영된 nodal head와 native cell hotspot은 같은 필드·support가 아니다.

검산 세부 수치와 해시는 동명의 JSON에 있다. GPU·Torch·PhysicsNeMo를 사용하지 않았고 예측이나 학습을 다시 실행하지 않았다. 미완료 결과의 정확도와 수렴은 확인하지 않았다.
