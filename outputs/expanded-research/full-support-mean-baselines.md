# Full-support train-mean 기준선

학습 case의 physical node×time×C 산술평균을 모든 held-out case에 그대로 적용했다. 별도 기준선 산출과 자체 검산이며 제3자 독립 PASS 판정이 아니다.

입력조건을 무시하므로 material·wall·loading·thickness 변화에 따른 case별 차이를 예측하지 않는다. NN 학습·test 예측·체크포인트 선택을 추가 실행하지 않았다.

작거나 0인 변위 성분의 relative L2는 물리 단위 절대오차와 truth 분모를 함께 읽는다. 분모 0은 null이며 성분별 퍼센트 순위를 만들지 않는다.

## phone

상태: `computed_and_self_checked`

5개 train case 평균을 float64로 독립 누적한 뒤 float32로 대조하여 bitwise 일치를 확인했다.

변위 벡터 relative L2: `0.0010736241225744`; centroid 제거 L2: `0.055563783172284`.

|필드|단위|RMSE|relative L2|P95 / 최대 절대오차|음수 출력|
|---|---|---:|---:|---:|---:|
|displacement_x|mm|3.996347327e-05|0.001007123181|6.731785834e-06 / 0.004548466764|None|
|displacement_y|mm|3.929793566e-05|0.0009028413537|8.076429367e-06 / 0.004758721218|None|
|displacement_z|mm|4.551427959e-05|0.001402367694|4.209578037e-07 / 0.002834283747|None|
|von_mises_GPS|MPa|0.04573966192|0.1026081388|0.04849283174 / 4.357067108|0|
|total_strain_frobenius|1|1.334359705e-05|0.06115375967|1.011606184e-06 / 0.001726981718|0|
|epsp|1|2.490792328e-06|0.7208524374|0 / 0.001362242037|0|

EPSP 양수 support·0인 영역의 오차·strict 시간 감소·개별 peak/위치는 동명 JSON에 기록했다. Bumper VM의 GPa 오차는 MPa 표시 시 ×1000이다.

## bumper

상태: `computed_and_self_checked`

112개 train case 평균을 float64로 독립 누적한 뒤 float32로 대조하여 bitwise 일치를 확인했다.

변위 벡터 relative L2: `0.20550271462252`; centroid 제거 L2: `0.33316222122059`.

|필드|단위|RMSE|relative L2|P95 / 최대 절대오차|음수 출력|
|---|---|---:|---:|---:|---:|
|displacement_x|mm|24.51114021|0.1698908525|49.2425156 / 105.1500092|None|
|displacement_y|mm|17.06444725|0.7609479407|37.01942394 / 69.00087929|None|
|displacement_z|mm|2.963337703|0.8906143981|6.171079564 / 30.30706644|None|
|inplane_vm_upper|GPa|0.1193934272|0.414693881|0.2597190343 / 0.6009079367|0|
|inplane_vm_lower|GPa|0.1202581835|0.4142904959|0.257989255 / 0.5973186046|0|
|EPSP_upper|dimensionless|0.01918174801|0.8382020679|0.02519863622 / 0.4373313785|0|
|EPSP_lower|dimensionless|0.01932097337|0.8465012249|0.02483643405 / 0.4978436641|0|

EPSP 양수 support·0인 영역의 오차·strict 시간 감소·개별 peak/위치는 동명 JSON에 기록했다. Bumper VM의 GPa 오차는 MPa 표시 시 ×1000이다.
