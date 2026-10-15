# 공개 휴대폰 낙하 예제의 실제 FEM 실행 및 학습 데이터

공개 휴대폰의 9개 독립 조건을 실제 OpenCourant로 해석하고, 동일한 결과에서 rear-cover 19,933개 노드와 전체 조립체 corner 76,065개 노드의 학습 데이터를 모두 만들었다. 현재 이 문서가 입증하는 범위는 **FEM 실행·라벨 생성·데이터 검증**이다. AI 학습 성공과 예측 정확도는 별도 학습 결과로 판단한다. 여기의 모든 FEM 실행은 로컬 Windows CPU에서 이루어졌으며 Codex Cloud의 실행 결과로 서술하지 않는다.

## 원본과 재현성

[NVIDIA의 고정 소스](https://github.com/NVIDIA/physicsnemo/tree/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/openradioss_dataset_gen/drop_test)는 base deck을 포함하지 않는다. 공식 Cell Phone Drop 문서의 ZIP 주소는 실제 요청에서404였고 문서 공간도 잠겨 있었다. 따라서 [공개 미러의 고정 commit](https://github.com/rogerjdeangelis/altair-slc-openradioss-python-script-for-cell-phone-drop-simulation-no-gui/tree/274a4d71a845534fa27a927d7595c16a17c874fa)에서 원 starter를 확보했다. 미러 Git blob과 다운로드 ZIP 해시를 확인했으나, 접근 불가능한 공식 ZIP과의 byte 동일성은 확인하지 못했다. 미러의 engine는 수정된 파일이므로 original official engine로 주장하지 않는다.

Starter 42,378,237 bytes, SHA256 `435499d744ad560a843d9a21210483bbe74c9fc6e2775da4a232ce609cf097e2`. 원문 header는 Altair2022 **CC BY-NC4.0**이다. 원 모델의 이용 조건과 attribution을 유지해야 하며 이 모델을 연구 코드의 일반 라이선스로 재표시하지 않는다. 다운로드·solver·소스·입력 해시는 `source-provenance.json`, `environment.json`, `doe/frozen-case-manifest.json`에 남겼다.

## 실제 물리와 조건

원 모델은 17개 part, 109,600 TETRA10 및20,618 BRICK(총130,218 solid), 257,099 NODE,5개 재료,14개 TYPE2 tied interface와 rigid plane으로 구성된다. 단위는 Mg/mm/s, 응력 MPa, 에너지 원단위는mJ이다. 질량은27.01477g인 단순화한 휴대폰이며 실제 가전제품 데이터가 아니다. 초기 속도는(-3200.900476,-3511.241739,-2618.285239)mm/s, 모든 원 노드에 적용된다. 유리는 선형탄성이고 파손·접합분리 모델은 없으므로 유리파손이나 제품고장 확률을 직접 검증하는 예제가 아니다.

공식 generator는5개 재료E를 각각0.8/1.2배 조합하고6개 wall orientation을 적용해192조건을 생성한다. 그러나 실제 원 기하에 공식 ±10° 조건을 적용하니 초기침투가 발생했다. 따라서 이번 독립 실험은 원 wall origin과 모든 NODE를 유지한 채, **공통E배율0.8..1.2와wall Rx±1°의2차원 보간 범위**로 줄였다. 이 범위의 모든 조건은 원257,099 NODE를 대상으로 최소gap0/초기침투0을 계산해 확인했다. 독립5재료DOE나원공식192조건 재현으로 주장하지 않는다.

Train5=네 corner+중심, validation2/test2=서로 다른 내부점으로 새 해석 시작 전에 동결했다. 목표시간은2,4,...20µs이며 t0는 학습target에서 제외했다. 실제 ANIM 출력시각을 모두 보존하고, nominal시간은 명시한 실제 두 frame 사이에서 선형보간했다. 시간외삽은 없다. 종료조건20.02µs는20µs보간 상한을 확보하기 위한 두세 cycle의 출력 여유이며 원 물리카드는 바꾸지 않았다.

## 라벨 및 mesh 계약

| 항목 | 정의 |
|---|---|
| C6 | u_x,u_y,u_z[mm], GPS-derived nodal von Mises[MPa], total-strain tensor Frobenius[1], EPSP[1] |
| reference | 원RAD NODE_ID로 매핑한 원좌표. 모든9조건에공유. ANIM t0의최대1.907e-6mm차이는single precision표현오차로별도검사 |
| stress | solver의6개 nodalGPS stress로 VM 계산. 공식 Curator 연산순서와 같으며 native cellVM의nodal평균과구분 |
| strain | TETRA10의4개 validIP(111,112,121,211) 동등평균, BRICK Isolid1의IP111만사용 → 원corner체적가중 nodal globaltensor → Frobenius |
| EPSP | 원cell scalar를같은체적가중nodalprojection. Totalstrain과다른변수 |
| 원값 보존 | 마지막frame의cellVM/EPSP와validIP globalstress/strain tensor를별도nativeNPZ로보존. 총strain Frobenius가부호·방향정보를보존한다고주장하지않음 |
| 후처리 근사 | quadratic TETRA10 corner 및BRICK/wedge의선형corner체적으로가중. FEM적분체적의정확재구성으로주장하지않음 |
| rear support | part2 cover_rear,19,933 cornernodes,196,658 directedtrueTETedges,1component |
| full support | 전체solidcorner76,065 nodes,561,184 directedphysicaledges,17components.14TYPE2/tied및wallcontact를가로지르는graph edge는별도추가하지않음 |
| 제외노드 | 원181,034 TETRA10 midside nodes. Sourcedeck실파싱에서제외노드가정확히이집합임을검증했으며원solid미참조노드0 |

기존 solver/Curator가제공하는 원stress 출력은 stress/strain 라벨을 모두 자동보존하는 학습세트와 같지 않다. 이번 strain/EPSP projection은 명시적으로 추가한 경로다. mesh-only graph의17개분리성분은실제FEM의tied/contact물리가없다는뜻이아니며,새접촉배치일반화가검증됐다는뜻도아니다.

## 실측 결과

| Run | Split | 공통E | WallRx[°] | Engine[s] | peakRSS[MiB] | fullGPSVM최대[MPa] | fullnodalEPSP최대 |
|---|---|---:|---:|---:|---:|---:|---:|
| phone_e100_rx000 | train | 1.0 | +0.0 | 228.18 | 390.6 | 26.185 | 0.00055908 |
| phone_e080_rxm100 | train | 0.8 | -1.0 | 189.53 | 391.4 | 21.74 | 0 |
| phone_e080_rxp100 | train | 0.8 | +1.0 | 171.64 | 391.4 | 20.561 | 0 |
| phone_e120_rxm100 | train | 1.2 | -1.0 | 213.13 | 391.4 | 29.73 | 0.0051574 |
| phone_e120_rxp100 | train | 1.2 | +1.0 | 203.98 | 391.3 | 30.261 | 0.0010669 |
| phone_e090_rxm050 | val | 0.9 | -0.5 | 191.04 | 391.4 | 23.866 | 0.00023837 |
| phone_e110_rxp050 | val | 1.1 | +0.5 | 196.04 | 391.4 | 28.198 | 0.00083386 |
| phone_e090_rxp050 | test | 0.9 | +0.5 | 173.08 | 391.4 | 23.437 | 0 |
| phone_e110_rxm050 | test | 1.1 | -0.5 | 193.03 | 391.4 | 28.126 | 0.0027189 |

Engine9개 합계 1759.65s, 개별 171.64..228.18s. 새8개Starter 합계 85.36s. 각해석2CPUthreads, 단정밀도,Engine peakRSS약391MiB. 물리RAM이낮을때는대기하고600MiB프로세스상한/시스템free128MiB의정상중단조건을유지했다. 모델·출력·자원조건이다른경우시간은이값으로확정할수없다.

Baseline20µs의 native cellVM은51.36034MPa, nativeEPSP최대0.00272068이며소성발생solid4개(후면커버)다. 최대응력은마지막출력에서도상승중이었다. 따라서20µs를전체낙하최대응력·반발·에너지수지완료로해석하지않는다. 별도중심조건100µs재시작진단은 `100us/`의실행로그와summary로판단하며9조건AI학습범위는계속20µs다.

완료case추출은전체C6+rearC6+nativeNPZ+raw해시를포함하여실측12..15s,peakRSS약196MiB였다. Fulltarget9개는총164,301,552bytes. 공통fullgraph추가파일은약9.59MB,생성2.10s. Fulltarget을이미보존했으므로메모리추정만으로전체조립체학습불가라는판단을내리지않고,실제학습모델별자원probe와오차로판단할수있다.

## 재사용 파일

- `doe/canonical/manifest-rear-cover.json`, `manifest-full-assembly-corners.json`: 같은9개조건/5·2·2split/C6/실제graph를가리키는학습입력.
- `doe/canonical/dataset_READY_evidence.json`: 모든target해시,finite검사,rear==fullsubset의bitwise검사,조건별field범위·nonzero수.
- `doe/canonical/shared/`: 원좌표원본float64/입력float32,NODE_ID,cornerconnectivity/체적,두support의graph와staticmeshlength.
- `doe/canonical/runs/*/`: target.npy,target-full.npy,native-last-frame.npz,metadata.json. Metadata는12개rawANIM의actualtime/해시와보간brackets를포함한다.
- `doe/phone-nine-condition-fem-targets.png`: 실제FEM의조건별20µs반응비교. AI예측그림이아니다.
- `active-flag-and-node-coverage-correction.json`: legacy `solid_deleted_flags`의실제값은1=active인rawbyte였음을교정. 원NPZ/SHA는보존했고stress/strain/geometry배열불변을검증했다.
- `canonical-independent-review.md`: baseline캐시의별도읽기전용검토승인. Predictionaccuracy를대신하지않는다.

원rawANIM과solverrestart는 `work/expanded-drop/`에보존했다. GitHub에모든대용량raw가업로드됐다고이문서만으로주장하지않으며,원source취득·생성·실행·후처리코드및해시로재생성경로를남겼다.

## Stress/strain의 물리적 정의 검증

`solid-tensor-basis-and-shear-check.json`은 [실제 integration-point writer](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/output/anim/generate/tensor6.F#L1991)를 추적한다. BRICK1IP와TETRA10 모두 내부 engineering shear를0.5배하여 tensor shear로 만든 뒤 필요한 global좌표변환을 수행한다. 따라서 rawANIM을 읽은 후 shear를 또0.5배하지 않는다. Glass의 실제 유의 shear27,781개도 σ=2Gε와gain1.000001706/상대잔차0.00012857로일치했고, 약신호인PCB/기타부품은 이 convention추론에 사용하지 않았다. 독립 scientific gate도 원문분기를 확인했다.

[공식 GPS 설명](https://help.altair.com/2023.1/hwsolvers/rad/topics/solvers/rad/anim_gps_tens_engine_r.htm)에 따르면 integration-point stress의corner외삽 후 연결element의tensor를node에서평균한다. Baseline nominal20µs에서 이번학습GPS_VM최대는26.1854MPa, projectedEPSP최대0.000559076이며 nativecell최대51.3603MPa/0.00272068과다르다. 이 정의의 AI정확도가 nativecell/IP hotspot정확도를곧바로입증하지않는다. 제품설계의비평균hotspot/파손연구에는cell/IP직접target과출력정의를추가해야한다.

## 실제100µs 중심조건 확장결과

20.001940356µs정상종료checkpoint에서이어 **100.008663139µs까지정상완료**했다. 추가8,079cycles,2threads,721.59s,peakRSS388.82MiB,최소systemfree260.67MiB였다. 시작frame은기존20µs마지막과시간·좌표·연결·active·nativeVM/EPSP·GPS·validIPstress/strain이모두bitwise동일했다.

마지막저장frame의nativeVM은73.505MPa,totalstrainIP111Frobenius는0.108241,EPSP최대0.0625744,소성solid240개다. 최대VM은여전히마지막저장frame이었다. 실제history최종99.90963µs에서KE0.3512431J,IE0.04125439J,plasticwork0.01038432J였다. 외력일·contact·hourglass등을포함한수지는초기0.3975138J→0.397201625J,변화-0.07853%다. Plasticwork는IE의일부여서수지에중복가산하지않았다.

실제globalmomentum/mass의wallnormal속도는마지막-5.03212m/s이며저장history에서양수전환이없었다. 마지막physicalcorner5개에는국부양의속도가있지만전체반발과구분한다. 초기P/M=-5.41160m/s와원INIVEL=-5.42494m/s의작은차이는확인했으나원t0nodalvelocity가저장되지않아그원인은독립적으로특정하지못했다.

따라서 **100µs에서도전체최대응력·전체반발완료를입증하지못했다.** 이번연장은초기20µs이후에도응력과소성영역이계속확대됨을실제로보여준다. 9조건AI target은20µs로유지되고,100µs는학습되지않은단일FEM진단이다. 세부값과원시각frame해시/시각물은 `100us/summary.json`, `restart-boundary-bitwise-check.json`, `phone-100us-field-trends.png`, `phone-100us-energy-momentum.png`에있다.
