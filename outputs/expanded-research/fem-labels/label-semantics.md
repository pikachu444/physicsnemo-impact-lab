# Run224 범퍼 FEM 학습 라벨 의미 검토

검토일: 2026-10-15. 범위는 공개 Run224의 기존 OpenCourant 실행 결과와 출력 의미다. 이 문서는 새 학습 성공이나 strain 출력 재실행 성공을 주장하지 않는다. 공식 Altair 문서와 실제 실행 배포에 대응하는 OpenCourant commit `cb760c206e346eff12161a2268b134ef9d48a4ef`를 대조했다.

## 1. 확인 결과

| 항목 | 확인 사실 | 학습 데이터에 미치는 영향 |
|---|---|---|
| 재료·단위 | 원본 `Bumper_Beam_AP_meshed_0000.rad`는 kg–mm–ms, `/MAT/PLAS_JOHNS`(LAW2), E=210, ν=0.3이다. | 응력 단위는 kg/(mm·ms²)=GPa이며 MPa로 표시할 때 ×1000이다. strain과 EPSP는 무차원이다. |
| Shell 설정 | property 1,2,6,7에 Ishell=24(QEPH), 두께 적분점 N=5. 카드 주석의 Istrain 값은 0이다. 실제 Starter 출력은 네 property 모두 post-processing strain flag=1, small strain flag=2를 기록한다. | 카드 문자열의 0을 실제 실행 상태와 혼동하면 안 된다. |
| 기존 출력 요청 | Engine에는 `/ANIM/SHELL/TENS/STRESS/ALL`, `/ANIM/SHELL/EPSP/ALL`이 있고 `STRAIN` 요청은 없다. | 기존 VTK는 응력·EPSP 라벨을 가진다. 총변형률은 추가 출력 후 실제 비영값·유효성을 확인해야 한다. |
| 기존 VTK | 최종 VTK: POINT_DATA 13,882, CELL_DATA 13,889. 응력 tensor 7개와 EPSP scalar 7개(Upper, Lower, layer 1…5)가 CELL_DATA 아래 존재한다. | 숫자 행 순서만으로 node 라벨에 붙일 수 없다. ELEMENT_ID, PART_ID, cell 종류, 시간, 두께 위치를 키로 보존한다. |

위 표의 실행 증거는 기존 `work/bumper-fem-rerun/run-extended/Bumper_Beam_AP_meshed_0000.rad`, `_0001.rad`, `_0000.out`, `Bumper_Beam_AP_meshedA011.vtk`를 읽어 확인했다. 원본 또는 기존 로그는 수정하지 않았다. 단위 환산은 입력 단위계에서 차원 분석한 결과다. [LAW2 공식 정의](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/mat_law2_plas_johns_starter_r.htm)는 탄소성 모델과 true stress/strain 출력을 설명한다.

## 2. 총변형률 출력 flag의 버전 차이

Altair 2022.3 [post-processing FAQ](https://2022.help.altair.com/2022.3/hwsolvers/rad/topics/solvers/rad/faq_rad_post_processing_r.htm)는 Istrain=1을 후처리용 strain tensor 계산 옵션으로 설명한다. 그러나 이를 현재 OpenCourant에 그대로 적용하면 부정확하다.

실제 배포 commit의 [hm_read_prop01.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/starter/source/properties/shell/hm_read_prop01.F#L145)는 Istrain 입력 읽기를 주석 처리하고, 173행에서 `ISTRAIN=1`을 설정한 뒤 315행에서 property 상태에 저장한다. 기존 Run224 Starter 출력에서도 flag=1을 확인했다. 따라서 **이 버전의 Run224에서 Istrain 0→1 편집은 strain 출력의 필요조건이 아니다.** 필요한 첫 조치는 별도 Engine 파일에 `/ANIM/SHELL/TENS/STRAIN/ALL`을 추가하는 것이다.

[czstra3.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/shell/coquez/czstra3.F#L76)의 정상 LAW2 경로는 EXX/EYY/EXY 등 재료 계산용 증분을 Istrain 조건 밖에서 계산하고, 후처리용 GSTR 누적을 조건 안에서 수행한다. 이는 후처리 저장 목적이라는 문서 설명을 지지한다. 다만 모든 재료·모든 solver 버전에서 물리해석을 절대 바꾸지 않는다는 보장은 아니다. 특히 ISMSTR=10의 별도 total-strain 경로는 구분해야 한다. 현재 버전에서 legacy 필드 0→1만 바꿨을 때 응답이 같다는 구조적 근거와, 두 실행을 수치로 비교한 실증은 서로 다르다.

권장 검증은 원본 실행과 출력만 추가한 실행의 변위·응력·EPSP·에너지·종료시각을 비교하는 것이다. 변화가 없을 것으로 예상하더라도 실행 전에는 검증 완료로 쓰지 않는다.

## 3. 좌표계, 두께 위치, shear

[공식 /ANIM/SHELL/TENS](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/anim_shell_tens_engine_r.htm)는 응력·변형률 tensor가 **elemental coordinate system**으로 출력된다고 명시한다. QEPH의 local frame과 변형 갱신은 [czforc3.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/shell/coquez/czforc3.F#L415)의 global→local 변환 및 strain-rate 계산 경로와 [공식 좌표계 설명](https://2024.help.altair.com/2024/hwsolvers/rad/topics/solvers/rad/radioss_coordinate_system_intro_c.htm)을 함께 봐야 한다. local 성분을 고정된 전역 X/Y 응력으로 해석하면 안 된다. 시간별 좌표계 변환 행렬을 실제로 추출·검증한 작업은 아직 없다.

- STRESS Upper/Lower는 최외곽 상·하 적분점의 응력이다.
- STRAIN Upper/Lower는 막변형률과 곡률로 외표면까지 외삽한 값이다. 응력 Upper/Lower와 동일한 저장 연산이라고 가정하지 않는다.
- N=5인데 저장 배열이 7개인 것은 Upper/Lower와 1…5가 함께 저장되기 때문이다. **서로 독립적인 적분점이 7개라는 뜻이 아니다.** 위치 중복과 값 일치는 실제 배열별 비교로 확인한다.

정상 isotropic shell 출력 경로의 [tensorc.F 1388·1401행](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/output/anim/generate/tensorc.F#L1381)은 내부 XY strain에 HALF를 곱해 ANIM tensor 성분을 만든다. 따라서 이 경로의 출력 XY는 tensor shear `εxy`이며 engineering shear는 `γxy=2εxy`다. 응력 XY에는 이 strain용 1/2 환산을 적용하지 않는다. 모델 학습·정규화·오차 계산 메타데이터에 shear 관례를 명시해야 한다.

## 4. VTK 3×3 배열이 보장하지 않는 것

[공식 converter 소스](https://github.com/OpenCourant/Tools/blob/c58c805696f7364f0efe80b5dd1585f70670232e/output_converters/anim_to_vtk/src/anim_to_vtk.cpp#L1344)는 ANIM shell의 x, y, xy 세 성분을 아래처럼 채운다. 이 경로는 좌표계 회전을 수행하지 않는다.

```text
xx xy 0
xy yy 0
 0  0 0
```

이 검토가 읽은 converter 소스 revision은 `c58c805696f7364f0efe80b5dd1585f70670232e`다. 기존 실행기가 이 소스 revision으로 빌드되었다는 빌드 재현 증명은 별도다. 기존 VTK의 배열 구조는 위 방식과 일치한다.

따라서 VTK의 9개 항목을 독립적인 6성분 3D 라벨로 취급하면 안 된다. 특히 strain의 zz=0 패딩은 두께 변형률이 실제 0이라는 뜻이 아니다. LAW2 shell에서는 Poisson 효과와 소성 비압축성 때문에 두께 변형이 가능하며 [LAW2 shell 소스](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/materials/mat/mat002/sigeps02c.F)는 별도로 두께를 갱신한다. 현재 출력의 관측량은 local in-plane tensor다.

## 5. STRAIN과 EPSP의 의미·미확인 사항

EPSP는 LAW2에서 누적되는 scalar 등가소성변형률 라벨로 취급한다. [m2cplr.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/materials/mat/mat002/m2cplr.F#L195)는 von Mises 응력에 의한 소성 보정과 PLA 증분 누적을 수행하며, [공식 재료 예제](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/tensile_test_example_law_characterization_r.htm)는 LAW2의 failure parameter를 equivalent plastic strain으로 설명한다. EPSP는 방향을 가진 총변형률 tensor나 파손 확률이 아니다. 해당 물성·파손모델의 한계와 별개로 무차원 scalar다. [EPSP 공식 출력 정의](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/anim_shell_epsp_engine_r.htm)는 비복합 shell에서 번호가 두께 적분점 위치임을 설명한다.

STRAIN은 EPSP와 구분되는 solver의 strain tensor 출력이다. 정상 QEPH 경로는 [czstra3.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/shell/coquez/czstra3.F#L111)에서 strain-rate 기반 증분을 누적한다. [PROP/SHELL 공식 문서](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/prop_type1_shell_starter_r.htm)는 small-strain flag 1/3이면 engineering, 그 외에는 true strain/stress로 설명한다. Run224의 실제 flag는 2다.

**하지만 이 근거만으로 이 출력 tensor를 모든 유한변형 경로에서 정확한 Hencky/logarithmic, Almansi 또는 Green–Lagrange tensor라고 특정하지 않는다.** 좌표계 회전과 증분 누적의 상세 경로까지 수학적으로 검증하지 못했다. 임시 명칭은 `solver_total_strain_inplane_local`, measure 상태는 `solver_true_strain_increment_accumulation; exact_finite_measure_unverified`가 적절하다. FEM 출력과 동일 정의의 AI 라벨로 비교하는 것은 가능하지만, 변위 gradient에서 직접 계산한 Green strain과 즉시 동일시하면 안 된다.

## 6. von Mises 파생 라벨과 solver 출력 분리

현재 Engine에는 `/ANIM/SHELL/VONM`가 없다. 현재 저장한 in-plane 응력으로 다음 파생 scalar를 계산할 수 있다.

```text
vm_from_inplane_tensor = sqrt(σxx² - σxx·σyy + σyy² + 3·τxy²)
```

이는 관측된 2D tensor에 plane-stress 가정을 적용한 **독립 파생 라벨**이다. 이름을 `von_mises_derived_plane_stress_<location>`로 정하고 입력 응력 배열·단위·위치를 기록한다. 이를 원 solver에서 직접 읽은 VONM이라고 적지 않는다.

[공식 FAQ](https://2022.help.altair.com/2022.3/hwsolvers/rad/topics/solvers/rad/faq_rad_post_processing_r.htm)는 /ANIM의 기본 VONM이 두께 적분점 응력을 적분가중치로 평균한 응력에 적용된 값이라고 설명한다. 그러므로 layer별 von Mises의 평균, layer별 최대값, 평균 tensor에서 계산한 von Mises는 일반적으로 다른 목표다. 원 solver VONM를 새로 요청하면 `solver_vonm`라는 별도 필드로 검증하고 두께 위치·평균 정의를 함께 비교한다.

## 7. 권장 라벨 보존·학습 설계 — 제안

1. 기본 저장을 cell/element 라벨로 유지한다. `run_id, time_ms, element_id, part_id, element_type, location, frame, unit, measure, tensor_shear_convention`을 동반한다. 물리 의미가 없는 rigid/spring 패딩 cell은 shell 학습 mask에서 제외한다.
2. 응력은 local in-plane 세 성분, strain은 검증된 local in-plane 세 성분, EPSP는 scalar로 분리한다. Upper/Lower/1…5를 의미 없이 하나의 feature축으로 합치지 않는다.
3. node-only 모델에 붙이려면 element/dual graph 또는 node–element 연결 모델을 먼저 비교한다. cell→node 평균은 선택한 보간 연산이며 새 물리 측정이 아니다. 평균 방법(단순/면적가중), material·part 경계 분리, 법선·두께 위치를 고정한다.
4. 경계·용접·접촉 근처의 응력 불연속과 극값은 node 평균에서 희석될 수 있으므로 원 cell 오차도 반드시 평가한다. 학습/검증 partition을 나눈 뒤에 전처리 통계·스케일을 적합한다.
5. EPSP 오차 외에 plastic-zone 영역, peak 값·위치, 두께별 응력과 파생 von Mises를 병렬 평가한다. EPSP 크기만으로 실제 제품 파손을 단정하지 않는다.

출력 추가 실행이 끝나면 먼저 비영 strain, NaN/Inf, 시간정렬, cell ID 정렬, tensor 대칭, 패딩 성분, shear 1/2 관례, 기존 응답 보존을 확인해야 한다. 이 문서의 확인 상태를 넘어 실제 검증이 이뤄지면 해당 실행 기록을 별도로 연결한다.

