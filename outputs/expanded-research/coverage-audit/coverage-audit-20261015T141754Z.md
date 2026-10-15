# 구조 실험 실행 범위 — 새 완료 snapshot

기준 시각은 **2026-10-15 14:13 UTC (23:13 KST)**다. 후속 실행은 이 snapshot에 소급 합산하지 않는다. 이번 작업은 작은 JSON·Markdown 읽기와 표준 라이브러리 산술·SHA256 검사뿐이며, 새 학습·추론·test·FEM·배열 열람·배열 hash·ML import·하드웨어/프로세스 점검은 하지 않았다.

완료된 expanded 신경망 학습 artifact는 **24개**다. **24개 알고리즘**이라는 뜻은 아니다. 기존 [오전 snapshot](coverage-audit-2026-10-15.md)은 당시 부분 완료 상태의 역사로 보존했다.

| 완료 집계 | 수 | 해석 |
|---|---:|---|
| DP600 인장 시편 | 9 | 정적6설정 + TC/AR/TF 시간전략3 |
| 휴대폰 full C6 정적 | 6 | 76,065 corner-node support; 원257,099절점 전체가 아님 |
| 범퍼 C7 정적 | 6 | upper/lower VM·EPSP; total strain 출력과 구분 |
| DeformingPlate | 3 | 축소2 + 공식 크기128/15의 frozen32×8×5 실험1 |
| 합계 | **24** | reload·평가 사례·자원/기능 probe·감사를 중복 합산하지 않음 |
| 최종 ROM predictor | **12, 별도** | 3방법×4support; train POD basis 공유4그룹 |
| 과거 curated3 predictor | 3, 별도 | 동일 범퍼의 변위-only 개발 단계 |

정적 비교의 여섯 설정은 Transolver, FLARE, GeoTransolver GALE, GALE_FA, MeshGraphNet, FIGConvUNet이다. GALE_FA는 GeoTransolver 변형이어서 **5개 backbone 계열**로 센다. Plate의 HybridMeshGraphNet은 별도 구현 class지만 이 명시적 계열 분류에서는 MeshGraphNet에 포함한다. 시간전략, seed, support 변경을 새 알고리즘으로 세지 않는다. ROM의 phone rear는 같은9조건의 부분 support이며 독립 물리 사례군이 아니다.

고정 source pin의 **선정 recipe/configuration/numerical inventory는20행(기성 neural8 + 추가 설정11 + 수치1)**이고 archived Sym3행은 별도다. 이는 선정한 조사 범위이며 모든 PhysicsNeMo 구조 알고리즘·연구·학습 조합의 전체 목록이 아니다. 24artifact와20recipe는 분모가 달라 완료율로 나눌 수 없다. [선정 행렬](../source-matrix/official-execution-matrix.json), [상세 artifact ledger](coverage-audit-20261015T141754Z.json)

| 범위 | 실제 완료 | 감사와 한계 |
|---|---|---|
| Coupon 정적6 + 시간전략3 | 학습·FEM test·기존 fresh reload 완료 | 단일 seed·작은 고정 형상/조건축; AR/TF는 C3, TC는 C7. 기존 C7 AR/TF wrapper 차원 실패 보존 |
| Phone6 + Bumper6 | 정적12 학습·고정 checkpoint 별도 process test 완료 | saved-prediction 수치 감사12/12 및 예산 게이트12/12 통과. 수렴·공학적 정확도·일반 우열 인증이 아님 |
| Plate 축소2 | 두 fit 완료 | 현재 유용정확도 실패; 기존 pilot reload 범위와 noise0의 별도 reload 근거 차이 보존 |
| Plate 공식 크기1 | **1280updates, selected epoch5, fresh6 각각 AR399 한 번** | 실행 완료, **정확도·형상 실패**. pre-access 실패 evaluator와 corrected evaluator는 추가 fit이 아님 |
| Phone/Bumper native temporal6 | **full fit0** | 한 차례 Phone TC cold 자원시도 부분 진행 뒤 host floor 중단. formal3update 성공0, 실제 update 수는 **unknown/null** |
| 새 clean functional6 | 계획6, **실제 완료0** at14:13 | 준비 protocol·패키지 build가 기능 실행·학습·정확도 성공을 대신하지 않음 |
| Q1 component + full DDM | Q1 두 문제/독립13검사, 후속 DDM64반복/독립25검사 통과 | **수치 FEM 기준선, AI fit0**. 이전11반복 guard중단과 완료 attempt02를 함께 보존 |

[정적12 완료 검증](../full-support-audit-final-verification.json), [기존 ROM](../benchmarks/rom/README.md), [시간전략 cold 결과](../benchmarks/temporal_strategies/native-scalar-v3-cold-calibration-v1-result.json), [clean 계획](../reproduction-clean-runtime/newfunctional-protocol-v1.json), [완료 DDM02](../benchmarks/elasticity_numerical/full-ddm/attempt02/README.md)

Plate 공식 크기의 fresh6 pooled relative L2는 변위 **2.4121746817 (241.217%)**, source nodal scalar VM **0.6823590692 (68.236%)**다. Stress 단위는 미확정이며 MPa로 변환하지 않았다. 좌표에서 유도한 Green–Lagrange strain L2는 **1293.4536325**로, 여섯 사례 각각21 sampled frame의 dimensionless 파생량이다. 원 COMSOL strain 검증이 아니다. Reference inversion은0, prediction inversion은 **4/6사례**다.

Plate metadata **68검사/실패0**과 source/saved-field independent numeric **297검사/mismatch0**는 대응·무결성 감사 통과다. 이 통과가 모델 정확도 실패를 뒤집지 않는다. [Metadata](../benchmarks/deforming_plate/review/official32-final-metadata-audit.json), [saved-field numerical audit](../benchmarks/deforming_plate/review/official32-streaming-numeric-audit-v1.json), [최종 closure](../benchmarks/deforming_plate/review/official32-final-closure-audit.json)

신규 native TC/AR/TF6의 자원 절차12개 중 실제 호출은 Phone TC의 중단된1개이며, 나머지 **11개 호출0**이다. 성공한 formal functional probe0, distinct-case stream profile0, fullfit0, validation/heldout0이다. 완료 receipt가 없으므로 update 수0 또는 array access0이라고 쓰지 않는다. V3 native scalar response head는 연구 확장이며 수정 없는 공식 recipe 재현이 아니다. 기존 Coupon 시간전략3과 중복 합산하지 않는다.

휴대폰·범퍼 외에 DP600 시편, Plate, Q1/DDM과 과거 BOX1BL MiniQA 실행이 있다. TopoDiff·Carbon LatticeGraphNet·UVA MFGNN_PDE·small_boule·TYPE25 hexa8 등의 추가 조사는 소스/공개성 조사이며 이번 완료 fit이 아니다. Sinter/VFGN, BIW, archived Sym, 구조 FNO의 해당-data neural 학습까지 완료했다고 확대할 수 없다. [확장 구조 조사](../broader-structural-examples.md)

이 snapshot은 완료 evidence와 기존 독립감사를 인용한 **metadata coverage 감사**다. 새 raw-field 수치 재계산을 수행한 것으로 쓰지 않는다. 읽은 작은 파일의 byte 수·SHA256·읽은 시각,24artifact ledger,12ROM ledger,20선정 row와 범위 제한은 [JSON](coverage-audit-20261015T141754Z.json)에 있다. 원 snapshot·보고서·source·Git은 수정하지 않았다.
