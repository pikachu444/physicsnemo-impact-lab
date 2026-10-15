# 원 소스 조사와 후속 실제 실행의 교차표

추가일: **2026-10-15**. 이 문서는 당시 조사자의 소스 검토를 후속 실험 담당자의 실제 실행으로 바꾸지 않고, 각각의 증거를 연결한다. 원 조사 본문과 JSON 행렬·99파일 blob 목록은 당시 판단으로 보존한다. [원 README 스냅샷](README.source-investigation-2026-10-15.md)의 SHA256은 `d188b9cba2b582a7c7a93c0bb4ae760f7197869c349fffa74058051aee2b847f`다.

아래 “완료”는 해당 행의 범위에만 적용한다. 데이터 추출·CUDA 역전파·학습·정답 비교·유용 정확도·제품 검증은 서로 다른 단계다. 현행20개 설정 조합과 archived Sym3개를 모두 학습했다는 뜻이 아니다. 이 addendum 자체는 문서·JSON·파일 hash만 읽었고 ML import·추론·GPU 실행을 하지 않았다.

## 환경을 실행별로 구분

| 기록 | 확인된 범위 | 현재 해석 |
|---|---|---|
| 원 조사 README의 PhysicsNeMo2.2.2 / Torch2.13.0+cu126 | 당시 기존 가상환경·metadata를 기술한 역사 기록 | 뒤에 새로 만든 GPU 환경의 버전으로 대체하거나 모든 실험에 소급 적용하지 않는다. |
| 후속 native GPU 학습 환경 | Windows, RTX3060 Laptop6GiB, PyTorch **2.10.0+cu126**, PyG2.8.1, 공식 torch-scatter2.1.2+pt210cu126, 설치 PhysicsNeMo2.2.2 | 별도 전용환경이다. [설치 manifest](../benchmarks/bumper/environment/installation-manifest.json), [실행 환경 안내](../benchmarks/bumper/environment/README.md), [판 runtime 기록](../benchmarks/deforming_plate/README.md). 원2.13 환경은 보존했다. |
| Q1 CPU 수치 탄성 | 기본 Python3.13.5, **Torch2.13.0+cpu**, float64, CPU1 thread, CUDA미초기화 | GPU 환경과 다르다. 설치 PhysicsNeMo 패키지가 아니라 pin의 원 `elasticity.py`를 직접 load했다. [실제 실행 JSON](../benchmarks/elasticity_numerical/cpu-execution/execution.json). |
| 공식 예제 소스 | pin `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`, 소스 버전2.3.0a0 | 설치 배포본2.2.2와 같다고 표시하지 않는다. 생성자/API 호환 변경과 실제 source hash는 각 실행에서 확인한다. |

이 GPU 결과는 사용자 PC에서 실행한 결과다. 별도 Codex Cloud 검사에서 GPU가 노출되지 않았던 사실과 구분하며 Cloud GPU 실행·유료 자원 사용으로 표시하지 않는다.

## 실제 실행과 정답 확보

| 원 조사 항목 | 2026-10-15 후속 증거 | 완료 범위와 남은 한계 |
|---|---|---|
| 범퍼 RAW에 응력/EPSP 출력 경로 존재, 당시 아직 추출 전 | [131조건 실제 구축](../fem-labels/README.md), [최종 구축 요약](../fem-labels/dataset-build-summary.json) | 1,441 ANIM을 확보해13,675절점·11시점·C7 정답을 추출하고 writer exit0/전수 hash/READY를 동결했다. 학습112/검증12/평가7. C7은 변위XYZ(mm), 상·하층 면내 VM(GPa), 상·하층 EPSP다. 총변형률131조건을 추출했다는 뜻은 아니다. |
| 범퍼 총변형률은 원 RAW에 없는 별도 출력 | [Run224 실제 추가 FEM 및 대조](../fem-labels/total-strain-validation.json), [정의·단위·층 설명](../fem-labels/README.md) | 같은 Run224에 총변형률 카드를 추가해 OpenCourant를 실제 실행했다. 변위·응력·EPSP의 기존 같은 solver 결과 대비 차이0. **총변형률1조건**이며 조건 일반화 데이터131개나 AI학습 완료로 세지 않는다. |
| 공개 원폰 deck 확보, 완성 DOE 미확보 | [실제 FEM·학습 자료 설명](../benchmarks/drop_test/PHONE-FEM-AND-DATASET.md), [원9조건 분할](../benchmarks/drop_test/doe/frozen-case-manifest.json), [공통 provenance 정정](../benchmarks/drop_test/common-cache-provenance-clarification.json) | 공개 RAD를 바탕으로 **신규 OpenCourant9조건**을 생성했다. 기존 공개 완성 DOE를 받은 것이 아니다. 고정 형상, 공통 E배율/Rx 두 변수,5/2/2 분할. 전체17부품의 선형화 corner76,065절점과 후면 part2 19,933절점을 구분하며 원257,099개 중 midside181,034개는 제외한다. |
| 원폰 직접 응력 출력, strain 추출 별도 필요 | [실제 canonical C6](../benchmarks/drop_test/doe/canonical/manifest-full-assembly-corners.json), [물리장 정의 메모](../benchmarks/rom/phone9-metadata-clarification.json) | 실제 C6는 변위XYZ(mm), native6성분 nodal GPS에서 **계산한** VM(MPa), validIP의 global total-strain tensor를 초기 corner체적으로 절점 투영한 뒤 구한 Frobenius, 별도 EPSP다. 실제 저장12시점→명목2..20µs 10시점의 선형 보간이며 외삽하지 않았다. 원 cell/IP peak와 nodal 투영값은 다르다. |
| 원폰 초기 짧은 실행만으로 전체 낙하 판단 불가 | [100µs 후속 FEM 요약](../benchmarks/drop_test/100us/summary.json), [실행 측정](../benchmarks/drop_test/100us/execution-metrics.json) | 기준1조건을100µs까지 실제 정상 완주했다. 마지막 저장 frame에서 native VM73.505MPa/EPSP0.0625744/소성solid240개다. 최대가 여전히 마지막이고 전체 COM 반발도 없어 **full-drop peak/rebound 완료가 아니다**. 20µs AI를100µs에 평가한 결과도 아니다. |
| 공개 구조 인장 coupon로 실제 stress/strain 검증 필요 | [13조건 FEM](../benchmarks/structural_coupon/README.md), [6설정 전체 결과](../benchmarks/structural_coupon/static-six-model-comparison.json) | 새 FEM13조건·물리절점919개·10시점·C7,5/4/4 분할에서 정적6설정의 실제 학습·별도 평가·새 프로세스 추론이 끝났다. Transolver/FLARE/GeoTransolver GALE/GALE_FA/MGN/FIG이며 **6설정·5 backbone 계열**이다. GALE_FA를 별도 독립 class로 세지 않는다. 고정 형상·두 변수·단일seed이며 수렴·가전제품 성능 인증이 아니다. |
| FEM과 비교하는 수치 축약·회귀 대안 | [ROM 전체 기록](../benchmarks/rom/README.md), [범퍼C7](../benchmarks/rom/bumper7/metrics.json), [전체 폰C6](../benchmarks/rom/phone9-full/metrics.json) | POD+RBF/POD+Ridge/직접RBF를 coupon·범퍼·phone full/rear에서 실제 fit·검증선택·시험·체크포인트 재추론했다. train-only 기저/정규화, val-only 설정선택을 지켰다. 변위 오차가 작아도 EPSP/음수/누적성 한계가 남는다. Phone rank4는 양수4mode 전부여서 POD+RBF≈directRBF이며 독립적인 우월성 두 건으로 세지 않는다. |
| Plate 원본 stress가 loader에 있고 strain은 없음 | [실제 두 학습·실패 기록](../benchmarks/deforming_plate/README.md), [첫 학습 지표](../benchmarks/deforming_plate/pilot8/metrics.json), [noise0 지표](../benchmarks/deforming_plate/noise0_pilot8/metrics.json) | 공식 HybridMeshGraphNet **축소 구조** 두 설정을 실제 학습하고399step AR로 비교했다. 유용 정확도는 둘 다 실패했다. 원 stress를 저장하고 reference기준 변위를 계산하도록 별도 driver를 사용했다. GL strain은 두 좌표장에서 유도한 비교이며 원 COMSOL strain 출력 검증이 아니다. noise0는 정규화/손실 가중치도 바뀐 민감도 실험이다. |
| Plate 공식128폭/15블록의 실행 자원 미확인 | [공식 크기 실제 자원 측정](../benchmarks/deforming_plate/official128-resource-summary.md), [probe](../benchmarks/deforming_plate/probe128_default/probe.json) | 실제 Adam5회·warm forward/backward·GPU메모리 측정을 완료했다. OOM 없이 맞았지만 **정확도 본학습은 미완료**다. 확대32train 자료와 새 final test2..7을 확보했어도, 확대 본학습이나 새 final 정확도를 실행했다고 쓰지 않는다. 축소모델 실패를 공식 전체 recipe의 실패로 확대하지 않는다. |
| TC/AR/TF의 stress 지원과 실제 시간 전략 구분 | [공식 범위·실제 결과](../benchmarks/temporal_strategies/source-notes.md), [3학습 요약](../benchmarks/temporal_strategies/execution-summary.json) | coupon에서 time-conditional C7·AR C3·one-step teacher-forcing C3을 각각30epoch 실제 학습·자유추론·재추론했다. **원 AR/TF는 위치3 전용으로 응력 head를 지원하지 않는다.** CPU/CUDA의 AR7/TF7 차원 실패도 보존했다. TC의 stress 오차는 평균장보다 나쁘고 AR/TF 자유rollout도 기준값보다 나빴다. 이전 FEM 이력을 주는 teacher 진단과 자유 미래 예측을 섞지 않는다. |
| TCAD elasticity는 구조 FNO가 아닌 수치 baseline | [실제 CPU 보고](../benchmarks/elasticity_numerical/execution-report.md), [독립 검산](../benchmarks/elasticity_numerical/execution-review.md) | 원 Q1 FEM/BiCGSTAB으로 두 문제를 계산해 독립 해석·힘/모멘트/에너지13검사를 통과했다. 두 solve0.591971초, import포함10.89775초. **AI/FNO 학습이나 전체2×2 DDM 실행은 아니다.** 해당 폴더의 준비 README도 역사 기록으로 보존했다. |

## 아직 정확도 본학습 완료로 세지 않는 검사

| 검사 | 확인된 증거 | 정확한 단계 |
|---|---|---|
| 범퍼 전체13,675절점 C7, 정적6설정 B1 | [원본 지원점 자원 요약](../benchmarks/bumper/probes-c7/native-support-resource-summary.json) | 실제 CUDA forward/backward·Adam 갱신·시간·메모리의 기능/자원 검사다. coupon6개 완료 결과를 범퍼C7 신경망의 본학습·정확도로 옮기지 않는다. 범퍼의 완료 ROM은 별도 행이다. |
| Phone 전체76,065코너 절점 C6, 정적6설정 B1 | [원본 지원점 자원 요약](../benchmarks/drop_test/probes-full-c6/native-support-resource-summary.json), [입력 접근 범위 정정](../benchmarks/drop_test/probes-full-c6/data-access-clarification.json) | 실제 전체 지원점의 기능/자원 검사다. 모든 split hash 확인은 있었지만 val/test 응답을 학습 feature로 쓰거나 정확도를 평가한 것이 아니다. Phone ROM 완료와 phone 다물리량 신경망 정확도 본학습을 구분한다. |
| 범퍼/Phone native B2·B5 배치 | [정적 준비·검사 규칙](../benchmarks/batching/PREPARATION.md), 결과 예정 경로 `benchmarks/batching/allocated-slot-01/` | 교차표 작성 시점에 배치 실측이 별도 진행 중이며 최종 동결 완료표를 수령하지 않았다. partial event/준비 문서는 배치 성공이나 정확도 학습 완료 증거로 세지 않는다. 완료 후 native batch/B1 동등성·gradient·시간·메모리·OOM 결과와 source hash를 별도 확인해야 한다. 처리량이 좋아도 동등 정확도·수렴을 보장하지 않는다. |
| Sinter VFGN·BIW·archived Sym | [원 행렬](official-execution-matrix.json), [소결·Sym 조사](sinter-sym-notes.md) | 이 addendum에 새 데이터 기반 실제 학습·정확도 완료 증거를 추가하지 않았다. 조사에서 확인한 recipe 존재나 구조 탄성 일반 가능성으로 실행 칸을 채우지 않는다. |

## 보존·재현과 이 교차표의 확인 범위

원 공식 소스는 pin·Git blob 근거로, 실행 driver는 실행별 source snapshot/hash로 확인한다. 다물리량 추가 head·단위·시간·초기속도·투영·정규화를 맞춘 연구용 adapter를 **수정 없는 공식 전체 recipe**라고 표현하지 않는다. 특히 원 소스의 phone8globals와 이번9조건의 E/Rx2globals, 원 AR scalar초기속도 기본값과 실제 구조 초기조건을 혼용하지 않는다.

[실행 코드 복원 묶음](../reproduction/README.md)은 소스·설정·실행 버전 profile을 복원하고 덮어쓰기 거부를 검사한 증거다. 새 환경 설치→자료 재획득/새 FEM→모든 학습까지 처음부터 재실행한 완료 증거는 아니다. 가중치·대형 FEM 배열·바이너리를 공개한 배포본으로도 취급하지 않는다.

[교차참조 hash 목록](executed-evidence-crosswalk-2026-10-15.json)은 이 문서 작성 때 읽은 주요 고정 증거의 bytes/SHA256을 기록한다. 진행 중인 배치 파일은 완료 hash 목록에 넣지 않는다. 이 문서화 작업에서는 표준 파일·JSON·hash 검사만 수행했으며 모델 재학습·FEM 재실행·새 GPU 실행은 하지 않았다. 이후 생성되는 결과가 이 표의 당시 상태를 바꾸면 새 날짜/후속 기록으로 연결하고 원 증거를 덮어쓰지 않는다.
