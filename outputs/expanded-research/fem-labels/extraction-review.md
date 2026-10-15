# Run224 ANIM 라벨 추출 독립 검토

검토일: 2026-10-15. 대상은 `extract-anim-fields.py`와 실제 `work/expanded-fem-labels/cache/Run224`이다. 공유 추출 코드와 기존 캐시는 변경하지 않았다. 단일 CPU thread 설정, NumPy/SciPy/VTK와 실제 배포 converter를 사용했으며 Torch·GPU·학습은 실행하지 않았다.

현재 Run224 캐시에서 수치 또는 대응 오류를 발견하지 못했다. 검증 범위는 원본 ANIM 11개 파일의 hash·캐시 전체의 배열 형태/유한성, A011 한 프레임의 독립 원본 변환·canonical 배열·투영 재계산이다. 131개 case 전체를 감사한 것은 아니다.

## 실제 재계산 결과

원본 `source-family/RAW_DATA/VALIDATION_DATA/Run224/Bumper_Beam_AP_meshedA011`을 converter의 표준출력으로 다시 변환하여 메모리에서 읽었다. 기존 VTK 파일을 정답으로 재사용하지 않았다. 원본 ANIM 11개의 SHA-256이 캐시 manifest의 source_files와 모두 일치했다.

| 항목 | 독립 확인 결과 |
|---|---|
| 대상 노드 | 원본 VTP의 thickness > 0인 13,675점. 저장 curated indices와 float32 좌표가 정확히 일치 |
| 실제 시간 | FIELD TIME = 100.001 ms. 명목 100 ms와 구분되어 기록됨 |
| 실제 shell | 13,626개. `(ELEMENT_ID, PART_ID)`와 실제 cell type 5/9로 식별. 요구 shell key 중복 0개 |
| 연결 | 13,626개 모두 실제 NODE_ID의 corner 집합이 RAD와 일치. 28개 degenerate triangle은 converter에 마지막 corner가 반복된 형태로 저장되어 있으며, unique corner로 투영할 때 반복 가중하지 않음 |
| canonical local stress | upper/lower/layer 1…5의 xx, yy, xy 모두 원본 A011 변환값과 정확히 일치: 최대 절대 차이 0 |
| canonical EPSP | 7개 위치 모두 원본 A011 변환값과 정확히 일치: 최대 절대 차이 0 |
| nodal VM/EPSP | 초기 RAD 면적과 unique corner별 area/corner-count 가중치로 독립 재계산. upper/lower 네 scalar 채널 모두 float32 저장값과 최대 절대 차이 0 |
| 변위 | A011 원본 변환값과 nodal cache의 변위 세 채널 최대 절대 차이 0 mm |
| 원본 ANIM 대 curated 변위 | A011 component RMSE = 0.0001348278583662763 mm. manifest와 일치 |
| 초기 shell 면적 | RAD로 다시 계산한 면적의 float32 값이 캐시와 정확히 일치 |
| 캐시 유한성 | nodal targets, canonical stress, canonical EPSP 전체에서 NaN/Inf 없음. EPSP 전체 범위 0…0.6488549709 |

`nodal_targets`의 의미는 `[13,675 nodes, 11 saved times, 7 channels]`이며 변위 x/y/z, in-plane von Mises upper/lower, EPSP upper/lower다. canonical stress는 `[11,13,626,7,3]`, canonical EPSP는 `[11,13,626,7]`로 보존된다. 추출 코드의 투영은 local tensor 성분을 직접 평균하지 않는다. 위치별 in-plane invariant를 먼저 계산하고 scalar만 노드로 투영한다. 면적은 변형 후 면적이 아닌 초기 RAD 면적이다.

## 물리 의미와 한계

입력 kg/mm/ms에서 변위 mm, 응력 GPa, 시간 ms가 일관된다. EPSP는 무차원 등가소성변형률 scalar이며 total strain tensor가 아니다. 원본 캐시에 total strain이 없다는 manifest 표기도 타당하다. 보존한 7개 위치는 upper/lower와 layer 1…5라는 출력 슬롯이며, 독립적인 두께 적분점이 7개라는 뜻으로 해석하면 안 된다.

VM 채널은 `sqrt(xx² - xx·yy + yy² + 3xy²)`로 계산한 in-plane 파생 invariant다. 직접 출력된 solver VONM과 동일하다고 주장하지 않는다. local shell tensor는 global 응력 세 성분이 아니며 converter의 3×3 zero padding도 완전한 3D stress/strain 관측을 뜻하지 않는다. scalar 노드 평균은 공유 part 경계를 가로질러 응력 불연속·hotspot을 완화할 수 있다. 현재 manifest가 이 한계를 밝히고 canonical cell 값을 별도로 보존하는 방식은 적절하다.

## 추후 재사용 시 보강할 사항

현재 Run224 결과를 무효화하는 발견은 없지만, 범용 추출기로 재사용하기 전에 다음 provenance 보강이 필요하다.

1. `dataset_revision`이 고정 문자열이고 `provenance`가 `--include-total-strain` 유무로 결정된다. 새 FEM ANIM에 strain flag를 끄거나 다른 revision의 입력을 주면 출처 명칭이 잘못 기록될 수 있다. 원본 source manifest 검증 또는 명시적인 provenance 입력으로 결정해야 한다. 이번 원본 Run224의 source hash는 독립적으로 모두 확인했다.
2. converter 경로는 기록 가능한 실행 입력이지만 manifest에는 converter binary SHA-256이 없다. source→canonical 재현성을 보존하려면 converter hash와 추출 스크립트 hash를 함께 저장하는 것이 좋다. local tensor/shear/upper-lower 의미의 문헌 근거는 `label-semantics.md`에 있으며 binary와 해당 source revision의 빌드 대응까지 증명한 것은 아니다.
3. 현 연결 검사는 corner 집합만 비교한다. 다른 mesh나 converter를 허용할 때 local tensor frame의 방향을 검증하려면 반복 corner를 정규화한 ordered connectivity와 normal/orientation도 검사해야 한다. 현재 감사에서 canonical 수치가 원본 converter 값과 일치한다는 사실이 모든 향후 입력의 방향 의미를 보장하지는 않는다.

이 검토는 원본 FEM 라벨의 보존과 지정한 scalar 투영을 검증한다. 새 모델 학습 성공, 응력·소성 예측 정확도, 파손 위험 예측 또는 실제 제품의 실험적 타당성은 별도 검증 대상이다.

## 부록: direct ANIM backend 전환 후 추가 검토

같은 날 추가된 `direct-anim-reader.py`와 extractor의 direct 분기를 좁게 재검토했다. 위 본문은 최초 VTK backend로 생성된 Run224 캐시의 감사 기록으로 유지한다. 이 부록은 direct backend 전체 구현의 독립 인증이나 전체 case 재검증을 뜻하지 않는다.

direct 분기는 binary 2D shell 블록의 element/part ID, 연결, scalar/tensor 배열을 같은 인덱스 체계로 읽고, extractor가 `(ELEMENT_ID, PART_ID)`로 구한 `ei`를 모든 shell field에 공통 적용한다. 따라서 코드에서 field별 cell 순서가 뒤섞이는 경로는 발견하지 못했다. 원 ANIM tensor의 local `[xx,yy,xy]`가 그대로 보존되고 scalar invariant 투영은 기존 연산을 재사용한다. binary float32를 유지하므로 VTK의 6자리 ASCII 변환 캐시와 마지막 자리까지 같아야 한다고 요구하면 안 된다.

Run100 A011을 direct reader로 독립 재독해하여 다음을 확인했다.

- 저장 backend는 direct이며 실제 시간 `100.00042724609375 ms`가 캐시의 실제 시간과 정확히 일치한다.
- 13,626개 real shell을 element/part key로 재정렬한 canonical stress 7개 위치, EPSP 7개 위치와 nodal 변위가 direct 원본 배열과 모두 최대 절대 차이 0이다.
- 반복 corner를 양쪽에서 제거한 ordered connectivity가 13,626개 모두 RAD와 일치한다. unique corner 3개인 shell 28개, 4개인 shell 13,598개다. 따라서 이번 실제 Run100 입력에서 local corner 순서 불일치는 발견하지 못했다.

### 조치가 필요한 상태 정보

검토 당시 extractor는 direct reader의 `shell_active`를 캐시에 보존하거나 nodal 투영 정책에 반영하지 않았다. 이는 향후 stress/plastic label의 의미를 위해 보강할 항목이다. 실제 Run100 A001에서 real shell 13,626개는 전부 True지만 A011에서 element 8376 / part 20이 False로 바뀐다. 해당 shell의 NODE_ID는 `[8903,7212,7211,8902]`, upper stress는 `[0,0,0]`, upper EPSP는 `0.3670252562`다. 따라서 dummy padding shell로 무시할 사례가 아니다.

원본 상태 bit를 `[time,element]` canonical 배열로 함께 보존하고, 원본 EROSION_STATUS와의 의미를 명확히 해야 한다. 비활성 shell의 EPSP 이력 보존 자체가 잘못됐다는 뜻은 아니다. 현재 초기 면적의 전체 real-shell 평균에 비활성 shell도 포함되는 연산을 유지하려면 이를 명시해야 하며, active-only 투영을 도입하면 기존 정의와 별도 채널/실험으로 구분해야 한다. 상태 변화만으로 실제 제품 파손을 입증했다고 해석해서도 안 된다.

### 남은 구현·출처 보강

- direct branch가 `cell_type=9`를 일괄 대입하므로 이를 실제 VTK cell type 검증과 동일하게 설명하면 부정확하다. direct 2D 블록에서 unique corner 수 3/4를 검증하고 triangle/quad를 유도·기록하는 방식이 적절하다.
- 현재 extractor의 연결 assertion은 여전히 corner 집합 비교다. 독립 Run100 검사에서 순서는 일치했으므로 현 수치 오류는 없지만, 반복 corner를 정규화한 순서 assertion을 코드에 추가하면 새 입력의 local frame 순서 변경을 잡을 수 있다.
- source revision/provenance를 strain 출력 flag로 결정하는 이전 권고는 그대로 유효하다. direct backend에서는 converter binary 대신 direct reader SHA-256, extractor SHA-256 및 공식 format source commit을 manifest에 보존해야 한다. VTK backend를 선택한 경우에는 converter binary hash를 보존한다.
- direct/VTK 속도 비교와 peak memory는 이 감사에서 독립 재측정하지 않았다. backend 정확도 확인과 성능 측정의 증거 범위를 구분해야 한다.

## 부록: 상태·출처 보강 후 최종 확인

보강된 extractor와 현 Run100 캐시를 다시 읽어 이전 지적의 실제 해결 여부를 확인했다. 이전 부록은 발견 당시의 상태 기록이며, 다음 확인 결과가 최신 상태다.

- `shell_animation_status.npy`가 실제 uint8 `[11,13626]`로 저장된다. Run100 A001의 0상태 shell 수는 0개, A011은 1개이고 element 8376 / part 20의 상태가 1→0으로 보존됐다. A011 상태 배열 전체가 원본 ANIM direct 독해·element/part 재정렬값과 정확히 일치하며, manifest의 시점별 zero count도 배열과 일치한다.
- manifest는 고정 초기 topology의 전체 real shell을 투영에 포함한다는 정책과 status 0의 EPSP 이력 해석 한계를 명시한다. 상태를 버리는 이전 문제는 해결됐다. 이를 active-only 투영 또는 실제 damage 검증으로 다시 명명하지 않는다.
- direct branch는 unique corner 수 3/4로 cell type을 유도하고, real shell에 대해 지원 type과 반복 corner를 제거한 **ordered connectivity**를 assertion으로 확인한다. 이전의 일괄 quad 표현 및 집합 검사만의 한계가 보완됐다.
- provenance를 별도 CLI로 받고 total-strain 출력에서는 명시적인 새 실행 provenance를 요구한다. extractor hash, direct reader hash, 공식 reader source commit 또는 선택한 VTK converter hash를 보존한다. 현 Run100 manifest의 extractor·direct reader hash가 현재 파일과 실제로 일치했다.
- 배열은 별도 staging에서 완성·flush/close한 뒤 파일별 replace하고 complete manifest를 마지막에 commit한다. 이는 다중 파일 전체가 한 번에 바뀌는 transaction은 아니므로, README처럼 전체 READY/hash 동결 전에 학습을 시작하지 않는 조건이 필요하다.

동결된 새 total-strain Run224 manifest의 SHA-256은 실제로 `12646c2e5c886c9caf38a4775f4b35028272ad3dd866538d060af3a28039d6ae`와 일치했다. 이 추가 검토에서는 이미 별도 scientific gate가 수행한 17개 NPY 전체 해시와 11프레임 재계산을 중복 수행하지 않았다. 원본 131조건 전체 완료 여부는 여전히 writer exit 및 최종 READY 판정의 범위이며 이 보고서로 완료를 대체하지 않는다.

### 공개 로그·재현 문서의 좁은 검토

`solver-log-provenance.json`에 기록된 원본·공개 로그 6쌍의 SHA-256을 모두 실제 파일과 대조했다. 공개 파일이 `preserve-solver-logs.py`의 지정된 경로·hostname 가림 연산과 Windows 개행 변환으로 생성된 결과임을 확인했고, 가림 후 비공백 행은 원본과 모두 일치했다. 숫자 내용을 변경한 증거는 없다. 현재 Windows `write_text` 처리는 CRLF 원본에 CRCRLF를 추가하므로 빈 줄 표시가 늘어날 수 있지만 수치 내용에는 영향이 없다.

공개 폴더의 Markdown/JSON/OUT/LOG에서 사용자 절대경로와 실제 hostname을 발견하지 못했다. EXE/DLL/ZIP·원본 RAD/ANIM/VTK/NPY는 이 폴더에 없었다. 검토 시점에는 Python `.pyc` 3개가 `__pycache__`에 남아 있어 최종 공개 정리가 필요하다.

README는 실제 FEM 총변형률, 원본 family 응력/EPSP, 파생 principal strain, actual/nominal 시간과 제한을 구분한다. 확인한 재현 명령과 CLI 사이의 치명적 불일치는 없다. 새 환경에서 바로 따라 하기 쉽게 하려면 필요한 Python 패키지에 다운로드 스크립트가 사용하는 `requests`도 적고, 첫 family 추출 전에 baseline RAD를 받는 `download-bumper.ps1` 실행을 명시하는 것이 좋다. Solver 다운로드·deck 확보가 본문에서 전제로 언급되므로 이 두 항목은 문서 편의 보완이다.
