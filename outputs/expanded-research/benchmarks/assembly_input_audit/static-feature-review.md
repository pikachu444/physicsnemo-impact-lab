# 휴대폰 조립체 입력 채널·연결 구조 정적 감사

이 문서는 기존 소스·JSON을 읽어 입력의 의미를 확인한 결과다. Python, PhysicsNeMo, Torch, GPU를 실행하거나 기존 FEM 라벨·캐시를 수정하지 않았다. 배열의 중복 좌표·동일 입력에 대한 상이 출력 여부는 별도 수치 감사 대상이다.

말미의 추가 검토는 이후 생성된 수치 감사 JSON/CSV를 대조한 것이다. 배열 계산을 다시 실행한 결과와 구분한다.

**확인 결과:** `static_feature_full.npy`는 재료·부품·질량·두께 정보가 아니라 **기준 corner-cell 체적의 입사 셀 평균에 세제곱근을 취한 단일 길이 채널**이다. 공통 좌표 3개와 이 길이 1개, 조건별 전역 변수 2개가 현재 캐시의 명시적인 입력 정보다. NODE_ID와 연결 성분 번호는 별도 보존 자료이며, feature에 자동 포함되지 않는다.

## 1. 캐시 채널의 정확한 의미

| 항목 | 현재 정의·역할 | 생성 소스 근거 |
|---|---|---|
| `coordinates_full.npy` | 원본 RAD `/NODE`의 물리 좌표를 ANIM의 원 NODE_ID로 정합한 `(76065,3)` float32 좌표, mm. 변형 후 좌표나 좌표 순번이 아니다. float64 원본도 별도 보관한다. | [cache-phone-case.py:39–62](../drop_test/cache-phone-case.py#L39), [114–119](../drop_test/cache-phone-case.py#L114) |
| `static_feature_full.npy` | `(76065,1)` float32, mm. 노드 i에 대해 `cbrt(sum(V_e for incident e) / count(incident e))`. 즉 평균 입사 셀 체적의 세제곱근이다. | [build-full-graph.py:43–50](../drop_test/build-full-graph.py#L43) |
| 기준 `V_e` | 원본 float64 corner 좌표로 만든 선형 체적. TETRA10은 4개 corner의 determinant/6, BRICK/wedge는 정해진 6개 tetra 분해 체적의 합. 원래 quadratic 요소의 적분 체적을 정확히 복원했다는 뜻은 아니다. | [cache-phone-case.py:64–85](../drop_test/cache-phone-case.py#L64), [134–136](../drop_test/cache-phone-case.py#L134) |
| 전역 변수 0: `e_scale_common` | 기존 5개 재료의 Young 계수에 똑같이 곱하는 무차원 공통 배율. 5개 재료별 독립 E 값 또는 material ID 채널이 아니다. | [generate-phone-doe.py:58–64](../drop_test/generate-phone-doe.py#L58), [90–110](../drop_test/generate-phone-doe.py#L90) |
| 전역 변수 1: `rwall_rx_deg` | 원래 rigid plane의 법선을 X축으로 회전하는 각도, degree. 물체 자체의 NODE 좌표나 벽 원점을 회전·이동하는 변수가 아니다. | [generate-phone-doe.py:64](../drop_test/generate-phone-doe.py#L64), [82–110](../drop_test/generate-phone-doe.py#L82) |
| `node_ids_full.npy` | 원본 FEM 노드 식별자. 좌표 정합·라벨 대응·출처 보존에 사용한다. `static_feature_full`에 붙이거나 모델의 수치 채널로 연결하는 코드는 없다. | [cache-phone-case.py:35–58](../drop_test/cache-phone-case.py#L35), [113–119](../drop_test/cache-phone-case.py#L113) |
| `connected_component_full.npy` | physical edge graph의 union-find 연결 성분 번호. 그래프 분석 결과로 별도 저장한다. 현재 phone manifest의 입력 필드에는 이를 지정하지 않는다. | [build-full-graph.py:31–51](../drop_test/build-full-graph.py#L31), [manifest:5–13](../drop_test/doe/canonical/manifest-full-assembly-corners.json#L5) |

따라서 이 정적 길이에는 node별 part/material ID, density, E, Poisson 비, plastic law, nodal mass, shell thickness, 벽까지의 signed distance, contact/tied 상대편 ID가 명시적으로 들어 있지 않다. 좌표·메시 크기·고정 배치와 상관관계가 있을 수는 있지만, 그 상관관계를 독립적인 물성 또는 접촉 입력으로 계산하면 안 된다. 생성 코드도 `is_shell_thickness: False`로 구분한다. [cache-phone-case.py:134](../drop_test/cache-phone-case.py#L134)

두 전역 변수는 모든 노드에 같은 조건 값이다. 원래 물성의 공간 배치, 초기 속도, 중력, mesh, 벽 원점은 이 9개 조건에서 고정되어 FEM 라벨에 반영된다. 이것들을 node별 입력으로 제공한 것은 아니다. 공식 generator의 5개 material ID와 E/벽 변경 기능은 [generate_dataset.py:45–51, 80–87](../../../../work/expanded-drop/official-generator/drop_test/generate_dataset.py#L45)에 있고, 이번 축소 DOE는 공통 E 배율과 X 회전만 사용한다. [generate-phone-doe.py:103–114](../drop_test/generate-phone-doe.py#L103)

## 2. 76,065 corner support와 17개 연결 성분

캐시는 ANIM solid connectivity에 참조되는 노드만 선택한다. `used == arange(76065)`를 검사하고, 원 NODE_ID의 유일성을 확인한다. [cache-phone-case.py:31–37](../drop_test/cache-phone-case.py#L31)

이 support는 원 RAD의 모든 TETRA10 corner와 BRICK 노드의 합집합이다. 원래 TETRA10 midside 중 이 support에 없는 181,034개 노드는 학습·예측 support에 포함되지 않는다. 이 차이는 요소 삭제나 임의 노드 샘플링이 아니다. [check-active-and-node-coverage.py:27–45](../drop_test/check-active-and-node-coverage.py#L27), [full-graph-contract.json:32–34](../drop_test/doe/canonical/shared/full-graph-contract.json#L32)

full graph 생성 코드는 다음 연결만 만든다.

1. TETRA10의 4개 corner 사이 6개 edge.
2. BRICK 8-slot connectivity의 표준 둘레·수직 12개 edge. wedge 퇴화 표현의 self-edge는 제거한다.
3. 중복 undirected edge를 제거한 뒤 양방향으로 저장한다.

근거: [build-full-graph.py:13–26](../drop_test/build-full-graph.py#L13). 저장 contract의 수치는 **280,592 undirected / 561,184 directed edges, 17 components, degree 3–23, 고립 노드 없음**이다. component는 위 edge에 대한 union-find로 구한다. 본 감사에서는 이 배열을 재계산하지 않았다. [build-full-graph.py:27–42](../drop_test/build-full-graph.py#L27), [full-graph-contract.json:2–28](../drop_test/doe/canonical/shared/full-graph-contract.json#L2)

원본에는 17개 part, 14개 `/INTER/TYPE2`와 `/RWALL/PLANE/1`이 있다. part와 material 개수는 [deck-inspection.json](../drop_test/deck-inspection.json)에 보존되며, 원본 TYPE2 첫·마지막 카드는 `work/expanded-drop/source/Cell_Phone_Drop_0000.rad:548772`, `:622564`, wall은 `:629252`이다.

**TYPE2/tied·wall cross-edge가 없는 이유는 캐시의 graph 정의 때문이다.** `build-full-graph.py`는 `solid_connectivity_8.npy`, node 수, corner 체적만 읽고, `/INTER/TYPE2` slave/master 관계나 wall 접촉 후보를 파싱·추가하지 않는다. 좌표 근접 검색으로 다른 part를 연결하는 단계도 없다. 그러므로 tied/contact를 FEM에서 계산했다는 사실이 학습 graph의 cross-component edge로 자동 변환되지 않는다. 생성 코드와 저장 contract가 이 한계를 명시한다. [build-full-graph.py:57–60](../drop_test/build-full-graph.py#L57), [full-graph-contract.json:30–32](../drop_test/doe/canonical/shared/full-graph-contract.json#L30)

17개 graph component는 FEM이 17개의 독립적인 물체처럼 계산되었다는 뜻이 아니다. solver는 기존 tied와 wall 물리를 포함한 해를 만들고, 그 결과가 target이다. 다만 현재 element-edge graph만으로는 component 사이에 message passing을 할 수 없다. global attention 계열이 전체 점을 혼합하는 것은 가능하지만, 그것은 명시적인 tied/contact 관계를 입력했다는 근거가 되지 않는다. component 수와 part 수가 같은 사실만으로 component↔part의 일대일 대응을 입증한 것도 아니다.

## 3. C6는 입력 6채널이 아니라 출력 물리량 6개

full target은 노드 × 시간 × field = `(76065,10,6)`이다. field 순서는 다음과 같다.

| 채널 | 의미 | 단위 |
|---|---|---|
| 0–2 | `u_x, u_y, u_z`: ANIM 현재 좌표 − 원 RAD 기준 좌표 | mm |
| 3 | `von_mises_GPS`: solver GPS nodal stress 6성분으로 계산한 von Mises | MPa |
| 4 | `total_strain_frobenius`: valid-IP 평균 → 기준 corner 체적가중 nodal global strain tensor → tensor Frobenius norm | 1 |
| 5 | `epsp`: cell Plastic_Strain을 같은 기준 corner 체적으로 nodal projection | 1 |

근거: [cache-phone-case.py:150–168](../drop_test/cache-phone-case.py#L150), [206–218](../drop_test/cache-phone-case.py#L206). total strain은 plastic strain과 다르며, VM은 cell VM을 node에 평균한 값과도 구분된다. TETRA10은 valid IP 111/112/121/211 동등평균, BRICK Isolid1은 IP111을 사용한다. [cache-phone-case.py:135–137](../drop_test/cache-phone-case.py#L135)

학습 target의 시간은 nominal 2,4,…,20 µs이고 t0는 제외한다. 실제 출력시간 양옆 frame에서 6개 물리 채널을 선형보간하며 시간외삽을 금지한다. 같은 값의 rear-cover subset도 함께 저장한다. [cache-phone-case.py:189–203](../drop_test/cache-phone-case.py#L189)

검토 말미에 생성된 [manifest-full-assembly-corners.json](../drop_test/doe/canonical/manifest-full-assembly-corners.json)은 위 좌표·graph·단일 static feature·두 global key와 C6 target을 연결한다. [dataset_READY_evidence.json](../drop_test/doe/canonical/dataset_READY_evidence.json)은 9개 case, train/val/test 5/2/2의 동결·hash 검증 결과를 보존한다. 본 감사는 해당 검증을 다시 실행한 것이 아니라 그 기록과 생성 코드를 읽었다. 동결 검사는 두 support shape, rear/full subset의 bitwise 일치, finite·비음수 물리 채널, target hash를 검사한다. [freeze-phone-cache.py:17–41](../drop_test/freeze-phone-cache.py#L17)

## 4. 모델에 전달되는 정보와 schema 경계

현재 공용 adapter의 정적 코드를 읽으면, 호환되는 cache loader가 연결되었을 때 node 정보는 **정규화 xyz 3 + 정규화 static scalar 1 + 정규화 global 2**로 해석된다. 구현 변수 이름 `thickness`는 이 scalar를 담는 API 이름이며, phone의 물리적 두께라는 뜻이 아니다.

- Transolver/FLARE: functional `[static, globals]`와 별도 xyz embedding.
- GeoTransolver/GeoFLARE: local `[xyz, static]`, 별도 xyz geometry와 global embedding.
- MeshGraphNet: node `[xyz, static, globals]`; edge는 정규화 xyz 차이 3성분과 그 거리 1개. 보존된 physical graph만 전달한다.
- FIGConvUNet: `[xyz, static, globals]`와 별도 xyz geometry.

근거: `work/expanded-ml/model_adapters.py:195–219`. 정규화는 `work/expanded-ml/run_benchmark.py:93–106`에 있으며, 같은 caller는 시간·C6를 한 번에 펼친 출력으로 만든다(`:109–111`). 이 검토에서는 모델을 import하거나 실행하지 않았다.

**현재 phone manifest와 공용 loader의 schema는 직접 호환되지 않는다.** phone은 top-level `static_feature`, `runs` list, row별 `split`을 사용한다([freeze-phone-cache.py:45–56](../drop_test/freeze-phone-cache.py#L45)). 반면 `work/expanded-ml/benchmark_data.py:44–78`은 top-level `split` dict, `runs` dict, run별 `thickness` 파일 키를 요구한다. 따라서 별도 변환/loader 연결 및 실행 증거 없이 이 캐시를 공용 trainer가 성공적으로 학습했다고 판단할 수 없다. cache READY와 모델 학습 성공은 별도 단계다.

NODE_ID·part/material/component를 명시적으로 주지 않는 point 모델에서, 서로 다른 부품의 점이 같은 xyz·static과 같은 global 조건을 갖는다면 구별 가능한 입력 정보가 부족할 수 있다. 실제로 그런 입력 쌍과 서로 다른 C6 target이 존재하는지는 별도 배열 감사 결과로 판단해야 한다. 이 문서는 그 충돌 수 또는 학습 오차를 추정하지 않는다.

## 검토한 주요 소스의 SHA256

| 소스 | SHA256 |
|---|---|
| `drop_test/cache-phone-case.py` | `289de22b51f32b0e2b5d8e4e98cf7c60e9ed32345da9e661f24ddd0fae81ca2e` |
| `drop_test/build-full-graph.py` | `f0322d36aff743a088422aeef4f5205c2acf3694a436357d2841b674ec746535` |
| `drop_test/freeze-phone-cache.py` | `87c3d132749d327fe446fc35986799a4ee0b8c3a5d2d993c74f815b08d7c0c48` |
| `drop_test/generate-phone-doe.py` | `ae480a953b226930f43fd8570d3e884b4dad148b9968444ac1e94cca2644e656` |
| `work/expanded-ml/benchmark_data.py` | `4888594826c9a517bb8fc876d2fa533daf7eb78d9f247c10af287821f70956c6` |
| `work/expanded-ml/model_adapters.py` | `fcfbfc72c4eb36793849410ba54e7e41c27a5261d7f0f3830800a774edd66e7d` |
| `work/expanded-ml/run_benchmark.py` | `6f11b318cc872b8e15401833db294f7bdfddc7a69ae78e0e8890dd20ece72eab` |

## 추가 검토: 좌표 쌍 수치 감사 기록의 해석

[assembly-input-audit.json](assembly-input-audit.json), [coordinate-pairs.csv](coordinate-pairs.csv), [native-pair-audit.json](native-pair-audit.json)을 JSON/CSV 읽기만으로 대조했다. 치명적인 count·채널 순서·단위 불일치는 확인하지 못했다. CSV는 중복 없는 unordered 쌍 334개이며, float64와 float32 exact 표시는 각각 132개다. 따라서 1e-6 mm 이내 334개는 exact 132개를 **포함**하고, exact를 제외한 near-only는 202개다.

exact 132개는 모두 서로 다른 NODE_ID·part·material 사이에 있다. 재료 조합은 material 8 PCB ↔ material 9 comps이며, **part ID 8 ↔ 9라는 뜻이 아니다.** 실제 part 조합은 27–28, 13–19, 13–17, 25–28, 23–28이다. 저장된 float32 xyz에 기존 static feature까지 붙인 exact 중복은 0개다. exact 좌표 쌍의 static feature 차이는 최소 0.10822022 mm, inclusive near 쌍은 최소 0.08647841 mm로 기록되어 있다. 그러므로 현재 관측은 **좌표만의 중복**이며, 검사한 저장 xyz+static 전체의 중복이라고 부를 수 없다. 실제 모델 정규화 후 tensor를 실행·검증한 결과도 아니다.

아래 값은 exact 132개 쌍의 **FEM 유래 출력 차이**다. 학습 모델의 예측 오차가 아니다.

| C6 | nominal 2..20 µs 보간 target의 최대 절대 차이 | 보간하지 않은 actual frame C6의 최대 절대 차이 | 단위 |
|---|---:|---:|---|
| u_x | 0 | 0 | mm |
| u_y | 1.89989805e-6 | 1.90734863e-6 | mm |
| u_z | 2.23517418e-8 | 1.19209290e-7 | mm |
| GPS nodal VM | 5.55290735e-5 | 5.55736633e-5 | MPa |
| total-strain Frobenius | 1.27637201e-8 | 1.27675222e-8 | dimensionless |
| EPSP | 0 | 0 | dimensionless |

native 기록의 9개 case는 각각 **12개 actual frame**이다. t0와 보존 terminal snapshot을 포함하며, nominal target의 시간 10개와 구분해야 한다. native 최대값은 20 µs를 약간 넘는 terminal frame도 포함하므로 위 두 열의 최대값 차이를 시간 보간만의 영향으로 분해할 수 없다. native라는 이름도 원시 tensor 성분을 직접 비교했다는 뜻은 아니다. 기존 GPS VM·strain tensor 평균/체적 projection/Frobenius 정의를 적용한 C6를 **시간 보간하기 전**에 비교한 것이다.

`pairs_nonzero_any_native_case_time_per_channel=[0,2,9,132,132,0]`는 9개 case와 actual frame 중 적어도 한 번 비영 차이를 가진 쌍 수다. 모든 frame에서 132개가 다르다거나, 차이가 공학적으로 유의함을 입증한 것은 아니다. EPSP 쌍 차이 0도 전체 휴대폰의 plastic strain이 0이라는 뜻이 아니다. 0개의 xyz+static 쌍에 대한 비어 있는 비교 집계의 0은 모델의 0 오차 검증이 아니다.

새 graph JSON은 component 17개 각각에 part 하나가 대응하는 `component_part_mapping`과 `cross_part_edges=0`을 기록한다. 이 추가 기록은 앞서 component 수만 읽었을 때보다 대응 근거가 늘어난 것이며, 본 추가 검토에서 graph 배열을 다시 계산한 것은 아니다.

이 결과만으로 모델의 표현 불가능성·성공·실패, 물성 채널의 불필요성 또는 새 조립체 일반화를 판단할 수 없다. 현재 static feature가 이 고정 메시의 중복 좌표 쌍을 구별한다는 관측과, 해당 feature가 part/material을 의미상 명시하지 않는다는 사실은 함께 성립한다.
