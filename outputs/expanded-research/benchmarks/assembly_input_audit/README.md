# 휴대폰 조립체의 좌표·입력 중복 읽기 전용 감사

**같은 초기 좌표를 가진 서로 다른 부품·재료의 노드가 실제로 존재한다. 그러나 현재 좌표와 static feature를 합친 저장 입력까지 동일한 쌍은 발견하지 못했다.** 따라서 좌표 중복만으로 현재 모델의 예측 오차나 학습 불가능성을 단정할 수 없다.

기존 완료 FEM·캐시는 변경하지 않았다. 9조건의 동결 C6 캐시와 원 RAD, 보관된 part 경계를 읽었고 CPU 1 thread만 사용했다. NumPy/SciPy/psutil 외 ML framework·GPU는 사용하지 않았다. 주 감사 1.43초·관측 최대 working set 99,090,432 B, 보간 전 프레임 추가 검사 0.46초·35,864,576 B였으며 두 프로세스 모두 종료되어 메모리를 해제했다.

## 확인된 좌표·부품 사실

대상은 원 RAD의 257,099 노드 중 전체 solid corner **76,065개**다. 원정밀도 float64와 실제 모델 좌표 float32를 따로 검사했다. 모든 노드의 NODE_ID는 고유하고 part 배정도 중복 없이 17개 part 중 하나로 정해졌다.

| 검사 | 중복 그룹 | 서로 다른 NODE_ID의 unordered 쌍 | 다른 part | 다른 material |
|---|---:|---:|---:|---:|
| 원 float64 좌표 bitwise 동일 | 132 | 132 | 132 | 132 |
| 모델 float32 좌표 bitwise 동일 | 132 | 132 | 132 | 132 |
| 원 float64 좌표의 Euclidean 거리 ≤ 1e−6 mm | — | 334 | 334 | 334 |
| 모델 float32 좌표 + static feature bitwise 동일 | 0 | **0** | 0 | 0 |

정확 중복 그룹은 모두 2노드다. 1e−6 mm 이내 334쌍에는 정확 중복 132쌍이 포함된다. **근접 좌표는 동일한 입력 비트값과 구분**했다. 동일 조건·동일 시각에서는 두 노드에 적용되는 전역조건이 같으므로 좌표+전역조건만의 중복은 좌표 중복과 같다.

정확 중복은 top_pcb–top_micro_chip_01 8쌍, top_pcb–top_micro_chip_02 8쌍, bottom_micro_chip_01–bottom_pcb 11쌍, bottom_micro_chip_02–bottom_pcb 11쌍, bottom_micro_chip_03–bottom_pcb 94쌍이다. 모두 material 8(`pcb_elastic`)과 material 9(`comps_elastic`) 사이에 있다. 좌표가 같은 두 노드의 static feature 차이는 최소 0.1082202 mm, 중앙값 0.2205637 mm, 최대 0.2491462 mm였다.

전체 334쌍의 원 nodeID·part·material·거리·static feature는 `coordinate-pairs.csv`, 모든 상세 결과·해시는 `assembly-input-audit.json`에 있다.

## 실제 FEM에서 나온 출력의 차이

C6는 변위 XYZ(mm), solver GPS nodal stress에서 계산한 von Mises(MPa), 체적 가중으로 절점 투영한 총변형률 텐서의 Frobenius norm, EPSP다. **총변형률 텐서 성분 전체나 원 셀 응력 성분의 비교가 아니다.** 마지막 두 라벨은 서로 다른 변형률 종류다.

주 검사는 실제 FEM 프레임을 물리 단위에서 선형 보간한 **명목 2,4,…,20 µs 학습 라벨**을 사용했다. 보간하지 않은 실제 출력 시각의 C6 raw-frame 캐시도 별도로 검사했다. 각 case의 원 프레임 12개는 t0와 20 µs를 약간 넘는 terminal snapshot까지 포함하므로, 두 표의 최대값 차이를 보간 효과만으로 단정하지 않는다. raw-frame 캐시 역시 기존의 GPS 불변량·체적 투영 정의를 거친 FEM 파생값이며 원 ANIM 텐서 자체와 구분한다. 출처 해시·프레임 실제 시각은 `native-pair-audit.json`에 보존했다.

아래는 정확 동일 좌표 132쌍에서 9조건·모든 해당 시각을 통틀어 관측한 **두 노드 사이 최대 절대 차이**다. 모델의 예측 오차가 아니다.

| 출력 | 명목 보간 라벨 | 보간 전 실제 시각 프레임 |
|---|---:|---:|
| uX (mm) | 0 | 0 |
| uY (mm) | 1.899898e−6 | 1.907349e−6 |
| uZ (mm) | 2.235174e−8 | 1.192093e−7 |
| GPS von Mises (MPa) | 5.552907e−5 | 5.557366e−5 |
| total-strain Frobenius (무차원) | 1.276372e−8 | 1.276752e−8 |
| EPSP (무차원) | **0** | **0** |

명목 라벨에서는 응력과 총변형률 norm이 132쌍 모두에서 수치상 달랐다. 변위 uY는 2쌍, uZ는 9쌍에서만 수치상 차이가 있었다. EPSP가 다른 쌍은 없었다. 이는 매우 짧은 초기 20 µs 구간의 작은 수치 차이이며, 해당 차이의 물리적 중요성·모델 예측 성능을 판정한 결과가 아니다. 1e−6 mm 이내 전체 334쌍에서도 최대 차이는 위 명목 값과 같았고 EPSP 차이는 없었다.

실제 예시는 다음과 같다.

- 노드 4171(top_pcb, material8)과 3804(top_micro_chip_02, material9)는 같은 원 좌표다. `phone_e120_rxm100`, 명목 20 µs에서 VM은 각각 6.368242e−5, 8.153345e−6 MPa이며 차이는 5.552907e−5 MPa다.
- 노드 4161(top_pcb)과 3794(top_micro_chip_02)는 같은 원 좌표다. `phone_e080_rxp100`, 명목 20 µs의 총변형률 Frobenius는 각각 2.421701e−9, 1.518542e−8이다.
- 노드 3277(top_pcb)과 2910(top_micro_chip_01)의 원 좌표는 둘 다 (−7.359996, −16.8, 2.0) mm다. `phone_e100_rx000`, 명목 14 µs에서 uY 차이는 1.899898e−6 mm다. 두 노드의 static feature는 0.69215244, 0.51586539 mm로 다르다.

## static feature와 graph가 표현하는 범위

`static_feature_full.npy`는 각 노드에 닿는 **선형 corner-cell 기준 체적 평균의 세제곱근**이다. 단위는 mm이며 두께·질량·part ID·material ID 채널이 아니다. NODE_ID와 connected component 파일은 별도로 보존되지만 이 단일 feature에 부호화되어 있지 않다. 현재 중복 좌표 쌍에서는 이 mesh 길이 대리값이 서로 달랐다는 사실과, 물성을 명시적으로 제공한다는 주장은 구분한다. 동일 쌍 0개는 저장 float32 조합의 검사이며 모델 정규화 후 입력 텐서를 실행 검증한 결과가 아니다. 실제 모델이 어떤 채널을 전달받았는지는 실행 adapter에서 별도로 확인해야 한다.

현재 graph는 **561,184 directed element-corner edges, 17개 connected components**를 가진다. 이번 검사에서도 cross-part edge는 0이고 각 성분은 한 part에 대응했다. 원 FEM의 **14개 TYPE2 tied interface와 1개 rigid wall**은 원 deck에서 확인했지만, 해당 tied/contact 연결을 가로지르는 edge는 이 mesh graph에 추가되지 않았다. graph가 17개로 나뉜다는 사실이 실제 FEM의 부품들이 물리적으로 연결되지 않았다는 뜻은 아니다.

원 257,099 노드에서 제외한 **181,034개는 TETRA10 midside 노드**다. 전체 corner support와의 관계는 기존 `drop_test/active-flag-and-node-coverage-correction.json`의 원 deck 파싱 증거에 근거하며, 이번 감사는 그 노드까지 예측했다고 주장하지 않는다. 원 solid는 109,600 TETRA10과 20,618 BRICK이다. TETRA10을 4corner로 표현하고 BRICK의 일부 퇴화 wedge를 corner 체적으로 투영한 후처리 정의도 유지한다.

입력 정의·코드 줄 근거와 현재 공용 loader/phone manifest schema의 경계는 `static-feature-review.md`에 기록했다. 이 감사는 full-phone 모델의 실행 성공·새 접촉배치·새 조립체·파손 예측 검증으로 확장하지 않는다.

## 재현

동결 phone canonical cache가 준비된 동일 연구 폴더에서 실행한다. 출력은 이 감사 폴더에만 쓰며 FEM·원 캐시는 읽기 전용이다. NumPy, SciPy, psutil이 필요하다.

```powershell
python outputs/expanded-research/benchmarks/assembly_input_audit/audit-assembly-inputs.py
python outputs/expanded-research/benchmarks/assembly_input_audit/audit-native-pairs.py
```

공개에는 코드·JSON·334행 쌍 목록·문서만 보존한다. 모델 데이터·원 deck·solver binary를 복사하지 않았다. 출처 모델의 사용 조건은 원 phone 연구 산출물의 고지를 따른다.
