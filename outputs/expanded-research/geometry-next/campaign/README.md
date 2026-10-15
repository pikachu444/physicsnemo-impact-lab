# 복잡한 초기 형상 변경 예측: 실행 전 문제 구성

[전체 연구 구성](../README.md) · [기존 코드의 형상·메시 제약](adapter-gaps.md)

**지금 준비된 것은 문제 정의·설정·메타데이터 검사 코드다. 새 차량 데이터를 받거나 학습한 결과는 없다.** 실제 학습은 SHIFT-Crash 접근 승인, 고정 재료군 확인, 데이터 로더 구현과 자원 측정 후 진행한다. 이 폴더의 Python 코드는 표준 라이브러리만 사용하며 다운로드·FEM·신경망 실행 기능이 없다.

## 검증할 질문

같은 충돌 조건, 같은 레일 두께·항복강도에서 **처음 보는 차량 초기 형상**의 변위·응력·소성변형률을 예측할 수 있는가? 전면 레일 길이, 실내 길이, 차폭, 지붕 높이의 네 형상 변수를 다룬다. 한 Yaris 기본 조립체를 변형한 범위이므로 다른 차종·새 부품 배치·접합 방식 변경까지 검증했다고 확대하지 않는다.

- 입력: 변형 전 좌표/구조 정보 + 형상 변수 + 알려진 재료/두께 + 공통 충돌 조건 + 예측할 시간.
- 출력: 해당 시간의 변위 3성분, von Mises 응력, 유효소성변형률(EPSP). 공개 카드의 단위는 각각 mm, Pa, 무차원이며 실제 파일에서 확인해야 한다.
- 추론에는 미래 FEM 좌표·응력·변형률을 넣지 않는다. 자유 시간 전개는 별도 실험이며, 정답을 매 단계 넣은 시험을 실제 미래 예측으로 보고하지 않는다.
- EPSP는 전체 변형률 텐서가 아니다. 원본에 없는 전체 변형률·파손 확률·접촉력은 이번 출력으로 선언하지 않는다.

## 데이터와 형상 분할

대상은 [SHIFT-Crash](https://huggingface.co/datasets/luminary-shift/SHIFT-Crash), 고정 revision `6faa776f7e363ce39d43eb8dc8b0d9c7d0c83049`이다. 공개 API 조사로 파일 목록·크기를 확인했지만 params/VTP는 아직 취득하지 않았다. 실제 필드·연결·단위·형상 대응관계는 미검증이다. 자료 근거는 [공개 데이터 조사](../complex-public-datasets.md)에 있다.

먼저 작은 `params.json`에서 **두께와 항복강도가 정확히 같은데 초기 형상이 다른 군**이 충분히 존재하는지 확인한다. 없으면 이 데이터에서 순수 형상 효과는 식별할 수 없다고 보고한다. 재료 변화까지 섞은 회귀 결과를 형상 단독 성능으로 바꾸어 설명하지 않는다.

`campaign.py split`은 네 형상 변수의 정확한 소수 문자열을 묶어 같은 형상의 모든 시간·재료 변형을 하나의 fold에 배치한다. 기본 학습 사례 수·분할 수·비율은 없다. 실제 메타데이터 검토 뒤 정한 학습/검증/시험 **형상 수**를 명시해야 한다. 기본 비교에는 지정한 고정 재료군만 들어가며, 다른 재료 변형의 fold도 함께 기록해 누출을 점검한다. 남는 형상은 사용하지 않았다고 기록한다. 새로운 형상이라는 이유만으로 외삽이라고 부르지 않는다.

## 비교할 모델: 5계열, 6설정

| 설정 | 실제 계열/구현 | 이 문제에서 아직 필요한 구현 |
|---|---|---|
| Transolver | PhysicsNeMo Transolver, `plus=False` | 형상별 입력·시간 조건·새 출력 계약 |
| FLARE | PhysicsNeMo FLARE | 같은 입력·출력 계약, 규모별 메모리 측정 |
| GeoTransolver GALE | GeoTransolver의 GALE | 형상/전역 특징 차원과 시간 조건 |
| GeoTransolver GALE_FA | 같은 클래스의 GALE_FA 설정 | 위와 같음; 별도 계열로 세지 않음 |
| MeshGraphNet | 실제 FEM 연결을 쓰는 그래프 | 형상별 그래프, 검증된 축약, 시간/출력 계약 |
| FIGConvUNet | 점과 factorized grid | 공통 공간 영역, 범위 밖 형상 정책, KNN 메모리 측정 |

실제 기존 `model_adapters.py`와 데이터·배치·추론 코드를 읽어 작성했다. 모델 함수가 가변 N을 받는 것과 전체 학습 경로가 새 형상을 지원하는 것은 다르다. 기존 캐시는 공통 N/연결을 가정하고 기존 추론은 좌표/연결 해시가 바뀌면 거부한다. **현재 차량 학습 명령은 구현되어 있지 않다.** 기존 검증을 약화시키지 않고 별도 형상별 로더/추론을 구현해야 한다. 현재 어댑터에는 동적 접촉 그래프도 없다.

이 여섯 신경망 설정이 가능한 알고리즘 전부는 아니다. 필드별 영(0) 예측, 학습 자료의 평균장, 형상 매개변수 회귀를 기준선으로 둔다. **POD+RBF**도 비교 후보이며, 노드/셀 대응 또는 공통 물리 query 전송이 실제 검증되고 학습 자료만으로 기저를 만든 경우에 비교한다. 고정 토폴로지의 기존 ROM 결과를 새 형상 성능으로 승계하지 않는다. 우승 방법은 미리 정하지 않는다.

기존 좌표 정규화도 이미 학습 사례 전체의 공통 상자를 사용한다. 새 구성에서도 형상별 bbox 정규화로 치수 차이를 지우지 않는다. 공통 물리 좌표계와 학습 자료에서 구한 공통 변환을 저장한다. `[0,1]` 검사와 FIG 고정 AABB 밖에 나오는 시험 형상은 clipping하거나 시험 자료로 다시 맞추지 않고 기록·처리 방침을 확정한다. PART_ID는 카드상 cell 정보이므로 여러 부품이 만나는 노드에 정수 하나로 무작정 옮기지 않는다.

## 메시 영향은 두 문제로 분리한다

| 시험 | 바꾸는 것 | 필요한 비교/현재 한계 |
|---|---|---|
| AI 입력 해상도 | 같은 형상·같은 FEM 정답에서 AI 점/그래프 수만 변경 | 같은 물리 query에서 면적 가중 오차·최대값/국부 hotspot·메모리/시간 비교. query 전송 오차도 측정해야 한다. 기존 어댑터는 입력 노드에서 출력하므로 공통 query 전송/decoder 구현이 먼저다. |
| 원본 FEM 재메시 | 같은 물리 형상을 독립적으로 다른 메시로 만들고 재해석 | 재료·접촉·하중을 고정한 FEM 수렴과 AI 전이 오차를 각각 측정. SHIFT의 형상 morph만으로는 이 시험을 할 수 없다. |

동일 노드 수만으로 동일 연결을 주장하지 않는다. 연결/좌표/part 해시를 확인한다. MGN은 점을 무작위로 버리고 남은 edge만 사용한 결과를 검증된 그래프 축약이라 부르지 않는다. 현재 지표의 단순 노드 평균은 조밀한 영역에 치우칠 수 있으므로 공통 물리 query 또는 면적 가중치가 필요하다. 변위는 강체 이동을 포함한 오차와 면적 가중 중심 이동을 제거한 변형 오차를 따로 기록한다.

## 지금 실행 가능한 검사

Python 3.10 이상이면 추가 패키지 없이 실행한다. 아래 명령은 **이 폴더에서** 사용한다. Windows에서는 실제 Python 실행 파일 경로를 지정해도 된다.

```powershell
python -S campaign.py validate --manifest manifest.template.json
python -S campaign.py selftest --out scratch/prep-selftest.json
python -S campaign.py preflight --manifest manifest.template.json
```

마지막 명령은 `PREPARED_NOT_EXECUTABLE`과 종료 코드 2를 반환한다. 미승인 접근·빈 실제 메타데이터·미구현 로더 등 남은 조건을 알리는 의도된 결과다. selftest의 사례·해시·조건은 **전부 합성 fixture**이며 실제 차량 데이터나 예측 정확도 증거가 아니다. 배포된 `prep-selftest.json`은 보존된 검사 증거이며, 재검사는 위처럼 scratch 경로에 쓴다. `prep-validation.json`은 정밀 소수 분할 수정 후 16개 검사와 수정 전 15개 검사를 구분해 기록한다.

실제 접근 승인 후 `manifest.template.json`을 복사해 다음 계약에 따라 채운다. 접속 동의 화면을 보는 것과 데이터 제공자의 승인을 받은 것은 다르다. 연락정보/자격증명은 이 파일에 쓰지 않는다.

```json
{
  "case_id": "yaris_x00001",
  "geometry_mm": {"front_rails": "<실제 값>", "cabin_length": "<실제 값>", "car_width": "<실제 값>", "roof_height": "<실제 값>"},
  "material": {"rail_thickness_mm": "<실제 값>", "rail_yield_strength_mpa": "<실제 값>"},
  "condition": {"impact_speed_kmh": "<확인값>", "barrier_id": "<확인한 동일 조건 ID>", "duration_ms": "<확인값>"},
  "files": [
    {"role": "params", "path": "yaris_x00001/params.json", "bytes": 283, "acquired": true, "sha256": "<실제 파일 SHA-256>"},
    {"role": "vtp", "path": "yaris_x00001/merged_surfaces.vtp", "bytes": 257016764, "acquired": false, "sha256": null}
  ]
}
```

위 크기는 공개 API로 확인한 case 00001의 값이다. 다른 사례에 복사해서 확인값으로 쓰면 안 된다. 문서 예시의 `<...>`는 실행용 값이 아니다. 모든 매개변수는 소수 문자열로 적고, 원본 params 키와 단위 변환을 대조한 뒤 access의 해당 확인 항목을 true로 기록한다. 승인·취득·키/조건 매핑이 확인되지 않으면 split 명령은 거부된다.

```powershell
# 값과 형상 수는 실제 군 확인 뒤 지정한다. 현재 실행 예제가 아니다.
python -S campaign.py split --manifest manifest.actual.json --thickness-mm <확인값> --yield-mpa <확인값> --counts <학습형상수> <검증형상수> <시험형상수> --seed <기록할시드> --out split.candidate.json
python -S campaign.py download-plan --manifest manifest.actual.json --case-id <확인사례> --role vtp --max-bytes <명시한바이트한도> --out download-plan.json
```

download-plan은 고정 revision의 정확한 파일 allowlist와 byte 합만 작성한다. 네트워크 기능이 없고 1.4TB 전체 다운로드를 실행하지 않는다. 실제 전송 전에는 로그인/접근 승인, 디스크 여유와 임시 파일·캐시 공간을 별도로 확인해야 한다.

## 실행을 재개할 순서와 완료 기준

접근 승인과 재료가 일치하는 형상군 확인 → VTP 일부의 필드/단위/연결/part·erosion 검사 → 범위를 제한한 추출기와 형상별 입력 구현 → 시간 조건·공통 query 계약 검증 → 작은 실제 forward/backward의 RAM/VRAM/시간 측정 → 그 근거로 사례 수·학습 예산·GPU 결정 → 검증 형상으로 선택 → 보류한 시험 형상에서 수치/그림 비교 순서다.

예산·사례 수·GPU·학습 시간은 아직 확정하지 않았다. 유료 자원을 생성하는 기능도 없다. 목표는 정확도까지 검증하는 것이며, 이번 준비 검사는 설치 성공·실제 학습 성공·형상 예측 검증을 대신하지 않는다.
