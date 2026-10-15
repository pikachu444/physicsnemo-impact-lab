# 현재 코드에서 확인한 형상·메시 제약

2026-10-16 실제 소스를 **읽어서 확인한 사실**이다. 새 데이터 취득·라이브러리 import·학습·정확도 시험 결과가 아니다. 경로와 행 번호는 작업 공간의 원본을 가리키며, 파일 식별 SHA-256은 아래에 기록했다. 기존 결과/제약은 유지하고 새 캠페인 입력 경로를 별도로 구현한다.

| 항목 | 실제 소스 근거 | 다음 문제 구성에 미치는 영향 |
|---|---|---|
| 모델 수 | `work/expanded-ml/model_adapters.py:20–31,148–149` | 5계열·6설정. `geoflare`는 GeoTransolver의 GALE_FA 설정이다. |
| 가변 노드 수 | `model_adapters.py:164–168,196,220` | 단일 모델 인터페이스는 입력에서 N을 읽는다. 가변 N 실행 가능성과 형상/메시 정확도 일반화는 다른 검증이다. |
| 고정 캐시 가정 | `work/expanded-ml/benchmark_data.py:44–55,77–83,90,98` | 공통 N·edge와 `[N,T,C]` 출력·노드별 평균장을 사용한다. 형상별 연결과 크기가 바뀌는 자료를 그대로 넣는 경로가 아니다. |
| 좌표/연결 보호 | `work/expanded-batch/train_probe_data.py:87`, `work/expanded-ml/predict_inputs.py:39–40` | 기존 경로는 좌표 동일성 및 좌표/edge 해시를 검사한다. 기존 CLI가 새 형상 예측까지 준비됐다고 주장할 수 없다. |
| 정규화 | `benchmark_data.py:115–117,133–140` | 이미 학습 사례 전체로 구한 공통 bbox다. 사례별 정규화로 잘못 설명하면 안 된다. |
| 공간 범위 | `model_adapters.py:79,172–173` | `[0,1]` 검사와 FIG AABB가 있다. 1% 여유의 학습 상자 밖 시험 형상은 명시적 처리 방침이 필요하다. clipping·시험별 재정규화로 크기 차이를 지우지 않는다. |
| 입력 정보 | `model_adapters.py:135,197–219` | xyz·단일 nodal thickness·전역 변수만 사용하며 PART_ID·부품별 재료·접촉 정보가 없다. SHIFT의 cell PART_ID를 공유 노드의 정수 하나로 무작정 바꾸면 모호해진다. |
| MGN 연결 | `model_adapters.py:209–214` | 실제 FEM edge와 좌표 차이/길이를 쓴다. 동적 접촉 edge 생성 기능은 없다. 점 삭제만으로 유효 축약 그래프가 만들어지는 것은 아니다. |
| 배치 | `work/expanded-batch/batch_adapters.py:8–16,46–54` | 공통 N·edge를 전제로 한다. 가변 그래프에는 단일 사례 처리 또는 새 collate가 필요하다. |
| 시간·출력 | `model_adapters.py:45–46,105–107` | caller가 정한 `T×C` head는 가능하다. SHIFT 시간 조건/필드/mask 계약과 새 head는 미구현이다. 기존 가중치를 호환된다고 가정하지 않는다. |
| 지표의 메시 의존성 | `benchmark_data.py:150–180,188–197` | 단순 노드 합/평균과 노드 평균 중심을 사용한다. 메시 밀도를 바꿀 때는 면적 가중치/공통 물리 query와 면적 가중 중심이 필요하다. |

| 작업 공간의 원본 파일 | SHA-256 |
|---|---|
| `work/expanded-ml/model_adapters.py` | `fcfbfc72c4eb36793849410ba54e7e41c27a5261d7f0f3830800a774edd66e7d` |
| `work/expanded-ml/benchmark_data.py` | `4888594826c9a517bb8fc876d2fa533daf7eb78d9f247c10af287821f70956c6` |
| `work/expanded-batch/batch_adapters.py` | `58a987382bd4039a1c0c81133bd41b088e558ebdf2e71d47bbdfd9f0fdf4c673` |
| `work/expanded-batch/train_probe_data.py` | `0287662371b5268a005b36a9cc3a5b278ea56a4b8b52a3a207e9725a4a9f78f5` |
| `work/expanded-ml/predict_inputs.py` | `fe25900e1531ac86120c5994656d3da0c66b609b335de9a614713cc47470ad7d` |

따라서 현재 코드가 다른 노드 수를 받는다는 사실만으로 “메시 영향이 작다”고 답할 근거는 없다. 형상 예측 성능과 AI 입력 해상도 민감도, 원본 FEM 재메시 수렴을 각각 측정해야 한다.
