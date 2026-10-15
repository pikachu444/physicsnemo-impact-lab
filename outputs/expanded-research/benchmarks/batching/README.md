# 배치 검산 결과

24개 설정을 실제 시도했다. 22개 통과, phone MeshGraphNet B5는 기존 실행의 실제 CUDA OOM, phone FIGConvUNet B5는 시스템 메모리 가드 중단이다. 원본 `allocated-slot-01/suite.json`은 21개 통과 시점 그대로 보존했으며, 원본 밖에 남아 있던 OOM 결과를 `merged-summary.json`에 합쳤다.

| 도메인 | Transolver | FLARE | GeoTransolver | GeoFLARE | MeshGraphNet | FIGConvUNet |
|---|---:|---:|---:|---:|---:|---:|
| Bumper | B5 | B5 | B5 | B5 | B2 | B5 |
| Phone | B5 | B5 | B5 | B5 | B1 | B2 |

추천은 동일 초기 가중치의 B1 반복 호출과 native batch를 비교해 두 번 모두 forward/backward가 빨랐고, 기록된 CUDA reserve가 명목 GPU 용량의 80% 이하인 후보 중 처리량이 가장 높은 배치를 골랐다. Phone MeshGraphNet B2는 약 0.0985배 속도와 큰 메모리 사용량 때문에 B1을 택했다. Phone FIG B2는 forward/backward 1.1015배, forward-only 0.9034배이므로 학습 작업에 대한 추천이다. 3회 Adam 검산은 동일 수렴이나 같은 epoch의 optimizer 갱신 횟수를 보장하지 않는다.

새 `allocated-slot-02`의 FIG B2는 통과했다. B5는 32.56초에 시스템 가용 물리 메모리가 11,423,744바이트로 내려가 `resource_guard_stop`으로 종료했다. CUDA OOM으로 분류하지 않았고, 부분 events와 OS 자원 로그를 보존했다. GPU는 종료 확인 후 주학습 담당자에게 반환했다.

- `batch-recommendations.json`: 12개 추천과 각각의 성능·메모리 근거.
- `merged-summary.json`: 24개 실제 시도, 원본 SHA, 입력·소스 provenance, 사례별 노출 회계.
- `recovery-audit.json`: 정적 검토, 중단된 FIG B5의 입력 metadata와 추가 소스 검증.
- `allocated-slot-02/recovery-source/`: 실제 복구 실행기와 집계 코드 snapshot.

23개 result의 138개 소스 snapshot과 중단된 FIG B5의 6개 snapshot, 총 144개 비교가 일치했다. 원본 소스와 기존 출력은 덮어쓰지 않았다. Strict loader는 선택된 train 파일만 읽고 hash하며, held-out target 파일은 읽거나 hash하지 않는다. 복구·집계 과정에서 주학습이나 held-out 평가는 수행하지 않았다.

완료된 검산 한 건은 각 선택 사례의 label loss를 9회 관측하고 그중 Adam을 3회 갱신한다. Forward는 사례당 17회이며 마지막 사례만 18회다. 마지막 사례의 2회는 global 입력 변경 검사다. 실패한 B5 둘은 각각 최소 26개 사례-forward와 5개 label loss가 완료됐고 Adam 갱신은 0회다. 중단 중인 연산의 추가 노출은 정확한 완료 횟수로 주장하지 않는다.

Phone의 실제 scalar는 `reference_mean_incident_corner_cell_volume_cuberoot`다. 기존 결과의 중첩 `model_source.input_information`에 남은 `nodal_thickness` 문구는 표시 metadata 오류로 별도 명시했으며, 실행 결과를 고쳐 쓰지 않았다. OS 로그의 RSS는 Windows venv launcher이므로 실제 Torch 메모리는 probe result/events의 RSS·Windows peak working set을 사용한다. OS 가용 물리 메모리와 commit 표본은 시스템 전체 측정값이다.
