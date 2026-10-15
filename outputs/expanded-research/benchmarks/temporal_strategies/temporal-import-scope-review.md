# 메모리 중단 후 import 경로 정적 검토

2026-10-15, 자원 검사 중단 뒤 파일만 읽어 확인했다. 추가 모델 import·배열 읽기·학습·하드웨어 검사는 수행하지 않았다. 측정된 메모리 절감량은 없다.

`run_multifield_stage.py` → TC shim → V2 → `run_domain_temporal.py` → `run_temporal.py` → 공식 drop/crash wrapper 경로를 추적했다. 확인한 driver와 pinned wrapper에는 직접적인 matplotlib/GUI/sklearn/scipy import가 없었다. OMP/OpenBLAS thread1은 고정하지만 `MPLBACKEND=Agg`는 고정하지 않는다. plotting import를 발견하지 못했으므로 이 환경변수 하나로 이번 메모리 문제가 해결된다고 주장하지 않는다.

두 공식 `sources/{drop_test,crash}/rollout.py`의20–23행은 Transolver·MeshGraphNet·FIGConvUNet·GeoTransolver를 모두 import한다. 시간 조건 전략이 선택하는 모델 외에도 eager import가 있지만, 다른 모델은120/135/154행 클래스의 직접 부모다. 단순히 import 문을 함수로 옮기는 수정으로는 충분하지 않다. 관련 클래스 정의를 지연하거나 모듈을 나누면 frozen wrapper와 source hash가 바뀌므로 기존 R4 동결 실험에 조용히 섞지 않는다. datapipe의 PyG는 이미 OptionalImport를 사용한다.

이 검토의 결론은 현재 동결 규약 안에서 적용할 명백한 단순 lazy-import 수정은 찾지 못했다는 것이다. 새 경량 wrapper를 연구하려면 별도 소스 버전·동등성 검사·실측이 필요하며, 이 검토로 구현·절감 효과를 검증한 것은 아니다. 동결 목록20개 밖의 전이 의존성을 전수 조사한 결과도 아니다. 기존 실제 중단 로그와 미인증 update 수는 `resource-calibration-v1/SUMMARY.json`에 그대로 보존한다.
