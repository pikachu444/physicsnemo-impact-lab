# PhysicsNeMo Impact Lab

NVIDIA PhysicsNeMo 기반 가전제품 낙하·충격해석 AI 연구와 공개 예제 실행 기록.

## 연구 목표

공개 FEM 데이터와 예제로 설치, 실행, 학습, 추론 및 FEM 대비 오차를 검증한다.
전체 가전제품, 부품 및 서브어셈블리의 적용 범위를 비교하며 알고리즘은 실험 근거로 선정한다.
추가 비용이 발생하는 자원은 사용자와 협의한 뒤 사용한다.

## 실행 환경 확인

```sh
python scripts/probe_environment.py --label codex-cloud --network --output results/cloud/environment.json
```

이 명령은 실행 중인 환경의 자원을 측정한다. 로컬 측정은 Codex Cloud 측정으로 해석하지 않는다.
GPU 드라이버 인식, CUDA PyTorch 동작, 실제 학습 성공은 별도 단계로 기록한다.

현재 조사와 실험은 진행 중이다. 대용량 원본 데이터와 타사 실행 파일은 별도 다운로드하며
출처·라이선스·해시를 남긴다. 데이터셋이 공개됐다는 사실만으로 재배포 권한을 가정하지 않는다.
