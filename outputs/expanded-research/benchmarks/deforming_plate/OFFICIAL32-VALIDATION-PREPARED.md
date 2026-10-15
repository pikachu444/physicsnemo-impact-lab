# 공식 크기 판 모델의 검증 전용 실험 준비

현재 상태는 **코드·프로토콜 준비 및 정적 확인 완료, GPU 실행 대기**다. 이 문서는 새 학습 결과가 아니다. 앞서 완료한 작은 모델 두 개의 정확도 실패, 체크포인트 재사용 결과 및 공식 크기 자원 측정은 그대로 보존한다.

실제 측정을 완료한 공식 크기 HybridMeshGraphNet(폭 128, processor 15, 3,605,636 parameters)을 사용한다. 원본 독립 학습 궤적 32건에서 궤적당 8개의 인접 단계 쌍을 5 epoch 학습한다. 현재 단계 anchor는 `0, 56, 113, 170, 227, 284, 341, 398`이며 각각 다음 단계를 예측한다. 총 1,280회 Adam update다. Gaussian noise 표준편차 0.003, 초기 학습률 0.0001, update당 감소율 0.9999917을 사용한다.

이 설정은 원 논문 학습 recipe 전체를 재현하지 않는다. 데이터·anchor 수를 줄였고, 학습 데이터만의 통계 및 매 update 새 noise에 대한 해석적 분산으로 정규화하며, 메모리 제한에 맞춰 CPU SciPy의 완전 반경 그래프를 지연 생성한다. NVIDIA example의 node/edge/target 계약과 설치된 PhysicsNeMo HybridMeshGraphNet 자체는 유지한다.

`official32-validation-protocol.json`은 학습 전에 고정했다. SHA256은 `a71497614937762d2ca7434d8b5c6e2f91d3fecb4c750ab9efd97192e776bf33`이다. 검증 전체 6건의 기존 metadata를 비교해 절점 수가 유일하게 가장 큰 `valid_0000`(1,645절점)과 원본 scalar stress 최댓값이 유일하게 가장 큰 `valid_0005`(3,531,090.75, 단위 미확정)를 진단 사례로 골랐다. 선택은 시험 성능을 보기 전에 이뤄졌다.

각 epoch에서는 검증 6건의 고정 12개 teacher-forced anchor에 대해 정규화 MSE 및 자유절점의 변위 증분·응력 원단위 오차를 기록한다. epoch 1·3·5에서는 고정 진단 2건을 각각 실제 399단계 자동회귀한다. 체크포인트는 다음 사전 규칙으로 선택한다.

1. 완전한 평가를 마친 후보를 실패 후보보다 우선한다.
2. 모든 예측 단계에 역전 요소가 없는 후보를 하나라도 역전된 후보보다 우선한다.
3. 두 사례 각각의 변위·scalar stress 상대 L2 중 가장 큰 값을 최소화한다. 모든 후보에 역전이 있더라도 같은 규칙을 유지하며, 오차가 같으면 앞선 epoch를 남긴다.

정답 norm이 0이면 절대 RMSE·최대 오차를 별도로 기록하고, 두 절대 오차가 모두 정확히 0일 때에만 상대 오차 0으로 인정한다. 원 응력 예측의 음수 값을 잘라내지 않는다. 매 단계의 예측·원본 요소 역전 수와 최소 det(F)를 남긴다. Green–Lagrange 변형률은 FEM/AI 절점 좌표에서 유도한 값이며 **원본 COMSOL 변형률 정답이 아니다**.

선택 후 검증 6건만 399단계 자동회귀로 평가한다. 최종 시험 `test_0002..0007`은 이 실행에서 파일 목록 조회·읽기·해시·예측을 하지 않는다. 시험은 모델 설정과 추가 학습 여부를 검증 결과로 확정한 뒤 별도 승인된 실행에서 한 번 평가할 예정이다.

단일 큰 학습 그래프의 실측 처리량을 적용하면 학습 update 약 8.87분, epoch별 teacher 검증 약 0.96분, 세 번의 진단 약 6.40분, 최종 검증 6건 약 6.40분으로 합계 **약 22.64분**이다. 이는 I/O, 초기화, 체크포인트, 매 단계 determinant 검사, 유도 변형률과 paging을 제외한 계산 참고치다. 자동회귀 형상이 붕괴해 그래프가 조밀해지면 시간이 늘거나 OOM이 발생할 수 있다. 정확도나 상한 시간을 보장하지 않는다.

다음 명령은 전용 GPU 순서가 배정된 뒤 실행한다. 새 출력 폴더가 이미 존재하면 진입점이 실행을 거부한다. 실패·OOM traceback, 실행 소스, 정규화, 체크포인트, 원단위 오차, 모델 선택 이력 및 메모리 측정을 보존한다.

```powershell
$env:PHYSICSNEMO_FORCE_TE = 'False'
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/run_validation_training.py --output outputs/expanded-research/benchmarks/deforming_plate/official32_valpilot --protocol outputs/expanded-research/benchmarks/deforming_plate/official32-validation-protocol.json --hidden 128 --processor 15 --checkpoint-segments 0 --epochs 5 --frames-per-trajectory 8 --noise 0.003 --seed 20261015 --lr 0.0001 --lr-decay-per-update 0.9999917 --rollout-steps 399 --train-count 32 --valid-count 6 --validation-only --checkpoint-selection rollout --selection-epochs 1,3,5 --selection-trajectories valid_0000,valid_0005 --evaluation-mode ar_only --evaluation-label exploratory
```

`official32_prepared/`는 준비 당시 소스와 정적 구문 검사의 해시를 담는다. 실제 실행 시 다시 실행된 소스 snapshot과 해시를 새 결과 폴더에 남긴다. 정적 구문 확인과 synthetic 선택 규칙 점검은 실제 CUDA 학습 검증을 대신하지 않는다.
