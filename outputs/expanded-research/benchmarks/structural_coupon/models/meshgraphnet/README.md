# Coupon C7 — MeshGraphNet 실제 학습

공개 RD-E1101 shell 인장 예제를 바탕으로 새 FEM 13조건을 계산한 고정 자료를 사용했다. 실제 노드 919개, 비영 시점 10개, 변위 XYZ·응력·등가소성변형률·총변형률 주값 2개의 7채널을 한 번에 예측한다. 원 shell 응력/변형률 불변량은 초기 면적 가중치로 노드에 투영한 정답이며 원 cell tensor도 별도로 보존한다.

입력은 초기 좌표, 실제 두께, 항복·UTS 동시 배율, 속도 배율이다. 예측 입력에 FEM 응답을 넣지 않는다. 동일 메시·재료계·하중 경로의 파라미터 보간 실험이며 가전제품 낙하 성능 검증은 아니다.

- 학습 FEM 5조건 / 내부 검증 4조건 / 평가 4조건
- PhysicsNeMo 2.2.2, 모델 파라미터 81,958개; 실제 구조 설정은 아래와 같다.
- 학습 전용 정규화: 변위 XYZ 공통 vector scale, 나머지 scalar별 scale. 원 예측을 자르거나 물리 라벨을 대체하지 않았다.
- 공통 예산 상한 1,000 epoch, 최소 100 epoch, 검증 개선 없음 100 epoch 중단
- 325 epoch에서 중단했고 내부 검증 최소는 225번째다. 검증 손실 기준의 조기 중단이며 물리적 수렴이나 실제 제품 검증을 뜻하지 않는다.
- 학습 77.08초; NVIDIA GeForce RTX 3060 Laptop GPU; CUDA allocator peak 39.19 MB; sampled RSS 최대 1.669 GB. allocator와 경계 샘플 RSS는 전체 GPU VRAM·실제 RSS 순간 최대와 구분한다.
- 별도 프로세스 체크포인트 재로드 결과에서 비트 단위로 동일하지 않다. 원 메타데이터의 전체 배열 maxabs=0.00018310546875는 혼합 단위 참고값이다. 채널별 차이는 아래와 같으며 서로 다른 단위를 합쳐 정확도로 해석하지 않는다. 훈련 scale로 나눈 최대 차이는 1.6137762486323481e-06다. 저장된 허용 기준은 1e-05이며 GPU 합산의 반올림 차이를 고려한다.



| 새 프로세스 재추론 채널 | 최대 절대 차이 | 단위 |
|---|---:|---|
| displacement_x | 2.86102294921875e-06 | mm |
| displacement_y | 9.30391252040863e-07 | mm |
| displacement_z | 7.897615432739258e-07 | mm |
| von_mises_shell_scalar_projected | 0.00018310546875 | MPa |
| equivalent_plastic_strain_projected | 1.4901161193847656e-08 | 1 |
| total_strain_principal_major_inplane_projected | 2.2351741790771484e-08 | 1 |
| total_strain_principal_minor_inplane_projected | 1.1175870895385742e-08 | 1 |

```json
{
  "input_dim_nodes": 6,
  "input_dim_edges": 4,
  "output_dim": 70,
  "processor_size": 3,
  "hidden_dim_processor": 48,
  "hidden_dim_node_encoder": 48,
  "hidden_dim_edge_encoder": 48,
  "hidden_dim_node_decoder": 48,
  "num_layers_node_encoder": 2,
  "num_layers_edge_encoder": 2,
  "num_layers_node_decoder": 2,
  "num_layers_node_processor": 2,
  "num_layers_edge_processor": 2,
  "mlp_activation_fn": "relu",
  "aggregation": "sum",
  "do_concat_trick": false,
  "num_processor_checkpoint_segments": 0,
  "checkpoint_offloading": false,
  "recompute_activation": false,
  "norm_type": "LayerNorm"
}
```

| 평가 물리량 | 상대 L2 | 물리 단위 RMSE |
|---|---:|---:|
| 변위 벡터 (성분 RMSE) | 6.948% | 0.16284 mm |
| 변위 x | 6.688% | 0.271446 mm |
| 변위 y | 110.477% | 0.0692805 mm |
| 변위 z | 거의 0인 정답: 해석 보류 | 0.0326715 mm |
| 투영 shell von Mises 응력 | 23.718% | 122.451 MPa |
| 투영 등가소성변형률 EPSP | 24.954% | 0.00766121 |
| 투영 in-plane 총변형률 주값 major | 24.127% | 0.00789325 |
| 투영 in-plane 총변형률 주값 minor | 24.262% | 0.00385834 |

**축별·물리적 한계:** y 방향 상대 L2는 110.477%, 절대 RMSE는 0.0692805 mm다. 평면 FEM의 z 변위 정답 최대 절댓값은 2.30011e-14 mm인 반면, 예측 최대 절댓값은 0.150442 mm, 최대 절대 오차는 0.150442 mm, RMSE는 0.0326715 mm다. z 상대 L2는 거의 0인 정답으로 나누므로 정확도 주장에 사용하지 않는다.

원 예측 전체의 음수 von Mises 응력은 87/36,760개이며 최소 -20.7057 MPa다. 음수 EPSP는 6,306/36,760개이며 최소 -0.0025553다. 이 값들은 숨기거나 0으로 보정하지 않았다.

참 EPSP >0 영역 28,897개의 RMSE는 0.00852127, 참 EPSP=0 영역 7,863개의 RMSE는 0.00274674다. 참 EPSP=0 영역에서 예측 절댓값 최대는 0.0500821다. 양수 support 판정은 정답·예측 각각 >0을 그대로 사용하며 임계값을 맞추지 않았다. 이 투영된 노드 support는 원 cell 파손 판정 기준이 아니다.

**EPSP 누적성 사후 검사:** 고정된 투영 노드에서 연속 저장 시점의 엄밀한 음수 증가분은 예측 6,842/33,084개(정확히 0인 증가분 0개), 정답 0/33,084개다. 예측 delta 최솟값은 -0.00233241, 감소 크기 분포는 p50=0.000564196, p90=0.00131767, p95=0.00158999, p99=0.00194739, max=0.00233241다. 정답 최대 감소는 0다. 저장 배열은 float32/float32이며 FP32 machine epsilon은 1.19209e-07이다(1 부근의 상대 간격이며 EPSP의 절대 허용 오차가 아니다). 엄밀한 음수 횟수는 반올림 규모의 변화도 셀 수 있으므로 정답의 감소 분포와 함께 읽어야 한다. 이 검사는 임의의 성공 임계값·보정·단조화 없이 원 예측을 진단하며, 언로딩 때 감소할 수 있는 총변형률 주값에는 적용하지 않았다.

**투영 주값 순서 사후 검사:** 예측 minor > major는 1,840개, 정답은 0개다. 예측의 최대 minor-major 위반 크기는 0.00185405이며 재정렬하지 않았다.

'metrics.json'은 학습 당시 원 보고서이고 'field_diagnostics.json'은 저장된 원 예측의 후속 진단이다. 상대 L2는 평가 전체 노드·시점의 정답 제곱합을 공통 분모로 사용하며 모든 노드에 동일 가중치를 준다. 변위 벡터 오차가 작더라도 지배적인 x 방향 응답이나 평균 성능만으로 국부 응력·소성의 정확도를 판정하지 않는다. 원 총변형률은 solver의 accumulated total strain 출력이며 특정 유한변형률 측정량과 정확히 같다는 추가 증명은 없다.

그림은 'field_comparison_1.png', 'field_comparison_2.png', 개별 'field_*.png', 'deformation_1x.png', 'learning_curve.png'에 있다. 비교장은 첫 평가 조건의 마지막 저장 시점이며 FEM과 예측의 공통 색 범위를 사용한다. 변형 그림은 1배이며 회색은 초기 형상이다. 전체 평가 수치는 그림 한 장의 조건에 한정되지 않는다.

13조건 고정 입력 manifest SHA256은 '08876879b2b003f8222b831bb51a15c63f8d69d30e6b83e077385f80850169b1'이며 학습 시작·종료 검증 결과가 일치했다. 'source/'와 'executed_*.py'에는 당시 코드가, 'best_checkpoint.pt'에는 구조·가중치·정규화·입력/분할 해시가 있다. 후처리는 학습·체크포인트·원 NPY를 수정하지 않는다. 예측 forward는 정답을 입력받지 않으며 FEM 정답은 수치 검증에 사용한다.

로컬 'predictions.npy', 'truth.npy'와 원 FEM/캐시는 공개 저장소 복사에서 제외한다. 재현에는 FEM 생성 기록·원 자료 출처·보존된 정규화가 필요하다. 이 결과를 읽은 뒤 동일 평가 집합에 맞춰 모델 설정이나 학습 예산을 바꿔 재학습하지 않았다.
