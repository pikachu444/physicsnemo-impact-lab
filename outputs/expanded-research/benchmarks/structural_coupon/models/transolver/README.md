# Coupon C7 — Transolver 실제 학습

공개 RD-E1101 shell 인장 예제를 바탕으로 새 FEM 13조건을 계산한 자료를 사용했다. 실제 노드 919개, 비영 시점 10개, 변위 XYZ·응력·등가소성변형률·총변형률 주값 2개의 7채널을 한 번에 예측한다. 원 shell 응력/변형률 불변량은 명시적인 초기 면적 가중치로 노드에 투영한 정답이며, 원 cell tensor도 별도로 보존한다.

입력은 초기 좌표, 실제 두께, 항복·UTS 동시 배율, 속도 배율이다. 예측 입력에 FEM 응답을 넣지 않는다. 동일 메시·재료계·하중 경로의 파라미터 보간 실험이며 가전제품 성능 검증은 아니다.

- 학습 FEM 5조건 / 내부 검증 4조건 / 평가 4조건
- 사전 고정한 모델: PhysicsNeMo 2.2.2 Transolver, 2층·hidden48·head4·slice8
- 학습 전용 정규화: 변위 XYZ 공통 vector scale, 나머지 scalar별 scale
- 예산 상한 1,000 epoch, 최소 100 epoch, 검증 개선 없음 100 epoch 중단
- 1,000 epoch 실행, 내부 검증 최소도 1,000번째: **유한 예산에 도달했으며 수렴을 입증한 것은 아니다.**
- 학습 206.50초, RTX3060 Laptop CUDA allocator peak 26.24 MB, sampled RSS 최대 1.481 GB
- 별도 프로세스에서 체크포인트를 재로드한 4조건·7채널 예측 최대 차이 0

| 평가 물리량 | 상대 L2 | 물리 단위 RMSE |
|---|---:|---:|
| 변위 벡터 | 1.911% | 성분 RMSE 0.04479 mm |
| 투영된 shell von Mises 응력 | 5.491% | 28.349 MPa |
| 투영된 등가소성변형률 EPSP | 2.094% | 0.0006429 |
| 투영된 in-plane 총변형률 주값 major | 2.059% | 0.0006737 |
| 투영된 in-plane 총변형률 주값 minor | 2.483% | 0.0003948 |

이 표만으로 물리적으로 타당한 모델이라고 판정하지 않는다. 원 평면 FEM의 z 변위는 최대 약 `2.3e-14 mm`인데 예측의 z 변위 RMSE는 `0.002512 mm`, 최대 절댓값은 `0.011694 mm`다. z 방향 상대 L2는 거의 0인 정답으로 나누므로 실용적인 정확도 지표가 아니다. y 방향 상대 L2도 55.17%이며 절대 RMSE는 0.03460 mm다.

원 예측을 자르거나 음수를 숨기지 않았다. 평가 전체에서 음수 응력 3,417개(최소 -57.88 MPa), 음수 EPSP 6,753개(최소 -0.001454)가 발생했다. 참 EPSP 양수 영역의 RMSE는 0.0006899, 참 EPSP=0 영역의 RMSE는 0.0004282다. 따라서 물리 제약과 국부 응답 개선이 필요하다. 이 결과를 본 뒤 모델 설정이나 학습 예산을 바꾸어 같은 평가 집합에 다시 맞추지 않았다.

`metrics.json`은 학습 당시의 원 보고서, `field_diagnostics.json`은 저장된 원 예측에 대한 추가 진단이다. 상대 L2는 평가의 모든 노드·시점에서 정답 제곱합을 분모로 하며 노드에 동일 가중치를 준다. 원 총변형률은 solver가 출력한 accumulated total strain으로, 특정 이름의 유한변형률 측정량과 정확히 같다는 추가 증명은 없다.

`source/`에는 실행 당시 코드가, `best_checkpoint.pt`에는 구조·가중치·정규화·메시/분할/파일 해시가 있다. 새 실행은 새 출력 폴더를 지정한다.

```powershell
$env:PYTHONUTF8='1'
$env:PHYSICSNEMO_FORCE_TE='False'
$env:TORCHDYNAMO_DISABLE='1'
work/expanded-ml/venv/Scripts/python.exe -X utf8 outputs/expanded-research/benchmarks/structural_coupon/models/transolver/source/run_benchmark.py --cache work/expanded-structural-coupon/cache-initial13-nodal7/manifest.json --out <new-directory> --mode infer --checkpoint outputs/expanded-research/benchmarks/structural_coupon/models/transolver/best_checkpoint.pt --reference outputs/expanded-research/benchmarks/structural_coupon/models/transolver/predictions.npy --require-ready --expected-manifest-sha256 08876879b2b003f8222b831bb51a15c63f8d69d30e6b83e077385f80850169b1
```

로컬 `predictions.npy`, `truth.npy` 및 원 데이터 캐시는 공개 저장소 복사에서 제외한다. 재현에는 FEM 생성 기록과 원 데이터 출처가 필요하다. 이 검증 명령은 저장된 정답과 수치 비교를 하기 위한 것이며 모델 forward 자체는 정답을 입력받지 않는다.
