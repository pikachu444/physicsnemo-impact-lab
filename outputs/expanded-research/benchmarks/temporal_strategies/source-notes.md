# 구조 FEM 시간 예측 전략의 공식 지원 범위

이 문서는 고정 소스 `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`를 직접 읽어 확인한 범위다. 이후 실제 CPU·CUDA 기능 검사와 구조 인장 데이터의 3개 시간 전략 학습·추론을 완료했다. 실행·정확도 증거는 별도 JSON에 보존하며, 마지막 절에 결과를 정리한다.

| 경로 | 원 wrapper의 출력과 학습 | 응력·변형률 범위 |
|---|---|---|
| GeoTransolverTimeConditional | 초기 형상과 미래 프레임 번호를 받아 각 프레임을 독립적으로 계산한다. 학습은 `[N,Fo]`, 추론은 `[N,T,Fo]`다. | 위치 3개 뒤의 추가 채널을 그대로 보존하므로 C7 학습이 가능하다. |
| GeoTransolverAutoregressiveRolloutTraining | 예측 가속도를 속도와 위치에 적분하고, 예측된 상태를 다음 입력에 쓴다. 학습에서도 전체 rollout을 사용한다. | 원 코드는 가속도와 위치 3개 전용이다. 7개 채널을 추가하는 것만으로 응력·변형률 모델이 되지 않는다. |
| GeoTransolverOneStepRollout | 학습에서는 직전 FEM 위치 쌍을 사용하고 추론에서는 예측 위치를 다시 입력한다. | 원 코드의 초기 좌표와 정답 시퀀스 연결 및 가속도 적분은 3개 위치 채널 전용이다. |

공식 근거: [crash rollout.py](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/rollout.py), [drop_test rollout.py](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/drop_test/rollout.py). 두 예제의 클래스 구현은 유사하지만 drop 예제는 time-conditional 추론 중 원본 `node_features` 사전을 복원하고, crash 예제는 `time` 키를 남긴다. 실제 실행 검사는 이를 기록한다.

시간 입력은 물리 시각 자체가 아니라 미래 프레임 인덱스다. 11개 프레임에서 미래 10개를 계산하면 첫 미래 프레임이 0, 마지막이 0.9로 인코딩된다. 학습과 추론의 인덱스는 서로 일치한다. 이를 물리 시각 0~10ms로 해석하면 안 된다. [crash datapipe.py](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/datapipe.py), [time-conditional wrapper](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/rollout.py).

원 AR/one-step YAML은 `dt=0.005`, 스칼라 `initial_vel=9.22`를 제공하며 데이터별 변경을 명시한다. 이 값은 이번 구조 인장 예제의 명목 1ms 간격과 휴지 초기조건에 사용할 수 없다. 원 클래스는 하나의 스칼라 `dt`를 전체 시퀀스에 적용하며 `initial_vel`은 등록된 buffer가 아니다. 따라서 장치 이동 전에 올바른 장치·단위의 초기 속도를 생성해야 한다. [AR 설정](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/conf/model/geotransolver_autoregressive_rollout_training.yaml), [one-step 설정](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/crash/conf/model/geotransolver_one_step_rollout.yaml).

이번 실행은 원 클래스 소스를 변경하지 않고 독립 데이터·학습 드라이버를 사용한다. 모델 크기와 Adam 학습, 경계 마스크, 단위·정규화 설정은 연구용 변경이다. 공식 전체 예제의 설정을 그대로 재현했다고 표현하지 않는다. 출력 성분을 음수에서 자르는 변환을 적용하지 않으므로 부호가 있는 주 변형률을 유지한다. 공식 drop 데이터 파이프의 선택적 `log1p(clamp(min=0))` 경로를 부호가 있는 변형률에 적용해서는 안 된다. [drop datapipe.py](https://github.com/NVIDIA/physicsnemo/blob/11685edb3ec1fb6f25665e96cf7a8c4d11a23f16/examples/structural_mechanics/drop_test/datapipe.py).

이번 시간 모델의 손실은 FEM 정답에 대한 정규화 MSE다. PhysicsNeMo를 사용했다는 사실만으로 균형식, 접촉 부등식, 에너지 보존이나 소성 구성식 제약이 자동 적용되지는 않는다. 이번 학습에는 그러한 물리 잔차·제약 손실이 추가되지 않았다.

생성한 OpenCourant 인장 데이터는 물리 셸 절점 919개, 미래 시각 1~10ms의 10개 프레임, 독립 FEM 조건 13개다. 초기 5개 조건은 학습, 내부 4개 조건은 검증, 별도 내부 4개 조건은 시험이다. C7은 변위 XYZ(mm), FEM von Mises 응력(MPa), 등가 소성변형률, FEM 누적 총변형률의 IP3 면내 주값 2개다. 이는 독립 FEM에서 추출한 정답이며, 위치에서 수치 미분한 변형률이 아니다. 셸의 국소 텐서는 서로 다른 좌표계 사이에서 직접 절점 평균하지 않고 스칼라 불변량을 면적으로 투영했다. 근거와 모든 파일 해시는 생산자의 동결 manifest 및 이 폴더 `source-audit.json`에 보존한다.

실제 애니메이션 출력 시각에는 최대 0.0002ms의 지터가 있다. 원 wrapper 실행은 명목 `dt=1ms`로 같은 프레임끼리 비교하고 보간하지 않는다. 모든 초기 절점의 실제 t=0 변위·속도는 정확히 0이며, t>0에 오른쪽 강체 그립에 계단 형태의 지정 속도가 작용한다. 미래 정답의 차분으로 초기 속도를 만들지 않는다. 원 입력에서 확인한 왼쪽 고정 절점 91개와 오른쪽 이동 그립 절점 91개를 입력 마스크로 제공한다. 응력·변형률과 초기조건의 근거는 원본 덱 자체, 초기 ANIM 확인 결과 및 FEM 생산자의 기록을 따른다.

AR/teacher-forcing 결과가 위치 3개 학습에 성공해도 응력·변형률 예측 성공을 의미하지 않는다. Time-conditional C7과 AR/teacher-forcing C3은 과제가 다르므로 결과를 시간 전략의 통제된 우열로 해석하지 않는다. 세 방법 모두 같은 학습 조건과 변위 공통 척도를 사용하지만, 정적 6개 모델과도 입력 경계 정보 및 좌표 표현이 다르다. 절대 오차와 기준 모델을 함께 보고해야 한다.

## 실제 실행 결과

설치된 PhysicsNeMo 2.2.2의 GeoTransolver는 고정 main YAML에 있는 `activation_checkpointing` 생성자 인자를 받지 않았다. 실제 최초 CPU 실패 기록을 보존했고, 해당 설정 인자만 제거했다. 원 시간 wrapper 클래스는 변경하지 않았으며, AR/one-step 자체의 step checkpointing은 유지했다. 이후 CPU와 CUDA 각각에서 crash/drop의 TC7·AR3·TF3 6개 조합은 실제 forward, backward 및 Adam 가중치 갱신에 성공했다. AR7·TF7 4개 조합은 예상한 7 대 3 차원 불일치로 실제 실패했다. 추론 중 정답의 값을 0으로 바꿔도 지원된 6개 조합의 출력 최대 차이는 0이었다. Drop wrapper를 인장 데이터로 검사한 것은 휴대폰 영역의 학습 증거가 아니다.

세 모델은 별도로 30 epoch를 학습했다. 검증 오차로 checkpoint를 선택한 다음 시험 4조건을 계산했다. 각 모델은 독립 모델이며, crash/drop의 동일 클래스를 중복 검사한 횟수를 모델 수에 더하지 않는다.

| 실제 학습 모델 | Adam 갱신 횟수 | 학습·검증 시간 | 시험 자유 예측 변위 L2 | 변위 성분 RMSE |
|---|---:|---:|---:|---:|
| Time-conditional C7 | 1,500 | 75.84s | 4.3140% | 0.10110mm |
| AR rollout C3 | 150 | 101.39s | 70.7092% | 1.65709mm |
| One-step teacher-forcing C3 | 150 | 98.10s | 74.9891% | 1.75739mm |
| 학습 평균장 기준값 | 학습 없음 | — | 12.9208% | 0.30280mm |

Time-conditional 모델의 시험 응력 RMSE는 **40.202MPa**, 소성변형률 RMSE는 **0.0019087**, 총변형률 major/minor RMSE는 **0.0020331/0.0010399**였다. 응력의 학습 평균장 기준 오차 **26.102MPa**보다 나쁘다. VM 응력 음수 예측 675개, EPSP 음수 예측 1,673개를 그대로 기록했다. 변위 y의 상대 L2는 77.06%이며, 실제 z 변위가 수치 영점 수준인 평면 문제에서 z 예측 RMSE는 0.03574mm였다. 벡터 전체 오차만으로 물리적 정확도가 충분하다고 해석하면 안 된다.

Teacher-forcing 모델은 시험 시점마다 이전 FEM 정답을 공급하는 진단에서는 변위 L2 4.5925%를 보였으나, 실제 자유 rollout은 74.9891%였다. 정답 이력이 필요한 진단을 독립 미래 예측 성능으로 제시하지 않는다. AR/teacher-forcing의 최저 검증 오차는 30 epoch 상한에서 나왔으므로 수렴을 입증하지 못했다. 급격한 t0 이후 지정 속도와 1ms 출력 간격에서 첫 유효 가속도가 크며, 원 통계 방식의 a2~a10 척도에는 첫 a1이 포함되지 않는다는 제한도 남는다.

새 프로세스에서 세 checkpoint를 각각 다시 구성하고 CUDA로 시험 4조건 전체를 재추론했다. 세 경우 모두 모든 출력 성분의 최대 차이가 **0**이었다. 독립 읽기 전용 감사자는 원본 정답 NPZ, checkpoint/예측 해시, 검증 checkpoint 선정과 L2/RMSE/P95/최대 오차를 재계산하여 일치함을 확인했다. 이는 실행·산술 검증이며 제품 충격 성능의 검증을 의미하지 않는다.

실행 결과는 `coupon13/<strategy>/result.json`, 곡선은 `history.json`, 재실행은 `fresh-process-reload.json`에 있다. 원 기능 검사와 실패 기록은 `official-wrapper-probe-*.json`에 있다. FEM와 AI의 장 비교 및 시간별 오차는 두 PNG 그림에 보존한다. 원본 데이터와 checkpoint는 현재 작업 폴더에 별도 보관하며, 공개 저장소에 원본 덱이나 대형 데이터·환경 파일을 함께 넣지 않는다.
