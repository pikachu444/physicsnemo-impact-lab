# 공식 128폭·15블록 Plate 자원 실측

2026-10-15 실제 GPU 실행을 마쳤다. **정확도 학습 완료가 아니라 자원·연산 경로의 실행 검증**이다. 기본 설정이 성공했으므로 activation checkpointing 버전은 반복하지 않았다. 새 최종 평가 6건의 예측이나 확대 본학습은 시작하지 않았다.

| 항목 | 실제 측정값 |
|---|---:|
| 모델 | 설치된 공식 HybridMeshGraphNet, 폭128 / processor15 / checkpoint segments0 |
| 학습 파라미터 | 3,605,636 |
| 학습 geometry | train_0011, frame319, 1,488절점 / 60,298 directed edges |
| Adam update | 준비2회 + 측정3회 |
| 평균 update 시간 | 415.859 ms |
| graph 구성 + forward + 결과 전송 | 160.416 ms / 호출 |
| GPU 입력이 준비된 순수 forward | 136.373 ms / 호출 |
| import·설정·데이터 준비를 포함한 cold wall | 45.404 s |
| Torch tensor allocation peak | 3,176,932,864 bytes |
| Torch allocator reserved peak | 3,498,049,536 bytes |
| Windows peak working set / 측정 RSS peak | 1,644,982,272 bytes |
| 표본 host 여유 메모리 최저 | 268,742,656 bytes |

독립된 첫 cold update의 시간은 따로 보존하지 않았다. 위 cold wall은 Python import·초기화·자료 준비·모든 측정의 합이며 첫 update 지연과 다르다. Windows peak pagefile/commit 값 6,714,429,440 bytes는 물리 RSS가 아니므로 RAM 사용량과 혼동하지 않는다. GPU 할당 수치는 Torch 관측값이며 다른 앱과 CUDA 라이브러리의 모든 메모리를 포함하지 않는다. 20ms는 요청한 RSS 표본 간격이며 실제 스레드 스케줄링 간격은 늘어날 수 있다.

프로세스 PID35608의 종료를 확인했으며 GPU 슬롯을 반환했다. 종료 후 host 여유 RAM은 2,136,461,312 bytes였다. 시스템 단독 성능이라고 주장하지 않는다. 이 슬롯은 root의 CPU Q1 병행이 허용된 상태였고 실제 시간 중첩 여부는 아직 확인하지 않았다.

## 측정 기반 후보 시간

같은 큰 학습 그래프의 측정 시간을 그대로 적용한 자원 proxy다. 일반 자료의 평균 속도나 보장된 상한 시간이 아니다. 매 epoch 내부 validation6건×12개 teacher-forced anchor, 본학습 후 validation6+fresh test6의 각399단계 AR 및 별도 teacher forcing까지 합산했다.

| 32개 학습 궤적의 후보 설정 | Adam 학습 | epoch 중 teacher 검증 | 마지막 전체 검증·평가 | 합계 proxy |
|---|---:|---:|---:|---:|
| 8 anchor × 5 epoch | 8.87분 | 0.96분 | 25.60분 | 35.44분 |
| 16 anchor × 10 epoch | 35.49분 | 1.92분 | 25.60분 | 63.01분 |
| 16 anchor × 20 epoch | 70.97분 | 3.85분 | 25.60분 | 100.43분 |

파일 읽기·압축해제, checkpoint 저장, 변형률 후처리, 그림과 OS paging은 이 proxy에 빠져 있다. 첫 실제 epoch와 validation 곡선으로 cap을 다시 판단해야 한다. 작은 모델에서 추정했던 10~12분을 공식 구조에 그대로 적용할 수 없다. 모델·학습 cap·checkpoint 선택은 training/validation으로 확정하고, 아직 쓰지 않은 source test2..7의 결과로 조정하지 않는다.

근거: `probe128_default/resource-attempt.json`, `probe.json`, `config.json`, `time-guide.json`, `gpu-release.json` 및 원 로그. 실행·설정·보조 코드의 스냅샷과 SHA256도 함께 보존했다. OOM은 발생하지 않았다. 이 결과만으로 공식 구조의 예측 정확도가 확보됐다고 판단할 수 없다.
