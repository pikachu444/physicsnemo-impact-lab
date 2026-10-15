# 유료 GPU 대안 가격 재확인

2026-10-15 07:56 UTC에 제공처 공식 가격 페이지를 다시 열어 확인했다. 계정 연결·인스턴스 생성·예약·결제는 하지 않았다. 이번 실험은 로컬 PC에서 실행하며 추가 유료 GPU 비용은0이다. 전기·기존 장비·기존 구독 비용을0으로 측정했다는 뜻은 아니다.

| 공식 페이지의 표시 구성 | VRAM | 표시 host RAM | GPU 시간당 USD | 4시간 단순 계산 | 20시간 단순 계산 |
|---|---:|---:|---:|---:|---:|
| Runpod Pods RTX A5000 | 24GB | 25GB | 0.27 | 1.08 | 5.40 |
| Runpod Pods RTX A6000 | 48GB | 50GB | 0.59 | 2.36 | 11.80 |
| Runpod Pods RTX4090 | 24GB | 41GB | 0.89 | 3.56 | 17.80 |
| Runpod Pods A100 PCIe | 80GB | 117GB | 1.79 | 7.16 | 35.80 |
| Lambda 1× A6000 | 48GB | 100GiB | 1.09 | 4.36 | 21.80 |

출처: [Runpod 공식 Pods 가격표](https://www.runpod.io/pricing), [Lambda 공식 Instances 가격표의1× 구성](https://lambda.ai/pricing). Runpod 페이지에는2026-10-08 갱신일이 표시되어 있었다. 표의 GB/GiB는 각 제공처 표시를 유지했다. Lambda의 GPU당 다중 GPU 가격을 단일 인스턴스 비용으로 바꾸지 않았다.

위 시간은 수렴 학습 시간을 예측한 값이 아니라 점유 시간 시나리오다. 이 연구의 모델별 로컬 optimizer240초를 다른 GPU의 wall 시간으로 그대로 환산하지 않는다. 설치·다운로드·데이터 준비·validation·추론·관측 비용을 포함한 실제 점유 시간을 따로 측정해야 한다. 재고·지역·선택한 Cloud 종류·세금·네트워크와 추가 저장 비용은 최종 계정 견적에서 확인해야 하며 위 합계에 포함하지 않았다.

Runpod Network Storage Standard의1TB 미만 표시 단가는0.07USD/GB·월이므로100GB의 단순 한 달 계산은7USD다. 이는 로컬 데이터와 설치 환경을 실제로100GB로 측정했다는 뜻이 아니다. Volume Disk는 실행·정지에 따라 요금이 다르다. [공식 저장 요금](https://www.runpod.io/pricing)

실제 메모리 제약은 VRAM과 host RAM 양쪽에서 발생했다. 휴대폰 MGN B5의 CUDA OOM과 FIG B5·별도 다운로드의 host 가용 메모리 가드 중단은 구별해서 기록했다. 향후 유료 환경을 선택할 때 GPU 이름이나 VRAM만으로 결정하지 않고 실제 학습의 host RAM·I/O·추론·장기 rollout 요구도 함께 비교한다. 추가 지출이 필요한 다음 연구는 사용자와 협의한 뒤 진행한다.
