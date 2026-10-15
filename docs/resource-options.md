# 실행 환경·GPU 자원 대안

조사 기준: 2026-10-15, 한국 시간. 이 문서는 공개 공식 문서·가격표 확인 결과와 실험 제안을 구분한다. 유료 GPU 인스턴스를 만들거나 결제하지 않았다. 아래 시간은 학습 완료 시간의 예측이 아니라 비용을 비교하기 위한 GPU 점유 시간 시나리오다.

**실행 결과에 따른 보완:** 이후 별도 실제 실험에서 보유 RTX3060 Laptop6GB로 공개 범퍼13,675절점의 Transolver 학습·평가를 완료했다. 60epoch260.53초, Torch 최대 할당383.35MB로, 이 작은 네트워크·현재 데이터 실험을 위해 유료GPU가 필수라는 근거는 없다. 큰 조립체나 다른 모델의 필요 자원은 별도 측정해야 한다. 아래 가격표는 후속 대안 비교다. [실제 실험](experiment-results.md), [로컬 GPU 환경 기록](../outputs/gpu-experiment/README.md)

## 1. Codex Cloud: 문서 확인과 실측은 별개

공식 OpenAI documentation의 현재 기본 VM 사양은 다음과 같다.

| 구독 | vCPU | RAM | 디스크 |
|---|---:|---:|---:|
| Plus, Edu Plus | 2 | 8 GiB | 8 GiB |
| Pro, Business, Enterprise, Edu, Edu Pro | 4 | 16 GiB | 32 GiB |

GPU/CUDA가 위 기본 사양 표에 명시되어 있지 않다. 따라서 **GPU 제공이 문서로 보장되지 않는다**고 판단한다. 문서 표만으로 실제 VM에 GPU가 없음을 관측했다고 기록할 수는 없다. 이후 별도로 만든 Cloud 환경에서는 `/dev/nvidia*`와 NVIDIA PCI 장치가 노출되지 않음을 실제 확인했다. 이는 해당 컨테이너의 측정이며 모든 Cloud 호스트의 GPU 유무를 단정하지 않는다. Enterprise의 더 큰 VM·맞춤 사양은 별도 문의 대상이다. [Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environments), [해당 환경의 실측](../results/cloud/README.md)

클라우드 환경은 저장소를 선택해 준비·검사·Publish한 뒤 새 작업에서 사용한다. 인터넷은 환경별 설정 대상이며, Package managers 프리셋에는 PyPI, PyTorch, GitHub 소스·릴리스 호스트와 NVIDIA 패키지 호스트 등이 포함된다. 그 외 데이터 서버·리다이렉션 호스트는 별도 허용이 필요할 수 있다. 저장된 VM 상태는 소스 관리의 대체물이 아니다. [환경 및 네트워크 설정](https://learn.chatgpt.com/docs/environments/cloud-environments)

**실행 상태:** GitHub 저장소를 연결해 Cloud 환경을 준비·게시했고 CPU·RAM·디스크·GPU 장치·네트워크를 검사했다. 이 컨테이너에는 PyTorch와 NVIDIA GPU가 없어 Cloud GPU 텐서 연산·학습은 수행하지 않았다. 사용자 로컬의 별도 CUDA 환경에서 GPU 텐서 역전파와 실제 공개 FEM 학습·평가를 완료했다. 후속 환경에서도 드라이버 인식·GPU 연산·학습·오차 검증을 각각 확인한다.

## 2. 유료 GPU 대안: 공식 공개 단가

단위는 USD이며 세금·환율·추가 디스크·네트워크/egress·체크포인트 보관 비용은 아래 GPU 시간 합계에 포함하지 않았다. 로그인 후 지역·재고·Cloud 종류·최종 견적을 다시 확인해야 한다. GPU 이용률이 낮아도 VM을 점유한 동안 비용이 발생할 수 있으므로, 데이터 준비와 분석 시간을 포함한 점유 시간을 예산으로 잡는다.

| 제공처·표시 GPU | VRAM | 공식 표시 단가 / GPU·h | 4 h 시나리오 | 20 h 시나리오 |
|---|---:|---:|---:|---:|
| Runpod Pods, RTX A5000 | 24 GB | $0.27 | $1.08 | $5.40 |
| Runpod Pods, RTX A6000 | 48 GB | $0.59 | $2.36 | $11.80 |
| Runpod Pods, RTX 4090 | 24 GB | $0.89 | $3.56 | $17.80 |
| Runpod Pods, A100 PCIe | 80 GB | $1.79 | $7.16 | $35.80 |
| Lambda, 1× A6000 | 48 GB | $1.09 | $4.36 | $21.80 |
| Lambda, 1× A100 PCIe/SXM | 40 GB | $1.99 | $7.96 | $39.80 |
| Lambda, 1× H100 PCIe | 80 GB | $3.29 | $13.16 | $65.80 |

Runpod 행은 가격 페이지의 **Pods** 표에 표시된 값이다. 페이지의 Cloud 선택과 공급 현황에 따라 실제 견적을 재확인한다. Network Storage Standard는 1 TB 미만에서 $0.07/GB·월로 표시되어, 100 GB를 한 달 보존하면 단순 계산 $7이다. Volume Disk는 실행/정지 상태별 단가가 다르다. 과거 검색 자료의 RTX 4090 $0.34/h를 현재 확정 단가로 쓰지 않았다. [Runpod 공식 가격표](https://www.runpod.io/pricing)

Lambda 행은 다중 GPU 표의 GPU당 가격을 단일 GPU 인스턴스 비용으로 혼동하지 않도록 **1× 표**에서 추렸다. A6000 1× 표는 RAM 100 GiB, SSD 512 GiB를 명시한다. [Lambda 공식 가격표](https://lambda.ai/pricing)

**제안:** 첫 유료 후보는 24–48 GB 단일 GPU이며, 실제 예제의 메모리 관측 결과를 보고 선정한다. A6000 48 GB의 공개 단가는 작은 메모리의 상위 연산 성능 GPU보다 낮을 수 있으므로, 카드 이름만으로 선택하지 않는다. 4 h를 설치/데이터 로딩/짧은 실제 학습/추론 측정 예산으로, 20 h를 후속 비교 실험 예산으로 협의할 수 있다. 이 숫자로 특정 예제의 수렴·정확도를 보장하지 않는다. 사용자 승인 전 리소스 생성·선결제·과금은 진행하지 않는다.

## 3. Colab은 추가 탐색 후보

공식 FAQ는 GPU 종류·사용 제한·가동 시간이 가변적이며 무료/유료 자원 모두 제공이 보장되지 않는다고 설명한다. 무료 실행은 사용 패턴·가용성에 따라 최대 12 h, Pro+는 충분한 compute units가 있으면 연속 코드 실행 최대 24 h로 설명한다. 따라서 고정 GPU·고정 가동 시간의 재현 실험을 계획할 때는 재개 가능한 체크포인트가 필요하다. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)

지역별 유료 가입 페이지는 이번 텍스트 조회에서 세부 요금이 노출되지 않았다. 정확한 KRW 구독료나 GPU별 compute-unit/h를 확인한 것으로 주장하지 않는다. 계정에서 GPU와 단위 소모율을 확인하기 전에는 4 h/20 h를 확정 금액으로 환산하지 않는다. 기존 유료 구독이나 compute units가 있다면 추가 결제 없이 가능한 범위를 확인할 가치가 있다. [Colab 유료 요금 페이지](https://colab.research.google.com/signup)

## 4. 로컬 RTX 3060 Laptop 6 GiB 활용 경로

초기 환경 검사에서 전달받은 로컬 사양은 RTX3060 Laptop6GiB·시스템 RAM약15.7GiB·전역 CPU용 PyTorch였다. 이후 기존 전역 환경을 보존한 별도 venv에서 CUDA PyTorch를 설치하고 실제 역전파·공개 FEM 학습을 완료했다. 초기 CPU 상태와 최종 GPU 실행 상태를 구분한다. [최종 GPU 실행 기록](../outputs/gpu-experiment/README.md)

최신 PhysicsNeMo 공식 요구사항은 Windows(native/WSL), Ubuntu 24.04와 RTX 30xx를 지원 대상으로 명시한다. 그러므로 Windows라는 이유만으로 실행 불가능하다고 단정할 근거는 없다. [PhysicsNeMo 요구사항](https://docs.nvidia.com/physicsnemo/latest/getting-started/system_requirements.html)

아래는 적용한 순서와 추가 대안이다. 현재 선택한 네이티브 Windows 경로는 실제 학습까지 완료했다.

1. 기존 Python 환경을 보존하고 프로젝트 전용 가상환경을 만든다.
2. PyTorch 공식 선택기에서 Windows·pip·호환 CUDA 빌드를 골라 설치한다. GPU 텐서 연산과 `torch.cuda.is_available()`을 확인한다. 설치 버전·드라이버·`torch.version.cuda`를 함께 기록한다. [PyTorch 공식 설치·검사](https://pytorch.org/get-started/locally/)
3. 그 환경에 PhysicsNeMo와 선택한 예제의 추가 의존성을 설치한다. 공식 문서는 PyTorch를 먼저 설치한 뒤 PhysicsNeMo를 설치하는 pip/uv 경로와, 의존성을 포함한 NGC 컨테이너 경로를 제공한다. 예제별 추가 패키지는 별도 확인 대상이다. [PhysicsNeMo 설치](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html)
4. 네이티브 Windows 추가 의존성에서 문제가 발생하면 WSL 2를 비교한다. NVIDIA는 Windows 드라이버를 WSL에서 공유하며 WSL 내부 Linux 디스플레이 드라이버를 설치하지 않도록 명시한다. WSL 설정·재부팅이 필요한 경우는 별도 사용자 작업으로 설명한다. [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

**실측과 한계:** 보유6GiB에서 현재 공개 범퍼·작은 batch의 실제 GPU 학습·평가를 완료했다. 그러나 공식 crash/drop 전체 구성, 대규모 조립체 또는 장기 rollout이 들어간다는 보장은 없다. 메모리는 절점 수뿐 아니라 간선 수, 상태 채널, 은닉 폭, 층 수, 배치, 시간창, 정밀도, 활성값 저장 방식에 좌우된다. 작은 성공 사례를 전체 제품 학습 가능성의 증거로 확대하지 않는다.

## 5. 자원 산정과 실행 기록 방법

다음은 특정 데이터셋 실측이 아닌 크기 계산 예시다. `float32` 변위 3성분을 10만 절점 × 100시점에 저장하면 출력 배열 하나만 약 120 MB(십진)다. 동일한 100조건은 약 12 GB다. 응력·변형률·속도·가속도·연결 정보·원본 결과·전처리 사본·체크포인트는 여기에 더해진다. 따라서 Codex Cloud의 8/32 GiB 디스크를 실제 데이터 전체 수용 가능성으로 가정하면 안 된다.

학습 GPU 메모리는 이 원본 배열 크기와 같지 않다. 설치 완료 뒤 실제 선택 예제의 설정에서 다음을 측정해 확대 여부를 정한다.

| 측정 대상 | 보존할 기록 |
|---|---|
| 설치 | OS·드라이버·GPU·프레임워크·잠긴 의존성·커밋 SHA |
| 데이터 | 원본 URL·라이선스·해시·케이스/절점/간선/시점 수·분할 기준 |
| 학습 | warm-up 이후 step 시간·epoch 시간·총 점유 시간·최대 GPU 메모리·프로세스 RAM |
| 추론 | GPU 동기화를 포함한 시간·전처리/로딩 포함 시간·배치·정밀도 |
| 정확도 | 독립 FEM 케이스의 필드 오차·최댓값·시간 이력·에너지/접촉/파손 관련 지표 |
| 비용 | GPU 점유 시간 × 최종 단가 + 저장/전송/세금, 미확인 항목 표시 |

**제안:** 설치 성공 → GPU 기능 검사 → 실제 FEM 데이터 짧은 학습 → 독립 케이스 추론 → 수렴 학습/오차 비교를 별도 상태로 기록한다. 초기에 10–50 step을 측정해 메모리와 처리량을 확인할 수 있지만, 그 짧은 실행은 수렴 또는 예측 정확도 검증의 대체가 아니다. 노드·시점을 무작위로만 나누면 동일 해석 이력이 train/test 양쪽에 섞일 수 있으므로, 설계·낙하 조건 단위 독립 검증을 우선한다.
