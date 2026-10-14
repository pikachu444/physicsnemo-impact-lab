# Codex Cloud 실제 환경 검사

실행일: 2026년 10월 15일 한국 시간. `physicsnemo-impact-lab` 저장소로 Cloud 환경 설정 세션을 생성하여 실제 명령을 실행했다. 이 기록은 로컬 PC 검사와 구분한다.

| 항목 | 실제 관측 |
|---|---|
| Python | 3.12.14 |
| CPU | 5개 표시, cgroup `400000 100000`으로 4코어 상당 할당 |
| 메모리 | cgroup `memory.max=34359738368`, 32 GiB |
| 디스크 여유 | 첫 검사 약 29.83 GiB |
| NVIDIA 장치 | `/dev/nvidia*` 없음; 확인 가능한 PCI 8개 중 vendor `0x10de` 없음 |
| 도구 | `nvidia-smi`, `nvcc`, PyTorch 없음 |
| GitHub·PyPI | HTTP 200 |
| Hugging Face 첫 검사 | 프록시 HTTP 403 |
| Hugging Face 허용 후 | 공개 API HTTP 200; global feature JSON 307→동일 호스트→200, 15,994 B |

현재 컨테이너에서 NVIDIA GPU 노출이 관측되지 않았다. 이는 도구 미설치만으로 GPU 부재를 추정한 결론과 다르며, 호스트 또는 다른 Cloud 상품의 GPU 유무를 판단한 것은 아니다.

사용자가 `huggingface.co` 허용을 명시 승인한 뒤 이 환경의 추가 허용 도메인에 적용·저장·게시했다. 재검사는 다른 호스트의 리다이렉션을 따라가거나 추가 도메인을 허용하지 않았다. 이 작은 JSON 접근 성공은 대용량 VTP의 CDN 접근까지 검증했다는 뜻이 아니다.

환경 설정에서는 Python 가상환경과 표준 라이브러리 진단만 검증했다. PhysicsNeMo 설치·학습·GPU 추론을 Cloud에서 성공했다고 주장하지 않는다. 이후 GPU 학습은 보유 로컬 GPU의 별도 가상환경을 시험한다. 유료 자원은 생성하지 않았다.

`setup-probe.json`과 `access-gpu-recheck.json`은 Cloud의 실제 명령 출력에서 JSON을 회수한 것이다. `command-evidence.json`과 `access-gpu-command-evidence.json`에 명령·종료상태·출력을 보존했다. `access-gpu-recheck.py`는 원격 검사에서 사용한 소스이며 출력 경로는 당시 세션 기준이다. `cloud-environment.jpg`는 설정과 재검사 결과를 보여주는 화면이다.
