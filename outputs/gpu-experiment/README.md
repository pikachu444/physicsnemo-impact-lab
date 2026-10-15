# 로컬 GPU 실행 환경 실제 검증

검증일: 2026-10-15, 한국 시간. 기존 전역 CPU PyTorch를 보존하고 프로젝트 전용 가상환경에 CUDA 빌드를 설치했다. 추가 결제·클라우드 인스턴스 생성은 없었다. 이 기록은 **설치 및 실제 GPU 연산 검증**이며, FEM 예측 정확도는 별도 학습 실험에서 평가한다.

## 실제 관측 결과

| 항목 | 결과 |
|---|---|
| OS / Python | Windows 11, Python 3.13.5 |
| GPU / 드라이버 | RTX 3060 Laptop GPU, 6144 MiB, 576.02 |
| 전역 환경 | `2.13.0+cpu`, 기존 사용자 site-packages 경로 유지 |
| 전용 환경 | `work/gpu-experiment/venv`, `--system-site-packages` 사용 |
| 전용 PyTorch / CUDA | `2.13.0+cu126` / CUDA 12.6 |
| PhysicsNeMo | `2.2.2`, Transolver import 성공 |
| CUDA 가용성 | `torch.cuda.is_available() == True` |
| 실제 연산 | 1024×1024 CUDA 텐서 제곱 평균, 역전파, 해석적 gradient 비교 assert 통과 |
| 연산 시간 | 0.4734555 s, 최초 텐서 연산·역전파·동기화 포함 |
| PyTorch 메모리 | 최대 할당 29,360,640 B; 최대 예약 48,234,496 B |
| nvidia-smi | 텐서 생존 중 169 MiB, GPU 사용률 5% 관측 |
| 프로세스 RAM | 테스트 후 RSS 824,283,136 B |
| 당시 시스템 RAM | 총 16,841,400,320 B; 가용 636,166,144 B |

0.473 s는 프레임워크 import, 데이터 전처리, 실제 학습 또는 제품 추론 시간이 아니다. 작은 텐서의 연산 성공과 해당 연산 메모리만 입증한다. `gpu-probe.json`, `probe.log`에 실제 값과 실행 환경 경로를 보존했다.

## 공식 휠 확인·다운로드·설치

[공식 CUDA 12.6 휠 index](https://download.pytorch.org/whl/cu126/torch/)에서 Python 3.13·Windows에 맞는 `torch-2.13.0+cu126-cp313-cp313-win_amd64.whl`을 확인했다. 기존 CPU 설치와 PyTorch 본 버전 2.13.0을 맞췄으므로 이전 버전으로 내릴 필요가 없었다.

- 직접 HEAD: HTTP 200.
- 파일 크기: 2,594,592,768 B.
- 압축 해제 파일 크기 합: 4,171,209,895 B.
- 공식 index SHA256 및 다운로드 파일 해시 일치: `cb91b2f91d053bb91e413459a13ec8b036ff348d03e411e9c29988cc6edb7b32`.
- 실제 다운로드: HTTP 200, 232.449 s, 평균 11,161,979 B/s.
- 로컬 휠 설치: 08:55:02–08:56:26 KST, 약 84.7 s, 종료 코드 0.

원본 index, HEAD, 해시/크기, 다운로드·설치 로그를 같은 폴더에 보존했다. [PyTorch 공식 설치 지침](https://pytorch.org/get-started/locally/), [PhysicsNeMo 공식 설치 지침](https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html)

## 수행한 설치 명령

아래는 프로젝트 루트에서 실행한 명령의 재현용 정리다. 기존 전역 Python 또는 다른 실험 가상환경에 설치하지 않는다. 기본 TEMP 경로의 sandbox 권한 오류가 발생하여 이 프로세스의 TEMP/TMP를 작업 폴더 내부로 바꿨다. 첫 실패와 수정 이유는 `setup-context.json`에 기록했다.

```powershell
New-Item -ItemType Directory -Path 'work/gpu-experiment/tmp' -Force
$env:TEMP = (Resolve-Path 'work/gpu-experiment/tmp').Path
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
python -m venv --system-site-packages work/gpu-experiment/venv

# 아래 두 명령은 CUDA 휠 설치 전에 수행했다.
& 'work/gpu-experiment/venv/Scripts/python.exe' -m pip install --no-cache-dir --no-deps 'nvidia-physicsnemo==2.2.2'
& 'work/gpu-experiment/venv/Scripts/python.exe' -m pip install --no-cache-dir warp-lang einops jaxtyping hydra-core nvtx treelib termcolor cftime importlib-metadata onnx s3fs 'tensordict[zarr]' timm 'h5py>=3.15.1' 'urllib3>=2.7.0'

# 공식 휠을 다운로드하고 index의 SHA256·해제 크기를 확인한 후 실행했다.
& 'work/gpu-experiment/venv/Scripts/python.exe' -m pip install --no-cache-dir --no-compile --no-deps --ignore-installed 'work/gpu-experiment/wheels/torch-2.13.0+cu126-cp313-cp313-win_amd64.whl'
& 'work/gpu-experiment/venv/Scripts/python.exe' -X utf8 'outputs/gpu-experiment/probe_gpu.py' 'outputs/gpu-experiment'
& 'work/gpu-experiment/venv/Scripts/python.exe' -X utf8 -c 'import torch,physicsnemo; from physicsnemo.models.transolver import Transolver; print(torch.__version__,physicsnemo.__version__,torch.cuda.is_available())'
```

이 환경은 전역의 NumPy·VTK·SciPy 등 기존 패키지를 상속한다. 다른 실험의 가상환경 경로나 PYTHONPATH는 재사용하지 않았다. 독립적인 재구축에는 `requirements-local.txt`뿐 아니라 `requirements-with-inherited.txt` 및 Python/플랫폼 정보도 필요하다. 위 명령은 수행 기록이며, 실제 설치 버전은 잠긴 목록을 기준으로 한다. CUDA 휠은 공식 index의 동일 URL·해시를 이용한다.

전역 Torch의 기존 origin과 `version.py`의 `2.13.0+cpu`, `cuda=None`을 다시 읽어 보존 상태를 확인했다. `disk-and-global-preservation.json`에 경로와 결과를 기록했다. CUDA 빌드는 전용 venv의 torch 경로에서 import되는 것을 별도로 확인했다.

## 자원·비용

| 항목 | 관측 또는 산정 |
|---|---:|
| CUDA 휠 다운로드 | 정확히 2,594,592,768 B |
| pip 로그의 추가 다운로드 합 | 약 174,335,000 B |
| 합산 다운로드 | 약 2.77 GB, 로그 반올림값 기반 |
| 전용 venv 파일 크기 | 4,698,797,493 B |
| 보관 중인 휠 | 2,594,592,768 B |
| 작업 하위 폴더 합 | 7,293,390,261 B, 약 7.29 GB |
| 클라우드·추가 결제 | 0 |

pip 로그 합계는 반올림된 전송량이며 빌드 의존성·일부 메타데이터의 정확한 네트워크 누적 계수는 수집하지 않았다. 다운로드 4 GB 및 작업 디스크 10 GB 한도에서 대용량 추가 설치 없이 완료했다. 디스크 합계는 GPU 학습 담당자에게 환경을 인계한 시점의 파일 크기이며, 이후 체크포인트·학습 결과는 별도다. 실험 중 CPU 학습과 설치가 겹쳤을 때 RAM 압박과 응답 지연이 있었고, CPU 학습을 중단한 뒤 GPU 기능 검증을 진행했다.

## 확인된 제한과 다음 단계

`warp-lang==1.18.0`은 CUDA driver 13.0 이상이 필요하다는 경고를 냈고, 로컬 드라이버가 표시하는 CUDA 지원은 12.9였다. **Warp GPU 커널이 동작한다고 확인하지 않았다.** PyTorch CUDA 12.6 연산은 실제 성공했고, Transolver import도 성공했다. Warp 의존 모델을 선택하면 호환 Warp 빌드·드라이버 경로를 따로 검증해야 한다. `triton not found`에 따른 Triton 커널 FLOP 계수 제한도 import 로그에 남았다.

GPU 기능 검증 프로세스는 정상 종료했고 텐서·CUDA 컨텍스트를 해제했다. Transolver import가 성공한 직후 별도 FEM 학습 담당자에게 이 venv와 `--device cuda`, UTF-8 실행 설정을 인계했다. 이후 venv 수정 소유권도 그 담당자에게 넘겼다. 이 문서의 기능 검증 메모리를 실제 모델의 VRAM 요구량으로 사용하면 안 된다.

## 정리 절차 — 실행하지 않음

1. 이 venv를 사용하는 학습·추론 프로세스가 모두 종료되었는지 확인한다.
2. 필요한 모델·데이터·학습 로그를 GitHub 또는 별도 보관처에 보존한다. 이 폴더의 `probe_gpu.py`, 로그, JSON, 환경 목록도 보존한다.
3. 설치 휠 재다운로드가 허용된다면 파일 탐색기에서 프로젝트의 `work/gpu-experiment/wheels`만 삭제하면 약 2.59 GB를 회수한다.
4. 환경 전체가 불필요해지면 절대 경로가 이 프로젝트 아래의 `work/gpu-experiment/venv`임을 확인한 후 해당 폴더만 삭제하면 추가 약 4.70 GB를 회수한다.
5. `outputs/gpu-experiment`와 별도 FEM 실험 결과는 보존한다. 다른 Python 설치·사용자 site-packages·다른 실험 venv는 정리 대상에 포함하지 않는다.

이 정리 절차는 문서화만 했으며 삭제는 수행하지 않았다.
