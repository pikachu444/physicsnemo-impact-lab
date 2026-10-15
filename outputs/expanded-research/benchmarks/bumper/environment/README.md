# 확장 벤치마크 격리 실행환경 설치 기록

2026-10-15에 별도 Python 3.13.5 가상환경을 구성했다. 이 기록의 검증 범위는 **설치, wheel 무결성, 패키지 메타데이터와 경로 우선순위**다. 이 설치 작업에서는 Torch·PhysicsNeMo를 import하거나 CUDA 연산·모델 forward·학습을 실행하지 않았다. 실행 가능성과 정확도는 별도 실험 결과로 판단해야 한다.

| 패키지 | 선택 버전 | 위치 |
|---|---|---|
| torch | 2.10.0+cu126 | 새 venv |
| torchvision | 0.25.0+cu126 | 새 venv |
| torch_scatter | 2.1.2+pt210cu126 | 새 venv |
| torch-geometric | 2.8.1 | 새 venv |
| xxhash | 4.0.1 | 새 venv |
| nvidia-physicsnemo | 2.2.2 | 기존 연구 venv에서 재사용 |

실행 파일은 작업 공간의 `work/expanded-ml/venv/Scripts/python.exe`다. `--system-site-packages`로 만든 새 venv의 `.pth` 파일에 기존 `work/gpu-experiment/venv/Lib/site-packages`를 추가했다. 메타데이터와 module spec 검사에서 Torch·vision·scatter·PyG·xxhash는 모두 새 venv를 우선했고, PhysicsNeMo는 기존 연구 venv에서 찾았다. 기존 Torch 2.13.0+cu126 메타데이터도 그대로 남아 있다. 새 venv는 기존 연구 venv와 시스템 패키지에 의존하므로 폴더 하나만 복사하면 재현되는 독립 배포물이 아니다.

공식 [PyTorch cu126 index](https://download.pytorch.org/whl/cu126)와 [PyG Torch 2.10/cu126 index](https://data.pyg.org/whl/torch-2.10.0+cu126.html)에서 CPython 3.13 Windows x64 wheel을 받았다. PyG와 xxhash는 공식 PyPI 배포물을 사용했다. Torch·vision은 공식 index의 SHA256와 일치했고, PyG·xxhash는 실시간 공식 PyPI 메타데이터의 SHA256·바이트 수와 일치했다. scatter는 공식 index에 공개 hash가 없으므로 받은 파일의 SHA256만 기록했다. 다섯 wheel의 정확 URL·크기·SHA256·호환 태그는 `installation-manifest.json`에 있다.

요청된 PyG 2.9.0은 실제 pip에서도 배포본을 찾지 못했다. 2026-10-15 02:47 UTC에 읽은 공식 PyPI JSON의 `releases`에도 2.9.0은 없었으며 당시 최신 배포본은 2.8.1이었다. 이 근거로 2.8.1을 고정했다. 실패한 2.9.0 설치 시도와 초기 sandbox DNS 실패 로그도 보존했다.

선택한 여섯 패키지의 활성 기본 의존성 검사에는 누락이나 버전 위반이 없었다. 전체 `pip check`는 상속된 다른 프로그램 패키지의 기존 문제 세 건 때문에 exit 1이었다: `questionary`의 `prompt-toolkit` 누락, `google-genai`와 `websockets 17.0.1`의 제약 불일치, `mixpanel`과 `pydantic 2.13.3`의 제약 불일치. 이 패키지들은 이번 설치에서 변경하지 않았다. 전체 venv가 모든 프로그램에 대해 일관된 환경이라고 주장하지 않는다.

설치 명령은 기존 환경의 제거·수정을 피하도록 `--no-deps --ignore-installed`를 사용했다. `installation-manifest.json`에는 명령 인수 배열·시작/종료 시각·exit code와 이전 실패 시도 기록이 있다. `selected-package-versions.txt`는 선택 버전, `package-inventory.json`은 상속된 패키지를 포함한 메타데이터 목록이다. `install-*.log`는 설치·다운로드·검사 원문 로그다. wheel·venv·임시 폴더는 이 결과 폴더에 복사하지 않았다.

`installation-script.ps1.txt`와 `metadata-audit.py.txt`는 실행 코드의 기록용 사본이다. 원본 스크립트는 `work/expanded-ml`의 venv·wheel·임시 폴더와 기존 연구 venv를 전제로 실행됐다. 다른 위치에서 재현할 때는 이 경로 관계와 기본 의존성을 먼저 구성해야 한다.
