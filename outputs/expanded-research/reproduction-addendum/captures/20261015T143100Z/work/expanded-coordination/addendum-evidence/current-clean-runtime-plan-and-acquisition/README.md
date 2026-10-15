# 독립 실행 환경 재현

현재 상태는 **새 독립 환경의 90개 고정 runtime 패키지 오프라인 설치, pip check 충돌 0개, core 40개 wheel·RECORD 바이트 검사 통과**다. 최신 결과는 [CPU 구축 완료 기록](CPU-clean-runtime-closure-r3.json)에 있다. Framework import·CUDA 기능·checkpoint validation reload는 아직 실행하지 않았고, 기존 전체 학습을 다시 수행했다는 주장은 하지 않는다. 초기 계획과 중단 기록은 당시 상태 그대로 남겼다.

ANTLR 공식 원본에서 별도 빌드를 한 번 수행했고, 새 wheel SHA256 `815dd98f0ae9191076788662e065b7bf86a7e8c53dc241f4c1ac588197bae439`를 [새 hash lock](requirements-lock-r2.txt)에 묶어 설치했다. [빌드 검증](antlr-build-verification.json)은 승인 원본 소스·새 wheel 코드·RECORD·실행 receipt를 확인한다. 이전 캐시 wheel을 새 빌드 증거로 사용하지 않았다. 실제 설치 metadata는 고정 runtime 90개와 bootstrap용 pip 한 개이며 기존 사용자 앱은 포함하지 않는다.

후속 [여섯 functional 검사 protocol](newfunctional-protocol-v1.json)은 backbone마다 3 Adam updates와 input-only validation/fresh reload만 준비했다. 최종 상태는 `NOT_EXECUTED_RESOURCE_ADMISSION_NOT_MET`이다. 실제 검사 완료 0개·실패한 ML 검사 0개이며 전체 fit·완료 모델 수에 더하지 않는다. CPU 완료 후 10초 간격 관측은 1,976,545,280B와 1,553,977,344B였고 두 번째가 functional 운영 기준 1.5GiB를 충족하지 않았다. 현재 GPU는 배정되지 않았으며 추가 메모리 관측·자동 재시도·ML 실행을 하지 않는다. [고정된 종료 판단](newfunctional-disposition-v1.json)에 두 관측과 source/protocol hash를 보존한다.

`CPU-runtime-evidence-r3/`는 작은 공개용 증거 묶음이다. `evidence-index.json`은 원본과 공개 사본의 경로·크기·SHA256 관계, `source-freeze.json`은 수정하지 않은 코드 바이트를 기록한다. 로그·metadata의 계정 경로만 token으로 가리며 원본 파일은 그대로 보존한다. wheel·환경·배열·checkpoint는 포함하지 않는다. 이 묶음은 기존 addendum capture와 별개이며 전체 학습 재실행을 입증하지 않는다.

기존 `expanded-ml/venv`와 `gpu-experiment/venv`는 모두 전역 패키지를 상속한다. 앞 환경의 `.pth`는 뒤 환경도 참조하고, 실제 NumPy·SciPy 등 일부는 사용자 site에서 읽는다. 새 환경은 이 상속을 사용하지 않고 Python 3.13.5/Windows amd64, Torch 2.10.0+cu126, PhysicsNeMo 2.2.2와 같은 실행 버전을 별도로 설치하는 것이 목표다.

- [설치·검증 계획](plan.json)은 metadata 조사와 실제 환경 검증의 단계를 구분한다. 초기 계획 승인 이후에도 framework/GPU 실행에는 별도 자원 배정이 필요하다.
- [설치 metadata 조사](installed-metadata-inventory.json)는 명시한 네 site의 실제 우선순위와 default Windows 의존성, 필요한 `tensordict[zarr]`를 반영한 90개 패키지를 기록한다. 무관한 사용자 앱은 포함하지 않는다.
- [후보 제약 파일](constraints-installed.txt)은 현재 설치 버전 목록이며 설치 가능한 hash lock이 아니다. 공식 wheel metadata·호환성·파일 hash를 확인한 뒤 별도의 lock을 만든다.
- [캐시 wheel 조사](cached-wheel-inventory.json)는 최초 조사 시점의 hash와 원 URL을 기록한다. 이후 정확한 Torch wheel은 공식 URL에서 새로 받아 SHA256을 확인했다. ANTLR4 4.9.3의 기존 pure-Python wheel은 과거 source archive에서 빌드된 캐시다. 이 한 개에만 공식 source와 고정 build 도구를 사용한 별도 1회 빌드를 수행하고 새 hash로 구별했다.
- [Core source 계약](core-source-contract.json)은 사용 모델과 공통 core/distributed의 40개 파일이 캐시의 PhysicsNeMo 2.2.2 wheel과 정확히 일치함을 확인한다. 공식 예제의 개발 source pin과 설치된 2.2.2 core를 혼동하지 않는다.

CPU 구축은 [R3 admission protocol](CPU-admission-protocol-r3.json)에 따라 CPU 1개/BelowNormal, 후손 합산 RSS 350MiB 제한, 시작 전 가용 RAM 1GiB 이상을 10초 간격으로 두 번 관측하고 disk 15GiB를 확인했다. 실행 중 가용 RAM 512MiB와 disk 5GiB 하한, Windows Job 소유권·후손 회수를 유지했다. R3는 원 R2의 시작 상수 1.5→1GiB 한 곳만 바꿨고 원본은 보존했다. 이 수치는 CPU 작업 운영 여유 기준이며 모델의 최소 RAM 사양이 아니다. 이후 GPU functional admission 1.5GiB는 별도다. build와 pip 임시 파일도 새 작업 영역에 뒀고 기존 환경과 사용자 앱은 변경하지 않았다.

기존 Plate driver는 `work/expanded-plate/deps`를 앞에 넣는다. 후속 검증은 보존한 source를 깨끗한 scratch workspace에 복원해 그 fallback이 없는 조건에서 수행하고, 실제 import 위치가 새 환경으로 한정되는지 확인해야 한다. Python 표준 라이브러리·애플리케이션 source 위치는 별도로 기록한다. `pip check`의 충돌 0개, source/RECORD 일치, 대표 train-only CUDA 기능과 이미 동결된 validation anchor reload를 각각 통과해야 환경 검증 완료로 표시한다. 추가 held-out test는 수행하지 않는다.

큰 wheel·환경·데이터·배열·가중치는 공개 묶음에 넣지 않는다. 공개 저장소에는 source·계획·hash/URL lock·작은 검사 결과만 보존한다.

90개 정확 버전의 공식 metadata와 호환 wheel은 [조회 결과](wheel-resolution.json)에 URL·hash·크기·캐시 출처를 기록했다. [07:38 metadata 검사](metadata-20261015T073809537237Z.json)는 25.03초, 관측 peak RSS 54.43MB로 완료했다. [첫 Torch 다운로드](download-20261015T074014937926Z.json)는 가용 RAM 501,207,040B에서 guard가 중단했고 받은 1,150,287,872B를 보존했다. 이후 별도 CPU 창에서 [전체 취득](download-20261015T083044142375Z.json)을 완료했다. 당시 검증한 90개 wheel은 합계 2,869,977,825B이며 Torch 2,589,881,709B의 SHA256은 `91e21e7ad572bf0136e5b7f192714f120c8abde8e128f1a0759f158951643822`다. 이후 새 ANTLR wheel과 별도 runtime/build 환경을 생성했으며 원 wheelhouse와 부분 파일은 보존했다.

다운로드 코드는 이전에 기록한 강한 ETag, `Range`/`If-Range`, 응답의 ETag·`Content-Range` 시작점·총 크기를 모두 확인한 경우만 이어받는다. 초기 partial에는 ETag receipt가 없어 원 파일을 보존한 별도 이름으로 전체 재다운로드했다. 최종 크기와 SHA256이 맞기 전에는 성공으로 분류하지 않는다. 이어받기 분기들은 작은 가짜 HTTP 응답으로 검사했으며 서버의 실제 Range 재개는 수행하지 않았다. 새로운 clone에서 과거 ANTLR 캐시 wheel이 없으면 그 정확 버전·원 source hash 한 개만 별도 source-build 단계로 넘기고 나머지 89개 검증과 구별한다. 새 wheel hash를 얻기 전 R2 설치는 실행할 수 없다.

ANTLR 공식 source archive는 117,034B, SHA256 `f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b`로 취득했다. [첫 정적 검사](antlr-source-audit.json)는 packaging 선언 3건을 과도하게 차단해 실패했고 당시 build는 시작하지 않았다. 원 source는 수정하지 않고 정확히 고정한 `pygrun` CLI 복사와 Python 3.13에서 비활성인 `typing` marker만 허용하도록 검사기를 수정했다. 후속 [source audit](antlr-source-audit-r2.json)와 독립 검토가 통과한 뒤 별도 build venv의 setuptools 80.9.0/wheel 0.45.1, `--no-index --no-deps --no-build-isolation`, 고정 `SOURCE_DATE_EPOCH=1704067200`/`PYTHONHASHSEED=0`/`TZ=UTC`로 정확히 한 번 빌드했다. 첫 최종화는 Windows 경로 표기 차이를 거부해 중단했고, [경로 identity 수정 검토](finalizer-path-identity-review-r3.json) 뒤 재빌드 없이 최종화했다. 기존 캐시 wheel hash를 새 wheel에 재사용하지 않았으며, wheel 자체는 공개 저장소에 넣지 않는다. 같은 빌드를 두 번 수행한 결정성 검증은 하지 않았다.

[wheel 시작 hook 검사](wheel-startup-hook-audit.json)는 90개 ZIP을 읽고 정확한 setuptools hook 한 개만 허용했으며 package 코드를 실행하지 않았다. [작은 도구 검사](tools-selfcheck-r2.json)는 copy/hash·부분 파일 보존·엄격한 가짜 HTTP 재개·경로/충돌 거부·정확 ANTLR 예외 범위 9개를 통과했다. 이 정적 결과는 실제 clean 환경의 import 위치나 CUDA 기능 통과를 뜻하지 않는다.
