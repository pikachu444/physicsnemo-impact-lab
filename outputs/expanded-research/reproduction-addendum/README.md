# 추가 실행 소스 복원

원본 `../reproduction/`(169 source files / 22 historical profiles)을 수정하지 않고 후속 full-support 정적 학습, 배치 검사, Phone 0.5ms FEM 제어·분석, Plate completion, Temporal V2, 독립 수치 감사의 작은 실행 소스·설정을 보존했다. `source-manifest.json`의 `created_utc`가 관측 시점이다. `profiles`에 있는 완료 실행만 실제 학습과 최종 시험 완료로 분류한다. Plate·Temporal·FEM은 `observations_as_of_capture`를 읽는다. 완료 profile도 정확도·수렴 성공을 뜻하지 않는다.

## 별도 빈 workspace에서 복원

Python 표준 라이브러리만 필요하다. 이 폴더의 `bootstrap.py`는 원본 번들의 안전 복원 코드를 재사용한 독립 실행본이며, 같은 선택 내 복원 목적지가 충돌하면 명시적으로 거부하는 검사만 추가했다. 원본/public bootstrap hash와 변경 내용은 `package-manifest.json`에 기록했다.

```powershell
python outputs/expanded-research/reproduction-addendum/bootstrap.py --verify-only
python outputs/expanded-research/reproduction-addendum/bootstrap.py --list-profiles
python outputs/expanded-research/reproduction-addendum/bootstrap.py --workspace <EMPTY_DIRECTORY> --profile fullsupport-phone-transolver --dry-run
python outputs/expanded-research/reproduction-addendum/bootstrap.py --workspace <EMPTY_DIRECTORY> --profile fullsupport-phone-transolver
```

하나의 빈 폴더에 하나의 profile을 먼저 선택한다. `current`는 관측 시점의 소스 모음이며 완료 실험의 동결 실행본을 대신하지 않는다. 원본 번들과 같은 대상에 연속 복원하면 중복 파일 때문에 거부될 수 있으므로 각각 별도의 빈 폴더를 쓴다. 두 묶음을 합치는 자동 덮어쓰기 기능은 없다. 대상 파일은 내용이 같아도 덮어쓰지 않으며 경로 이탈·symbolic link·junction·환경/데이터 경로를 차단한다.

실행된 `.py`와 `.ps1`은 원본 바이트 그대로다. 모든 복사 경로, 원 위치, 대상 위치, 원본 SHA256, 공개 SHA256, profile 관계는 manifest에 기록한다. JSON 설정의 계정 경로는 `${WORKSPACE}`와 `${USER_HOME}` token으로 가렸으므로 설정은 기록용 template이다. 새 장비에서 해당 token과 로컬 환경/데이터 경로를 명시적으로 바인딩하고, 영향을 받는 설정 hash를 새 실행용으로 재고정해야 한다. 과거의 hash를 변경된 설정에 붙이면 안 된다. 동결 완료 profile은 당시 `source/` snapshot의 hash를 최종 `test_report.json`의 source hash와 맞춘다. `checkpoint-frozen.json`의 model-config/normalization/contract hash는 canonical JSON 객체 hash이며 파일 byte hash와 다를 수 있다. 공식 NVIDIA wrapper의 저작권 header와 `licenses/` Apache-2.0 고지는 유지했다. 코드 라이선스를 FEM 모델·데이터에 적용하지 않는다.

## 보존 범위와 제한

데이터·raw FEM deck·예측 배열·학습 checkpoint·solver binary·가상환경·패키지·비밀값은 포함하지 않는다. `work/`의 원래 상대 구조로 복원하지만 framework 설치·upstream 코드·데이터·solver·재시작 checkpoint는 각 연구 안내에 따라 별도로 확보해야 한다. 이 패키지는 신규 환경 설치부터 학습까지의 독립 재실행을 입증하지 않는다. GPU·framework import·데이터 로드·FEM·학습·추론은 packer/복원 검사에서 실행하지 않는다. FEM 제어 소스가 있어도 실제 0.5ms 종료나 중간 재시작 경계의 bitwise 일치를 뜻하지 않는다.

`bootstrap-acceptance.json`은 source bytes/hash, 모든 Python AST, 빈 임시 폴더 복원, overwrite 거부와 경로 검사를 실제 수행한 결과다. 검증 시점 이후 새 결과는 이 capture에 자동으로 추가되지 않는다.

## 새 결과를 보존할 때

원 workspace에서 `work/expanded-coordination/addendum_pack.py --as-of <UTC_ISO_TIME> --output outputs/expanded-research/reproduction-addendum/captures/<NEW_NAME>`으로 **없는 새 경로**를 선택한다. 같은 입력·같은 `--as-of`는 같은 manifest/바이트를 만든다. 기존 capture와 실험 source는 덮어쓰지 않는다. 완료된 full-support `test_report.json`만 새 profile로 검출하고, snapshot hash가 맞지 않으면 거부한다. 생성 후 `addendum_selfcheck.py --package <NEW_CAPTURE>`를 실행한다. 공개 저장소에서는 보존된 `capture-tool.py`가 원 capture 코드를 설명하며 원 workspace의 입력 없이는 refresh할 수 없다.
