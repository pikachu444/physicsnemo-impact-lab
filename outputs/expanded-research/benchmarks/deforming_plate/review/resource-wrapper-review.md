# Resource wrapper static review

Reviewed `work/expanded-plate/run_resource_probe.py`, SHA256 `4e38b664f09913a200bda30990e15b35394ee98eb04c6853ad9b680542a6b586`, against the current training entry point. Syntax compilation only; no framework import, model construction, simulated output, or GPU operation occurred.

The wrapper rejects measured update counts outside 1–10, including zero, and explicitly forwards defaults hidden width 128, processor blocks 15, checkpoint segments 0, and the measured probe count. The trainer performs `probe + 2` actual Adam updates, excludes two warmups from timing, and returns before training/evaluation. Its current timing includes CPU graph preparation and synchronized forward/backward/Adam. Default versus changed activation checkpointing is recorded separately.

The attempt record covers source hashes/snapshots, exceptions and CUDA OOM classification, traceback, elapsed framework/setup time, sampled RSS every 20 ms, sampled available memory, Windows peak working set/pagefile, PyTorch allocator peaks, and parameters/versions if configuration creation was reached. Import failures are inside the guarded region. Hard process termination cannot execute `finally`; a remaining `running` record must not be interpreted as completion. Sampler RSS is a lower bound, not a continuously measured high-water mark.

Two fixes are required before execution:

1. **Existing-evidence protection is incomplete.** Checking only `resource-attempt.json` allows entry into a directory containing other evidence and overwriting `resource_*.py` before the trainer refuses an existing result. Require a fresh directory, preferably an atomic `mkdir(exist_ok=False)`, or an equally strict exclusive empty-directory contract.
2. **Final metadata reading can mask a failure.** Reading/parsing `config.json` in `finally` is unguarded. A partial or malformed file can prevent final attempt persistence and replace the original exception. Record a metadata-read error inside a guarded block and still persist the final failure record.

These issues were reported to the writer. The static review does not establish a successful resource probe or measured capacity.

## 수정본 재확인

`run_resource_probe.py` 수정본 SHA256은 `f695c2de61309d6583a80109d91ade04ab92f4bc792df150d11df8e33a07dd49`입니다. 기존 출력 폴더를 거부하고 `mkdir(exist_ok=False)`를 사용하므로 앞서 지적한 증거 덮어쓰기 문제가 해결됐습니다. `finally`의 설정 파일 읽기도 `try/except`로 감싸, 읽기 실패를 `config_read_error`에 남기고 최종 기록을 계속 저장하도록 수정됐습니다. 두 지적 사항의 수정은 확인했습니다.

현재 학습 진입점의 probe는 측정 Adam update 수에 2회 준비 update만 더하고, 추가로 graph/forward/output 5회와 미리 만든 GPU 입력의 pure forward 5회를 각각 측정해 앞 2회를 제외합니다. 추가 forward 측정은 optimizer update나 정확도 평가가 아닙니다. 이것은 코드상 확인이며 실제 측정이 아닙니다.

`training_time_guide.py` 초기 검토본 SHA256은 `c4f3a54b27ec404c69f73fa3501ae8c0783e5ceb737f333603e462924d3f17f1`입니다. 완료 상태와 128/15 용량을 확인하고, 32/6/6 분할의 9개 후보(8/16/32 anchor × 5/10/20 epoch) 시간을 제시합니다. 매 epoch의 6×12 teacher validation과 학습 후 validation/test 12건×399단계×2종 추론을 더하는 산식은 일관됩니다. 특정 후보를 선택하지 않고, I/O·재압축 해제·저장·변형률 후처리·페이징 제외와 단일 형상 시간의 추정 한계를 표시합니다.

시간 가이드에는 추가 보강을 권고했습니다. 자체 소스와 설정·probe 파일 해시를 보존하고, attempt와 probe 측정의 일치·측정 수/준비 수·양의 유한 시간·encoder/decoder를 포함한 전체 hidden=128을 확인하는 것이 좋습니다. 출력은 배타적 생성(`open(..., 'x')`)을 쓰면 기존 여부 검사와 쓰기 사이의 경합도 막을 수 있습니다. 이 권고가 반영되기 전까지 위 시간 가이드 평가는 해당 초기 검토본에 한정됩니다.

세 파일은 AST 문법 컴파일만 수행했습니다. 프레임워크 import, 모델 생성, 가짜 측정 파일이나 출력 시뮬레이션, GPU 실행은 하지 않았습니다.

## 추가 권고 구현 확인

최종 읽기 검토본의 SHA256은 wrapper `592d493619a1be138d2ed9cab57ce6cb86d79767d774290f3614049977645456`, 시간 가이드 `2bc9e44fae3ed4d826f7103760ccbf39c39b527d4f11c4b429bc3d79c7ec029a`입니다. wrapper가 실제 `config.json`과 `probe.json` 해시를 attempt에 기록하도록 바뀌었습니다. 시간 가이드는 완료 상태, attempt/probe 측정 내용 일치, 두 파일의 해시 일치, encoder·processor·decoder의 hidden 128과 processor 15, 1–10회 측정 update 및 준비 2회, 유한한 양의 시간, 32/6/6 분할의 내부 중복과 분할 간 중첩 부재를 확인합니다. 출력에는 입력 파일·분할·시간 가이드 자체 소스 해시를 남기고, `open('x')`로 기존 결과 덮어쓰기를 거부합니다.

앞서 전달한 추가 권고의 구현을 코드에서 확인했습니다. 이 마지막 확인은 파일 읽기와 해시 계산만으로 수행했으며 가이드·wrapper·모델을 실행하거나 측정 fixture를 만들지 않았습니다. 실제 128/15 자원 측정 결과나 학습 시간 예측치가 확보됐다는 뜻은 아닙니다. GPU 실측 여부와 실행 순서는 root의 자원 슬롯 결정 사항으로 남아 있습니다.
