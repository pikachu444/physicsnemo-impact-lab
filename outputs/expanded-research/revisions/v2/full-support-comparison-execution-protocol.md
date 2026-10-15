# Full-support 비교 그림 실행 기록과 보정 프로토콜

첫 실행은 2026-10-15 08:31 UTC에 정상 종료했다. 실제 PNG를 열어 Bumper VM 상대L2 축에 숫자 눈금이 없는 문제를 확인했다. 원 소스·프로토콜·PNG/JSON·QA 및 해시는 revisions/v1/에 복사 보존했으며 원 실행 PNG/JSON도 그대로 둔다.

Root의 명시 승인으로 locator/formatter만 보정해 한 번 추가 렌더한다. 모든 저장 지표값·입력·선택·정규화·단위변환·축 범위·배치·모델 호출 수는 그대로다. 로그 눈금은 범위 안의1/2/5 계열에서 최대5개를 숫자로 표시한다. Bumper VM 상대L2의 범위10.0396–62.2041%는 유지하며20·50% 눈금을 표시한다. 이는 축 label QA 보정이며 새로운 모델·데이터 선택 실험이 아니다.

최종 보정 출력은 revisions/v2/full-support-algorithm-comparison.png/json이다. 원 실행 증거 덮어쓰기를 피하기 위해 기계판독 프로토콜에 저장한 stdlib wrapper로 출력 위치만 메모리에서 지정한다. 소스 고정 해시와12개 완료·동결 checkpoint/소스·같은 support/split/time/fields/normalization/mean/truth metadata 검증을 유지한다. 원 입력131개 중 변경은 plot 소스와 본 실행 프로토콜뿐이다.

CPU1, 사전 가용 RAM1.5GiB, 실행 중 host512MiB·peakRSS250MiB 가드를 유지한다. 저장 JSON 스칼라만 읽으며 배열·checkpoint·새 fit·NN 추론·테스트·clipping·metric 재계산은 없다. 전체 plot 렌더는 승인된2회, 모델 호출은0회다. 안전 gate 미달이면 자동 재시도하지 않는다.

Phone은76,065 corner·2 testcase·2–20µs·총변형률 Frobenius/EPSP이고 Bumper는13,675 node·7 testcase·10–100ms·상하층 EPSP다. Nodal/projected VM과 native cell/IP peak를 구분하고 MPa RMSE와 상대L2%를 별도 축으로 표시한다. Training mean은 computed/selfchecked 범위이며 별도12개 NN의 독립 감사와 구분한다. NN은240s bounded budget·일 seed이고 ROM은 POD basis를 공유한다.

보정 PNG도 실제 view_image로 열어12개 축의 숫자 눈금·10방법명·범례·단위·시간창·field 차이·footer 가독성을 확인한 뒤 QA를 기록한다. Root README/Git/Page/coverage는 수정하지 않는다.

보정 실행 완료: 2026-10-15 08:36:25–26 UTC, exit0. 실제 view_image QA PASS이며 모든12축에2–5개 숫자 눈금이 표시된다. 최종 채택 PNG/JSON은 revisions/v2/에 있고 raw-render-record.json에 QA 추가 전 실행 기록도 보존했다. 원 v1 PNG/JSON은 변경하지 않았다. 총 plot2회·모델 호출0회이며 finalcoverage는 갱신하지 않았다.
