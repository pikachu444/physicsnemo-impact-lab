# Full-support 비교 그림 실행 프로토콜

상태: **준비 완료·그림 미생성**. Static12개 test report가 모두 완료되고 source가 동결된 뒤 root의 CPU window 배정을 받아 한 번만 실행한다. 현재 준비시점 완료 수와 missing경로는 JSON에 보존한다. 이 문서는 실행 승인이 아니다.

그림은 Phone/Bumper 두 행, 여섯 metric축, 모델6+학습평균장+POD/RBF·Ridge·직접RBF의10방법을 배치한다. 로그 metricX축을 사용해 Phone의 큰 범위를 읽을 수 있게 하고, VM RMSE(MPa)와 상대L2(%)는 별도축으로 나눈다. Bumper상·하층VM은 다른 marker로 보여 평균하지 않는다. Phone총변형률Frobenius와EPSP, Bumper상·하층EPSP를 구분한다.

SavedJSON만 읽는다. 모델·배열·checkpoint를 열거나 refit/inference/test/clipping/metric재계산을 하지 않는다. %변환과GPa→MPa×1000은 표시단위 변환뿐이다. Null은0으로 바꾸지 않으며 log축에서 제외해 표시한다. 정확한0도 양수로 clipping하지 않고 text로 표기한다.

실행기는12완료·frozencheckpoint/소스SHA·같은support/split/time/fields/normalization/mean/truth metadata를 확인한다. Root배정flag와 사전고정scriptSHA가 있어야 Matplotlib를 지연 import한다. 1CPU affinity, 사전1.5GiB 가용RAM, 실행중512MiB hostfloor·250MiBpeakRSS를 실제로 감시하고 위반/감시실패는 자기작업만 중단해 별도status를 기록한다. 기존PNG/JSON을 덮어쓰지 않는다.

실행명령과 정확한 sourceSHA는 [기계판독protocol](full-support-comparison-execution-protocol.json)에 있다. 준비 중 `--check-inputs`만 실행할 수 있으며 stdlib만 사용하고 그림을 만들지 않는다. 실제render는 root신호 전 실행하지 않는다.

출력은 full-support-algorithm-comparison.png/json이다. JSON에는 입력hash·각source값/표시단위/축값·실제resource표본·boundedbudget/일seed/ROMsharedbasis 주의를 기록한다. 마지막에는 실제PNG를 `view_image`로 열어 축·범례·단위·글자/색상·null표시 가독성을 확인해야 한다. 그 전에는 `rendered_pending_actual_visual_review`이며 최종시각QA완료로 쓰지 않는다. Nativecellpeak와nodalhead를 섞거나 이 비교를 수렴·제품검증·새독립알고리즘수로 확대하지 않는다.

이번준비에서는 AST·단위/null/zero분기·가짜resourceguard selfcheck와 일부완료보고서의 metadata schema만 검증했다. NumPy/Matplotlib/ML/GPU를 import/실행하지 않았고 rootREADME·Git·Page·suite·active코드는 수정하지 않았다.


독립 정적검토에서 두 줄 축 제목과 domainheading의 겹침 가능성을 찾아 실제render전에 위쪽 여백과Bumperheading 위치를 조정했다. 실제 xlimits·visible logticks·zero/null수를 출력metadata에 기록하도록 보완했으며 sourceSHA를 갱신했다. 실제PNG가 생성된 것은 아니다.
