# 중앙 조건 0.5ms FEM 독립 감사

판정: **범위를 한정한 독립 수치검산 PASS**. 완료시각 2026-10-15T07:10:43.951720+00:00, 원결과 동결 2026-10-15T07:01:27.383349+00:00. GPU/ML을 실행하지 않았으며 CPU affinity1개, 최대 RSS 99.22MiB에서 검사했다.

실제 stdout의 NORMAL TERMINATION, exit0, 38680 cycles, engine 3201.10s와 마지막 native ANIM 500.006426591µs를 확인했다. 저장된 원 TH converter CSV를 다시 합쳐 최종 history와 모든 행이 같음을 확인하고, 실제 P/M을 원 RAD wall normal에 투영했다. 마지막 COM normal 속도는 -3.440977807m/s이며 저장 history의 양의 부호 전환은 **없음**다. 이는 contact separation이나 전체 낙하 요구의 완결 판정과 다르다.

Energy는 KE+IE+rotation+spring+contact+hourglass−external work에0.001J 변환을 적용해 다시 계산했다. Plastic work를 IE에 중복 가산하지 않았다. 원 t0 대비 마지막 drift는 -0.151024035%, 마지막 accounted energy는 0.396913458620J다.

중복 시각을 정리한63개 native ANIM에서 cell VM/EPSP scalar와130,218 active flags만 제한적으로 다시 읽었다. 관측 native VM 최대는 84.767242432MPa, 실제 150.016057887µs이며 EPSP의 저장 시점 간 감소는 없었다. 최종·VM피크·strain피크 후보에서 실제 stress/strain4validIP tensor를 chunk로 별도 재계산했다. 다른 모든 시점의 전체 tensor를 재스캔하지 않았으므로 strain 연속시간/전수 native peak 독립 증명으로 확대하지 않는다.


| 직접 확인한 물리량 | 최종500.006427µs | 저장 관측값 |
|---|---:|---|
| Native cell VM | 57.573257446MPa | 63시각 최대 84.767242432MPa / 150.016057887µs |
| Native EPSP | 0.080029912293; plastic326cells | 63시각 최대 0.080029912293, 처음 관측 170.014711330µs |
| Valid-IP total-strain Frobenius | 0.090927016875 | metadata피크 후보 170.014711330µs에서 0.121876074917 직접 재계산 |

최종 corner절점 중4780개가 국소적으로 바깥 방향이지만, 전 조립체의 실제 COM normal 속도는−3.440977807m/s로 안쪽이다. 국소 반향을 조립체 반발로 바꾸어 해석하지 않는다.

원 RAD257,099 node ID와130,218 corner connectivity·17part를 실제 최종 ANIM과 대조했다. 물리 corner76,065개가 native 앞부분 support라는 것도 연결에서 직접 확인했다. 원20µs ANIM과 최종 ID/연결은 동일하다. 모든 원 canonical20µs 동결파일 33개의 실제 SHA를 다시 계산해 보존을 확인했다.

재시작 근거는 분리했다. 100µs→첫500µs의 같은 시각 native field bytes를 다시 비교해 일치했고, **117µs guard stop 경계는 같은 시각 ANIM이 없어 bitwise 일치를 주장하지 않는다**. 최종0.5ms checkpoint의 추가 reload는 **NOT_VERIFIED**다. 첫 시도는 access violation이었고 재시도는 RAM preflight에서 시작하지 못했다. 같은 시각 field 비교를 수행하거나 성공했다고 주장하지 않는다. 이 항목은 메인0.5ms 정상완주와 분리한다. 큰 RST 파일 hash는 worker 동결기록을 인용하고 중복 재독립 hash하지 않았다.

이것은 **단일 중앙 조건의0.5ms FEM**이며0.5ms AI 검증이 아니다. AI9조건 target은 여전히 명목2..20µs다. Native cell scalar VM, IP응력에서 계산한 VM, GPS절점 VM, 투영된 strain의 지지점과 정의를 섞지 않는다. 자세한 직접 계산값·동결 SHA·자원·검산하지 않은 범위는 [감사 JSON](500us-final-review.json)에 있다.
