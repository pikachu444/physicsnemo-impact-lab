# Run224 실행·정렬·오차 검증 기록

2026-10-15. 구현 담당과 별도의 감사 에이전트가 원본 파일과 실제 출력으로 읽기 전용 수치 검증을 수행했다. 추가 학습·GPU 추론 없이 단일 CPU thread로 재계산했다.

## 확인한 증거

- 실제 `run-extended` Starter/Engine 입력 2개의 SHA256가 고정 Run224 manifest와 일치했다.
- source/rerun Engine의 정상종료, 마지막 cycle100000/summary100001, source16threads·rerun2threads와 최종 에너지/질량을 `.out`에서 직접 대조했다.
- source/rerun 양쪽 초기 penetration638개·212nodes 경고와 최종 energy error -0.4%, mass error fraction0.0002304/0.0002315를 확인했다.
- 원본 VTP, 원본 RAD 초기 좌표 및 실제 A011 `NODE_ID`/`Displacement`를 사용해13,675점 매핑·제외 index13675를 독립 재구성했다. 최종 actual time100.001ms, 명목100ms 차이0.001ms와 보간 없음이 기록되어 있다.
- 순수 FEM 비교의 vector RMSE0.012518114490905938mm, component RMSE0.0072273367710710985mm, relative displacement L2 0.00008861547652069487를 독립 재현했다.
- NN 저장 NPZ의 node index hash가 mesh contract와 일치하고 Run224 좌표/truth가 원 VTP와 float32 bitwise 일치했다. 실제10개 VTK frame의 NODE_ID를 다시 정렬해 NN vs새FEM component RMSE4.857321296171668mm, vector RMSE8.413127273655643mm, relative L2 0.06439172891071719와 최종 component RMSE7.666988743646143mm를 독립 재현했다.
- kg/mm/ms 입력과 GPa·J 파생 단위가 일관되고 그림의 단위·FEM/NN 구분·서로 다른 오차 축이 명확함을 확인했다.

## 발견 사항과 수정

1. FEM 비교 메타데이터의 rerun log가 Starter를 가리키던 부분을 Engine으로 수정하고 다시 실행했다. 수치 계산에는 영향이 없었다.
2. NN 비교의 최근접거리 메타데이터가 두 번째 후보 거리까지 포함해 max를 집계하던 부분을 첫 번째 거리 열로 수정하고 다시 실행했다. 실제 대응거리 max는0.000033046267503180924mm이며 차근접 min1.9396634333360867mm와 분리했다. NN 오차 계산에는 영향이 없었다.
3. 공개하지 않는 NPZ에 대한 재현 경로를 보완했다. `--training-dir`과 `--output-dir`을 추가하고 README에 manifest download→별도 폴더 학습 재실행→mesh contract→새 폴더 비교 절차를 명시했다. 이번에는 추가 학습하지 않았고 기존 저장 예측을 사용한 기본 비교만 재실행해 같은 값을 확인했다.

## 공개 정리

공개 텍스트 로그에서 실제 사용자 경로·컴퓨터 이름을 제거했다. 원본은 `work/`에 보존했다. 공개 폴더에서 EXE/DLL/ZIP, RAD/restart/ANIM/VTK 원시 결과, NPZ/checkpoint 및 pycache가 없음을 마지막으로 검사했다. 포함 파일은 로그·정량 JSON·PNG·재현 코드·문서·`.gitignore`다. 모델 출처와 CC BY-NC4.0 조건을 README에 명시했다.

## 남은 적용 한계

원본 변환의 모든 내부 이력과 원 OpenRadioss binary build는 확인하지 못했다. 비교 시점에는0.001ms 차이가 있고 solver build/threads/동시 작업 조건도 다르다. 이 결과는 동일 공개 범퍼1개 조건의 재현과 이미 학습한 NN의 추가 변위 검증이며 실험 정확도, 실제 가전제품, 새 topology, 파손 또는 통제된 solver 성능 우위의 근거로 확대하지 않는다.
