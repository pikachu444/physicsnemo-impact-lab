# DP600 구조 인장 coupon: 출처와 FEM field 해석

작성·확인일: 2026-10-15. 이 문서는 공식 예제에서 출발한 소형 구조 FEM pilot의 입력 출처, 단위, 재료 변화, native 정답 field의 의미를 고정한다. 실행 시간·오차·학습 성능은 각 case의 별도 결과 파일을 따른다.

## 원본과 재현 식별자

선정 사례는 Altair **RD-E: 1101 Elasto-plastic Material Law Characterization**의 `1_LAW2/tensile_LAW2`이다. 공식 설명은 DP600 강재의 dogbone 인장, shell mesh, 고정단과 imposed velocity를 사용한다. [공식 예제 설명](https://2022.help.altair.com/2022.2/hwsolvers/rad/topics/solvers/rad/tensile_test_example_law_characterization_r.htm)

원본 다운로드: [RD-E-1101_Tensile_test.zip](https://2022.help.altair.com/2022.2/simulation/demos/hwsolvers/radioss/example/RD-E-1101_Tensile_test.zip). 75,508 bytes이며, 아래 SHA256은 실제 로컬 원본을 읽어 재확인했다.

| 대상 | SHA256 |
| --- | --- |
| 다운로드 ZIP | `b873a56df00aba34f44a80a0fa443e188dd0d0bf3844d274feb4b06494c72564` |
| `1_LAW2/tensile_LAW2_0000.rad` | `36272441339572e8a2d63a7f89cf3a953277ed6ace9c7f6fad89215ae0130ee4` |
| `1_LAW2/tensile_LAW2_0001.rad` | `b85e569d7421eef5b00df7e2a9868aa99a096acd75cdf786003e7bbf45c09054` |

원본 ZIP에서 별도 재배포 허가를 찾지 못했고 공식 문서는 Altair의 저작권과 All Rights Reserved를 표시한다. 따라서 원본 ZIP 및 원본·수정 `.rad` deck은 로컬 `work/expanded-structural-coupon/`에만 보관한다. 이 산출물 묶음에는 취득·준비 코드, 해석 설명과 결과를 두며 원본 deck을 복사하여 재배포하지 않는다. 공개 다운로드 가능 여부가 재배포 허가를 뜻하지는 않는다. 이 판단은 확인한 허가 범위에 관한 보수적 취급이며, 예제에 OpenCourant solver의 AGPL이나 다른 QA deck의 CC 라이선스를 적용한 것이 아니다. [원본 제공·저작권 표시](https://2022.help.altair.com/2022.2/hwsolvers/rad/topics/solvers/rad/tensile_test_example_law_characterization_r.htm)

## Mesh와 단위

원본의 고정 폭 필드를 파싱하면 `/NODE`는 **920개**, 물리 shell은 **808개(4절점 804개 + 3절점 4개)**이다. shell 연결에 사용되는 절점은 919개이며 나머지는 rigid-body 보조 절점이다. **canonical raw-cell C8**의 공간 단위는 원래 element ID로 정렬한 808개 shell element이며, 이 보존 데이터에서는 응력이나 strain을 절점으로 평활화하지 않는다. 아래 C7 primary target은 별도로 scalar invariant의 면적 가중 절점 projection을 사용한다. 이 경우에도 서로 다른 local basis의 tensor 성분 자체를 평균하지 않는다. 숫자 필드 사이에 공백이 없는 행도 있어 공백 기반 정규식으로 node 수를 세면 누락될 수 있다.

원본 `/BEGIN`의 기본 단위는 **kg, mm, ms**이다. 다음은 차원에서 직접 유도한 단위다.

| 물리량 | native 단위 | 변환·의미 |
| --- | --- | --- |
| 길이·시간·질량 | mm, ms, kg | 원본 기본 단위 |
| 속도 | mm/ms | 1 mm/ms = 1 m/s |
| 힘 | kg·mm/ms² = kN | 1 native force = 1000 N |
| 응력·탄성계수 | kg/(mm·ms²) = GPa | 학습 target의 응력에는 ×1000하여 MPa 사용 |
| 에너지 | kg·mm²/ms² = J | kN·mm = J |
| 선형 impulse | kg·mm/ms = kN·ms | ms에 대해 미분하면 kN |
| strain·EPSP | 1 | 무차원 |

공식 설명 페이지는 단위 목록에 힘을 N으로 표기하지만 그 목록은 kg–mm–ms와 차원적으로 일치하지 않는다. 여기서는 **실제 deck의 `/BEGIN`을 기준**으로 kN, GPa, J를 사용한다. 이 문제를 다른 예제의 단위 추정으로 보정하지 않는다.

## 물성 및 연구 민감도 조건

원본 이름은 `Steel_DP600`, `/MAT/PLAS_JOHNS/2`이다. 기본 입력은 밀도 `7.8e-6 kg/mm³`, E `210 GPa`, ν `0.3`, yield `0.3 GPa`, engineering UTS `0.686 GPa`, engineering strain at UTS `0.129`이다. 본 pilot은 source LAW2 기본형을 사용하며 BIQUAD 손상 사례와 구분한다.

`Iflag=1`에서는 입력 두 열을 Johnson–Cook A/B로 읽으면 안 된다. yield, engineering UTS, UTS engineering strain에서 solver가 A, B, n을 계산한다. [LAW2 공식 정의](https://2022.help.altair.com/2022.2/hwsolvers/rad/topics/solvers/rad/mat_law2_plas_johns_starter_r.htm), [Iflag=1 설명](https://2022.help.altair.com/2022/hwsolvers/rad/topics/solvers/rad/explicit_structural_fea_elasto_plastic_c.htm)

준비 코드는 strength scale `s`를 yield와 UTS에 함께 곱하고 E, ν, 밀도, UTS strain은 유지한다. **각 case에서 solver가 A/B/n을 다시 계산하므로 A와 B가 모두 정확히 s배이고 n이 일정하다고 가정하지 않는다.** 각 starter report에 출력되는 실제 계수를 결과 metadata에 보존한다. base starter에서 확인된 값은 A `0.3000000000000 GPa`, B `0.7205175297228 GPa`, n `0.1980449212604`이다.

4-corner 조건 `s∈{0.9,1.1}`, velocity scale `v∈{0.75,1.25}`는 **알고리즘 검증을 위한 국소 연구 민감도 가정**이다. 측정된 DP600 재료 산포, 제조 공차, 실물 허용 범위 또는 실험으로 보정된 확률분포가 아니다. `v`는 imposed velocity의 원본 상수 함수에 곱한다. 원본은 rate coefficient C=0이므로 속도 변화의 응답을 실험적으로 검증된 DP600 strain-rate 물성으로 설명하지 않는다. 원본 40 ms 중 첫 10 ms를 사용하는 pilot이며, 실제 frame 시간은 각 ANIM 파일의 값을 보존한다.

## Native total strain, 좌표계와 적분점

본 설정은 QEPH shell `I_shell=24`, 두께 2.5 mm, 두께 방향 적분점 5개이다. 준비 시 `Istrain=1`을 명시한다. pilot starter report에서도 `POST PROCESSING STRAIN FLAG=1`, `SMALL STRAIN FLAG=2`, `SHELL FORMULATION FLAG=24`를 확인했다. `I_smstr=2`는 full geometric nonlinearities 설정이다. 공식 문서는 small-strain 옵션 1/3과 구분하여 나머지의 응력·strain을 true 값으로 설명한다. [SHELL property 및 comment 4](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/prop_type1_shell_starter_r.htm)

따라서 `total_strain_*_local`은 **solver가 출력한 native 누적 total strain**이다. displacement로 다시 계산한 small strain, engineering gauge strain, EPSP와 섞지 않는다. 이 pilot에서 해당 누적 tensor가 특정한 finite-strain measure와 정확히 일치한다는 독립 증명은 수행하지 않았으므로 Hencky, Green–Lagrange 등의 이름을 추가하지 않는다.

요청은 `/ANIM/SHELL/TENS/STRESS/3`, `/ANIM/SHELL/TENS/STRAIN/3`, `/ANIM/SHELL/EPSP/3`이다. 여기서 `/3`은 **5개 중 세 번째 두께 적분점(IP3)**이다. 좌표계 번호나 tensor 성분 번호가 아니다. 5점 shell의 가운데 위치를 선택한 것이며 UPPER/LOWER 표면 외삽 출력과 구분한다. [ANIM tensor 위치 정의](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/anim_shell_tens_engine_r.htm)

응력·strain은 **element local co-rotational basis**로 보존한다. 즉 초기 mesh의 고정 전역 X/Y 성분으로 주장하지 않는다. 공식 출력 문서는 element 좌표계를 명시하며, 좌표계 설명과 solver의 local→global 변환 경로도 변형 geometry에 따른 회전 해석을 뒷받침한다. [좌표계 설명](https://2024.help.altair.com/2024/hwsolvers/rad/topics/solvers/rad/radioss_coordinate_system_intro_c.htm), [SHLROTG 코드](https://github.com/OpenCourant/OpenCourant/blob/main/engine/source/output/anim/generate/tensor6.F#L4650). 정확한 global 재구성은 `/SHFRA`·`ISHFRAM`·`INVERT` 등의 convention을 따라야 하므로 이번 추출에는 포함하지 않는다.

## Shear와 2D VTK 확장

저장하는 strain XY는 **tensor shear `εxy = γxy/2`**이다. solver의 ANIM 생성 코드에서 XY에 `HALF`를 곱한 다음 세 번째 tensor 성분에 넣는다. membrane과 layer 경로 모두 같은 half-factor를 적용한다. [tensorc.F, 확인한 revision](https://github.com/OpenCourant/OpenCourant/blob/3c2bfe8636ce556c0aac1ee830c04e70882cedcb/engine/source/output/anim/generate/tensorc.F#L1377)

공식 `anim_to_vtk`는 raw 2D `[xx,yy,xy]`를 아래 대칭 행렬로 확장한다. 추가 shear scaling 또는 좌표 회전은 하지 않는다. 읽은 skew 배열도 이 tensor 출력 계산에 쓰지 않는다. [3값 읽기](https://github.com/OpenCourant/Tools/blob/4e52942e191d3b1ede4b320fb0f1780f4e41b59a/output_converters/anim_to_vtk/src/anim_to_vtk.cpp#L451), [VTK 확장 코드](https://github.com/OpenCourant/Tools/blob/4e52942e191d3b1ede4b320fb0f1780f4e41b59a/output_converters/anim_to_vtk/src/anim_to_vtk.cpp#L1364)

```text
[[xx, xy, 0],
 [xy, yy, 0],
 [ 0,  0, 0]]
```

따라서 추출 시 XY에 다시 2 또는 1/2을 곱하지 않는다. **변환기가 채운 zz/yz/zx의 0은 측정된 FEM field가 아니므로 물리 정답 label로 사용하지 않는다.** 특히 strain zz=0이라고 결론 내리지 않는다. element-native 보존 target은 stress xx/yy/xy, total strain xx/yy/xy, equivalent plastic strain, 별도 native von-Mises shell scalar의 8채널이다. native von-Mises scalar의 출력 정의는 IP3 stress tensor invariant와 구별하여 보존하며 두 값의 완전 일치를 전제하지 않는다.

사용 중인 bundled `anim_to_vtk_win64.exe` SHA256은 `238278b91341168f517eb1547efdb249705af8080cd1f0beca0ba61d3bc65e12`이다. 위 설명은 공식 소스를 검토한 결과이며, 이 EXE의 정확한 빌드 커밋과 그 소스의 일치는 별도로 입증하지 않았다.

## Nodal displacement와 C7 primary target

`/ANIM/VECT/DISP`에서 읽는 919개 물리 절점의 `(uX,uY,uZ)`는 **global Cartesian basis의 raw nodal FEM displacement**, 단위 mm로 취급한다. 공식 [ANIM/VECT 문서](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/anim_vect_engine_r.htm)는 DISP를 displacement로 정의하며 frame ID를 받는 형식을 제시하지 않는다. 다만 이 페이지에 DISP의 global basis를 직접 명시한 문장은 없다. 이 basis 판단은 공식 solver의 [GENANI displacement 경로](https://github.com/OpenCourant/OpenCourant/blob/3c2bfe8636ce556c0aac1ee830c04e70882cedcb/engine/source/output/anim/generate/genani.F#L3562)가 D 배열을 VELVEC에 전달하고, [VELVEC의 원절점 경로](https://github.com/OpenCourant/OpenCourant/blob/3c2bfe8636ce556c0aac1ee830c04e70882cedcb/engine/source/output/anim/generate/velvec.F#L70)가 세 성분을 frame 투영 없이 쓰는 코드 근거를 함께 사용한다. 추출 검증에서는 물리 절점의 `current_coordinate - initial_coordinate`와 이 displacement의 일치를 확인한다.

원본 `/FRAME/MOV/1`은 node 102/616/148로 정의되고 `/TH/NODE/2`의 node 102와 616 history가 명시적으로 frame 1을 참조한다. 이 **time-history gauge의 상대변위**와 animation의 global nodal displacement를 구분한다. 공식 [TH/NODE 정의 및 comments 3–4](https://www.2024.help.altair.com/2024/hwsolvers/rad/topics/solvers/rad/th_node_starter_r.htm)는 기본 global 출력, skew 지정 시 투영, frame 지정 시 상대운동 출력을 구분한다. FRAME 카드의 존재만으로 ANIM DISP 전체가 moving-frame 출력이 되는 것은 아니다.

C7 primary target은 `uX,uY,uZ` 세 raw nodal 채널과 **면적 가중 element→node projection으로 얻는 네 scalar 채널**의 joint target이다. 네 채널은 native shell von-Mises scalar, IP3 EPSP, IP3 in-plane total-strain principal major/minor에서 출발한다. strain의 두 principal 값은 각 element에서 `[[εxx,εxy],[εxy,εyy]]`의 고유값을 먼저 계산하고 큰 값/작은 값으로 정렬한 다음 scalar로 projection한다. 서로 다른 co-rotating local basis의 tensor 성분 자체를 평균하지 않는다. projection 가중치·연결 관계·기준 geometry·각 채널 단위는 실제 추출 metadata에 보존해야 한다.

면적 projection을 거친 네 값은 **FEM에서 유래한 nodal projected scalar**이며 solver가 직접 출력한 raw nodal stress/strain/EPSP라고 부르지 않는다. C7 전체를 일괄적으로 raw nodal solver fields라고 표시하지 않는다. 또한 in-plane principal minor를 padding된 3×3 tensor의 최소 고유값으로 대체하지 않는다. 2D converter의 zz=0은 물리 principal strain이 아니기 때문이다.

## T01 impulse와 force–extension 진단

`/TH/RBODY`의 T01 계열은 힘 자체가 아니라 **linear/rotational impulse**를 저장한다. HyperGraph는 읽을 때 미분하여 force/moment를 계산하고, 일반 ASCII 추출에는 사용자가 미분해야 한다. ABF의 직접 force와는 구별한다. [공식 TH/RBODY comments 3–5](https://help.altair.com/hwsolvers/rad/topics/solvers/rad/th_rbody_starter_r.htm)

현재 진단 곡선은 RBODY X impulse를 **실제 time(ms)에 대해 수치 미분한 후 부호를 반전**하여 인장 reaction(kN)으로 표시한다. 필터는 적용하지 않는다. 미분은 끝점·초기값·시간 샘플 간격에 민감하므로 이것은 force–extension **진단 곡선**이며 직접 native force label이 아니다. 공식 문서도 impulse 미분으로 time=0의 force가 0이 아닐 수 있음을 설명한다.

Gauge extension은 원본 moving-frame 설정의 node 616 output을 사용하고 초기 gauge length는 node 102–616 사이 **80 mm**이다. engineering gauge strain은 extension/80, engineering stress 진단값은 force(kN)×1000/초기면적(12×2.5=30 mm²)으로 MPa를 구한다. 이것들은 element의 native true/local tensor와 구분하며 학습용 total strain을 대체하지 않는다. [공식 gauge·engineering/true 비교 설명](https://2022.help.altair.com/2022.2/hwsolvers/rad/topics/solvers/rad/tensile_test_example_law_characterization_r.htm)

