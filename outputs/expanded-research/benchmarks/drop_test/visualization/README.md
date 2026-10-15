# 휴대폰 전체 corner FEM의 3D 후처리 파일

다음 두 고정 test 조건의 **명목20µs FEM 정답만** 실제 VTU로 저장했다. 이 파일에는 AI 예측값이나 가상 오차값이 없다.

- `phone_e090_rxp050_t20us_FEM.vtu`
- `phone_e110_rxm050_t20us_FEM.vtu`

각13,759,876bytes. 전체76,065개 physical solid corner NODE_ID 순서와130,218개 cell 순서를 유지한다. 원257,099개 NODE 중181,034개 TETRA10 midside는 제외된 시각화다. 해석 자체가 TETRA4였다는 뜻은 아니다. 원해석은 TETRA10을 사용했다.

## 형상과 필드

Points는 `ReferenceCoordinates_mm + FEM_U_mm`이며 **변위 배율1로 이미 변형된 형상**이다. ParaView에서 `FEM_U_mm`로 다시 Warp하면 변위가 두 번 적용되므로 이 파일은 그대로 표시한다. 명목20µs 값은 기존 target의 마지막time step을 사용하며 actual solver frame 하나와 같다고 주장하지 않는다. 각 파일의 같은이름 JSON에 실제 두 frame 시각·보간가중치·원target SHA를 기록했다.

| 데이터 | 위치 | 단위/정의 |
|---|---|---|
| NODE_ID | PointData | 원 RAD node ID, 기존 full support 순서 |
| PART_ID | PointData/CellData | 원 part ID. 모든 표시노드는 단일 part 소유임을 검사 |
| ReferenceCoordinates_mm | PointData vector | 원 RAD에 매핑한 canonical float32 reference를Float64로표현 |
| FEM_U_mm | PointData vector | 변위x/y/z,mm |
| FEM_GPS_VM_MPa | PointData scalar | solver nodalGPS tensor의 von Mises,MPa. 원 cell/IP peak와 다른정의 |
| FEM_TOTAL_STRAIN_FROB | PointData scalar | validIP평균→체적가중nodal globaltensor→Frobenius,무차원 |
| FEM_EPSP | PointData scalar | 체적가중nodal accumulatedplasticstrain,무차원 |
| ANIM_SOLID_INDEX | CellData | 원 ANIM solid행번호,0기준. 원 FEM element ID라는뜻은아님 |
| TimeValue/NominalTime_s | FieldData | 0.00002s |

TETRA10은109,600개의 `VTK_TETRA`4corner로 선형화했고 일반 BRICK20,150개는 `VTK_HEXAHEDRON`8corner다. DegenerateBRICK468개는 실제slot패턴[a,b,c,c,d,e,f,f]를 모두검사하여 `VTK_WEDGE`6corner[a,b,c,d,e,f]로변환했다. 원node집합·모든9개물리edge가정확히유지된다. Reference의다른tet분할대각선과float32 quad의작은비평면성으로 두 체적계산의최대상대차3.8895e-6이있으며 bitwise적분체적동일성은주장하지않는다. Reference와두변형형상에서표시cell의signedvolume은모두양수였다.

## 검증 및 실행

`write_phone_vtu.py`는NumPy와Python표준라이브러리만사용한다. CPU1/BLAS1, GPU미사용. 두파일생성을합쳐실측2.84s,peakRSS74.52MiB였다. `execution-metrics.json`과각case JSON에측정값·해시가있다. 원raw/학습cache는수정하지않았다.

별도 `meshio5.3.5` reader로두VTU를실제다시열었다. 76,065points/130,218cells와3가지celltype수, NODE_ID순서,모든FEMpointfield, reference+u배율1형상, 각cell의원corner집합,point/cellPART_ID를원동결자료와검사했다. `independent-reader-validation.json`을참조한다. **ParaView GUI를직접열어검사한것은아니다.** 표준 [VTKXML](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html)과 [celltypes10/12/13](https://vtk.org/doc/nightly/html/vtkCellType_8h.html)을사용한다.

`phone-test20us-FEM-3D-preview.png`는같은자료의cornerpoint미리보기다. VTU에는실제선형solidcells가포함되므로ParaView에서surface,clip,slice,part분리등으로확인할수있다. 표시값은nativecell/IP가아닌nodalcanonical label이다.

## 실제 AI 결과를 추가할 때

`write_case(run_id, destination, prediction, prediction_node_ids)`가동일함수로추가배열을지원한다. Prediction은physical단위의[N76065,6]이고 NODE_ID가원순서와완전히일치해야한다. 다른순서·비유한값·다른shape는중단한다. 함수는 AI_U/GPS_VM/STRAIN_FROB/EPSP, AI−FEM부호오차,절대오차,변위오차norm을추가한다. None이기본이며지금생성된두VTU에는그배열이없다. 형상기준은계속FEM변형형상이므로향후AI형상을보려면FEM형상에AI−FEM변위vector를배율1로적용할수있다.

## 보존 경계

**두 `.vtu`는로컬대용량자료로만보존하고publicGitHub/package에서제외한다.** Public에는writer/validator/preview코드,JSON해시·검증·측정기록,이README와PNG만포함한다. 원모델의Altair2022 CC BY-NC4.0 notice와출처는상위MODEL-NOTICE.txt/PHONE-FEM-AND-DATASET.md를따른다. 새FEM이나추가시간연장은수행하지않았다.
