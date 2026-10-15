# 시각검증 추가 기록

검토시각: **2026-10-15T07:50:15.960836+00:00**. 판정: **가독성·표시 계약 PASS, 해석 주의사항은 아래와 같음**. 실제 `view_image`로 Phone6장과 Bumper2장을 모두 열었다. Python 표준 라이브러리로 작은 JSON·PNG헤더·파일hash만 읽었으며 GPU·NumPy·프레임워크·추론·test·metric 재계산은 수행하지 않았다.

## Phone EPSP 두 시험 사례 확대6장

모든 그림의 두 row는 `phone_e090_rxp050`, `phone_e110_rxm050`이며 FEM/Raw모델/Absoluteerror의 세 column, 최종명목20µs, referenceXY[mm], EPSP무차원[1]이 읽힌다. 제목의 ROI는 **두 testcase에서 모든 저장시각의 양의 FEM EPSP 합집합+1mm 표시 여유**라고 명시돼 있어 오차가 큰 지점만 고르는 표시가 아니다. 여섯 metadata의 같은bounds·89절점/전체76,065절점·최종GT양성0/11개가 일치하며 imageSHA도 모두 일치했다.

첫 testcase의 FEM은 균일한0색이고 둘째의 작은 양성 군집은 보인다. Raw예측은 이 군집보다 낮은색으로 나타나며 Absoluteerror에서 군집이 드러난다. 따라서 대부분의0영역이 비슷해 보이는 사실을 소성장 정확도 성공으로 읽으면 안 된다. 정확한89개·0/11개는 metadata를 확인한 값이다. XY에서 겹치는 점 때문에 픽셀만으로 모든 점을 독립적으로 세지는 않았다.

| 모델 | FEM/Raw공통범위[1] | Absoluteerror범위[1] | 시각판정 |
|---|---|---|---|
| transolver | -1.50020162e-06 .. 0.00271891663 | 0 .. 0.00263193203 | PASS |
| flare | 0 .. 0.00271891663 | 0 .. 0.00270334864 | PASS |
| geotransolver | -2.24593555e-06 .. 0.00271891663 | 0 .. 0.00265481835 | PASS |
| geoflare | 0 .. 0.00271891663 | 0 .. 0.00271475851 | PASS |
| meshgraphnet | 0 .. 0.00271891663 | 0 .. 0.00260533718 | PASS |
| figconvunet | 0 .. 0.00271891663 | 0 .. 0.00270806043 | PASS |

여섯 그림의 FEM/Raw최대는0.00271891663으로 같지만 **error최대는 모델별**이다. 서로 다른 PNG의 같은 오차색을 같은 오차값으로 비교하지 않는다. Transolver와 GeoTransolver의 작은 음수 raw값은 metadata범위에 보존돼 있으며0.0000으로 반올림된 하단tick만으로 clipping을 판단하지 않는다. 그림 title에는 exact89개·0/11개가 없으므로 caption/metadata와 함께 읽는 것이 좋다. 제목·축·colorbar 잘림이나 겹침은 보이지 않았다.

근거: [Phone모델별그림과metadata](phone-visual-comparison/), [세부8장검토·hash](visual-review-addendum-20261015.json).

## Bumper 첫 neural heldout 응력·EPSP2장

Transolver의 첫 heldout `Run224`, 최종100ms, 전체13,675support의 upper층 응력·EPSP를 실제로 열었다.

- [Upper응력](benchmarks/bumper/models-c7/transolver/heldout/inplane_vm_upper_final_FEM_prediction_error.png): 제목·referenceXY[mm]·**GPa**·FEM/Raw공통척도·0중심의 signed `Prediction−FEM` 척도가 읽힌다. MPa로 잘못 읽지 않는다.
- [UpperEPSP](benchmarks/bumper/models-c7/transolver/heldout/EPSP_upper_final_FEM_prediction_error.png): **dimensionless**, 같은FEM/Raw범위와 signed오차가 구분된다. EPSP를 총변형률로 부르지 않는다.

원본은2210×765이며 host화면에는2048×709로 표시됐다. 제목·단위·colorbar 잘림이 없고 긴 얇은 구조의 물리 aspectratio를 유지했다. Fullfield패턴·예측의 평활화·부호오차는 볼 수 있지만 좁은 panel에서 세밀한 hotspot 형상을 확대 검증한 것은 아니다. 이는 첫사례의 display이며7testcase전체의 성능을 대체하지 않는다. [완료 testmetadata](benchmarks/bumper/models-c7/transolver/test_report.json)

기존 root시각검토·전체README·Git·Page·coverage snapshot은 수정하지 않았다. 최종12모델/Plate/Temporal 완료 뒤 coverage갱신 요청을 기다린다.
