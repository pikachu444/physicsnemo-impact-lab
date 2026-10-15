"""Generate only README.md from preserved metrics and posthoc diagnostics."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
m = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
d = json.loads((root / "field_diagnostics.json").read_text(encoding="utf-8"))
arch = m["architecture"]
v, df = m["metrics"][arch], d["fields"]
names, shape = m["contract"]["fields"], m["contract"]["shape"]
title = {"flare": "FLARE", "geotransolver": "GeoTransolver (GALE)",
         "geoflare": "GeoTransolver (GALE_FA / GeoFLARE 방식)",
         "meshgraphnet": "MeshGraphNet", "figconvunet": "FIGConvUNet"}[arch]
split, cfg, early = m["contract"]["split"], m["model_config"], m["early_stopping"]
actual, cap, best = m["actual_epochs"], m["epochs_budget"], m["best_epoch"]
def num(value):
    return "정의되지 않음" if value is None else f"{value:.6g}"
def pct(value):
    return "정의되지 않음" if value is None else f"{value*100:.3f}%"
def row(label, key, metric="rmse", relative=True):
    x = v[key]
    rel = pct(x.get("relative_l2")) if relative else "거의 0인 정답: 해석 보류"
    unit = "" if x["unit"] == "1" else " " + x["unit"]
    return f"| {label} | {rel} | {num(x[metric])}{unit} |"
rows = [row("변위 벡터 (성분 RMSE)", "displacement_vector", "component_rmse"),
        row("변위 x", "displacement_x"), row("변위 y", "displacement_y"),
        row("변위 z", "displacement_z", relative=False),
        row("투영 shell von Mises 응력", names[3]),
        row("투영 등가소성변형률 EPSP", names[4]),
        row("투영 in-plane 총변형률 주값 major", names[5]),
        row("투영 in-plane 총변형률 주값 minor", names[6])]
if best == cap:
    stop = f"{actual:,} epoch 실행, 내부 검증 최소도 상한인 {best:,}번째다. **유한 예산에 도달했으며 수렴을 입증한 것은 아니다.**"
elif m["stopping_reason"] == "finite_epoch_cap":
    stop = f"상한 {actual:,} epoch까지 실행했고 내부 검증 최소는 {best:,}번째다. 예산 상한에 도달했으므로 수렴을 확정하지 않는다."
else:
    stop = f"{actual:,} epoch에서 중단했고 내부 검증 최소는 {best:,}번째다. 검증 손실 기준의 조기 중단이며 물리적 수렴이나 실제 제품 검증을 뜻하지 않는다."
fresh_path = root / "fresh-inference" / "metrics.json"
fresh_table = ""
if fresh_path.exists():
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    physical_differences = fresh["fresh_checkpoint_prediction_max_absolute_difference_by_field"]
    physical_note = ("7개 채널의 물리 단위 최대 차이가 모두 0이다." if all(value == 0 for value in physical_differences)
                     else f"비트 단위로 동일하지 않다. 원 메타데이터의 전체 배열 maxabs={fresh['fresh_checkpoint_prediction_max_absolute_difference']!r}는 혼합 단위 참고값이다. 채널별 차이는 아래와 같으며 서로 다른 단위를 합쳐 정확도로 해석하지 않는다.")
    fresh_note = (f"별도 프로세스 체크포인트 재로드 결과에서 {physical_note} "
                  f"훈련 scale로 나눈 최대 차이는 {fresh['fresh_checkpoint_prediction_max_scale_normalized_difference']!r}다. "
                  f"저장된 허용 기준은 {num(fresh['fresh_checkpoint_prediction_tolerance_in_scale_normalized_units'])}이며 GPU 합산의 반올림 차이를 고려한다.")
    if any(value != 0 for value in physical_differences):
        fresh_table = ("| 새 프로세스 재추론 채널 | 최대 절대 차이 | 단위 |\n|---|---:|---|\n" +
                       "\n".join(f"| {field} | {value!r} | {unit} |" for field, value, unit in
                                 zip(names, physical_differences, m["contract"]["units"])))

else:
    fresh_note = "별도 프로세스 체크포인트 재로드 검증 보고서는 이 보고서 생성 시점에 아직 없다."
z, y = df["displacement_z"], v["displacement_y"]
vm, eps, ed = v[names[3]], v[names[4]], df[names[4]]
history = ed["history_diagnostics"]
ph, th = history["prediction_raw_delta_distribution"], history["reference_raw_delta_distribution"]
quantiles = ph["decrease_magnitude_quantiles"]
qtext = ("감소 없음" if quantiles is None else ", ".join(
    f"{label}={num(quantiles[key])}" for key, label in
    [("0.5","p50"),("0.9","p90"),("0.95","p95"),("0.99","p99"),("1.0","max")]))
history_note = (
    f"**EPSP 누적성 사후 검사:** 고정된 투영 노드에서 연속 저장 시점의 엄밀한 음수 증가분은 예측 "
    f"{ph['negative_count']:,}/{ph['count']:,}개(정확히 0인 증가분 {ph['zero_count']:,}개), "
    f"정답 {th['negative_count']:,}/{th['count']:,}개다. 예측 delta 최솟값은 {num(ph['minimum_delta'])}, "
    f"감소 크기 분포는 {qtext}다. 정답 최대 감소는 {num(history['reference_maximum_time_decrease'])}다. "
    f"저장 배열은 {history['stored_prediction_dtype']}/{history['stored_reference_dtype']}이며 FP32 machine epsilon은 "
    f"{num(history['float32_machine_epsilon'])}이다(1 부근의 상대 간격이며 EPSP의 절대 허용 오차가 아니다). 엄밀한 음수 횟수는 반올림 규모의 변화도 셀 수 있으므로 정답의 감소 분포와 함께 읽어야 한다. "
    "이 검사는 임의의 성공 임계값·보정·단조화 없이 원 예측을 진단하며, 언로딩 때 감소할 수 있는 총변형률 주값에는 적용하지 않았다.")

order = d.get("principal_order_diagnostics", {})
order_note = ("**투영 주값 순서 사후 검사:** 예측 minor > major는 "
              f"{order['prediction_minor_greater_than_major_count']:,}개, 정답은 "
              f"{order['reference_minor_greater_than_major_count']:,}개다. 예측의 최대 minor-major 위반 크기는 "
              f"{num(order['prediction_maximum_order_violation'])}이며 재정렬하지 않았다.") if order else ""
context_note = ("이 실행의 동시 CPU 작업 등 자원 맥락은 별도 execution-context.json에 기록되어 있다. "
                "여기의 시간은 해당 실행 조건에서 측정한 값이며 격리된 장비의 속도 순위로 해석하지 않는다.") if (root / "execution-context.json").exists() else ""

verified = m["frozen_data_verification"]
assert verified == m["frozen_data_end_verification"], "Frozen source changed during training"
assert verified["expected_cases"] == 13
manifest_sha, fence = verified["manifest_sha256"], chr(96)*3
readme = f"""# Coupon C7 — {title} 실제 학습

공개 RD-E1101 shell 인장 예제를 바탕으로 새 FEM 13조건을 계산한 고정 자료를 사용했다. 실제 노드 {shape[0]:,}개, 비영 시점 {shape[1]}개, 변위 XYZ·응력·등가소성변형률·총변형률 주값 2개의 {shape[2]}채널을 한 번에 예측한다. 원 shell 응력/변형률 불변량은 초기 면적 가중치로 노드에 투영한 정답이며 원 cell tensor도 별도로 보존한다.

입력은 초기 좌표, 실제 두께, 항복·UTS 동시 배율, 속도 배율이다. 예측 입력에 FEM 응답을 넣지 않는다. 동일 메시·재료계·하중 경로의 파라미터 보간 실험이며 가전제품 낙하 성능 검증은 아니다.

- 학습 FEM {len(split['train'])}조건 / 내부 검증 {len(split['validation'])}조건 / 평가 {len(split['test'])}조건
- PhysicsNeMo {m['physicsnemo']}, 모델 파라미터 {m['parameters']:,}개; 실제 구조 설정은 아래와 같다.
- 학습 전용 정규화: 변위 XYZ 공통 vector scale, 나머지 scalar별 scale. 원 예측을 자르거나 물리 라벨을 대체하지 않았다.
- 공통 예산 상한 {cap:,} epoch, 최소 {early['minimum_epochs']} epoch, 검증 개선 없음 {early['patience']} epoch 중단
- {stop}
- 학습 {m['training_seconds']:.2f}초; {m['device_name']}; CUDA allocator peak {m['cuda_peak_allocated_bytes']/1e6:.2f} MB; sampled RSS 최대 {m['peak_sampled_rss_bytes']/1e9:.3f} GB. allocator와 경계 샘플 RSS는 전체 GPU VRAM·실제 RSS 순간 최대와 구분한다.
- {fresh_note}

{context_note}

{fresh_table}

{fence}json
{json.dumps(cfg, ensure_ascii=False, indent=2)}
{fence}

| 평가 물리량 | 상대 L2 | 물리 단위 RMSE |
|---|---:|---:|
{chr(10).join(rows)}

**축별·물리적 한계:** y 방향 상대 L2는 {pct(y['relative_l2'])}, 절대 RMSE는 {num(y['rmse'])} mm다. 평면 FEM의 z 변위 정답 최대 절댓값은 {num(z['truth_absolute_peak'])} mm인 반면, 예측 최대 절댓값은 {num(z['prediction_absolute_peak'])} mm, 최대 절대 오차는 {num(z['absolute_error_max'])} mm, RMSE는 {num(z['component_rmse'])} mm다. z 상대 L2는 거의 0인 정답으로 나누므로 정확도 주장에 사용하지 않는다.

원 예측 전체의 음수 von Mises 응력은 {vm['prediction_negative_count_for_nonnegative_field']:,}/{vm['count']:,}개이며 최소 {num(vm['prediction_min'])} MPa다. 음수 EPSP는 {eps['prediction_negative_count_for_nonnegative_field']:,}/{eps['count']:,}개이며 최소 {num(eps['prediction_min'])}다. 이 값들은 숨기거나 0으로 보정하지 않았다.

참 EPSP >0 영역 {ed['reference_positive_support_count']:,}개의 RMSE는 {num(ed['rmse_on_reference_positive_support'])}, 참 EPSP=0 영역 {ed['reference_zero_support_count']:,}개의 RMSE는 {num(ed['rmse_on_reference_zero_support'])}다. 참 EPSP=0 영역에서 예측 절댓값 최대는 {num(ed['prediction_absolute_peak_on_reference_zero_support'])}다. 양수 support 판정은 정답·예측 각각 >0을 그대로 사용하며 임계값을 맞추지 않았다. 이 투영된 노드 support는 원 cell 파손 판정 기준이 아니다.

{history_note}

{order_note}

'metrics.json'은 학습 당시 원 보고서이고 'field_diagnostics.json'은 저장된 원 예측의 후속 진단이다. 상대 L2는 평가 전체 노드·시점의 정답 제곱합을 공통 분모로 사용하며 모든 노드에 동일 가중치를 준다. 변위 벡터 오차가 작더라도 지배적인 x 방향 응답이나 평균 성능만으로 국부 응력·소성의 정확도를 판정하지 않는다. 원 총변형률은 solver의 accumulated total strain 출력이며 특정 유한변형률 측정량과 정확히 같다는 추가 증명은 없다.

그림은 'field_comparison_1.png', 'field_comparison_2.png', 개별 'field_*.png', 'deformation_1x.png', 'learning_curve.png'에 있다. 비교장은 첫 평가 조건의 마지막 저장 시점이며 FEM과 예측의 공통 색 범위를 사용한다. 변형 그림은 1배이며 회색은 초기 형상이다. 전체 평가 수치는 그림 한 장의 조건에 한정되지 않는다.

13조건 고정 입력 manifest SHA256은 '{manifest_sha}'이며 학습 시작·종료 검증 결과가 일치했다. 'source/'와 'executed_*.py'에는 당시 코드가, 'best_checkpoint.pt'에는 구조·가중치·정규화·입력/분할 해시가 있다. 후처리는 학습·체크포인트·원 NPY를 수정하지 않는다. 예측 forward는 정답을 입력받지 않으며 FEM 정답은 수치 검증에 사용한다.

로컬 'predictions.npy', 'truth.npy'와 원 FEM/캐시는 공개 저장소 복사에서 제외한다. 재현에는 FEM 생성 기록·원 자료 출처·보존된 정규화가 필요하다. 이 결과를 읽은 뒤 동일 평가 집합에 맞춰 모델 설정이나 학습 예산을 바꿔 재학습하지 않았다.
"""
(root / "README.md").write_text(readme, encoding="utf-8")
print(json.dumps({"status": "model_readme_written", "architecture": arch,
                  "actual_epochs": actual, "best_epoch": best,
                  "relative_l2": {key: v[key]["relative_l2"] for key in ("displacement_vector", names[3], names[4])}},
                 ensure_ascii=False))

