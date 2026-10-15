"""Summarize frozen EPSP time-history diagnostics, without loading field arrays."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]/"outputs/expanded-research/benchmarks/rom"
rows=[]
for trial in ["coupon13","bumper7","phone9-full","phone9-rear"]:
    path=root/trial/"frozen_support_peak_diagnostics.json"
    report=json.loads(path.read_text(encoding="utf-8"))
    for model,fields in report["physical_diagnostics"].items():
        for name,field in fields.items():
            history=field.get("accumulated_epsp_history")
            if not history:continue
            for kind in ["truth","prediction"]:
                cases=[case[kind] for case in history]
                rows.append({"trial":trial,"model":model,"field":name,"values":kind,
                    "time_interval_count":sum(case["interval_count"] for case in cases),
                    "negative_increment_count":sum(case["negative_increment_count"] for case in cases),
                    "minimum_delta":min(case["minimum_delta"] for case in cases),
                    "largest_case_decrease_p95":max((case["decrease_magnitude_quantiles"][1] for case in cases if case["decrease_magnitude_quantiles"]),default=None),
                    "largest_case_decrease_in_endpoint_ulp_p95":max((case["decrease_to_max_endpoint_float32_ulp_quantiles"][1] for case in cases if case["decrease_to_max_endpoint_float32_ulp_quantiles"]),default=None),
                    "case_diagnostics":str(path.relative_to(root)),"case_diagnostics_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
result={"scope":"동결된 raw 정답/예측의 EPSP 누적 시간 일관성을 사후 관찰한다. 총변형률 Frobenius 및 signed principal strain에는 누적성 요구를 적용하지 않는다.",
    "definition":"저장 float32 endpoint를 float64로 올린 뒤 np.diff(time)의 정확한 음수 부호를 모두 센다. 임의 허용오차·성공기준·clipping을 두지 않는다. 감소 P95는 전체 혼합분포의 P95가 아니라 사례별 P95 중 최대다.",
    "float32_interpretation":"endpoint float32 ULP 대비 감소 크기를 함께 기록한다. 단순 반올림 해상도와 큰 history 감소를 구분할 근거이며, 임계값으로 성공/실패를 나누지 않는다.",
    "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":rows}
(root/"epsp-history-summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
for row in rows:
    if row["values"]=="prediction":print(json.dumps({key:row[key] for key in ["trial","model","field","negative_increment_count","minimum_delta","largest_case_decrease_p95"]}),flush=True)
