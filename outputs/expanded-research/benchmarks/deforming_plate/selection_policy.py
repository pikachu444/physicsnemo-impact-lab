"""Validation-only checkpoint ranking, using standard-library logic only.

This ranks two fixed autoregressive diagnostics; it is not an accuracy pass.
Any reported strain comes from source/AI nodal arrays, not an original COMSOL
strain tensor. This module neither opens trajectories nor imports a framework.
"""
import math
import re


STRAIN_CAVEAT = (
    "Strain is kinematically derived from source FEM/AI nodal position arrays; "
    "no original COMSOL strain tensor label is available."
)


def parse_selection_epochs(text, epochs):
    """Return unique sorted checkpoints, always including the final epoch.

    Empty text means final epoch only. Nonempty text is comma-separated positive
    integers; values outside 1..epochs and empty comma fields are rejected.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(text, str):
        raise ValueError("selection epochs must be comma-separated text")
    selected = {epochs}
    if text.strip():
        for token in text.split(","):
            token = token.strip()
            if not re.fullmatch(r"[0-9]+", token):
                raise ValueError("selection epochs must contain positive integers")
            epoch = int(token)
            if not 1 <= epoch <= epochs:
                raise ValueError("selection epoch is outside 1..epochs")
            selected.add(epoch)
    return sorted(selected)


def validate_diagnostic_names(names, allowed_valid_names):
    """Return the two fixed unique validation names, preserving caller order."""
    if isinstance(names, (str, bytes)):
        raise ValueError("diagnostic names must be a sequence of two names")
    names = tuple(names)
    allowed = set(allowed_valid_names)
    if len(names) != 2 or any(not isinstance(name, str) for name in names):
        raise ValueError("exactly two string diagnostic names are required")
    if len(set(names)) != 2:
        raise ValueError("exactly two unique fixed diagnostics are required")
    if any(not isinstance(name, str) or not re.fullmatch(r"valid_[0-9]{4}", name)
           or name not in allowed for name in names):
        raise ValueError("diagnostics must belong to the allowed validation split")
    return names


def _finite_relative_l2(case, field):
    metrics = case.get(field)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("relative_l2")
    if metrics.get("truth_l2") == 0:
        return (0.0 if metrics.get("component_rmse") == 0
                and metrics.get("max_abs_component_error") == 0 else None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _metric_diagnostic(case, field):
    metrics = case.get(field)
    metrics = metrics if isinstance(metrics, dict) else {}
    relative = _finite_relative_l2(case, field)
    note = None
    if relative is None:
        note = "relative L2 is missing, nonfinite, negative, or undefined (zero truth norm with nonzero/unconfirmed error)"
    elif metrics.get("relative_l2") is None:
        note = "zero relative error assigned only for zero truth norm AND exactly zero RMSE and maximum error"
    absolute = {}
    for name in ["component_rmse", "max_abs_component_error", "truth_l2"]:
        value = metrics.get(name)
        absolute[name] = (value if not isinstance(value, bool) and isinstance(value, (int, float))
                          and math.isfinite(value) and value >= 0 else None)
    return {"relative_l2": relative, **absolute, "note": note}


def comparison_key(record):
    """Internal sortable key; infinity never appears in the public record."""
    failed, invalid, score = record["selection_key"]
    return failed, invalid, math.inf if score is None else score


def _counts(case, field, steps):
    values = case.get(field)
    if not isinstance(values, (list, tuple)) or len(values) != steps:
        return None
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
        return None
    return values


def rollout_selection_record(results):
    """Rank complete AR results with (failure, invalid geometry, worst error).

    Each of exactly two cases needs status='completed', positive steps, finite
    displacement/stress relative L2, and one nonnegative integer inversion count
    per frame for both prediction and reference. A missing/malformed item fails
    the candidate. Nonfinite metrics or malformed prediction counts also flag
    invalid predicted geometry conservatively. Reference inversions are flagged
    separately and do not masquerade as predicted inversions.

    The worst error is max over both fields AND both cases. Failed candidates
    have a null score in the public sorting record; comparison_key() substitutes
    infinity internally. Zero truth norm is assigned relative zero only when
    both RMSE and maximum absolute error are exactly zero. Baseline comparisons
    and negative von Mises counts never alter the sorting key. If every complete
    candidate has inversions, the same predeclared worst-error minimax decides
    among their equal geometry flags; inversion counts remain diagnostics.
    """
    cases = list(results) if isinstance(results, (list, tuple)) else []
    reasons, scores, baseline_diagnostics, case_diagnostics = [], [], [], []
    invalid_geometry = False
    reference_inversion = False
    try:
        validate_diagnostic_names([c.get("trajectory") for c in cases],
                                  [c.get("trajectory") for c in cases])
    except (AttributeError, TypeError, ValueError):
        reasons.append("exactly two unique validation diagnostic cases are required")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            reasons.append(f"case {index}: result is not a mapping")
            continue
        name = case.get("trajectory", f"case {index}")
        if case.get("status") != "completed":
            reasons.append(f"{name}: status is not completed")
        steps = case.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            reasons.append(f"{name}: positive rollout steps are required")
            steps = 0
        disp = _finite_relative_l2(case, "rollout_free_displacement")
        stress = _finite_relative_l2(case, "rollout_free_stress")
        if disp is None or stress is None:
            reasons.append(f"{name}: displacement/stress relative L2 must be finite and nonnegative")
            invalid_geometry = True
        else:
            scores.extend([disp, stress])
        predicted = _counts(case, "all_frame_predicted_inverted_tetrahedra", steps)
        truth = _counts(case, "all_frame_truth_inverted_tetrahedra", steps)
        if predicted is None or not steps:
            reasons.append(f"{name}: predicted inversion counts must cover every frame")
            invalid_geometry = True
        elif any(predicted):
            invalid_geometry = True
        if truth is None or not steps:
            reasons.append(f"{name}: reference inversion counts must cover every frame")
        elif any(truth):
            reference_inversion = True
        zero = _finite_relative_l2(case, "zero_increment_baseline_free_displacement")
        mean = _finite_relative_l2(case, "training_mean_constant_stress_baseline")
        baseline_diagnostics.append({
            "trajectory": name,
            "zero_motion_displacement_relative_l2": zero,
            "training_mean_constant_stress_relative_l2": mean,
            "displacement_better_than_zero_motion": disp < zero if disp is not None and zero is not None else None,
            "stress_better_than_training_mean_constant": stress < mean if stress is not None and mean is not None else None,
        })
        minimum = case.get("predicted_von_mises_minimum")
        minimum = (minimum if not isinstance(minimum, bool) and isinstance(minimum, (int, float))
                   and math.isfinite(minimum) else None)
        negative = case.get("predicted_negative_von_mises_values")
        negative = negative if isinstance(negative, int) and not isinstance(negative, bool) and negative >= 0 else None
        case_diagnostics.append({
            "trajectory": name,
            "displacement": _metric_diagnostic(case, "rollout_free_displacement"),
            "von_mises_stress": _metric_diagnostic(case, "rollout_free_stress"),
            "predicted_negative_von_mises_values": negative,
            "predicted_von_mises_minimum": minimum,
            "stress_clipped": False,
            "all_frame_predicted_inverted_tetrahedra": list(predicted) if predicted is not None else None,
            "all_frame_truth_inverted_tetrahedra": list(truth) if truth is not None else None,
        })
    failed = bool(reasons)
    worst = max(scores) if scores else None
    return {
        "selection_key": (int(failed), int(invalid_geometry), None if failed else worst),
        "failure_flag": failed,
        "geometry_invalid_flag": invalid_geometry,
        "reference_inversion_flag": reference_inversion,
        "worst_per_case_displacement_or_stress_relative_l2": worst,
        "failure_reasons": reasons,
        "baseline_diagnostics": baseline_diagnostics,
        "case_diagnostics": case_diagnostics,
        "interpretation": "Validation checkpoint ranking only; not an accuracy qualification or fresh-test result.",
        "strain_label_caveat": STRAIN_CAVEAT,
    }


def _selfcheck():
    from copy import deepcopy

    assert parse_selection_epochs("3,1,3", 5) == [1, 3, 5]
    assert parse_selection_epochs("", 5) == [5]
    for text in ["0", "6", "1,,3", "1.0", "-1"]:
        try:
            parse_selection_epochs(text, 5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid schedule accepted: {text}")
    assert validate_diagnostic_names(["valid_0000", "valid_0001"], ["valid_0000", "valid_0001"]) == ("valid_0000", "valid_0001")
    for names in [["valid_0000", "test_0002"], ["valid_0000", "valid_0000"]]:
        try:
            validate_diagnostic_names(names, ["valid_0000", "valid_0001", "test_0002"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid diagnostic names accepted")
    cases = [{"trajectory": f"valid_{i:04d}", "status": "completed", "steps": 2,
              "rollout_free_displacement": {"relative_l2": 0.1},
              "rollout_free_stress": {"relative_l2": 0.2},
              "all_frame_predicted_inverted_tetrahedra": [0, 0],
              "all_frame_truth_inverted_tetrahedra": [0, 0],
              "zero_increment_baseline_free_displacement": {"relative_l2": 1.0},
              "training_mean_constant_stress_baseline": {"relative_l2": 0.3}}
             for i in range(2)]
    assert rollout_selection_record(cases)["selection_key"] == (0, 0, 0.2)
    bad_field = deepcopy(cases)
    bad_field[1]["rollout_free_stress"]["relative_l2"] = 9.0
    assert rollout_selection_record(bad_field)["selection_key"] == (0, 0, 9.0)
    inverted = deepcopy(cases)
    inverted[0]["all_frame_predicted_inverted_tetrahedra"][1] = 1
    assert rollout_selection_record(bad_field)["selection_key"] < rollout_selection_record(inverted)["selection_key"]
    failed = deepcopy(cases)
    failed[0]["status"] = "failed"
    assert comparison_key(rollout_selection_record(inverted)) < comparison_key(rollout_selection_record(failed))
    nonfinite = deepcopy(cases)
    nonfinite[0]["rollout_free_stress"]["relative_l2"] = math.nan
    assert rollout_selection_record(nonfinite)["failure_flag"]
    assert rollout_selection_record(nonfinite)["geometry_invalid_flag"]
    reference_inverted = deepcopy(cases)
    reference_inverted[0]["all_frame_truth_inverted_tetrahedra"][0] = 1
    record = rollout_selection_record(reference_inverted)
    assert record["reference_inversion_flag"] and not record["geometry_invalid_flag"]
    assert record["selection_key"] == (0, 0, 0.2)
    missing_frame = deepcopy(cases)
    missing_frame[0]["all_frame_predicted_inverted_tetrahedra"] = [0]
    assert rollout_selection_record(missing_frame)["failure_flag"]
    zero_truth = deepcopy(cases)
    zero_truth[0]["rollout_free_stress"] = {"relative_l2": None, "truth_l2": 0,
                                           "component_rmse": 0, "max_abs_component_error": 0}
    assert not rollout_selection_record(zero_truth)["failure_flag"]
    zero_truth[0]["rollout_free_stress"]["max_abs_component_error"] = 1
    assert rollout_selection_record(zero_truth)["failure_flag"]
    zero_truth[0]["rollout_free_stress"]["relative_l2"] = 0
    assert rollout_selection_record(zero_truth)["failure_flag"]
    import json
    json.dumps(rollout_selection_record(zero_truth), allow_nan=False)
    assert rollout_selection_record(failed)["selection_key"][-1] is None
    assert math.isinf(comparison_key(rollout_selection_record(failed))[-1])
    print("Synthetic selection-policy logic checks passed; no data, framework, or GPU was used.")


if __name__ == "__main__":
    _selfcheck()
