"""Stdlib-only mocks of the corrected option gate; no modules/arrays are imported."""
import ast
import copy
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent
BENCHMARK = ROOT.parents[1] / "outputs/expanded-research/benchmarks/deforming_plate"


def main():
    source = ROOT / "run_frozen_final_corrected.py"
    tree = ast.parse(source.read_text(encoding="utf8"))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and ast.unparse(node.iter) == "protocol['required_options'].items()"]
    assert len(loops) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=loops, type_ignores=[])),
                   str(source) + "::required_options", "exec")
    protocol = json.loads((BENCHMARK / "official32-validation-protocol.json").read_text(encoding="utf8"))
    baseline = copy.deepcopy(protocol["required_options"])
    baseline["selection_epochs"] = list(protocol["selection_epochs"])
    baseline["selection_trajectories"] = list(protocol["selection_trajectories"])

    def passes(config):
        try:
            exec(code, {"protocol": protocol, "config": config})
        except (AssertionError, KeyError, TypeError, ValueError):
            return False
        return True

    assert passes(baseline), "The exact canonical runtime configuration must pass"
    mutations = {
        "wrong_training_epoch_count": {"epochs": 4},
        "wrong_selection_epoch": {"selection_epochs": [1, 3, 4]},
        "reversed_epoch_order": {"selection_epochs": [5, 3, 1]},
        "added_epoch": {"selection_epochs": [1, 2, 3, 5]},
        "missing_epoch": {"selection_epochs": [1, 5]},
        "string_epoch_form_is_not_runtime_list": {"selection_epochs": "1,3,5"},
        "wrong_trajectory": {"selection_trajectories": ["valid_0000", "valid_0001"]},
        "reversed_trajectory_order": {"selection_trajectories": ["valid_0005", "valid_0000"]},
        "added_trajectory": {"selection_trajectories": ["valid_0000", "valid_0005", "valid_0001"]},
        "missing_trajectory": {"selection_trajectories": ["valid_0000"]},
        "test_trajectory_rejected": {"selection_trajectories": ["valid_0000", "test_0002"]},
        "string_trajectory_form_is_not_runtime_list": {"selection_trajectories": "valid_0000,valid_0005"},
        "unrelated_architecture_option_unchanged": {"hidden": 64},
        "unrelated_rollout_option_unchanged": {"rollout_steps": 398},
    }
    for name, changes in mutations.items():
        config = copy.deepcopy(baseline)
        config.update(changes)
        assert not passes(config), f"Malformed configuration accepted: {name}"
    missing = copy.deepcopy(baseline)
    del missing["selection_epochs"]
    assert not passes(missing)
    print(json.dumps({"status": "PASS", "canonical_configuration_passed": True,
                      "rejected_mutations": list(mutations) + ["missing_required_key"],
                      "framework_imported": False, "fresh_array_opened": False}))


if __name__ == "__main__":
    main()
