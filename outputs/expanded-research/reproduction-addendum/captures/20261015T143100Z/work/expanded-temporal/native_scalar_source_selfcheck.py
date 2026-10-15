"""Stdlib-only structural checks; never import or execute the ML wrapper."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "native_scalar_rollout_v3.py"
FRAMEWORKS = ("torch", "numpy", "physicsnemo")


def dump(node):
    return ast.dump(node, include_attributes=False)


def expr(source):
    return ast.parse(source, mode="eval").body


def method(cls, name):
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


def assigned(fn, name):
    return [node.value for node in ast.walk(fn) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]


class AccelerationSlice(ast.NodeTransformer):
    """Compare official arithmetic after the intentional first-three slice."""

    def visit_Name(self, node):
        if node.id == "outf":
            return expr("raw[..., :3]")
        return node


def main():
    assert not any(name in sys.modules for name in FRAMEWORKS)
    text = WRAPPER.read_text(encoding="utf8")
    tree = ast.parse(text, filename=str(WRAPPER))
    compile(tree, str(WRAPPER), "exec")
    compile(Path(__file__).read_text(encoding="utf8"), str(Path(__file__)), "exec")
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_model")
    assert [arg.arg for arg in factory.args.args] == ["core_class", "strategy"]
    assert factory.args.kwarg.arg == "config"
    classes = {node.name: node for node in factory.body if isinstance(node, ast.ClassDef)}
    assert set(classes) == {"NativeScalarAutoregressiveRollout", "NativeScalarOneStepRollout"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"eval", "exec", "type", "setattr"} for node in ast.walk(tree))

    modes = [
        ("NativeScalarAutoregressiveRollout", "GeoTransolverAutoregressiveRolloutTraining", "y_t2",
         ("vel", "vel_norm", "fx_t", "acc", "y_t2")),
        ("NativeScalarOneStepRollout", "GeoTransolverOneStepRollout", "y_t2_pred",
         ("vel", "vel_norm", "fx_t", "acc", "vel_pred", "y_t2_pred")),
    ]
    reference_hashes = {}
    for domain in ("crash", "drop_test"):
        reference = HERE / "sources" / domain / "rollout.py"
        reference_hashes[domain] = hashlib.sha256(reference.read_bytes()).hexdigest()
        reference_tree = ast.parse(reference.read_text(encoding="utf8"))
        for name, official_name, position, names in modes:
            cls = classes[name]
            assert [dump(base) for base in cls.bases] == [dump(expr("core_class"))]
            fn = method(cls, "forward")
            official_cls = next(node for node in reference_tree.body
                                if isinstance(node, ast.ClassDef) and node.name == official_name)
            official_fn = method(official_cls, "forward")
            for variable in names:
                expected = [dump(AccelerationSlice().visit(copy.deepcopy(value)))
                            for value in assigned(official_fn, variable)]
                assert [dump(value) for value in assigned(fn, variable)] == expected, (domain, name, variable)
            assert [dump(value) for value in assigned(fn, "fx_t")] == [dump(expr("torch.cat([vel_norm, features], dim=-1)"))]
            append = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == "append"]
            assert len(append) == 1
            assert dump(append[0].args[0]) == dump(expr(f"torch.cat([{position}, raw[..., 3:]], dim=-1)"))
            checkpoint = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == "ckpt"]
            assert len(checkpoint) == 1
            assert any(keyword.arg == "use_reentrant" and dump(keyword.value) == dump(expr("False"))
                       for keyword in checkpoint[0].keywords)
            checkpoint_guard = [node for node in ast.walk(fn) if isinstance(node, ast.If)
                                and dump(node.test) == dump(expr("self.training"))
                                and any(checkpoint[0] is child for statement in node.body for child in ast.walk(statement))]
            assert len(checkpoint_guard) == 1
            step = next(node for node in ast.walk(fn)
                        if isinstance(node, ast.FunctionDef) and node.name == "step_fn")
            call = step.body[0].value
            assert dump(call.func.value) == dump(expr(f"super({name}, self)"))
            assert [(kw.arg, dump(kw.value)) for kw in call.keywords] == [
                ("local_embedding", dump(expr("local_emb"))), ("geometry", dump(expr("geometry"))),
                ("local_positions", dump(expr("local_pos"))), ("global_embedding", dump(expr("global_emb")))
            ]
            assert dump(fn.body[-1].value) == dump(expr("torch.stack(outputs, dim=0).transpose(0, 1)"))

    ar = method(classes["NativeScalarAutoregressiveRollout"], "forward")
    tf = method(classes["NativeScalarOneStepRollout"], "forward")
    assert not any(isinstance(node, ast.Attribute) and node.attr == "node_target" for node in ast.walk(ar))
    assert [dump(value) for value in assigned(ar, "y_t1")] == [dump(expr("coords"))]
    assert [dump(value) for value in assigned(ar, "y_t0")] == [dump(expr("y_t1 - self.initial_vel * self.dt"))]
    ar_loop = next(node for node in ar.body if isinstance(node, ast.For))
    assert dump(ar_loop.body[-1]) == dump(ast.parse("y_t1, y_t0 = y_t2, y_t1").body[0])
    target_access = [node for node in ast.walk(tf) if isinstance(node, ast.Attribute) and node.attr == "node_target"]
    assert len(target_access) == 1
    target_guard = next(node for node in tf.body if isinstance(node, ast.If)
                        and dump(node.test) == dump(expr("self.training"))
                        and any(target_access[0] is child for statement in node.body for child in ast.walk(statement)))
    assert not target_guard.orelse
    assert dump(assigned(tf, "gt_seq")[0]) == dump(expr(
        "torch.cat([coords0.unsqueeze(0), sample.node_target[..., :3].transpose(0, 1)], dim=0)"))
    assert [dump(value) for value in assigned(tf, "y_t0")] == [dump(expr("coords0 - self.initial_vel * self.dt"))]
    assert [dump(value) for value in assigned(tf, "y_t1")] == [dump(expr("coords0"))]
    teacher = next(node for node in ast.walk(tf) if isinstance(node, ast.If)
                   and dump(node.test) == dump(expr("self.training and t > 0")))
    assert dump(teacher.body[0].value) == dump(expr("(gt_seq[t - 1], gt_seq[t])"))
    gt_reads = [dump(node) for node in ast.walk(tf) if isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name) and node.value.id == "gt_seq"]
    assert sorted(gt_reads) == sorted([dump(expr("gt_seq[t - 1]")), dump(expr("gt_seq[t]"))])
    inference = next(node for node in ast.walk(tf) if isinstance(node, ast.If)
                     and dump(node.test) == dump(expr("not self.training")))
    assert dump(inference.body[0].value) == dump(expr("(y_t1, y_t2_pred)"))
    assert not any(name in sys.modules for name in FRAMEWORKS)
    print(json.dumps({
        "status": "pass",
        "scope": "AST/compile only; no wrapper execution, arrays, framework import, GPU, or dataset access",
        "wrapper_sha256": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
        "selfcheck_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_rollout_sha256": reference_hashes,
        "checks": ["core subclass without source monkeypatch", "official AR/TF integration and velocity stats",
                   "training checkpoint use_reentrant=False", "acceleration first3 only", "native scalar suffix output",
                   "position-only recurrent inputs", "TF targets only under training", "TF past pair only", "N,T,C stacking"],
        "not_executed": ["gradient probe", "poison inference", "C3 numerical equivalence", "training"],
    }, indent=2))


if __name__ == "__main__":
    main()
