# tests/step2/test_orchestrate_step2_errors.py

import numpy as np
import pytest


# ------------------------------------------------------------
# Test 1 — _to_json_compatible converts functions AND callables
# (covers line 33)
# ------------------------------------------------------------
def test_json_compatible_converts_functions_and_callables():
    from src.step2.orchestrate_step2 import _to_json_compatible

    def dummy():
        pass

    class CallableObj:
        def __call__(self):
            return 42

    obj = CallableObj()
    obj.__name__ = "callable_obj"   # ensures line 33 is hit

    converted = _to_json_compatible({"f": dummy, "c": obj})

    assert "<function dummy>" in converted["f"]
    assert "<function callable_obj>" in converted["c"]


# ------------------------------------------------------------
# Test 2 — Step‑1 schema validation failure (covers lines 49–50)
# ------------------------------------------------------------
def test_orchestrate_step2_step1_schema_validation_failure():
    from tests.helpers.schema_dummy_state import SchemaDummyState
    from src.step2.orchestrate_step2 import orchestrate_step2

    state = SchemaDummyState(4, 4, 4)
    state.pop("boundary_table")   # required by Step‑1 schema

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step2(state)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 3 — Step‑2 schema validation failure (covers lines 90–91)
# ------------------------------------------------------------
def test_orchestrate_step2_step2_schema_validation_failure(monkeypatch):
    from tests.helpers.schema_dummy_state import SchemaDummyState
    import src.step2.orchestrate_step2 as orch

    state = SchemaDummyState(4, 4, 4)

    real_json = orch._to_json_compatible

    # Break ONLY Step‑2 schema validation by removing "mask"
    def break_json(obj):
        out = real_json(obj)
        if isinstance(out, dict) and "mask" in out:
            out = dict(out)
            out.pop("mask")   # required by Step‑2 schema, NOT Step‑1
        return out

    monkeypatch.setattr(orch, "_to_json_compatible", break_json)

    with pytest.raises(RuntimeError) as excinfo:
        orch.orchestrate_step2(state)

    assert "Output schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 4 — _to_json_compatible handles arrays, functions, nested structures
# ------------------------------------------------------------
def test_json_compatible_full_conversion():
    from src.step2.orchestrate_step2 import _to_json_compatible

    def f():
        pass

    obj = {
        "arr": np.zeros((2, 2)),
        "func": f,
        "nested": {"x": np.array([1, 2])},
        "list": [np.array([3, 4])]
    }

    converted = _to_json_compatible(obj)

    assert converted["arr"] == [[0.0, 0.0], [0.0, 0.0]]
    assert "<function f>" in converted["func"]
    assert converted["nested"]["x"] == [1, 2]
    assert converted["list"] == [[3, 4]]
