# tests/step2/test_orchestrate_step2_errors.py

import numpy as np
import pytest


# ------------------------------------------------------------
# Test 1 — _to_json_compatible converts functions AND callables
# ------------------------------------------------------------
def test_json_compatible_converts_functions_and_callables():
    from src.step2.orchestrate_step2 import _to_json_compatible

    def dummy():
        pass

    class CallableObj:
        def __call__(self):
            return 42

    obj = CallableObj()

    converted = _to_json_compatible({"f": dummy, "c": obj})

    assert "<function dummy>" in converted["f"]
    assert "<function" in converted["c"]


# ------------------------------------------------------------
# Test 2 — Step‑1 schema validation failure (covers lines 49–50)
# ------------------------------------------------------------
def test_orchestrate_step2_step1_schema_validation_failure():
    from src.step2.orchestrate_step2 import orchestrate_step2

    # Missing required Step‑1 keys → schema validation must fail
    bad_state = {
        "grid": {
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1,
            "z_min": 0, "z_max": 1,
            "nx": 1, "ny": 1, "nz": 1,
            "dx": 1, "dy": 1, "dz": 1,
        },
        "config": {
            "fluid": {"density": 1.0, "viscosity": 0.1},
            # MISSING "simulation" → Step‑1 schema violation
        },
        "fields": {
            "P": [[[0.0]]],
            "U": [[[0.0]]],
            "V": [[[0.0]]],
            "W": [[[0.0]]],
            "Mask": [[[1]]],
        },
        "constants": None,
    }

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step2(bad_state)

    assert "Input schema validation FAILED" in str(excinfo.value)


# ------------------------------------------------------------
# Test 3 — Step‑2 schema validation failure (covers lines 90–91)
# ------------------------------------------------------------
def test_orchestrate_step2_step2_schema_validation_failure(monkeypatch):
    from tests.helpers.schema_dummy_state import SchemaDummyState
    import src.step2.orchestrate_step2 as orch

    state = SchemaDummyState(4, 4, 4)

    # Monkeypatch _to_json_compatible to break output JUST before validation
    def break_json(obj):
        if isinstance(obj, dict) and "grid" in obj:
            broken = dict(obj)
            broken.pop("grid", None)  # remove required key
            return broken
        return obj

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
