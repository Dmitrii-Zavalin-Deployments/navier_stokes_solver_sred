# tests/step2/test_orchestrate_step2_errors.py

import numpy as np
import pytest


# ------------------------------------------------------------
# Test 1 — _to_json_compatible converts functions to strings
# ------------------------------------------------------------
def test_json_compatible_converts_functions():
    from src.step2.orchestrate_step2 import _to_json_compatible

    def dummy():
        pass

    converted = _to_json_compatible({"f": dummy})

    assert isinstance(converted["f"], str)
    assert "<function dummy>" in converted["f"]


# ------------------------------------------------------------
# Test 2 — Malformed Step‑1 input fails early in precompute_constants
# ------------------------------------------------------------
def test_orchestrate_step2_step1_validation_failure():
    from src.step2.orchestrate_step2 import orchestrate_step2

    # Missing required Step‑1 keys → should fail inside precompute_constants
    bad_state = {"fields": {}, "grid": {}, "config": {}}

    with pytest.raises(KeyError) as excinfo:
        orchestrate_step2(bad_state)

    # Ensure the failure is due to missing Step‑1 fluid block
    assert "fluid" in str(excinfo.value)


# ------------------------------------------------------------
# Test 3 — Step 2 validation failure triggers RuntimeError
# ------------------------------------------------------------
def test_orchestrate_step2_step2_validation_failure(monkeypatch):
    from tests.helpers.schema_dummy_state import SchemaDummyState
    from src.step2.orchestrate_step2 import orchestrate_step2

    state = SchemaDummyState(4, 4, 4)

    # Monkeypatch compute_initial_health to break the output schema
    def break_output(s):
        # Remove a required key so Step‑2 schema validation fails
        s.pop("grid", None)

    monkeypatch.setattr(
        "src.step2.orchestrate_step2.compute_initial_health",
        break_output
    )

    with pytest.raises(RuntimeError) as excinfo:
        orchestrate_step2(state)

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
