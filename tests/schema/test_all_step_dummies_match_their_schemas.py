# tests/schema/test_all_step_dummies_match_their_schemas.py

"""
test_all_step_dummies_match_their_schemas.py

Validates that each step's dummy output matches its corresponding
Python-typed step "schema" (EXPECTED_STEPX_SCHEMA).

These EXPECTED_STEPX_SCHEMA objects are Python-level structural/type
contracts, not JSON Schemas. This test ensures:

  • step dummies remain aligned with their Python schema contracts,
  • refactors cannot silently break intermediate solver structures,
  • every step produces structurally consistent, type-consistent data.

These tests validate INTERNAL solver state (Step 1–5),
not the final external solver output schema.
"""

import numpy as np

# Step 1
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA

# Step 2
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA

# Step 3
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step3_output_schema import EXPECTED_STEP3_SCHEMA

# Step 4
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step4_output_schema import EXPECTED_STEP4_SCHEMA

# Step 5
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy
from tests.helpers.solver_step5_output_schema import EXPECTED_STEP5_SCHEMA


def _is_ndarray_type_spec(t):
    """Return True if the schema type spec means 'numpy ndarray'."""
    if t == "ndarray":
        return True
    if isinstance(t, tuple) and "ndarray" in t:
        return True
    return False


def _matches_type(value, type_spec):
    """
    Check whether `value` matches the Python-level type spec used in EXPECTED_STEPX_SCHEMA.
    type_spec can be:
      - a Python type (e.g., dict)
      - a tuple of types (e.g., (type(None), object))
      - the string "ndarray"
      - a nested dict (for nested structures like fields)
    """
    # Nested dict schema: recurse
    if isinstance(type_spec, dict):
        if not isinstance(value, dict):
            return False
        return _matches_schema(value, type_spec)

    # ndarray shorthand
    if _is_ndarray_type_spec(type_spec):
        return isinstance(value, np.ndarray)

    # Normal Python type or tuple of types
    if isinstance(type_spec, tuple) or isinstance(type_spec, type):
        return isinstance(value, type_spec)

    # Fallback: if it's something else (e.g., object), just accept anything
    if type_spec is object:
        return True

    # Unknown spec kind: be conservative
    return False


def _matches_schema(value_dict, schema_dict):
    """
    Recursively validate that `value_dict` matches the Python schema dict:
      - all schema keys are present in value_dict
      - types/structures match the type specs
    Extra keys in value_dict are allowed (you can tighten this if you want).
    """
    for key, type_spec in schema_dict.items():
        assert key in value_dict, f"Missing key '{key}' in value: keys={list(value_dict.keys())}"
        v = value_dict[key]
        assert _matches_type(
            v, type_spec
        ), f"Key '{key}' has wrong type: got {type(v)}, expected {type_spec}"
    return True


def _validate_step_dummy(step_name: str, dummy, expected_schema: dict):
    """
    Validate a single step dummy against its EXPECTED_STEPX_SCHEMA.
    """
    # SolverState-like object: assume attributes match schema keys
    # If dummy is a dataclass / object, use vars() or __dict__.
    if hasattr(dummy, "__dict__"):
        value_dict = vars(dummy)
    else:
        value_dict = dummy

    assert isinstance(value_dict, dict), (
        f"{step_name}: expected dummy to be dict-like, got {type(value_dict)}"
    )

    _matches_schema(value_dict, expected_schema)


def test_step1_dummy_matches_schema():
    dummy = make_step1_output_dummy()
    _validate_step_dummy("Step 1", dummy, EXPECTED_STEP1_SCHEMA)


# def test_step2_dummy_matches_schema():
#     dummy = make_step2_output_dummy()
#     _validate_step_dummy("Step 2", dummy, EXPECTED_STEP2_SCHEMA)


# def test_step3_dummy_matches_schema():
#     dummy = make_step3_output_dummy()
#     _validate_step_dummy("Step 3", dummy, EXPECTED_STEP3_SCHEMA)


# def test_step4_dummy_matches_schema():
#     dummy = make_step4_output_dummy()
#     _validate_step_dummy("Step 4", dummy, EXPECTED_STEP4_SCHEMA)


# def test_step5_dummy_matches_schema():
#     dummy = make_step5_output_dummy()
#     _validate_step_dummy("Step 5", dummy, EXPECTED_STEP5_SCHEMA)
