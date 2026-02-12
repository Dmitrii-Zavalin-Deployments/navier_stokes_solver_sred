# tests/contract/test_dummy_schema_validation.py

import json
import os
import numpy as np
import pytest
from jsonschema import validate, ValidationError

from tests.helpers.minimal_step1_input import minimal_step1_input
from tests.helpers.step1_schema_dummy_state import Step1SchemaDummyState
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState
from tests.helpers.step4_schema_dummy_state import Step4SchemaDummyState


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SCHEMA_DIR = os.path.join(ROOT, "schema")


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    return obj


def _load_schema(name):
    path = os.path.join(SCHEMA_DIR, name)
    with open(path, "r") as f:
        return json.load(f)


def _validate_dummy(dummy, schema_name):
    schema = _load_schema(schema_name)
    json_safe = _to_json_safe(dummy)
    validate(instance=json_safe, schema=schema)


# ----------------------------------------------------------------------
# Positive tests
# ----------------------------------------------------------------------

def test_minimal_step1_input_matches_schema():
    dummy = minimal_step1_input()
    _validate_dummy(dummy, "input_schema.json")


def test_step1_dummy_matches_schema():
    dummy = Step1SchemaDummyState(nx=3, ny=3, nz=3)
    _validate_dummy(dummy, "step1_output_schema.json")


def test_step2_dummy_matches_schema():
    dummy = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _validate_dummy(dummy, "step2_output_schema.json")


def test_step3_dummy_matches_schema():
    dummy = Step3SchemaDummyState(nx=3, ny=3, nz=3)
    _validate_dummy(dummy, "step3_output_schema.json")


def test_step4_dummy_matches_schema():
    """
    Step‑4 schema validation test.
    This stays active because Step‑4 schema exists,
    even though Step‑4 implementation is still evolving.
    """
    dummy = Step4SchemaDummyState(nx=3, ny=3, nz=3)
    _validate_dummy(dummy, "step4_output_schema.json")


# ----------------------------------------------------------------------
# Negative tests
# ----------------------------------------------------------------------

def test_step2_dummy_fails_when_missing_required_field():
    dummy = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    del dummy["fields"]["U"]  # break it

    with pytest.raises(ValidationError):
        _validate_dummy(dummy, "step2_output_schema.json")


def test_step4_dummy_fails_when_missing_required_field():
    """
    Negative test for Step‑4 schema.
    This remains active because Step‑4 schema is defined,
    even though Step‑4 implementation is still in progress.
    """
    dummy = Step4SchemaDummyState(nx=3, ny=3, nz=3)
    del dummy["fields"]["P"]  # break it

    with pytest.raises(ValidationError):
        _validate_dummy(dummy, "step4_output_schema.json")
