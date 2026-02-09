# tests/contract/test_round_trip_schemas.py

import json
import os
import numpy as np
from jsonschema import validate

from tests.helpers.minimal_step1_input import minimal_step1_input
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState

from src.step1.construct_simulation_state import construct_simulation_state as orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import step3 as orchestrate_step3


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
    with open(os.path.join(SCHEMA_DIR, name), "r") as f:
        return json.load(f)


def _validate(instance, schema_name):
    schema = _load_schema(schema_name)
    validate(instance=_to_json_safe(instance), schema=schema)


def test_round_trip_step1_to_step2():
    inp = minimal_step1_input()
    _validate(inp, "input_schema.json")

    step1_out = orchestrate_step1(inp)
    _validate(step1_out, "step1_output_schema.json")

    step2_out = orchestrate_step2(step1_out)
    _validate(step2_out, "step2_output_schema.json")


def test_round_trip_step2_to_step3():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    _validate(s2, "step2_output_schema.json")

    step3_out = orchestrate_step3(s2, current_time=0.0, step_index=0)
    _validate(step3_out, "step3_output_schema.json")
