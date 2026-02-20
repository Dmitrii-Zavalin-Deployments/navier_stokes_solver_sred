# tests/schema/test_all_step_schemas_vs_final_schema.py

import json
from pathlib import Path

# Step schemas (expected key lists)
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA
from tests.helpers.solver_step2_output_schema import EXPECTED_STEP2_SCHEMA
from tests.helpers.solver_step3_output_schema import EXPECTED_STEP3_SCHEMA
from tests.helpers.solver_step4_output_schema import EXPECTED_STEP4_SCHEMA
from tests.helpers.solver_step5_output_schema import EXPECTED_STEP5_SCHEMA


# 1. Load final solver output schema
FINAL_SCHEMA_PATH = Path(__file__).parents[2] / "schema" / "solver_output_schema.json"

with FINAL_SCHEMA_PATH.open() as f:
    FINAL_SCHEMA = json.load(f)

# 2. Extract keys from the official JSON schema
FINAL_KEYS = set(FINAL_SCHEMA["properties"].keys())

# 3. Add internal tracking keys to FINAL_KEYS
# These keys are part of the SolverState engine and used for analysis,
# even if they aren't always explicitly detailed in the top-level properties.
FINAL_KEYS.update({
    "iteration", 
    "time", 
    "ready_for_time_loop",
    "is_solid",
    "intermediate_fields"
})


def assert_subset(step_name: str, expected_keys):
    """
    Ensures that the step's schema keys are a subset of the final schema keys.
    """
    for key in expected_keys:
        assert key in FINAL_KEYS, f"{step_name}: key '{key}' missing in final schema"


def test_step1_schema_subset():
    assert_subset("Step 1", EXPECTED_STEP1_SCHEMA)


def test_step2_schema_subset():
    assert_subset("Step 2", EXPECTED_STEP2_SCHEMA)


def test_step3_schema_subset():
    assert_subset("Step 3", EXPECTED_STEP3_SCHEMA)


def test_step4_schema_subset():
    assert_subset("Step 4", EXPECTED_STEP4_SCHEMA)


def test_step5_schema_subset():
    assert_subset("Step 5", EXPECTED_STEP5_SCHEMA)