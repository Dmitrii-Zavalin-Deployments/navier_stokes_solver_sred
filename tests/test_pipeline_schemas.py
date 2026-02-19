# tests/test_pipeline_schemas.py

import pytest
import jsonschema
from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1_state

# Importing your dummy inputs and schema definitions
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA
# Add other schema imports as we implement steps 2-5

def validate_dict_against_schema(data: dict, schema: dict):
    """
    Validates a dictionary against a JSON schema.
    Raises jsonschema.exceptions.ValidationError if it fails.
    """
    jsonschema.validate(instance=data, schema=schema)

class TestSolverPipelineSchemas:

    @pytest.fixture(scope="class")
    def step1_state(self):
        """
        Runs the actual Step 1 orchestrator once.
        This provides the state object for all Step 1 related tests.
        """
        input_dict = solver_input_schema_dummy()
        return orchestrate_step1_state(input_dict)

    def test_step1_to_json_safe_integrity(self, step1_state):
        """
        Verifies that Step 1 output contains all keys required by the 
        Step 1 output schema after being converted to JSON-safe format.
        """
        json_output = step1_state.to_json_safe()
        
        # Test 1: Check if 'operators' and other keys exist (The fix we just made)
        assert "operators" in json_output, "operators key missing from JSON output"
        assert "grid" in json_output
        
        # Test 2: Deep validation against the Python-defined schema
        # Assuming EXPECTED_STEP1_SCHEMA is a dictionary matching JSON Schema format
        try:
            validate_dict_against_schema(json_output, EXPECTED_STEP1_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Step 1 JSON output does not match schema! Error: {e.message}")

    def test_solver_state_attribute_types(self, step1_state):
        """
        Integration check: Ensures internal numpy types haven't leaked 
        into the dictionary keys or non-array fields.
        """
        json_output = step1_state.to_json_safe()
        
        # Grid values should be standard Python types (int/float), not np.int64
        assert isinstance(json_output["grid"]["nx"], int)
        assert isinstance(json_output["grid"]["x_max"], (int, float))
        
        # Fields should be lists (converted from numpy)
        assert isinstance(json_output["fields"]["U"], list)

    def test_step2_placeholder(self, step1_state):
        """
        Placeholder for Step 2 schema validation.
        Once orchestrate_step2 is ready, we will feed step1_state into it.
        """
        # state_v2 = orchestrate_step2_state(step1_state)
        # validate_dict_against_schema(state_v2.to_json_safe(), EXPECTED_STEP2_SCHEMA)
        pass