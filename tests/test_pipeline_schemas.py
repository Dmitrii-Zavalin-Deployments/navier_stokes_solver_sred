# tests/test_pipeline_schemas.py

import pytest
import jsonschema
import numpy as np
from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1_state

# Import your existing schema definitions and dummies
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA

def validate_state(state: SolverState, schema: dict):
    """Converts state to JSON-safe dict and validates against a schema."""
    data = state.to_json_safe()
    jsonschema.validate(instance=data, schema=schema)

class TestSolverPipelineSchemas:

    @pytest.fixture(scope="class")
    def step1_completed_state(self):
        """Runs Step 1 once to provide a state for all tests in this class."""
        input_data = solver_input_schema_dummy()
        return orchestrate_step1_state(input_data)

    def test_step1_schema_compliance(self, step1_completed_state):
        """
        Ensures Step 1 output matches the schema, specifically checking 
        that 'operators' exists in the JSON output.
        """
        # This will now pass because we updated SolverState.to_json_safe()
        validate_state(step1_completed_state, EXPECTED_STEP1_SCHEMA)

    def test_internal_state_consistency(self, step1_completed_state):
        """Verifies internal attributes are correctly typed before serialization."""
        assert isinstance(step1_completed_state.fields["U"], np.ndarray)
        assert isinstance(step1_completed_state.operators, dict)
        
    def test_serialization_types(self, step1_completed_state):
        """Confirms that to_json_safe actually converts numpy types to Python types."""
        json_data = step1_completed_state.to_json_safe()
        # JSON standard doesn't know numpy.float64, must be python float
        assert type(json_data["grid"]["dx"]) is float
        assert isinstance(json_data["fields"]["U"], list)