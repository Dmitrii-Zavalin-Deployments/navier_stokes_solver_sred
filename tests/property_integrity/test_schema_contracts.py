# tests/property_integrity/test_schema_contracts.py

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy


# Instead of importing a non-existent utility, we define how to find the schemas
def load_schema(schema_name: str) -> dict:
    """Helper to load schemas from the project's /schema directory."""
    # This finds the 'schema' folder at the root of your repo
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / schema_name
    
    if not schema_path.exists():
        pytest.fail(f"Schema not found at {schema_path}")
        
    with open(schema_path) as f:
        return json.load(f)

class TestSchemaContracts:
    """
    SYSTEM AUDITOR: Verifies that the SSoT helpers generate data 
    that strictly adheres to the JSON schemas.
    """

    def test_input_dummy_matches_schema(self):
        schema = load_schema("solver_input_schema.json")
        input_obj = create_validated_input(nx=4, ny=4, nz=4)
        payload = input_obj.to_dict()
        
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Input Contract Violation: {e.message}")

    def test_output_dummy_matches_schema(self):
        # 1. Load the Output Schema
        schema = load_schema("solver_output_schema.json")
        
        # 2. Generate the Output Dummy (SolverState)
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        
        # 3. Transform to dict using the correct method
        payload = state.to_dict()
        
        # 4. MANUALLY SANITIZE NUMPY (Since to_json_safe is missing)
        # We must convert any numpy arrays to lists for jsonschema to read them
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_safe_payload = sanitize(payload)
        
        # 5. Validate Contract
        try:
            jsonschema.validate(instance=json_safe_payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            pytest.fail(f"Output Contract Violation: {e.message}")