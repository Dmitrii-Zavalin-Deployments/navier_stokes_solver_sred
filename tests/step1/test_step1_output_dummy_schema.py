# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_schema import EXPECTED_STEP1_SCHEMA
from src.step1.orchestrate_step1 import orchestrate_step1_state

def test_step1_output_matches_schema():
    """
    Verifies that the orchestrator returns a SolverState object containing
    all required Step 1 data, and that its JSON-safe representation
    matches the expected schema structure.
    """
    # 1. Build canonical input (frozen dummy)
    input_dict = solver_input_schema_dummy()

    # 2. Run real Step 1 (returns SolverState object)
    state = orchestrate_step1_state(input_dict)

    # 3. Check object attribute existence (instead of dictionary keys)
    for key in EXPECTED_STEP1_SCHEMA:
        assert hasattr(state, key), f"SolverState object missing attribute: {key}"

    # 4. Validate fields (Object attribute access)
    assert isinstance(state.fields, dict)
    for f in ["P", "U", "V", "W"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray)

    # 5. Mask semantics (Internal boolean arrays for Step 2 logic)
    assert isinstance(state.mask, np.ndarray)
    assert isinstance(state.is_fluid, np.ndarray)
    assert isinstance(state.is_boundary_cell, np.ndarray)
    assert isinstance(state.is_solid, np.ndarray)

    # 6. Constants and Boundary Conditions
    assert isinstance(state.constants, dict)
    assert isinstance(state.boundary_conditions, dict)

    # 7. Empty containers Step 1 must initialize for Step 2/3
    assert isinstance(state.operators, dict)
    assert isinstance(state.ppe, dict)
    assert isinstance(state.health, dict)

    # 8. Schema Validation via JSON-safe conversion
    # This proves the object can be serialized/deserialized correctly
    json_state = state.to_json_safe()
    for key in EXPECTED_STEP1_SCHEMA:
        assert key in json_state, f"JSON-safe output missing key: {key}"
    
    # Verify that arrays were converted to lists in the JSON-safe version
    assert isinstance(json_state["mask"], list)
    assert isinstance(json_state["fields"]["U"], list)