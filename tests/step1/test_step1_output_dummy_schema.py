# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
import pytest
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import solver_step1_output_dummy
from src.step1.orchestrate_step1 import orchestrate_step1_state

def test_step1_output_matches_dummy_foundation():
    """
    The Auditor: Verifies the SolverState object produced by Step 1 matches 
    the 'Foundation' dummy used by the rest of the verification bridge.
    """
    # 1. Build canonical input (The Genesis)
    input_dict = solver_input_schema_dummy()

    # 2. Run real Step 1 (returns SolverState object)
    state = orchestrate_step1_state(input_dict)

    # 3. Retrieve the expected structure (The Foundation)
    # We use the keys from the dummy dictionary to audit the real object's attributes
    expected_foundation = solver_step1_output_dummy()

    # 4. Check object attribute existence (Constitutional Alignment)
    for key in expected_foundation.keys():
        assert hasattr(state, key), f"SolverState object missing required Foundation attribute: {key}"

    # 5. Validate fields (Numpy Array Handlers)
    # Staggered grid requirement: P, U, V, W must be present as Tensors
    assert isinstance(state.fields, dict)
    for f in ["P", "U", "V", "W"]:
        assert f in state.fields
        assert isinstance(state.fields[f], np.ndarray), f"Field {f} must be a NumPy array."

    # 6. Mask semantics (Logical Gates for Step 2 Discretization)
    # Boolean masks are required for high-performance logical indexing
    assert isinstance(state.mask, np.ndarray)
    assert isinstance(state.is_fluid, np.ndarray)
    assert state.is_fluid.dtype == bool

    # 7. Placeholder Container Integrity
    # These must be initialized as dictionaries to prevent Step 2/3 crashes
    assert isinstance(state.operators, dict)
    assert isinstance(state.ppe, dict)
    assert isinstance(state.health, dict)

    # 8. Serialization Contract (JSON-Safe Roundtrip)
    # Proves the object can be serialized for checkpoints or export
    json_state = state.to_json_safe()
    
    # Verify array-to-list conversion for JSON compliance
    assert isinstance(json_state["mask"], list)
    assert isinstance(json_state["fields"]["U"], list)