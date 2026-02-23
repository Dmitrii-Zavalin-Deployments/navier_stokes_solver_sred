# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
import pytest
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from src.step1.orchestrate_step1 import orchestrate_step1_state

def test_step1_output_matches_foundation_contract():
    """
    The Auditor: Verifies the SolverState object produced by Step 1 matches 
    the 'Foundation' dummy, allowing for NumPy array masks.
    """
    # 1. Build canonical input
    input_dict = solver_input_schema_dummy()

    # 2. Run real Step 1
    state = orchestrate_step1_state(input_dict)

    # 3. Check for critical departments
    critical_departments = [
        "grid", "fields", "mask", "is_fluid", "constants", 
        "boundary_conditions", "ppe", "health", "history"
    ]
    for dept in critical_departments:
        assert hasattr(state, dept), f"SolverState missing: {dept}"

    # 4. Grid and Shape Logic
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    total_expected_cells = nx * ny * nz

    # 5. Mask Logic (Fixed for NumPy Array vs List flexibility)
    # The real orchestrator returns a 3D array; we verify its total element count
    mask_as_array = np.asanyarray(state.mask)
    assert mask_as_array.size == total_expected_cells, f"Mask size mismatch: {mask_as_array.size} != {total_expected_cells}"
    
    # 6. Field Allocation (Staggered Grid Check)
    assert state.fields["U"].shape == (nx + 1, ny, nz)
    assert state.fields["V"].shape == (nx, ny + 1, nz)
    assert state.fields["W"].shape == (nx, ny, nz + 1)

    # 7. Verification of JSON Roundtrip (The actual flattening test)
    # This ensures that even if the internal state is a 3D array, 
    # the serialization method handles the conversion to list.
    json_state = state.to_json_safe()
    assert isinstance(json_state["mask"], list), "JSON output must provide a flattened list"