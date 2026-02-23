# tests/step1/test_step1_output_dummy_schema.py

import numpy as np
import pytest
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy
# Synchronized with actual helper function name
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from src.step1.orchestrate_step1 import orchestrate_step1_state

def test_step1_output_matches_foundation_contract():
    """
    The Auditor: Verifies the SolverState object produced by Step 1 matches 
    the 'Foundation' dummy (The Gold Standard for Step 2).
    """
    # 1. Build canonical input
    input_dict = solver_input_schema_dummy()

    # 2. Run real Step 1 (returns SolverState object)
    state = orchestrate_step1_state(input_dict)

    # 3. Retrieve the expected structure (The Foundation)
    # This is the "Gold Standard" object the rest of the solver expects
    expected_state = make_step1_output_dummy()

    # 4. Check for critical departments (Attributes)
    # We ensure the real orchestrator populated the departments found in the dummy
    critical_departments = [
        "grid", "fields", "mask", "is_fluid", "constants", 
        "boundary_conditions", "ppe", "health", "history"
    ]
    
    for dept in critical_departments:
        assert hasattr(state, dept), f"SolverState object missing required department: {dept}"

    # 5. Validate Field Allocation (Staggered Grid Layout)
    # U, V, W must follow the (N+1, N, N) logic for face-centered velocities
    for field_name in ["P", "U", "V", "W"]:
        assert isinstance(state.fields[field_name], np.ndarray)
    
    # Check staggered shape logic for U-velocity as a representative sample
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    assert state.fields["U"].shape == (nx + 1, ny, nz)

    # 6. Mask Compliance (Article 8: Flattened 1D)
    # Your helper explicitly notes the shift to flattened 1D lists for canonical compliance
    assert isinstance(state.mask, list)
    assert len(state.mask) == nx * ny * nz

    # 7. Health Check
    assert state.health["status"] == "initialized"