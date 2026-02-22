# tests/property_integrity/test_boundary_condition_integrity.py

import pytest
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Per Property Tracking Matrix: BC types are critical for Step 2 (Matrix A) 
# and Step 3 (Face velocity constraints).
BC_INTEGRITY_STAGES = [
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

VALID_BC_TYPES = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]

@pytest.mark.parametrize("stage_name, factory", BC_INTEGRITY_STAGES)
def test_boundary_type_persistence_and_validity(stage_name, factory):
    """
    Integrity: Verify BC types persist as standardized strings across 
    the computation and output phases.
    """
    state = factory()
    
    # 1. Structure Check: Must be a list of objects per Solver Input Schema
    assert isinstance(state.boundary_conditions, list), f"{stage_name}: BCs must be a list"
    assert len(state.boundary_conditions) > 0, f"{stage_name}: BC list is empty"

    for bc in state.boundary_conditions:
        # 2. Key Presence
        assert "location" in bc, f"{stage_name}: BC missing 'location' key"
        assert "type" in bc, f"{stage_name}: BC missing 'type' key"
        
        # 3. Enum Validation: Ensure type matches the Schema allowed values
        # This prevents "noslip" vs "no-slip" drift.
        assert bc["type"] in VALID_BC_TYPES, f"{stage_name}: Invalid BC type '{bc['type']}'"

def test_step2_matrix_bc_logic():
    """
    Physics: Verify that in Step 2, BCs are available to set the Laplacian A.
    """
    state = make_step2_output_dummy()
    # If BCs are missing, the PPE solver in Step 2 cannot build a valid matrix
    assert "A" in state.ppe, "Step 2: PPE Matrix A missing"
    assert len(state.boundary_conditions) >= 6, "Step 2: 3D domain requires 6 boundary faces"