# tests/property_integrity/test_boundary_values_propagation.py

import pytest
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

@pytest.mark.parametrize("stage_name, factory", [
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_boundary_values_presence_and_type(stage_name, factory):
    """
    Integrity: Verify boundary_conditions.values sub-dictionary persists.
    This ensures that specific inflow/outflow values are available for the PPE 
    Source Term (Step 2) and Ghost Cell assignment (Step 3).
    """
    state = factory()
    
    # Adaptive Access
    if isinstance(state, dict):
        bc_list = state.get("boundary_conditions", [])
    else:
        bc_list = getattr(state, "boundary_conditions", [])

    assert len(bc_list) > 0, f"Setup Error: No boundaries defined at {stage_name}"

    for bc in bc_list:
        # 1. Structural Check: 'values' key must exist for schema parity
        assert "values" in bc, f"Parity Failure: 'values' sub-dictionary missing at {stage_name} for {bc['location']}"
        
        # 2. Functional Check: Extract the values
        vals = bc["values"]
        
        # 3. Numeric Integrity Check (e.g., u, v, w, or p must be float/int if present)
        for component in ["u", "v", "w", "p"]:
            if component in vals:
                assert isinstance(vals[component], (int, float)), \
                    f"Type Error: Boundary value '{component}' at {stage_name} must be numeric."

def test_boundary_values_logic_match():
    """
    Specific Logic: Ensure that an 'inflow' type has corresponding velocity values.
    """
    state = make_step3_output_dummy()
    bc_list = state.get("boundary_conditions", []) if isinstance(state, dict) else state.boundary_conditions
    
    inflow_bc = next((bc for bc in bc_list if bc["type"] == "inflow"), None)
    
    if inflow_bc:
        assert "u" in inflow_bc["values"], "Logic Error: Inflow boundary must define at least one velocity component."