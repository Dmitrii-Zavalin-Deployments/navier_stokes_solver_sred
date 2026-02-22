# tests/property_integrity/test_theory_of_grid_lifecycle.py

import pytest
import numpy as np

# Explicitly importing the factories to ensure direct initialization
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Define the lifecycle stages for parametrization to ensure recursive coverage
LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_theory_grid_spacing_derivation_integrity(stage_name, factory):
    """
    Theory: Verify that Δx = (x_max - x_min) / nx is consistent across all steps.
    This validates the logic: Bound Definitions -> Metric Calculation -> Departmental Integrity.
    
    We verify that 'dx' lives in the 'grid' department and is consistent through the entire lifecycle.
    """
    nx, ny, nz = 50, 20, 10
    
    # Expected derived metrics based on the domain [0, 1]^3
    exp_dx = 1.0 / nx
    exp_dy = 1.0 / ny
    exp_dz = 1.0 / nz

    state = factory(nx=nx, ny=ny, nz=nz)

    # 1. Verification of Departmental Integrity
    assert "dx" in state.grid, f"{stage_name}: Geometry department missing 'dx'"
    assert "dy" in state.grid, f"{stage_name}: Geometry department missing 'dy'"
    assert "dz" in state.grid, f"{stage_name}: Geometry department missing 'dz'"

    # 2. Verification of Derivation Consistency
    # Formula: Δx must equal the range divided by cell count
    calc_dx = (state.grid["x_max"] - state.grid["x_min"]) / nx
    assert np.isclose(state.grid["dx"], calc_dx), f"{stage_name}: Grid spacing drift from bounds"
    assert np.isclose(state.grid["dx"], exp_dx), f"{stage_name}: Grid spacing mismatch with theory"

def test_theory_extended_geometry_consistency():
    """
    Verify that extended fields in Step 4 and Final Output maintain coordinate alignment.
    Staggered check: U-velocity faces = nx + 1, plus 2 ghosts = nx + 3.
    """
    nx, ny, nz = 10, 10, 10
    
    # Test both Step 4 and the Final Step 5 Output Schema
    for factory in [make_step4_output_dummy, make_output_schema_dummy]:
        state = factory(nx=nx, ny=ny, nz=nz)
        
        # Logic: If interior is nx, extended pressure must be nx + 2 ghosts
        assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2)
        
        # Staggered Logic: 
        # U faces (nx+1) + 2 ghosts = nx + 3
        # V faces (ny+1) + 2 ghosts = ny + 3
        # W faces (nz+1) + 2 ghosts = nz + 3
        assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2)
        assert state.V_ext.shape == (nx + 2, ny + 3, nz + 2)
        assert state.W_ext.shape == (nx + 2, ny + 2, nz + 3)