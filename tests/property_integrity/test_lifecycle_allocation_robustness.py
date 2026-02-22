# tests/property_integrity/test_lifecycle_allocation_robustness.py

import pytest
import numpy as np

# Import the full factory suite
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Define the lifecycle stages for parametrization
LIFECYCLE_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """
    Robustness: Verifies that core field shapes (P, U, V, W) match 
    the grid dimensions (nx, ny, nz) across all 5 stages.
    """
    nx, ny, nz = 8, 6, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    expected_shape = (nx, ny, nz)
    expected_total = nx * ny * nz

    # 1. Metadata Agreement
    assert state.grid["nx"] == nx
    assert state.grid["total_cells"] == expected_total, f"Total cells mismatch at {stage_name}"

    # 2. Core Field Shapes (verify_shape_consistency logic)
    for field in ["P", "U", "V", "W"]:
        assert state.fields[field].shape == expected_shape, (
            f"{stage_name}: Field '{field}' shape mismatch. "
            f"Expected {expected_shape}, got {state.fields[field].shape}"
        )

@pytest.mark.parametrize("stage_name, factory", LIFECYCLE_STAGES)
def test_lifecycle_ppe_dimension_intent(stage_name, factory):
    """
    Robustness: Verifies that the PPE dimension (N_total) is 
    consistent and present in every stage of the state.
    """
    nx, ny, nz = 4, 3, 2
    state = factory(nx=nx, ny=ny, nz=nz)
    expected_dim = nx * ny * nz
    
    # Validates that the PPE department 'plan' survives every orchestrator
    assert state.ppe["dimension"] == expected_dim, f"PPE dimension lost at {stage_name}"

def test_step3_intermediate_field_allocation():
    """
    Specific check for Step 3: Projection requires intermediate velocity 
    storage (U*, V*, W*) which must match internal grid dimensions.
    """
    nx, ny, nz = 5, 5, 5
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    for comp in ["U", "V", "W"]:
        assert state.intermediate_fields[comp].shape == (nx, ny, nz), \
            f"Step 3 intermediate field {comp} shape mismatch."

def test_ghost_cell_allocation_logic():
    """
    Verify the 'Ghost Cell' expansion logic in Step 4 and Final Output.
    Formula: Extended = Interior + 2 (one ghost on each side).
    Note: Staggered velocity faces (e.g., U) have nx+1 faces internally.
    """
    nx, ny, nz = 10, 10, 10
    
    # Check both Step 4 and the Final Output wrapper
    s4 = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    s_final = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)
    
    for stage_name, state in [("Step 4", s4), ("Final Output", s_final)]:
        # Pressure (Cell-centered): 10 + 2 = 12
        assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2), f"{stage_name} P_ext failure"
        
        # Velocity U (Face-centered in X): (10+1) + 2 = 13
        # centered in Y, Z: 10 + 2 = 12
        assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2), f"{stage_name} U_ext failure"
        assert state.V_ext.shape == (nx + 2, ny + 3, nz + 2), f"{stage_name} V_ext failure"
        assert state.W_ext.shape == (nx + 2, ny + 2, nz + 3), f"{stage_name} W_ext failure"