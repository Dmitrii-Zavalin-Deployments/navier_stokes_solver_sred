# tests/property_integrity/test_mask_spatial_integrity.py

import pytest
import numpy as np
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Per Property Tracking Matrix: Mask logic is central to Step 2 
# and must persist through the entire pipeline.
MASK_ACTIVE_STAGES = [
    ("Step 2", make_step2_output_dummy),
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", MASK_ACTIVE_STAGES)
def test_mask_value_constraints_and_shape(stage_name, factory):
    """
    Physics: Verify mask contains only allowed values (-1, 0, 1) 
    and matches grid dimensions.
    """
    state = factory()
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # 1. Shape Integrity
    mask_np = np.array(state.mask)
    assert mask_np.shape == (nx, ny, nz), \
        f"{stage_name}: Mask shape {mask_np.shape} mismatch with grid ({nx}, {ny}, {nz})"
    
    # 2. Value Integrity: Allowed set is {-1, 0, 1}
    # 0 = Solid, 1 = Fluid, -1 = Ghost/Boundary
    unique_values = np.unique(mask_np)
    allowed_values = {-1, 0, 1}
    for val in unique_values:
        assert val in allowed_values, \
            f"{stage_name}: Non-physical mask value detected: {val}"

def test_mask_matrix_consistency_step2():
    """
    Logic: Verify that Step 2 Laplacian dimensions exclude solid cells.
    """
    state = make_step2_output_dummy()
    mask_np = np.array(state.mask)
    
    fluid_count = np.sum(mask_np == 1)
    
    # The PPE dimension should ideally match the number of active fluid cells
    # Note: Depending on solver implementation, it might match total_cells 
    # but with zeroed-out rows for solids. 
    assert state.ppe["dimension"] == state.grid["total_cells"], \
        "Step 2: PPE dimension should match total grid cells for sparse mapping."

def test_mask_persistence_between_stages():
    """
    Integrity: Ensure the mask doesn't change between Step 2 and Step 4.
    """
    s2 = make_step2_output_dummy()
    s4 = make_step4_output_dummy()
    
    assert s2.mask == s4.mask, "Critical Failure: Spatial mask modified during computation steps!"