# tests/property_integrity/test_lifecycle_allocation_robustness.py

import pytest

from src.common.field_schema import FI
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Only stages that return a SolverState with a .fields manager
STATE_BASED_STAGES = [
    ("Step 1", make_step1_output_dummy),
    ("Step 2", make_step2_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

# Stages that return a StencilBlock
BLOCK_BASED_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
]

@pytest.mark.parametrize("stage_name, factory", STATE_BASED_STAGES)
def test_lifecycle_grid_dimensions_match_fields(stage_name, factory):
    """
    Robustness: Verifies buffer size allocation against grid cell count (nx * ny * nz).
    Applies to stages using the monolithic SolverFieldManager.
    """
    nx, ny, nz = 8, 6, 4
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    state = factory(nx=nx, ny=ny, nz=nz)
    
    data = state.fields.data
    
    # Assertions verify that the monolithic column length matches expected cell count
    for field_idx in [FI.P, FI.VX, FI.VY, FI.VZ]:
        assert data[:, field_idx].size == n_cells, f"{stage_name}: Field index {field_idx} size mismatch"

@pytest.mark.parametrize("stage_name, factory", BLOCK_BASED_STAGES)
def test_block_allocation_integrity(stage_name, factory):
    """
    Validation: Verify that StencilBlocks in Step 3 and 4 allocate 
    individual component arrays correctly.
    """
    nx, ny, nz = 5, 5, 5
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    block = factory(nx=nx, ny=ny, nz=nz)
    
    # Verify the internal slots of the StencilBlock
    assert block._u.size == n_cells, f"{stage_name}: Velocity U size mismatch"
    assert block._v.size == n_cells, f"{stage_name}: Velocity V size mismatch"
    assert block._w.size == n_cells, f"{stage_name}: Velocity W size mismatch"
    assert block._p.size == n_cells, f"{stage_name}: Pressure P size mismatch"

def test_step3_intermediate_predictor_allocation():
    """
    Validation: Predictor fields (u*, v*, w*) in Step 3 block match dimensions.
    """
    nx, ny, nz = 5, 5, 5
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    block = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Accessing predictor slots directly from StencilBlock
    assert block._u_star.size == n_cells
    assert block._v_star.size == n_cells
    assert block._w_star.size == n_cells

def test_ghost_cell_capacity_parity():
    """
    Verify storage capacity across different stage architectures.
    """
    nx, ny, nz = 10, 10, 10
    n_cells = (nx + 2) * (ny + 2) * (nz + 2)
    
    # Check a Block (Step 4)
    block = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    assert block._p.size == n_cells, "Step 4: Block pressure array size mismatch"
    
    # Check a State (Final Output)
    state = make_output_schema_dummy(nx=nx, ny=ny, nz=nz)
    assert state.fields.data.shape[0] == n_cells, "Final Output: Monolithic buffer size mismatch"