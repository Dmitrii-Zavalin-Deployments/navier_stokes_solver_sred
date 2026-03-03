# tests/logic_gate/test_step4_mms.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_logic_gate_4_linear_advection_transport():
    """
    LOGIC GATE 4: Linear Advection.
    Constitutional Role: Verify the advection structure is initialized.
    Success Metric: Advection weights and indices must be populated.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Execute Step 2 which builds the advection stencils/weights
    state = orchestrate_step2(state)
    
    # Verification: Per src/solver_state.py, advection is an AdvectionStructure
    assert state.advection is not None, "Advection structure not initialized."
    
    # Verify weights were allocated (Total velocity DOF x 8 neighbors)
    weights = state.advection.weights
    indices = state.advection.indices
    
    assert weights is not None
    assert indices is not None
    assert weights.ndim == 2
    assert weights.shape[1] == 8, "Staggered advection should utilize 8-point interpolation."
    
    # Verify that we actually have non-zero data
    assert np.any(weights != 0), "Advection weights are all zero (Identity failure)."