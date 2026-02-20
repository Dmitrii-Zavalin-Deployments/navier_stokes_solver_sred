# tests/step3/test_predict_velocity.py

import numpy as np
import pytest
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from scipy.sparse import csr_matrix

def test_predict_velocity_structural_integrity():
    """Verify that U_star fields are created with correct shapes."""
    # 1. Setup from Step 2 Dummy
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.constants["dt"] = 0.1
    state.constants["rho"] = 1.0
    state.constants["mu"] = 0.001
    
    # Ensure advection operators exist (even if empty for dummy)
    state.operators["advection_u"] = csr_matrix(state.operators["lap_u"].shape)
    state.operators["advection_v"] = csr_matrix(state.operators["lap_v"].shape)
    state.operators["advection_w"] = csr_matrix(state.operators["lap_w"].shape)

    # 2. Run Prediction
    predict_velocity(state)

    # 3. Assertions
    assert "U_star" in state.intermediate_fields
    assert state.intermediate_fields["U_star"].shape == state.fields["U"].shape
    assert state.intermediate_fields["V_star"].shape == state.fields["V"].shape
    assert state.intermediate_fields["W_star"].shape == state.fields["W"].shape

def test_predict_velocity_viscous_diffusion():
    """Test if velocity increases correctly when using an identity Laplacian."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.constants["dt"] = 1.0
    state.constants["mu"] = 1.0 
    state.constants["rho"] = 1.0
    
    # Manually inject an Identity Matrix as a Laplacian dummy
    # This simulates a case where Lap(U) = U
    size = state.fields["U"].size
    state.operators["lap_u"] = csr_matrix(np.eye(size))
    state.fields["U"][:] = 1.0
    
    # Set advection to zero matrix
    state.operators["advection_u"] = csr_matrix((size, size))
    
    predict_velocity(state)
    
    # Formula check: U* = U + dt * ( (mu/rho) * Lap(U) )
    # U* = 1.0 + 1.0 * ( 1.0 * 1.0 ) = 2.0
    assert np.allclose(state.intermediate_fields["U_star"], 2.0)