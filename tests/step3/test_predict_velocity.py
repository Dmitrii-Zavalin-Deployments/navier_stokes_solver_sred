# tests/step3/test_predict_velocity.py

import numpy as np
import pytest
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from scipy.sparse import csr_matrix

def test_predict_velocity_structural_integrity():
    """Verify that U_star fields are created with correct shapes and keys."""
    # 1. Setup from Step 2 Dummy
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.constants["dt"] = 0.1
    state.constants["rho"] = 1.0
    state.constants["mu"] = 0.001
    
    # Define shapes based on staggered grid
    shape_u = state.fields["U"].shape
    shape_v = state.fields["V"].shape
    shape_w = state.fields["W"].shape

    # Explicitly add velocity-specific operators that the predictor expects
    state.operators["lap_u"] = csr_matrix((state.fields["U"].size, state.fields["U"].size))
    state.operators["lap_v"] = csr_matrix((state.fields["V"].size, state.fields["V"].size))
    state.operators["lap_w"] = csr_matrix((state.fields["W"].size, state.fields["W"].size))
    
    state.operators["advection_u"] = csr_matrix((state.fields["U"].size, state.fields["U"].size))
    state.operators["advection_v"] = csr_matrix((state.fields["V"].size, state.fields["V"].size))
    state.operators["advection_w"] = csr_matrix((state.fields["W"].size, state.fields["W"].size))

    # 2. Run Prediction
    predict_velocity(state)

    # 3. Assertions
    assert "U_star" in state.intermediate_fields
    assert state.intermediate_fields["U_star"].shape == shape_u
    assert state.intermediate_fields["V_star"].shape == shape_v
    assert state.intermediate_fields["W_star"].shape == shape_w

def test_predict_velocity_viscous_diffusion():
    """Test if velocity increases correctly while respecting boundary zeroing."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.constants["dt"] = 1.0
    state.constants["mu"] = 1.0 
    state.constants["rho"] = 1.0
    
    # Setup Identity Laplacian: Lap(U) = U
    size_u = state.fields["U"].size
    state.operators["lap_u"] = csr_matrix(np.eye(size_u))
    state.operators["advection_u"] = csr_matrix((size_u, size_u))
    
    # Initialize internal field to 1.0
    state.fields["U"][:] = 1.0
    
    predict_velocity(state)
    
    # Expected Physics: U* = U + dt * ( (mu/rho) * Lap(U) )
    # U* = 1.0 + 1.0 * (1.0) = 2.0
    
    # We must slice [1:-1] because the solver enforces 0.0 at the domain boundaries
    internal_u_star = state.intermediate_fields["U_star"][1:-1, :, :]
    
    assert np.allclose(internal_u_star, 2.0), "Internal velocity should be 2.0"
    
    # Verify that the boundaries were correctly zeroed out by the predictor
    assert np.all(state.intermediate_fields["U_star"][0, :, :] == 0.0)
    assert np.all(state.intermediate_fields["U_star"][-1, :, :] == 0.0)