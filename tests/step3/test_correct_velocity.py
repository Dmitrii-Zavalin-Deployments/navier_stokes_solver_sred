# tests/step3/test_correct_velocity.py

import numpy as np
import pytest
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_correction_math():
    """Verify projection math with correct staggered shapes."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.config["dt"] = 1.0
    state.constants["rho"] = 1.0
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    
    # Mock gradients with staggered shapes
    state.operators["grad_x"] = lambda p: np.ones((nx+1, ny, nz))
    state.operators["grad_y"] = lambda p: np.ones((nx, ny+1, nz))
    state.operators["grad_z"] = lambda p: np.ones((nx, ny, nz+1))
    
    U_star = np.full((nx+1, ny, nz), 5.0)
    V_star = np.full((nx, ny+1, nz), 5.0)
    W_star = np.full((nx, ny, nz+1), 5.0)
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, np.zeros((nx,ny,nz)))
    
    assert np.allclose(U_new, 4.0)
    assert np.allclose(V_new, 4.0)
    assert np.allclose(W_new, 4.0)

def test_correction_neighbor_masking():
    """Verify staggered masking at solid boundaries."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.config["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # Cell (1,1,1) is solid
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_fluid[1, 1, 1] = False
    
    state.operators["grad_x"] = lambda p: np.zeros((nx+1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny+1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz+1))
    
    U_star = np.full((nx+1, ny, nz), 5.0)
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, np.zeros((nx,ny,nz)))
    
    # Face at index 1 (between cell 0 and 1) and index 2 (between cell 1 and 2) should be 0
    assert U_new[1, 1, 1] == 0.0
    assert U_new[2, 1, 1] == 0.0
    # Boundary faces (0 and 3) should remain 5.0 (as they aren't 'internal' to this mask)
    assert U_new[0, 1, 1] == 5.0