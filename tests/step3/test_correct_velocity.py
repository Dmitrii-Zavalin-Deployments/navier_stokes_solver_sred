# tests/step3/test_correct_velocity.py

import numpy as np
import pytest
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_correction_math():
    """
    Verify the projection step math: u_new = u_star - (dt/rho) * grad_p.
    With dt=1, rho=1, u_star=5, and grad_p=1, the result must be 4.0.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.config["dt"] = 1.0  # Synced with source code expectation
    state.constants["rho"] = 1.0
    
    # Define domain as all fluid to prevent masking from interfering
    state.is_fluid = np.ones((3, 3, 3), dtype=bool)
    
    # Inject mock gradient operators that return 1.0
    state.operators["grad_x"] = lambda p: np.ones_like(state.fields["U"])
    state.operators["grad_y"] = lambda p: np.ones_like(state.fields["V"])
    state.operators["grad_z"] = lambda p: np.ones_like(state.fields["W"])
    
    U_star = np.full_like(state.fields["U"], 5.0)
    V_star = np.full_like(state.fields["V"], 5.0)
    W_star = np.full_like(state.fields["W"], 5.0)
    P_dummy = np.zeros_like(state.fields["P"])
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_dummy)
    
    # Math: 5.0 - (1.0/1.0)*1.0 = 4.0
    assert np.allclose(U_new, 4.0)
    assert np.allclose(V_new, 4.0)
    assert np.allclose(W_new, 4.0)

def test_correction_neighbor_masking():
    """Verify that velocity is zeroed if adjacent to a solid cell."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.config["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # Create a solid wall at x=1
    state.is_fluid = np.ones((3, 3, 3), dtype=bool)
    state.is_fluid[1, :, :] = False 
    
    # Mock zero gradients
    state.operators["grad_x"] = lambda p: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda p: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda p: np.zeros_like(state.fields["W"])
    
    U_star = np.full_like(state.fields["U"], 5.0)
    
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, np.zeros((3,3,3)))
    
    # The U face between i=0 and i=1 should be zeroed because cell i=1 is solid
    assert U_new[0, 1, 1] == 0.0
    # The U face between i=1 and i=2 should be zeroed because cell i=1 is solid
    assert U_new[1, 1, 1] == 0.0
    # A face far from the solid (e.g., at i=2 if it existed) would remain 5.0

def test_correction_zero_gradient():
    """Verify that a zero pressure gradient leaves velocity unchanged in fluid."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.config["dt"] = 0.1
    state.constants["rho"] = 1.0
    state.is_fluid = np.ones((3, 3, 3), dtype=bool)
    
    state.operators["grad_x"] = lambda p: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda p: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda p: np.zeros_like(state.fields["W"])
    
    U_star = np.random.rand(*state.fields["U"].shape)
    
    U_new, V_new, W_new = correct_velocity(state, U_star, U_star, U_star, np.zeros((3,3,3)))
    
    assert np.allclose(U_new, U_star)