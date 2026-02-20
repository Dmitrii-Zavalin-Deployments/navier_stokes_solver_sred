# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _wire_mock_gradients(state, value=0.0):
    """Inject mock gradient operators that return a specific constant value."""
    state.operators["grad_x"] = lambda P: np.full_like(state.fields["U"], value)
    state.operators["grad_y"] = lambda P: np.full_like(state.fields["V"], value)
    state.operators["grad_z"] = lambda P: np.full_like(state.fields["W"], value)

def test_correction_math():
    """Verify u = u* - (dt/rho)*grad_p math logic."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["dt"] = 1.0
    state.constants["rho"] = 1.0
    _wire_mock_gradients(state, value=1.0)
    
    U_star = np.full_like(state.fields["U"], 5.0)
    V_star = np.full_like(state.fields["V"], 5.0)
    W_star = np.full_like(state.fields["W"], 5.0)
    P_dummy = np.zeros_like(state.fields["P"])
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_dummy)
    
    # Internal staggered faces should be 5.0 - (1.0/1.0)*1.0 = 4.0
    assert np.allclose(U_new[1:3, :, :], 4.0)
    assert np.allclose(V_new[:, 1:3, :], 4.0)
    assert np.allclose(W_new[:, :, 1:3], 4.0)

def test_internal_solid_mask_neighbor_rule():
    """Verify that if a cell is solid, its surrounding faces are zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_mock_gradients(state, value=0.0)
    state.is_fluid[1, 1, 1] = False 

    U_new, _, _ = correct_velocity(state, np.ones_like(state.fields["U"]), 
                                  np.ones_like(state.fields["V"]), 
                                  np.ones_like(state.fields["W"]), np.zeros_like(state.fields["P"]))
    assert U_new[1, 1, 1] == 0.0
    assert U_new[2, 1, 1] == 0.0