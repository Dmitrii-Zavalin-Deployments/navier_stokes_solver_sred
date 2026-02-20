# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _wire_mock_gradients(state, value=0.0):
    """Inject mock gradient operators with staggered-compliant shapes."""
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
    
    # Check internal staggered faces (for nx=3, index 1 and 2 are internal)
    assert np.all(U_new[1:3, :, :] == 4.0)
    assert np.all(V_new[:, 1:3, :] == 4.0)
    assert np.all(W_new[:, :, 1:3] == 4.0)

def test_internal_solid_mask_neighbor_rule():
    """Verify that if a cell is solid, its surrounding faces are zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_mock_gradients(state, value=0.0)

    state.is_fluid[1, 1, 1] = False # Set center cell to solid

    U_star = np.ones_like(state.fields["U"])
    V_star = np.ones_like(state.fields["V"])
    W_star = np.ones_like(state.fields["W"])
    P_dummy = np.zeros_like(state.fields["P"])

    U_new, _, _ = correct_velocity(state, U_star, V_star, W_star, P_dummy)

    # U-faces adjacent to cell [1,1,1]
    assert U_new[1, 1, 1] == 0.0
    assert U_new[2, 1, 1] == 0.0

def test_minimal_grid_no_crash():
    """Check stability on 1x1x1 grid."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    _wire_mock_gradients(state, value=0.0)

    U_star = np.zeros_like(state.fields["U"])
    V_star = np.zeros_like(state.fields["V"])
    W_star = np.zeros_like(state.fields["W"])
    P_dummy = np.zeros_like(state.fields["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_dummy)

    assert U_new.shape == U_star.shape