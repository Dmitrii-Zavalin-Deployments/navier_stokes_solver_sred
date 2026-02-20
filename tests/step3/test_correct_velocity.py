# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _wire_mock_gradients(state, value=0.0):
    """Utility to inject mock gradient operators into the state."""
    state.operators["grad_x"] = lambda P: np.full_like(state.fields["U"], value)
    state.operators["grad_y"] = lambda P: np.full_like(state.fields["V"], value)
    state.operators["grad_z"] = lambda P: np.full_like(state.fields["W"], value)

def test_correction_math():
    """Verify u = u* - (dt/rho)*grad_p math logic."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # Wire gradients to 1.0
    _wire_mock_gradients(state, value=1.0)
    
    # U* = 5.0, Grad = 1.0 -> New U should be 4.0
    U_star = np.full_like(state.fields["U"], 5.0)
    P_dummy = np.zeros_like(state.fields["P"])
    
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, P_dummy)
    
    # Check internal nodes (avoiding boundary logic)
    assert np.allclose(U_new[1:-1, 1:-1, 1:-1], 4.0)

def test_internal_solid_mask_neighbor_rule():
    """Verify that if a cell is solid, its surrounding faces are zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    _wire_mock_gradients(state, value=0.0)

    # Set center cell to solid
    state.is_fluid[1, 1, 1] = False

    U_star = np.ones_like(state.fields["U"])
    P_dummy = np.zeros_like(state.fields["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, U_star, U_star, P_dummy)

    # The U-face at index 1 and 2 in the x-axis touches the solid cell [1,1,1]
    # Faces are between cells: Face 1 is between cell 0 and 1. Face 2 is between cell 1 and 2.
    assert U_new[1, 1, 1] == 0.0
    assert U_new[2, 1, 1] == 0.0

def test_minimal_grid_no_crash():
    """Check stability on 1x1x1 grid."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    _wire_mock_gradients(state, value=0.0)

    U_star = np.zeros_like(state.fields["U"])
    P_dummy = np.zeros_like(state.fields["P"])

    U_new, V_new, W_new = correct_velocity(state, U_star, U_star, U_star, P_dummy)

    assert U_new.shape == U_star.shape