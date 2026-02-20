# tests/step3/test_correct_velocity.py

import numpy as np
from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_correction_math():
    """
    Verify the projection step math: u_new = u_star - (dt/rho) * grad_p.
    With dt=1, rho=1, u_star=5, and grad_p=1, the result must be 4.0.
    """
    # 1. Setup state with unit constants for clean math
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # 2. Inject mock gradient operators that return 1.0 everywhere
    # This ensures (dt/rho) * grad_p = (1.0/1.0) * 1.0 = 1.0
    state.operators["grad_x"] = lambda p: np.ones_like(state.fields["U"])
    state.operators["grad_y"] = lambda p: np.ones_like(state.fields["V"])
    state.operators["grad_z"] = lambda p: np.ones_like(state.fields["W"])
    
    # 3. Setup input fields
    U_star = np.full_like(state.fields["U"], 5.0)
    V_star = np.full_like(state.fields["V"], 5.0)
    W_star = np.full_like(state.fields["W"], 5.0)
    P_dummy = np.zeros_like(state.fields["P"]) # Content irrelevant due to mock
    
    # 4. Execute correction
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, P_dummy)
    
    # 5. Assertions
    # Expected: 5.0 - 1.0 = 4.0
    # We check a slice of the internal domain to avoid boundary edge cases
    expected_val = 4.0
    actual_val = U_new[1, 1, 1]
    
    assert np.allclose(U_new[1:2, 1:2, 1:2], expected_val), f"Expected {expected_val}, got {actual_val}"
    assert np.allclose(V_new[1:2, 1:2, 1:2], expected_val)
    assert np.allclose(W_new[1:2, 1:2, 1:2], expected_val)

def test_correction_zero_gradient():
    """Verify that a zero pressure gradient leaves velocity unchanged."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # Mock gradient as zero
    state.operators["grad_x"] = lambda p: np.zeros_like(state.fields["U"])
    
    U_star = np.full_like(state.fields["U"], 5.0)
    P_dummy = np.zeros_like(state.fields["P"])
    
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, P_dummy)
    
    assert np.array_equal(U_new, U_star), "Velocity should not change when grad(P) is zero."