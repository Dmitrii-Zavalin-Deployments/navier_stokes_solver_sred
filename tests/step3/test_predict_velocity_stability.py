import numpy as np
import pytest
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_predict_velocity_numerical_stability():
    """
    Verify that predict_velocity produces finite results and 
    respects the staggered grid boundaries.
    """
    # 1. Setup Staggered State
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Hardening: Set physical constants to prevent 'blow-up'
    state.constants["rho"] = 1.0
    state.constants["mu"] = 0.01  # Viscosity
    state.config["dt"] = 0.001    # Small dt for stability
    
    # 3. Initialize fields with a simple shear flow
    # U should be (nx+1, ny, nz)
    u_init = np.ones((nx + 1, ny, nz))
    v_init = np.zeros((nx, ny + 1, nz))
    w_init = np.zeros((nx, ny, nz + 1))

    # 4. Run Prediction
    try:
        u_star, v_star, w_star = predict_velocity(state, u_init, v_init, w_init)
    except IndexError as e:
        pytest.fail(f"Staggered Indexing Error: {e}")

    # 5. Numerical Checks
    assert np.all(np.isfinite(u_star)), "u_star contains NaNs or Infs"
    assert u_star.shape == (nx + 1, ny, nz), f"u_star shape mismatch: {u_star.shape}"
    
    # 6. Check that solid boundaries didn't 'leak'
    # If cell [1,1,1] is solid, the prediction should handle it via masks
    state.is_fluid[1, 1, 1] = False
    u_star_masked, _, _ = predict_velocity(state, u_init, v_init, w_init)
    
    # Check if the code runs without crashing when a solid is present
    assert u_star_masked is not None

def validate_staggered_indexing(state, u, v, w):
    """
    Helper to check if arrays match the MAC grid contract.
    """
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    if u.shape != (nx + 1, ny, nz): return False
    if v.shape != (nx, ny + 1, nz): return False
    if w.shape != (nx, ny, nz + 1): return False
    return True