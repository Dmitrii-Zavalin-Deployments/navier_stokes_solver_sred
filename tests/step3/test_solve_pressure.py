# tests/step3/test_solve_pressure.py

import numpy as np
import pytest
from scipy.sparse import eye
from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_config_key_mapping_and_logic():
    """Verify solver correctly utilizes ppe_tolerance and ppe_max_iter from config."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    p_size = state.fields["P"].size
    
    # Setup Identity Matrix as mock Laplacian (Non-singular case)
    state.ppe["A"] = eye(p_size, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # Inject specific JSON-style config
    state.config["solver_settings"] = {
        "solver_type": "PCG",
        "ppe_tolerance": 1e-10,
        "ppe_max_iter": 500
    }
    
    # Input RHS = 5.0 everywhere. For A=I, P should be 5.0.
    rhs = np.full((2, 2, 2), 5.0)
    P_new, meta = solve_pressure(state, rhs)

    assert meta["tolerance_used"] == 1e-10
    assert np.allclose(P_new, 5.0)
    assert meta["converged"] is True
    assert meta["is_singular"] is False

def test_singular_mean_subtraction_physics():
    """Ensure that for singular systems, the mean pressure is zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    p_size = state.fields["P"].size
    state.ppe["A"] = eye(p_size, format="csr")
    state.ppe["ppe_is_singular"] = True
    
    # RHS = 10.0 results in P = 10.0 before mean subtraction
    rhs = np.full((3, 3, 3), 10.0)
    P_new, meta = solve_pressure(state, rhs)

    # After mean subtraction, result should be 0.0
    assert abs(np.mean(P_new)) < 1e-12
    assert meta["is_singular"] is True

def test_minimal_grid_no_crash():
    """Minimal 1x1x1 grid check for solve_pressure."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    state.ppe["A"] = eye(1, format="csr")
    state.ppe["ppe_is_singular"] = False

    rhs = np.zeros((1, 1, 1))
    P_new, meta = solve_pressure(state, rhs)

    assert P_new.shape == (1, 1, 1)
    assert "converged" in meta