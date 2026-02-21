# tests/property_integrity/test_grid_propagation.py

import pytest
import numpy as np
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from src.step1.initialize_grid import initialize_grid
from src.step1.allocate_fields import allocate_fields
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step3.correct_velocity import correct_velocity

def test_step1_metric_calculation_integrity():
    """Theory Step 1: Verify Δx calculation reaches the state constants."""
    nx = 50
    # Use dummy to get a valid structure, then trigger the math
    state = make_step4_output_dummy(nx=nx)
    state.config["grid"].update({"x_min": 0.0, "x_max": 5.0}) 
    
    initialize_grid(state)
    
    # Verify: 5.0 / 50 = 0.1
    assert np.isclose(state.grid["dx"], 0.1)
    assert np.isclose(state.constants["dx"], 0.1)

def test_step1_staggered_allocation_integrity():
    """Theory Step 2: Verify MAC grid staggered shapes from Dummy."""
    nx, ny, nz = 10, 5, 2
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Verify the staggered face-centered logic
    assert state.fields["U"].shape == (nx + 1, ny, nz)
    assert state.fields["V"].shape == (nx, ny + 1, nz)
    assert state.fields["W"].shape == (nx, ny, nz + 1)
    assert state.fields["P"].shape == (nx, ny, nz)

def test_step2_laplacian_metric_scaling():
    """Theory Step 3: Verify 1/dx^2 scaling in operators."""
    def get_laplacian_diagonal(res):
        st = make_step4_output_dummy(nx=res, ny=res, nz=res)
        # Manually set metrics to ensure builder uses resolution-specific math
        st.grid.update({"dx": 1.0/res, "dy": 1.0/res, "dz": 1.0/res})
        st.constants.update({"dx": 1.0/res, "dy": 1.0/res, "dz": 1.0/res})
        
        A = build_laplacian_operators(st)
        return abs(A.diagonal()[0])

    # If resolution doubles (10->20), dx halves, 1/dx^2 must quadruple (4x)
    val_low = get_laplacian_diagonal(10)
    val_high = get_laplacian_diagonal(20)
    
    assert np.isclose(val_high / val_low, 4.0), "Laplacian diagonal failed to scale with 1/dx^2"

def test_step3_gradient_metric_scaling():
    """Theory Step 4: Verify Velocity Correction uses dt/dx."""
    nx = 10
    state = make_step4_output_dummy(nx=nx)
    state.constants["dt"] = 0.1
    state.constants["dx"] = 0.5
    state.constants["rho"] = 1.0
    
    # Setup a unit pressure drop across one cell
    p_field = np.zeros((nx, 4, 4))
    p_field[1, :, :] = 1.0 
    u_star = np.zeros((nx+1, 4, 4))
    
    # Math: u_new = u_star - (dt/rho) * (dp/dx)
    # u_new = 0 - (0.1/1.0) * (1.0/0.5) = -0.2
    u_new, _, _ = correct_velocity(state, u_star, np.zeros_like(state.fields["V"]), 
                                   np.zeros_like(state.fields["W"]), p_field)
    
    assert np.isclose(u_new[1, 0, 0], -0.2)

# =================================================================
# FUTURE IMPLEMENTATION (STEP 4)
# =================================================================
# def test_step4_ghost_zone_integrity():
#     """Theory Step 5: Verify Extended Fields (Step 4) account for resolution."""
#     nx, ny, nz = 8, 8, 8
#     state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
#     
#     # Extended fields must include 2 ghost layers for Pressure, 3 for Velocity
#     assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2)
#     assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2)