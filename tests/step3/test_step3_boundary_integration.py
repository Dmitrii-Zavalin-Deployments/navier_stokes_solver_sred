# tests/step3/test_step3_boundary_integration.py

import numpy as np
import pytest
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def setup_mock_operators(state):
    """Inject mock gradient operators to satisfy correct_velocity requirements."""
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # grad_x: P(nx,ny,nz) -> U(nx+1,ny,nz)
    state.operators["grad_x"] = lambda p: np.zeros((nx + 1, ny, nz))
    # grad_y: P(nx,ny,nz) -> V(nx,ny+1,nz)
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny + 1, nz))
    # grad_z: P(nx,ny,nz) -> W(nx,ny,nz+1)
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz + 1))

def test_bc_precedence_inflow_vs_solid():
    """
    PROVE: Internal Solid Masks take priority over Domain Inflow.
    If a solid block is placed exactly on an inflow boundary, 
    the velocity must be 0.0 (Solid), not 5.0 (Inflow).
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 0.01
    setup_mock_operators(state)
    
    # 1. Define an Inflow at x_min
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 5.0}}
    ]
    
    # 2. Place a solid block at the first cell
    state.is_fluid.fill(True)
    state.is_fluid[0, 0, 0] = False 
    
    U_star = np.ones_like(state.fields["U"]) 
    V_star = np.zeros_like(state.fields["V"])
    W_star = np.zeros_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"])

    # --- EXECUTION ---
    # Correction Step
    U_c, V_c, W_c = correct_velocity(state, U_star, V_star, W_star, P_new)
    
    # Final Post-Enforcement Step
    fields_final = apply_boundary_conditions_post(state, U_c, V_c, W_c, P_new)
    U_final = fields_final["U"]

    # --- VERIFICATION ---
    # Face at U[0,0,0] must be 0.0 because it touches a solid cell
    assert U_final[0, 0, 0] == 0.0, "Solid mask failed to override Inflow"

def test_projection_reduces_divergence_logic():
    """
    PROVE: correct_velocity uses grad_x to modify U_star.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 1.0
    state.is_fluid.fill(True)
    
    # Mock grad_x to return a constant 1.0 gradient
    state.operators["grad_x"] = lambda p: np.ones((4, 3, 3))
    state.operators["grad_y"] = lambda p: np.zeros((3, 4, 3))
    state.operators["grad_z"] = lambda p: np.zeros((3, 3, 4))

    U_star = np.ones((4, 3, 3))
    P = np.zeros((3, 3, 3))
    
    # U_new = U_star - (dt/rho) * grad_x = 1.0 - (1.0/1.0)*1.0 = 0.0
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, P)
    
    assert np.all(U_new[1:-1, :, :] == 0.0)

def test_solid_mask_indexing_consistency():
    """
    PROVE: apply_boundary_conditions_post handles staggered shapes without IndexError.
    Specifically checks the W field (staggered in Z).
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.is_fluid[:, :, 1] = False # Solid 'floor' at k=1
    
    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])
    
    # This should not raise IndexError
    try:
        fields = apply_boundary_conditions_post(state, U, V, W, P)
    except IndexError as e:
        pytest.fail(f"IndexError raised in BC post-processing: {e}")
    
    # W[i, j, 1] and W[i, j, 2] touch the solid cell at k=1
    assert np.all(fields["W"][:, :, 1] == 0.0)
    assert np.all(fields["W"][:, :, 2] == 0.0)