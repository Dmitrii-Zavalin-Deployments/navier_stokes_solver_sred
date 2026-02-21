import numpy as np
import pytest
from src.step3.correct_velocity import correct_velocity
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_bc_precedence_inflow_vs_solid():
    """
    PROVE: Internal Solid Masks take priority over Domain Inflow.
    If a solid block is placed exactly on an inflow boundary, 
    the velocity must be 0.0 (Solid), not 5.0 (Inflow).
    """
    # 1. Setup 3x3x3 grid
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 0.01
    
    # 2. Define an Inflow at x_min
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 5.0}}
    ]
    
    # 3. Place a solid block at the very first cell (0,0,0)
    state.is_fluid.fill(True)
    state.is_fluid[0, 0, 0] = False 
    
    # 4. Create dummy intermediate fields (U*) and Pressure
    U_star = np.ones_like(state.fields["U"]) # Initial guess 1.0
    V_star = np.zeros_like(state.fields["V"])
    W_star = np.zeros_like(state.fields["W"])
    P_new = np.zeros_like(state.fields["P"]) # No pressure gradient for this test

    # --- EXECUTION LOOP ---
    
    # Step A: Correction (subtracts grad P and zeroes internal solid faces)
    U_corr, V_corr, W_corr = correct_velocity(state, U_star, V_star, W_star, P_new)
    
    # Step B: Final Enforcement (Applies Domain BCs and re-checks solids)
    fields_final = apply_boundary_conditions_post(state, U_corr, V_corr, W_corr, P_new)
    U_final = fields_final["U"]

    # --- VERIFICATION ---
    
    # Face at U[0,0,0] is on the boundary AND touches a solid cell.
    # It must be 0.0 because the Solid Mask in apply_boundary_conditions_post 
    # should run after (or alongside) the domain BC logic.
    assert U_final[0, 0, 0] == 0.0, "Solid mask should override Inflow at the boundary"

    # Face at U[1,0,0] is internal but touches the solid cell [0,0,0].
    # It must also be 0.0.
    assert U_final[1, 0, 0] == 0.0, "Internal solid face must be zeroed"

def test_projection_reduces_divergence():
    """
    PROVE: The correct_velocity step actually uses the pressure gradient 
    to modify the velocity field.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants = {"rho": 1.0}
    state.config["dt"] = 1.0
    state.is_fluid.fill(True)

    # Mock a pressure gradient where P increases linearly in X
    # P = [0, 1, 2], so grad_x = 1.0
    P = np.zeros((3,3,3))
    P[0,:,:] = 0.0
    P[1,:,:] = 1.0
    P[2,:,:] = 2.0
    
    U_star = np.ones_like(state.fields["U"]) # Initial U = 1.0
    
    # If U_new = U_star - (dt/rho)*grad_P
    # U_new = 1.0 - (1.0/1.0)*1.0 = 0.0
    U_new, _, _ = correct_velocity(state, U_star, U_star, U_star, P)
    
    # Check interior face (index 1)
    assert np.allclose(U_new[1, :, :], 0.0), "Pressure gradient was not subtracted correctly"

def test_no_slip_tangential_remains_unchanged():
    """
    PROVE: Domain BCs for 'no-slip' only affect the Normal component.
    Tangential components are left to the mask or friction logic.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    # x_min no-slip: only affects U.
    state.config["boundary_conditions"] = [{"location": "x_min", "type": "no-slip"}]
    
    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    
    fields = apply_boundary_conditions_post(state, U, V, V, np.zeros_like(state.fields["P"]))
    
    assert fields["U"][0, :, :].max() == 0.0  # Normal component zeroed
    assert fields["V"][0, :, :].max() == 1.0  # Tangential component untouched