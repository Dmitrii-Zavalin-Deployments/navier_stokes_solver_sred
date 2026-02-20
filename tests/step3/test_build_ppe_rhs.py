# tests/step3/test_build_ppe_rhs.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def _get_total_vel_dof(state):
    """Helper to calculate the total velocity vector length."""
    u_size = state.fields["U"].size
    v_size = state.fields["V"].size
    w_size = state.fields["W"].size
    return u_size + v_size + w_size

def test_zero_divergence():
    """If divergence operator is a zero matrix, RHS must be zero."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    
    # Create an actual zero sparse matrix of the correct shape [P_size, Vel_size]
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    state.operators["divergence"] = csr_matrix((p_size, vel_size))

    rhs = build_ppe_rhs(state, state.fields["U"], state.fields["V"], state.fields["W"])
    assert np.allclose(rhs, 0.0)

def test_uniform_divergence():
    """If every velocity contributes to a divergence of 1, RHS must be scaled."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    
    # Create a 'Sum' matrix: when multiplied by a vector of 1s, it returns 1 for every P cell.
    # We do this by putting a 1.0 in the first column for every row.
    data = np.ones(p_size)
    indices = np.zeros(p_size) # all point to the first velocity DOF
    indptr = np.arange(p_size + 1)
    state.operators["divergence"] = csr_matrix((data, indices, indptr), shape=(p_size, vel_size))

    # Set velocity to have 1.0 at index 0, 0.0 elsewhere
    U = np.zeros_like(state.fields["U"])
    U[0,0,0] = 1.0
    V = state.fields["V"]
    W = state.fields["W"]

    rhs = build_ppe_rhs(state, U, V, W)

    rho = state.constants["rho"]
    dt = state.constants["dt"]
    assert np.allclose(rhs, rho / dt)

def test_solid_zeroing():
    """RHS must be zeroed inside solid cells even if divergence is non-zero."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    
    # Matrix that returns 1.0 everywhere
    data = np.ones(p_size)
    indices = np.zeros(p_size)
    indptr = np.arange(p_size + 1)
    state.operators["divergence"] = csr_matrix((data, indices, indptr), shape=(p_size, vel_size))

    # Mark a solid cell (is_fluid = False)
    state.is_fluid[1, 1, 1] = False

    # Input velocity that triggers the "1.0" result
    U = np.zeros_like(state.fields["U"]); U[0,0,0] = 1.0
    
    rhs = build_ppe_rhs(state, U, state.fields["V"], state.fields["W"])

    # The cell at [1,1,1] must be 0.0 because it is solid
    assert rhs[1, 1, 1] == 0.0
    # Other cells should be non-zero
    assert rhs[0, 0, 0] > 0.0

def test_minimal_grid_no_crash():
    """Minimal 1×1×1 grid check."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    
    state.operators["divergence"] = csr_matrix((p_size, vel_size))

    rhs = build_ppe_rhs(state, state.fields["U"], state.fields["V"], state.fields["W"])

    assert rhs.shape == (1, 1, 1)