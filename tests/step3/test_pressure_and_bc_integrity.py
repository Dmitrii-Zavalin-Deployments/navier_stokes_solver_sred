# tests/step3/test_pressure_and_bc_integrity.py

import numpy as np
import pytest
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_ppe_solid_pressure_zeroing():
    """Verify solve_pressure respects solid masks and non-zero fluid pressure."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.is_fluid.fill(True)
    state.is_fluid[1, 1, 1] = False
    
    # Balanced RHS (sum=0) to ensure solver convergence in singular systems
    rhs_ppe = np.zeros((nx, ny, nz))
    rhs_ppe[0, 0, 0] = 1.0
    rhs_ppe[2, 2, 2] = -1.0
    
    P_new, metadata = solve_pressure(state, rhs_ppe)

    assert P_new[1, 1, 1] == 0.0
    assert np.any(P_new[state.is_fluid] != 0.0)

def test_ppe_mean_subtraction_logic():
    """Ensures singular systems have a zero-mean pressure field."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.ppe["ppe_is_singular"] = True
    
    rhs_ppe = np.zeros((3,3,3))
    rhs_ppe[0,0,0], rhs_ppe[-1,-1,-1] = 1.0, -1.0
    P_new, _ = solve_pressure(state, rhs_ppe)
    
    assert np.isclose(np.mean(P_new), 0.0, atol=1e-10)

def test_ppe_solver_metadata_contract():
    """Verify the solver returns the expected metadata."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    rhs_ppe = np.zeros((2, 2, 2))
    
    _, metadata = solve_pressure(state, rhs_ppe)
    
    for key in ["converged", "solver_status", "method", "tolerance_used"]:
        assert key in metadata