import numpy as np
import pytest
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.step3.solve_pressure import solve_pressure
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_domain_boundary_isolation_fluid_first():
    """
    Verify domain BCs are enforced on a purely fluid domain.
    Ensures logic is independent of the solid-masking system.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)  # Fluid-First fix
    
    # Target: y_max normal component (V)
    state.config["boundary_conditions"] = [{"location": "y_max", "type": "free-slip"}]

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    # Verify: Boundary face is zeroed, interior face remains 1.0
    assert fields_out["V"][:, -1, :].max() == 0.0
    assert fields_out["V"][:, 1, :].min() == 1.0

def test_ppe_solid_pressure_zeroing():
    """
    Verify solve_pressure respects solid masks and doesn't fight velocity zeros.
    If a cell is solid, its pressure should be anchored to zero.
    """
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Setup a solid 'island' in the center cell (1,1,1)
    state.is_fluid.fill(True)
    state.is_fluid[1, 1, 1] = False
    
    # 2. Mock a source term (RHS) that is non-zero everywhere
    rhs_ppe = np.ones((nx, ny, nz))
    
    # 3. Solve (Note: make_step2_output_dummy provides the Laplacian A in state.ppe)
    P_new, metadata = solve_pressure(state, rhs_ppe)

    # 4. Verification:
    # Pressure at the solid island must be exactly 0.0 per Step 6 of solve_pressure
    assert P_new[1, 1, 1] == 0.0
    # Pressure at fluid cells should be adjusted by mean subtraction (non-zero)
    assert np.any(P_new[state.is_fluid] != 0.0)

def test_ppe_mean_subtraction_logic():
    """
    Ensures that for singular systems (pure Neumann), the pressure 
    field is normalized so the mean is zero.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.ppe["ppe_is_singular"] = True
    
    # Constant RHS results in a specific pressure field
    rhs_ppe = np.ones((3, 3, 3))
    P_new, _ = solve_pressure(state, rhs_ppe)
    
    # The mean of the pressure field should be effectively zero
    assert np.isclose(np.mean(P_new), 0.0, atol=1e-10)

def test_ppe_solver_metadata_contract():
    """Verify the solver returns the expected metadata for orchestration logging."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    rhs_ppe = np.zeros((2, 2, 2))
    
    _, metadata = solve_pressure(state, rhs_ppe)
    
    required_keys = ["converged", "solver_status", "method", "tolerance_used"]
    for key in required_keys:
        assert key in metadata
    assert metadata["method"] == "PCG (Jacobi)"