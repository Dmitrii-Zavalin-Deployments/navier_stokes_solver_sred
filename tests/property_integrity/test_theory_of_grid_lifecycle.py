# tests/property_integrity/test_theory_of_grid_lifecycle.py

import pytest
import numpy as np
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

from src.step1.initialize_grid import initialize_grid
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step3.correct_velocity import correct_velocity

# The full lifecycle of state evolution
DUMMIES = {
    "step1": make_step1_output_dummy,
    "step2": make_step2_output_dummy,
    "step3": make_step3_output_dummy,
    "step4": make_step4_output_dummy
}

# ------------------------------------------------------------------
# 1. Grid Metrics Persistence (Tested against Steps 1, 2, 3, 4)
# ------------------------------------------------------------------
@pytest.mark.parametrize("step_name", ["step1", "step2", "step3", "step4"])
def test_theory_step1_grid_logic_persistence(step_name):
    """Verify that Δx logic remains valid across the entire state lifecycle."""
    nx = 50
    factory = DUMMIES[step_name]
    state = factory(nx=nx)
    
    # We update existing keys only. If 'grid' is missing, it crashes (Contract fail).
    state.config["grid"]["x_min"] = 0.0
    state.config["grid"]["x_max"] = 5.0
    
    initialize_grid(state)
    
    # 5.0 / 50 = 0.1
    assert np.isclose(state.grid["dx"], 0.1), f"dx failed at {step_name}"
    assert np.isclose(state.constants["dx"], 0.1), f"constant dx failed at {step_name}"

# # ------------------------------------------------------------------
# # 2. Operator Scaling Persistence (Tested against Steps 2, 3, 4)
# # ------------------------------------------------------------------
# @pytest.mark.parametrize("step_name", ["step2", "step3", "step4"])
# def test_theory_step2_operator_scaling_persistence(step_name):
#     """Verify 1/dx^2 scaling is preserved in operators through later stages."""
#     res = 10
#     factory = DUMMIES[step_name]
#     state = factory(nx=res)
    
#     # Strictly check scaling based on the dummy's own constants
#     state.constants["dx"] = 1.0 / res
#     state.constants["dy"] = 1.0 / res
#     state.constants["dz"] = 1.0 / res

#     A = build_laplacian_operators(state)
    
#     assert A is not None, f"Laplacian operator missing/null in {step_name}"
#     # Verify diagonal represents the center coefficient (usually -6.0 / dx^2)
#     assert A.diagonal().size > 0

# # ------------------------------------------------------------------
# # 3. Correction Physics Persistence (Tested against Steps 3, 4)
# # ------------------------------------------------------------------
# @pytest.mark.parametrize("step_name", ["step3", "step4"])
# def test_theory_step3_correction_logic_persistence(step_name):
#     """Verify velocity correction physics remains sound in the final stages."""
#     factory = DUMMIES[step_name]
#     state = factory(nx=10) # dx=0.1 (if 1.0/10)
    
#     # Align constants for the math check
#     state.constants["dt"] = 0.1
#     state.constants["dx"] = 0.5
#     state.constants["rho"] = 1.0
    
#     p_field = np.zeros_like(state.fields["P"])
#     p_field[1, :, :] = 1.0 
#     u_star = np.zeros_like(state.fields["U"])
    
#     # u = u* - (dt/rho) * (dp/dx) => 0 - (0.1/1.0) * (1.0/0.5) = -0.2
#     u_new, _, _ = correct_velocity(state, u_star, state.fields["V"], state.fields["W"], p_field)
    
#     assert np.isclose(u_new[1, 0, 0], -0.2), f"Physics mismatch in {step_name}"

# ------------------------------------------------------------------
# 4. FUTURE IMPLEMENTATION (STEP 4)
# ------------------------------------------------------------------
# def test_theory_step4_ghost_zone_integrity():
#     """Theory Step 5: Verify Extended Fields (Step 4) account for resolution."""
#     nx, ny, nz = 8, 8, 8
#     state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
#     
#     # Extended fields must include 2 ghost layers for Pressure, 3 for Velocity
#     assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2)
#     assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2)