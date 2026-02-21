# tests/property_integrity/test_theory_of_grid_lifecycle.py

import pytest
import numpy as np
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

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
    """
    Verify that Δx logic is correctly represented in the Dummy.
    This respects the 'Frozen Dummy' rule—no manual substitution.
    """
    nx = 50
    # The dummy itself must calculate dx correctly (1.0 / 50 = 0.02)
    factory = DUMMIES[step_name]
    state = factory(nx=nx)
    
    expected_dx = 1.0 / nx
    
    # Verify the Dummy's internal consistency
    assert np.isclose(state.grid["dx"], expected_dx), f"Grid dx mismatch in {step_name}"
    assert np.isclose(state.constants["dx"], expected_dx), f"Constants dx mismatch in {step_name}"

# # ------------------------------------------------------------------
# # 2. Operator Scaling Persistence (Tested against Steps 2, 3, 4)
# # ------------------------------------------------------------------
# @pytest.mark.parametrize("step_name", ["step2", "step3", "step4"])
# def test_theory_step2_operator_scaling_persistence(step_name):
#     """Verify operators in dummies are pre-built and accessible."""
#     res = 10
#     factory = DUMMIES[step_name]
#     state = factory(nx=res)
    
#     # In a Pure Path, we don't build them here; we verify they EXIST in the dummy
#     operators = state.operators
#     A = operators.get("P_laplacian")
    
#     assert A is not None, f"Laplacian operator missing in {step_name} dummy"
#     assert A.diagonal().size > 0

# # ------------------------------------------------------------------
# # 3. Correction Physics Persistence (Tested against Steps 3, 4)
# # ------------------------------------------------------------------
# @pytest.mark.parametrize("step_name", ["step3", "step4"])
# def test_theory_step3_correction_logic_persistence(step_name):
#     """Verify field dimensions in dummies support the staggered math."""
#     nx = 10
#     factory = DUMMIES[step_name]
#     state = factory(nx=nx)
    
#     # Verify the staggered dimensions provided by the dummy
#     # U-velocity should be (nx+1, ny, nz)
#     assert state.fields["U"].shape[0] == nx + 1
#     assert state.fields["P"].shape[0] == nx

# # ------------------------------------------------------------------
# # 4. Step 4 Ghost Zone Integrity
# # ------------------------------------------------------------------
# def test_theory_step4_ghost_zone_integrity():
#     """Verify Extended Fields (Step 4) follow the N+2/N+3 Ghost Rule."""
#     nx, ny, nz = 8, 8, 8
#     state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
#     # Extended fields must include 2 ghost layers for Pressure, 3 for Velocity
#     assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2)
#     assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2)