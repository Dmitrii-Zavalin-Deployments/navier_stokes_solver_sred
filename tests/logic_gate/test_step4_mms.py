# tests/logic_gate/test_step4_mms.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_logic_gate_4_linear_advection_transport():
    """
    LOGIC GATE 4: Linear Advection.
    Input: u(x) = x, v=0, w=0
    Analytical Truth: (u ∂u/∂x) = x * 1 = x
    Success Metric: Builder must return the original velocity values.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    dx = state.grid["dx"]
    
    # Input field: u = x
    U_analytic = np.zeros((nx+1, ny, nz), order='F')
    for i in range(nx+1):
        U_analytic[i, :, :] = i * dx
    state.fields["U"] = U_analytic.flatten(order='F')
    
    # Build Advection Operator (Logic typically handled in Step 2 or 4)
    state = orchestrate_step2(state)
    adv_op = state.operators.get("advection_u")
    
    if adv_op is None:
        pytest.skip("Advection operator not implemented in state.")
        
    # Execute advection
    result = adv_op.dot(state.fields["U"]).reshape((nx+1, ny, nz), order='F')
    
    # Verification on internal nodes
    np.testing.assert_allclose(result[1:-1, 1:-1, 1:-1], U_analytic[1:-1, 1:-1, 1:-1], 
                               rtol=1e-10, err_msg="Logic Gate 4 Failure: Advection failed linear transport")