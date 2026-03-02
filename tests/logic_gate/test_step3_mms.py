# tests/logic_gate/test_step3_mms.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_logic_gate_3_divergence_pulse():
    """
    LOGIC GATE 3: Divergence Pulse.
    Input: u(x,y,z) = (x, 0, 0) -> ∇·u = 1.0
    Analytical Truth: ||∇·u_new|| < 10⁻¹² after projection.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state = orchestrate_step2(state)
    
    # Setup Mass Leak: u = x
    dx = state.grid.dx
    U = np.zeros((nx+1, ny, nz), order='F')
    for i in range(nx+1):
        U[i, :, :] = i * dx
    state.fields["U"] = U.flatten(order='F')
    
    # Action: Pressure Projection
    state_out = orchestrate_step3(state)
    
    # Verification: Calculate Final Divergence
    D = state.operators.divergence
    v_total = np.concatenate([state_out.fields["U"], state_out.fields["V"], state_out.fields["W"]])
    div_norm = np.linalg.norm(D.dot(v_total), np.inf)
    
    assert div_norm < 1e-12, f"Logic Gate 3 Failure: Mass balance not restored. Div Norm: {div_norm}"