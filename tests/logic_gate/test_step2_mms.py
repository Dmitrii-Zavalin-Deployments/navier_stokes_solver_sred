# tests/logic_gate/test_step2_mms.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_logic_gate_2_quadratic_laplacian():
    """
    LOGIC GATE 2: Quadratic Field MMS.
    Manufactured Solution: p(x,y,z) = x² + y² + z²
    Analytical Truth: ∇²p = 2 + 2 + 2 = 6.0
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state = orchestrate_step2(state)
    
    L = state.operators.get("laplacian")
    dx, dy, dz = state.grid["dx"], state.grid["dy"], state.grid["dz"]
    
    # Manufacture p field at cell centers
    p_analytic = np.zeros((nx, ny, nz), order='F')
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x, y, z = (i+0.5)*dx, (j+0.5)*dy, (k+0.5)*dz
                p_analytic[i,j,k] = x**2 + y**2 + z**2
    
    # Execute Matrix-Vector Multiplication
    res_vec = L.dot(p_analytic.flatten(order='F'))
    res_3d = res_vec.reshape((nx, ny, nz), order='F')
    
    # Success Metric: 6.0 on internal nodes (full stencil)
    internal_res = res_3d[1:-1, 1:-1, 1:-1]
    np.testing.assert_allclose(internal_res, 6.0, rtol=1e-10, 
                               err_msg="Logic Gate 2 Failure: Laplacian failed to recover 6.0")