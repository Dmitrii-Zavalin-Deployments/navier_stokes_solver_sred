# tests/logic_gate/test_step2_mms.py

import pytest
import numpy as np
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.build_gradient_operators import build_gradient_operators
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

"""
test_step2_mms.py
Constitutional Role: Logic Gate 2 (The Quadratic Laplacian Field) & Gate 2.A (Calculus Identity).
Compliance: Phase C (MMS Verification) and Operator Consistency Audit.

Verification:
1. (L @ p_vector) == 6.0 for p = x² + y² + z²
2. (D @ G) == L (The fundamental staggered grid identity)
"""

def test_laplacian_quadratic_field_mms():
    """
    LOGIC GATE 2: Quadratic Field MMS.
    Tests if the discrete Laplacian operator recovers the analytical 
    second derivative (∇²p = 6.0) for p = x² + y² + z².
    """
    # 1. Setup a controlled grid (nx=4, ny=4, nz=4)
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Build the Laplacian operator
    operators = build_laplacian_operators(state)
    # Support both key formats to ensure test flexibility
    L = operators.get("L") if operators else state.operators.get("laplacian")
    
    if L is None:
        pytest.fail("Operator 'L' or 'laplacian' not found in state.operators.")
    
    # 3. Manufacture the Solution: p = x² + y² + z²
    dx, dy, dz = state.grid["dx"], state.grid["dy"], state.grid["dz"]
    p_analytic = np.zeros((nx, ny, nz), order='F')
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                z = (k + 0.5) * dz
                p_analytic[i, j, k] = x**2 + y**2 + z**2
    
    p_vec = p_analytic.flatten(order='F')
    
    # 4. Execute the Operator: result = L * p
    laplacian_result = L.dot(p_vec)
    
    # 5. Parity Audit: ∇²p = 2 + 2 + 2 = 6.0
    expected_value = 6.0
    res_3d = laplacian_result.reshape((nx, ny, nz), order='F')
    
    # Audit internal 2x2x2 core where the 7-point stencil is full
    internal_result = res_3d[1:-1, 1:-1, 1:-1]
    
    np.testing.assert_allclose(
        internal_result, 
        expected_value, 
        rtol=1e-10, 
        err_msg="Logic Gate 2 Failure: Discrete Laplacian failed to recover analytical truth (6.0)."
    )

def test_gate_2a_calculus_identity_verification():
    """
    GATE 2.A: Identity Verification.
    Verifies the staggered grid consistency: Div(Grad(p)) == Laplacian(p).
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Generate the Calculus Triple
    build_laplacian_operators(state)
    build_divergence_operator(state)
    build_gradient_operators(state)
    
    L = state.operators.get("laplacian")
    D = state.operators.get("divergence")
    G = state.operators.get("gradient")
    
    if any(op is None for op in [L, D, G]):
        pytest.fail("Calculus Triple missing: Ensure L, D, and G are populated in state.operators.")
    
    # Composite the Divergence and Gradient: L_computed = D @ G
    L_computed = D @ G
    
    # Identity Check: The difference must be numerically negligible
    # Note: We compare internal stencil coefficients (the 'data' of the sparse matrix)
    diff = L - L_computed
    
    if diff.nnz > 0:
        max_err = np.max(np.abs(diff.data))
        assert max_err < 1e-12, f"Gate 2.A Failure: Identity mismatch. Max Error: {max_err}"
    else:
        # If diff.nnz is 0, they are perfectly identical
        assert True

if __name__ == "__main__":
    pytest.main([__file__])