import pytest
import numpy as np
from src.step2.build_laplacian_operators import build_laplacian_operators
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

"""
test_step2_mms.py
Constitutional Role: Logic Gate 2 (The Quadratic Laplacian Field).
Compliance: Phase C (MMS Verification).

Verification: (Laplacian_Matrix @ p_vector) must be exactly 6.0 everywhere
for the manufactured solution p(x,y,z) = x² + y² + z².
"""

def test_laplacian_quadratic_field_mms():
    """
    CLEAN ROOM VERIFICATION:
    Tests if the discrete Laplacian operator correctly recovers the analytical 
    second derivative of a quadratic field.
    """
    # 1. Setup a controlled grid (nx=4, ny=4, nz=4)
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Build the Laplacian operator (Step 2 logic)
    # This will likely fail currently as build_laplacian_operators is incomplete
    operators = build_laplacian_operators(state)
    L = operators["L"]
    
    # 3. Manufacture the Solution: p = x² + y² + z²
    # We need the coordinates for every cell center
    dx, dy, dz = state.grid["dx"], state.grid["dy"], state.grid["dz"]
    p_analytic = np.zeros((nx, ny, nz), order='F')
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                z = (k + 0.5) * dz
                p_analytic[i, j, k] = x**2 + y**2 + z**2
    
    # Flatten p to vector form (must match the solver's internal flattening)
    p_vec = p_analytic.flatten(order='F')
    
    # 4. Execute the Operator: result = L * p
    laplacian_result = L.dot(p_vec)
    
    # 5. Parity Audit
    # For p = x² + y² + z², ∇²p = 2 + 2 + 2 = 6.0
    # We only check internal cells (away from boundaries) where the stencil is full
    expected_value = 6.0
    
    # Reshape result back to 3D to check internal nodes easily
    res_3d = laplacian_result.reshape((nx, ny, nz), order='F')
    
    # Audit internal 2x2x2 core (indices 1 to n-2)
    internal_result = res_3d[1:-1, 1:-1, 1:-1]
    
    np.testing.assert_allclose(
        internal_result, 
        expected_value, 
        rtol=1e-10, 
        err_msg="Logic Gate 2 Failure: Discrete Laplacian failed to recover analytical truth (6.0)."
    )

if __name__ == "__main__":
    pytest.main([__file__])
