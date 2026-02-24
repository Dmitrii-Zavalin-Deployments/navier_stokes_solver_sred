# tests/logic_gate/test_step2_integration.py

import pytest
import numpy as np
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_step2_quadratic_field_mms():
    """
    MMS Verification for Step 2: Discrete Operators
    Scenario: The Quadratic Field p = x^2 + y^2 + z^2
    Verification of: Laplacian operator accuracy and Sparse Matrix assembly.
    
    Constitutional Compliance: 
    - Phase D.9: Per-Step MMS
    - Phase C.7: Scale Guard (Sparse check)
    """
    # 1. Setup Input: Start from the frozen Step 1 Output Dummy
    # We use a 4x4x4 grid to ensure we have enough interior points for a valid stencil
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 2. Execute Step 2 Pipeline
    # This should build state.operators['laplacian'], 'divergence', etc.
    state = orchestrate_step2(state)
    
    # 3. Mathematical Verification (The 'Constitutional' Truths)
    
    # TRUTH A: Scale Guard Compliance
    # The Laplacian MUST be a sparse matrix, not a dense array (Article 7)
    from scipy.sparse import issparse
    assert issparse(state.operators['laplacian']), "Scale Guard Violation: Laplacian must be a sparse matrix"

    # TRUTH B: The Quadratic MMS
    # Map the analytical field p = x^2 + y^2 + z^2 onto the grid
    dx, dy, dz = state.grid['dx'], state.grid['dy'], state.grid['dz']
    
    # Create coordinate arrays for cell centers
    x = np.linspace(dx/2, 1.0 - dx/2, nx)
    y = np.linspace(dy/2, 1.0 - dy/2, ny)
    z = np.linspace(dz/2, 1.0 - dz/2, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Manufactured Solution: p = x^2 + y^2 + z^2
    p_analytical = X**2 + Y**2 + Z**2
    p_flat = p_analytical.flatten()
    
    # Apply the discrete Laplacian operator: L * p
    # In a perfect world (central difference on quadratic), this is exact
    laplacian_result = state.operators['laplacian'] @ p_flat
    
    # Verification: Result should be a constant field of 6.0
    # We ignore boundary rows for now or ensure BCs are handled in orchestrate_step2
    # For a simple interior check:
    expected_value = 6.0
    
    # Reshape back to 3D to check interior accuracy
    lap_3d = laplacian_result.reshape((nx, ny, nz))
    
    # Check interior (indices 1 to -1) to avoid boundary condition noise in initial logic
    interior_result = lap_3d[1:-1, 1:-1, 1:-1]
    
    assert np.allclose(interior_result, expected_value, atol=1e-10), \
        f"MMS Failure: Laplacian of quadratic field should be 6.0, got {np.mean(interior_result)}"

    # TRUTH C: Operator Dimensions
    # Laplacian should be (Total Cells x Total Cells)
    expected_dim = nx * ny * nz
    assert state.operators['laplacian'].shape == (expected_dim, expected_dim)

    print("\n[MMS PASS] Step 2: Quadratic Field results in exact Laplacian curvature.")