# tests/physics_gate/test_operators.py

import pytest
import numpy as np
from scipy.sparse import issparse
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from src.step2.orchestrate_step2 import orchestrate_step2

def test_vector_identity_div_grad_is_laplacian():
    """
    PHYSICS GATE: Phase E.10
    Identity: div(grad(p)) == laplacian(p)
    
    This ensures that our discrete Gradient and Divergence operators
    are perfectly adjoint and match the Laplacian stencil.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state = orchestrate_step2(state)
    
    D = state.operators['divergence']
    G = state.operators['gradient']
    L = state.operators['laplacian']
    
    # 1. Create a random pressure field
    p = np.random.rand(nx * ny * nz)
    
    # 2. Compute L*p and Div(Grad(p))
    lap_result = L @ p
    div_grad_result = D @ (G @ p)
    
    # 3. Mandate: Identity must hold within 1e-12
    # This is the 'Physics Gate'—if this fails, Step 3 will never converge.
    np.testing.assert_allclose(lap_result, div_grad_result, atol=1e-12, 
                               err_msg="Fundamental Vector Identity Violation: Div(Grad) != Lap")

def test_null_space_of_gradient():
    """
    PHYSICS GATE: Null-Space Audit
    Identity: grad(constant) == 0
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state = orchestrate_step2(state)
    
    G = state.operators['gradient']
    constant_p = np.ones(nx * ny * nz) * 5.0 # Constant field
    
    grad_result = G @ constant_p
    
    assert np.max(np.abs(grad_result)) < 1e-14, "Ghost Physics Detected: Grad(constant) is not zero!"

def test_laplacian_symmetry():
    """
    PHYSICS GATE: Symmetry & Dissipation
    Property: L == L.T (Symmetric Negative Definite)
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    state = orchestrate_step2(state)
    
    L = state.operators['laplacian']
    
    # Check if L is symmetric: L - L.T should be zero
    diff = L - L.T
    assert diff.nnz == 0, "Thermodynamic Violation: Laplacian is not symmetric!"