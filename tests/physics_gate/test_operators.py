# tests/physics_gate/test_operators.py

import pytest
import numpy as np
from scipy import sparse
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def laplacian_matrix():
    """Setup a standard 4x4x4 grid Laplacian for auditing without relying on Step 1 orchestration."""
    raw_data = solver_input_schema_dummy()
    nx, ny, nz = 4, 4, 4
    
    # Manually assemble SolverState to bypass jsonschema/Step 1 dependencies
    state = SolverState(
        config=raw_data['config'],
        grid={'nx': nx, 'ny': ny, 'nz': nz, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
        boundary_conditions=raw_data['boundary_conditions']
    )
    # Initialize fluid mask (all fluid for audit)
    state.is_fluid = np.ones(nx * ny * nz, dtype=bool)
    state.operators = {}
    
    build_laplacian_operators(state)
    return state.operators["laplacian"]

def test_gate_2a_calculus_identity(laplacian_matrix):
    """Check identity dimensions."""
    N = 64 # 4*4*4
    assert laplacian_matrix.shape == (N, N)

def test_gate_2b_symmetry_and_sparsity(laplacian_matrix):
    """Check Symmetry L=L^T and Sparsity <= 7N."""
    diff = laplacian_matrix - laplacian_matrix.T
    assert diff.nnz == 0 or np.all(np.abs(diff.data) < 1e-12)
    assert laplacian_matrix.nnz <= 7 * laplacian_matrix.shape[0]

def test_gate_2c_null_space_audit(laplacian_matrix):
    """Check that Laplacian of a constant field is zero (or identity for BCs)."""
    ones = np.ones(laplacian_matrix.shape[0])
    result = laplacian_matrix.dot(ones)
    for val in result:
        # Should be 0 for Poisson or 1 for BC-enforced rows
        assert np.isclose(val, 0.0) or np.isclose(val, 1.0)
