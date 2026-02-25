import pytest
import numpy as np
from scipy import sparse
from src.step2.build_laplacian_operators import build_laplacian_operators
from tests.helpers.solver_input_schema_dummy import make_step1_output_dummy

@pytest.fixture
def laplacian_matrix():
    """Setup a standard 4x4x4 grid Laplacian for auditing."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    # The function now returns the dict or populates state based on your last fix
    result = build_laplacian_operators(state)
    
    # Handle both return types (dict return or state population)
    if result and "L" in result:
        return result["L"]
    return state.operators["laplacian"]

def test_gate_2a_calculus_identity(laplacian_matrix):
    """
    Identity: Laplacian must be a square matrix representing 
    the divergence of the gradient.
    """
    N = 4*4*4
    assert laplacian_matrix.shape == (N, N), "Matrix must be NxN"
    assert isinstance(laplacian_matrix, (sparse.csr_matrix, sparse.csc_matrix)), "Matrix must be sparse"

def test_gate_2b_symmetry_and_sparsity(laplacian_matrix):
    """
    Symmetry: L = L^T (for internal fluid cells).
    Sparsity: Non-zero elements per row <= 7 (Central + 6 neighbors).
    """
    # Check Symmetry: L - L.T should be effectively zero
    diff = laplacian_matrix - laplacian_matrix.T
    # We use a small epsilon for floating point comparison
    assert diff.nnz == 0 or np.all(np.abs(diff.data) < 1e-12), "Matrix L is not symmetric"

    # Check Sparsity: nnz(L) <= 7N
    N = laplacian_matrix.shape[0]
    assert laplacian_matrix.nnz <= 7 * N, f"Matrix too dense: {laplacian_matrix.nnz} > {7*N}"

def test_gate_2c_null_space_audit(laplacian_matrix):
    """
    Null-Space Audit: The Laplacian of a constant field must be zero.
    (Sum of every row in L should be 0, except for Dirichlet boundary rows).
    """
    N = laplacian_matrix.shape[0]
    ones = np.ones(N)
    result = laplacian_matrix.dot(ones)
    
    # In a pure Neumann system, result is 0 everywhere. 
    # With boundaries/solid cells, some rows (like the 1.0 identity rows) 
    # might sum to 1.0. We check for consistent logic.
    rows_with_data = np.where(result != 0)[0]
    
    # Audit: If a row doesn't sum to 0, it must be a boundary or solid cell (identity 1.0)
    for row_idx in rows_with_data:
        assert np.isclose(result[row_idx], 0.0) or np.isclose(result[row_idx], 1.0), \
            f"Row {row_idx} sums to {result[row_idx]}, violating physical null-space."