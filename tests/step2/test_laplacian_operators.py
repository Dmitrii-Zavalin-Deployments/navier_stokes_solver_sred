# tests/step2/test_laplacian_operators.py

import numpy as np
import pytest
from scipy.sparse import issparse

from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.create_fluid_mask import create_fluid_mask
# Use the correct helper name and alias it for the test
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy as make_step1_dummy_state


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    """
    Create a canonical Step-1 dummy state and override only the fields
    relevant for Laplacian operator tests.
    """
    # Create Step-1 dummy
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)

    # Ensure grid spacing is consistent
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx

    # Ensure mask is all fluid (1)
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Recompute fluid masks (state.is_fluid, state.is_boundary_cell, state.is_solid)
    create_fluid_mask(state)

    return state


# ------------------------------------------------------------
# 1. Sparse Check: Verify operator is a CSR matrix
# ------------------------------------------------------------
def test_laplacian_is_sparse():
    state = make_state()
    build_laplacian_operators(state)
    
    lap = state.operators.get("laplacian")
    assert lap is not None, "Laplacian operator not found in state.operators"
    assert issparse(lap), "Laplacian must be a scipy sparse matrix"
    assert lap.shape == (64, 64)  # 4*4*4 = 64


# ------------------------------------------------------------
# 2. Symmetry Check: Laplacian must be symmetric (A = A.T)
# ------------------------------------------------------------
def test_laplacian_symmetry():
    state = make_state(nx=3, ny=3, nz=3)
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"]
    # Check if A - A.transpose is nearly zero
    diff = lap - lap.transpose()
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0, atol=1e-10)


# ------------------------------------------------------------
# 3. Sum of Rows: Internal fluid cells sum to 0 (Conservation)
# ------------------------------------------------------------
def test_laplacian_row_sum_internal():
    # Large enough grid to have an "internal" cell at index [1,1,1]
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"].toarray()
    
    # Get index of an internal cell (not on boundary)
    # i + j*nx + k*nx*ny
    idx = 1 + 1*nx + 1*nx*ny
    
    # The sum of coefficients for an internal Laplacian row should be 0
    # because the center -6/h^2 cancels the six +1/h^2 neighbors
    assert np.isclose(np.sum(lap[idx, :]), 0.0, atol=1e-10)


# ------------------------------------------------------------
# 4. Solid Handling: Identity on diagonal
# ------------------------------------------------------------
def test_laplacian_solid_handling():
    nx, ny, nz = 4, 4, 4
    state = make_state(nx, ny, nz)
    
    # Set one cell to solid
    state.mask[1, 1, 1] = 0
    create_fluid_mask(state)
    
    build_laplacian_operators(state)
    lap = state.operators["laplacian"].toarray()
    
    idx = 1 + 1*nx + 1*nx*ny
    # For solid cells, our implementation places 1.0 on the diagonal
    assert lap[idx, idx] == 1.0
    # All other entries in that row should be 0
    assert np.sum(lap[idx, :]) == 1.0


# ------------------------------------------------------------
# 5. Diagonal Dominance: Necessary for solver convergence
# ------------------------------------------------------------
def test_laplacian_diagonal_dominance():
    state = make_state(nx=3, ny=3, nz=3)
    build_laplacian_operators(state)
    lap = state.operators["laplacian"].toarray()
    
    for i in range(lap.shape[0]):
        diag = np.abs(lap[i, i])
        off_diag = np.sum(np.abs(lap[i, :])) - diag
        # Diagonal should be >= sum of off-diagonals
        assert diag >= off_diag - 1e-10