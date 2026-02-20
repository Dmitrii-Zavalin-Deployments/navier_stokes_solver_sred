# tests/step2/test_laplacian_bc_matrix_structure.py

import pytest
import numpy as np
import scipy.sparse as sp
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.solver_state import SolverState

@pytest.fixture
def ppe_state():
    """Sets up a 3x3x3 grid where x_max is Pressure and others are Walls."""
    grid = {
        'nx': 3, 'ny': 3, 'nz': 3,
        'dx': 1.0, 'dy': 1.0, 'dz': 1.0
    }
    # Initialize a fluid mask (all fluid)
    is_fluid = np.ones((3, 3, 3), dtype=bool)
    
    # Define BCs: x_max is Dirichlet (Pressure), x_min is Neumann (No-slip)
    boundary_conditions = {
        "x_max": {"type": "pressure", "p": 100.0},
        "x_min": {"type": "no-slip", "p": 0.0}
    }
    
    state = SolverState(grid=grid, boundary_conditions=boundary_conditions)
    state.is_fluid = is_fluid
    return state

def test_pressure_dirichlet_row_identity(ppe_state):
    """
    Theory Check: For Pressure-type BCs: Row is replaced with identity-like row (A[i,i]=1).
    """
    # Execute matrix construction
    build_laplacian_operators(ppe_state)
    A = ppe_state.operators["laplacian"]
    
    nx, ny = ppe_state.grid['nx'], ppe_state.grid['ny']
    
    # Pick a cell on the x_max face: i=2, j=1, k=1
    # Index = i + j*nx + k*nx*ny = 2 + 1*3 + 1*9 = 14
    target_idx = 2 + (1 * nx) + (1 * nx * ny)
    
    # Extract the row as a dense array for inspection
    row_data = A.getrow(target_idx).toarray().flatten()
    
    # ASSERTIONS
    assert row_data[target_idx] == 1.0, f"Diagonal at index {target_idx} should be 1.0"
    assert np.count_nonzero(row_data) == 1, f"Row {target_idx} should only have one non-zero element"

def test_wall_neumann_stencil_modification(ppe_state):
    """
    Theory Check: For Wall-type BCs, the Neumann condition reduces the stencil.
    At x_min (i=0, j=1, k=1), the 'left' neighbor is missing.
    Standard center is -6.0; Neumann center should be -5.0.
    """
    build_laplacian_operators(ppe_state)
    A = ppe_state.operators["laplacian"]
    
    nx, ny = ppe_state.grid['nx'], ppe_state.grid['ny']
    
    # Cell on x_min face: i=0, j=1, k=1 -> Index = 0 + 3 + 9 = 12
    target_idx = 0 + (1 * nx) + (1 * nx * ny)
    
    row_data = A.getrow(target_idx).toarray().flatten()
    
    # The center value (A[i,i]) should be -5.0 because one neighbor (i-1) is outside the domain
    # and is_fluid check prevents its contribution to center_val.
    assert row_data[target_idx] == -5.0, f"Neumann center at {target_idx} should be -5.0, got {row_data[target_idx]}"