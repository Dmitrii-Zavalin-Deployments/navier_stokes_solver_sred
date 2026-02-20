# tests/step2/test_laplacian_bc_compliance.py

import pytest
import numpy as np
from src.step2.build_laplacian_operators import build_laplacian_matrix
from src.step2.boundary_index_mapper import get_boundary_indices
from src.solver_state import SolverState

@pytest.fixture
def state_with_bcs():
    """Setup a small 4x4x4 grid with specific BC types."""
    grid = {"nx": 4, "ny": 4, "nz": 4, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    # x_min is a Wall (Neumann), x_max is Pressure (Dirichlet)
    bcs = {
        "x_min": {"type": "no-slip", "p": 0.0},
        "x_max": {"type": "pressure", "p": 10.0}
    }
    return SolverState(grid=grid, boundary_conditions=bcs)

class TestLaplacianBCCompliance:
    """
    Verifies 100% implementation of 'The Matrix Debt' in Step 2.
    """

    def test_dirichlet_pressure_matrix_row(self, state_with_bcs):
        """
        Theory Check: For Pressure-type BCs: A[i,i] = 1, A[i,j] = 0.
        """
        nx, ny, nz = 4, 4, 4
        A = build_laplacian_matrix(state_with_bcs)
        
        # Get indices for the x_max face (Pressure Dirichlet)
        p_indices = get_boundary_indices(nx, ny, nz, "x_max")
        
        for idx in p_indices:
            row = A.getrow(idx).toarray().flatten()
            # The diagonal must be 1.0
            assert row[idx] == 1.0, f"Dirichlet diagonal at {idx} should be 1.0"
            # All other elements in the row must be 0.0
            assert np.count_nonzero(row) == 1, f"Dirichlet row at {idx} must be identity-like"

    def test_neumann_wall_matrix_modification(self, state_with_bcs):
        """
        Theory Check: For Wall-type BCs: Neumann conditions applied.
        In a 7-point stencil, a Neumann boundary cancels the neighbor outside the domain,
        often reducing the central coefficient or adjusting neighbor weights.
        """
        nx, ny, nz = 4, 4, 4
        A = build_laplacian_matrix(state_with_bcs)
        
        # Get index for a cell on the x_min face (No-slip Neumann)
        # Avoid corners to keep the test simple: i=0, j=2, k=2
        wall_idx = 0 + 4 * (2 + 4 * 2) 
        
        row = A.getrow(wall_idx).toarray().flatten()
        
        # In a standard interior Laplacian, the center is -6 (for dx=dy=dz=1).
        # In a Neumann boundary at x_min, the 'left' neighbor (i-1) is missing.
        # The center coefficient should be adjusted to -5 to prevent flow through the wall.
        assert row[wall_idx] == -5.0, "Neumann boundary should adjust central coefficient to -5.0"
        # Ensure the neighbor at x+1 (i+1) still exists
        assert row[wall_idx + 1] == 1.0