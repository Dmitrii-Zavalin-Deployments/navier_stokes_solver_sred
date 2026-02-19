# src/step2/build_laplacian_operators.py

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from src.solver_state import SolverState

def build_laplacian_operators(state: SolverState) -> None:
    """
    Construct a sparse 7-point Laplacian operator for the Pressure Poisson Equation.
    
    This matrix represents the second derivatives (diffusion) of pressure.
    It is a square matrix of shape (nx*ny*nz, nx*ny*nz).
    """
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    
    # Grid spacing pulled from the grid dictionary
    dx2 = grid['dx']**2
    dy2 = grid['dy']**2
    dz2 = grid['dz']**2
    is_fluid = state.is_fluid
    
    num_cells = nx * ny * nz
    rows, cols, data = [], [], []

    def get_idx(i, j, k): 
        return i + j * nx + k * nx * ny

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                curr = get_idx(i, j, k)
                
                # Solid cell handling: 
                # Place 1.0 on the diagonal to keep the matrix invertible (non-singular).
                if not is_fluid[i, j, k]:
                    rows.append(curr)
                    cols.append(curr)
                    data.append(1.0)
                    continue

                center_val = 0.0

                # X-Neighbors: (i-1, j, k) and (i+1, j, k)
                for ni in [i - 1, i + 1]:
                    if 0 <= ni < nx and is_fluid[ni, j, k]:
                        rows.append(curr)
                        cols.append(get_idx(ni, j, k))
                        data.append(1.0 / dx2)
                        center_val -= 1.0 / dx2
                
                # Y-Neighbors: (i, j-1, k) and (i, j+1, k)
                for nj in [j - 1, j + 1]:
                    if 0 <= nj < ny and is_fluid[i, nj, k]:
                        rows.append(curr)
                        cols.append(get_idx(i, nj, k))
                        data.append(1.0 / dy2)
                        center_val -= 1.0 / dy2

                # Z-Neighbors: (i, j, k-1) and (i, j, k+1)
                for nk in [k - 1, k + 1]:
                    if 0 <= nk < nz and is_fluid[i, j, nk]:
                        rows.append(curr)
                        cols.append(get_idx(i, j, nk))
                        data.append(1.0 / dz2)
                        center_val -= 1.0 / dz2

                # Diagonal element (center of the 7-point stencil)
                rows.append(curr)
                cols.append(curr)
                data.append(center_val)

    # Store as a sparse CSR matrix for efficient linear solving in Step 3
    state.operators["laplacian"] = sp.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_cells, num_cells)
    )