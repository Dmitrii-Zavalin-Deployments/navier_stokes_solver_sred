# src/step2/operators.py

from __future__ import annotations

import scipy.sparse as sp

from src.solver_state import SolverState


def get_idx(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """Helper to map 3D indices to a 1D vector index for matrix operations."""
    return i + j * nx + k * nx * ny

def build_gradient(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float, axis: str = 'x') -> sp.csr_matrix:
    """
    Implements Theory Section 4: First derivative (central difference).
    Formula: phi'_i ≈ (phi_{i+1} - phi_{i-1}) / 2Δ
    Correctly selects Δ based on the requested axis (x, y, or z).
    """
    n_cells = nx * ny * nz
    data, rows, cols = [], [], []

    # 1. Determine the step size and neighbor logic based on the axis
    if axis == 'x':
        delta = dx
        # Stencil: i+1 and i-1
        range_i, range_j, range_k = range(1, nx - 1), range(ny), range(nz)
        di, dj, dk = 1, 0, 0
    elif axis == 'y':
        delta = dy
        # Stencil: j+1 and j-1
        range_i, range_j, range_k = range(nx), range(1, ny - 1), range(nz)
        di, dj, dk = 0, 1, 0
    elif axis == 'z':
        delta = dz
        # Stencil: k+1 and k-1
        range_i, range_j, range_k = range(nx), range(ny), range(1, nz - 1)
        di, dj, dk = 0, 0, 1

    # 2. Build the matrix using the selected delta
    weight = 0.5 / delta
    for k in range_k:
        for j in range_j:
            for i in range_i:
                idx = get_idx(i, j, k, nx, ny)
                rows.extend([idx, idx])
                # Neighbors are shifted by (di, dj, dk)
                cols.extend([
                    get_idx(i + di, j + dj, k + dk, nx, ny), 
                    get_idx(i - di, j - dj, k - dk, nx, ny)
                ])
                data.extend([weight, -weight])
                
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

def build_laplacian(nx: int, ny: int, nz: int, dx: float) -> sp.csr_matrix:
    """
    Implements Theory Section 4: Second derivative (central difference).
    Formula: phi''_i ≈ (phi_{i+1} - 2phi_i + phi_{i-1}) / Δ²
    Used for: Diffusion and Pressure Poisson Equation (Section 3, 4, 5.2).
    """
    n_cells = nx * ny * nz
    data, rows, cols = [], [], []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx - 1):
                idx = get_idx(i, j, k, nx, ny)
                # Stencil weights: [1, -2, 1] / Δ²
                rows.extend([idx, idx, idx])
                cols.extend([get_idx(i+1, j, k, nx, ny), idx, get_idx(i-1, j, k, nx, ny)])
                data.extend([1.0 / dx**2, -2.0 / dx**2, 1.0 / dx**2])
                
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

def build_numerical_operators(state: SolverState) -> None:
    """
    Step 2 Logic: Populate Discrete Calculus Operators based on Section 4.
    Orchestrates assembly of Gradient and Laplacian directly from stencils.
    """
    g = state.grid

    # 1. Build Gradient for X (can be extended to Y and Z using the same logic)
    Gx = build_gradient(g.nx, g.ny, g.nz, g.dx)
    
    # 2. Build Laplacian directly (used for Diffusion and PPE)
    L = build_laplacian(g.nx, g.ny, g.nz, g.dx)

    # 3. Apply PPE constraint: Pin p[0,0,0] to 0 to remove the nullspace (Section 5.2)
    L_lil = L.tolil()
    L_lil[0, :] = 0
    L_lil[0, 0] = 1.0

    # 4. State commit: SSoT (Single Source of Truth)
    state.operators.grad_x = Gx
    state.operators.laplacian = L_lil.tocsr()
    state.ppe._A = state.operators.laplacian
