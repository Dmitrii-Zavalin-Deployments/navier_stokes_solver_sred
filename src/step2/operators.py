# src/step2/operators.py

from __future__ import annotations

import scipy.sparse as sp

from src.solver_state import SolverState


def get_idx(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return i + j * nx + k * nx * ny

def build_gradient(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> sp.csr_matrix:
    """Builds the cell-centered central difference gradient operator."""
    n_cells = nx * ny * nz
    data, rows, cols = [], [], []
    
    # Example for X-direction: (P_{i+1} - P_{i-1}) / 2dx
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx - 1):
                idx = get_idx(i, j, k, nx, ny)
                rows.extend([idx, idx])
                cols.extend([get_idx(i+1, j, k, nx, ny), get_idx(i-1, j, k, nx, ny)])
                data.extend([0.5 / dx, -0.5 / dx])
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

def build_divergence(grad: sp.csr_matrix) -> sp.csr_matrix:
    """Computes divergence as the negative transpose of the gradient operator."""
    return -grad.transpose()

def build_laplacian(grad: sp.csr_matrix, div: sp.csr_matrix) -> sp.csr_matrix:
    """Constructs the Laplacian: L = Div @ Grad."""
    return div @ grad

def build_numerical_operators(state: SolverState) -> None:
    """
    Orchestrates operator assembly.
    Ensures SSoT and Zero-Debt compliance.
    """
    g = state.grid
    
    # Build components
    Gx = build_gradient(g.nx, g.ny, g.nz, g.dx, g.dy, g.dz)
    D = build_divergence(Gx)
    L = build_laplacian(Gx, D)
    
    # Apply constraint (Pin pressure at [0,0,0])
    L_lil = L.tolil()
    L_lil[0, :] = 0
    L_lil[0, 0] = 1.0
    
    # State commit
    state.operators.grad_x = Gx
    state.operators.divergence = D
    state.operators.laplacian = L_lil.tocsr()
    state.ppe._A = state.operators.laplacian