# src/step2/build_divergence_operator.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from src.solver_state import SolverState

def build_divergence_operator(state: SolverState) -> None:
    """
    Construct a sparse MAC-grid divergence operator.
    
    The operator D is a matrix of shape (N_cells, N_velocity_dofs).
    It operates on a flattened vector [U.ravel(), V.ravel(), W.ravel()].
    """
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    dx, dy, dz = state.constants['dx'], state.constants['dy'], state.constants['dz']
    is_fluid = state.is_fluid # Used to zero out divergence in solid cells
    
    num_cells = nx * ny * nz
    
    # Staggered velocity dimensions
    num_u = (nx + 1) * ny * nz
    num_v = nx * (ny + 1) * nz
    num_w = nx * ny * (nz + 1)

    # Helper to get flat cell index (pressure centers)
    def get_c_idx(i, j, k): return i + j * nx + k * nx * ny

    # --- Dx (U contribution) ---
    rows_u, cols_u, data_u = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid[i, j, k]: continue
                
                # U indices: u[i, j, k] is west face, u[i+1, j, k] is east face
                idx_w = i + j * (nx + 1) + k * (nx + 1) * ny
                idx_e = (i + 1) + j * (nx + 1) + k * (nx + 1) * ny
                
                rows_u.extend([cell, cell])
                cols_u.extend([idx_w, idx_e])
                data_u.extend([-1.0/dx, 1.0/dx])
    
    Dx = sp.csr_matrix((data_u, (rows_u, cols_u)), shape=(num_cells, num_u))

    # --- Dy (V contribution) ---
    rows_v, cols_v, data_v = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid[i, j, k]: continue
                
                # V indices: v[i, j, k] is south face, v[i, j+1, k] is north face
                idx_s = i + j * nx + k * nx * (ny + 1)
                idx_n = i + (j + 1) * nx + k * nx * (ny + 1)
                
                rows_v.extend([cell, cell])
                cols_v.extend([idx_s, idx_n])
                data_v.extend([-1.0/dy, 1.0/dy])

    Dy = sp.csr_matrix((data_v, (rows_v, cols_v)), shape=(num_cells, num_v))

    # --- Dz (W contribution) ---
    rows_w, cols_w, data_w = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid[i, j, k]: continue
                
                # W indices: w[i, j, k] is back face, w[i, j, k+1] is front face
                idx_b = i + j * nx + k * nx * ny
                idx_f = i + j * nx + (k + 1) * nx * ny
                
                rows_w.extend([cell, cell])
                cols_w.extend([idx_b, idx_f])
                data_w.extend([-1.0/dz, 1.0/dz])

    Dz = sp.csr_matrix((data_w, (rows_w, cols_w)), shape=(num_cells, num_w))

    # Combine into full Divergence matrix: D = [Dx | Dy | Dz]
    state.operators["divergence"] = sp.hstack([Dx, Dy, Dz]).tocsr()