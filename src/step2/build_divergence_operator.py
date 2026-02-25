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
    if state.is_fluid is None or not np.any(state.is_fluid):
        raise RuntimeError("Topology Error: No fluid cells detected for Divergence operator.")
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    # Scale Guard: Pull from grid dict
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
    is_fluid = state.is_fluid 
    
    num_cells = nx * ny * nz
    
    # Staggered velocity dimensions
    num_u = (nx + 1) * ny * nz
    num_v = nx * (ny + 1) * nz
    num_w = nx * ny * (nz + 1)

    def get_c_idx(i, j, k): return i + j * nx + k * nx * ny

    # --- Dx (U contribution) ---
    rows_u, cols_u, data_u = [], [], []
    is_fluid_3d = is_fluid.reshape((nx, ny, nz), order="F")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid_3d[i, j, k]: continue
                
                # U staggered indices
                idx_w = i + j * (nx + 1) + k * (nx + 1) * ny
                idx_e = (i + 1) + j * (nx + 1) + k * (nx + 1) * ny
                
                rows_u.extend([cell, cell])
                cols_u.extend([idx_w, idx_e])
                data_u.extend([-1.0/dx, 1.0/dx])
    
    Dx = sp.csr_matrix((data_u, (rows_u, cols_u)), shape=(num_cells, num_u))

    # --- Dy (V contribution) ---
    rows_v, cols_v, data_v = [], [], []
    is_fluid_3d = is_fluid.reshape((nx, ny, nz), order="F")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid_3d[i, j, k]: continue
                
                # V staggered indices
                idx_s = i + j * nx + k * nx * (ny + 1)
                idx_n = i + (j + 1) * nx + k * nx * (ny + 1)
                
                rows_v.extend([cell, cell])
                cols_v.extend([idx_s, idx_n])
                data_v.extend([-1.0/dy, 1.0/dy])

    Dy = sp.csr_matrix((data_v, (rows_v, cols_v)), shape=(num_cells, num_v))

    # --- Dz (W contribution) ---
    rows_w, cols_w, data_w = [], [], []
    is_fluid_3d = is_fluid.reshape((nx, ny, nz), order="F")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cell = get_c_idx(i, j, k)
                if not is_fluid_3d[i, j, k]: continue
                
                # W staggered indices
                idx_b = i + j * nx + k * nx * ny
                idx_f = i + j * nx + (k + 1) * nx * ny
                
                rows_w.extend([cell, cell])
                cols_w.extend([idx_b, idx_f])
                data_w.extend([-1.0/dz, 1.0/dz])

    Dz = sp.csr_matrix((data_w, (rows_w, cols_w)), shape=(num_cells, num_w))

    # Combine into full Divergence matrix: D = [Dx | Dy | Dz]
    state.operators["divergence"] = sp.hstack([Dx, Dy, Dz]).tocsr()