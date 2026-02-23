# src/step2/build_gradient_operators.py

from __future__ import annotations
import scipy.sparse as sp
from src.solver_state import SolverState

def build_gradient_operators(state: SolverState) -> None:
    """
    Construct a unified sparse MAC-grid gradient operator.
    
    The operator G maps cell-centered Pressure (size N_cells) to 
    staggered faces (size N_u + N_v + N_w).
    """
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
    is_fluid = state.is_fluid
    
    num_cells = nx * ny * nz
    num_u = (nx + 1) * ny * nz
    num_v = nx * (ny + 1) * nz
    num_w = nx * ny * (nz + 1)

    def get_c_idx(i, j, k): 
        return i + j * nx + k * nx * ny

    # --- Gx (Pressure -> U faces) ---
    rows_u, cols_u, data_u = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx): 
                u_idx = i + j * (nx + 1) + k * (nx + 1) * ny
                # Pressure gradient exists between two adjacent fluid cells
                if is_fluid[i-1, j, k] and is_fluid[i, j, k]:
                    c_left = get_c_idx(i-1, j, k)
                    c_right = get_c_idx(i, j, k)
                    rows_u.extend([u_idx, u_idx])
                    cols_u.extend([c_left, c_right])
                    data_u.extend([-1.0/dx, 1.0/dx])
    Gx = sp.csr_matrix((data_u, (rows_u, cols_u)), shape=(num_u, num_cells))

    # --- Gy (Pressure -> V faces) ---
    rows_v, cols_v, data_v = [], [], []
    for k in range(nz):
        for j in range(1, ny):
            for i in range(nx):
                v_idx = i + j * nx + k * nx * (ny + 1)
                if is_fluid[i, j-1, k] and is_fluid[i, j, k]:
                    c_bot = get_c_idx(i, j-1, k)
                    c_top = get_c_idx(i, j, k)
                    rows_v.extend([v_idx, v_idx])
                    cols_v.extend([c_bot, c_top])
                    data_v.extend([-1.0/dy, 1.0/dy])
    Gy = sp.csr_matrix((data_v, (rows_v, cols_v)), shape=(num_v, num_cells))

    # --- Gz (Pressure -> W faces) ---
    rows_w, cols_w, data_w = [], [], []
    for k in range(1, nz):
        for j in range(ny):
            for i in range(nx):
                w_idx = i + j * nx + k * nx * ny
                if is_fluid[i, j, k-1] and is_fluid[i, j, k]:
                    c_back = get_c_idx(i, j, k-1)
                    c_front = get_c_idx(i, j, k)
                    rows_w.extend([w_idx, w_idx])
                    cols_w.extend([c_back, c_front])
                    data_w.extend([-1.0/dz, 1.0/dz])
    Gz = sp.csr_matrix((data_w, (rows_w, cols_w)), shape=(num_w, num_cells))

    # Unified Gradient: Stack vertically [Gx; Gy; Gz]
    state.operators["gradient"] = sp.vstack([Gx, Gy, Gz]).tocsr()