# src/step2/advection.py

import numpy as np
from src.solver_state import SolverState

# Global Debug Toggle
DEBUG = True

def build_advection_stencils(state: SolverState) -> None:
    """
    Step 2 Logic: Populate 3D Trilinear Interpolation stencils.
    Maps U, V, and W DOFs to their physical 8-point P-cell neighborhoods.
    NO DEFAULTS: Uses grid topology and configured weights.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    if DEBUG:
        print(f"DEBUG [Step 2 Advection]: Mapping {total_vel_dof} DOFs (U:{dof_u}, V:{dof_v}, W:{dof_w})")

    # 1. Allocate buffers
    weights = np.zeros((total_vel_dof, 8), dtype=np.float64)
    indices = np.zeros((total_vel_dof, 8), dtype=np.int32)

    def get_p_idx(i, j, k):
        """Helper to get 1D index for Pressure cells with clamping for boundaries."""
        i_c = max(0, min(i, nx - 1))
        j_c = max(0, min(j, ny - 1))
        k_c = max(0, min(k, nz - 1))
        return i_c + j_c * nx + k_c * nx * ny

    current_dof = 0

    # 2. Map U-Components (i-staggered)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx + 1):
                indices[current_dof, :] = [
                    get_p_idx(i, j, k),     get_p_idx(i-1, j, k),
                    get_p_idx(i, j+1, k),   get_p_idx(i-1, j+1, k),
                    get_p_idx(i, j, k+1),   get_p_idx(i-1, j, k+1),
                    get_p_idx(i, j+1, k+1), get_p_idx(i-1, j+1, k+1)
                ]
                current_dof += 1

    # 3. Map V-Components (j-staggered)
    for k in range(nz):
        for j in range(ny + 1):
            for i in range(nx):
                indices[current_dof, :] = [
                    get_p_idx(i, j, k),     get_p_idx(i, j-1, k),
                    get_p_idx(i+1, j, k),   get_p_idx(i+1, j-1, k),
                    get_p_idx(i, j, k+1),   get_p_idx(i, j-1, k+1),
                    get_p_idx(i+1, j, k+1), get_p_idx(i+1, j-1, k+1)
                ]
                current_dof += 1

    # 4. Map W-Components (k-staggered)
    for k in range(nz + 1):
        for j in range(ny):
            for i in range(nx):
                indices[current_dof, :] = [
                    get_p_idx(i, j, k),     get_p_idx(i, j, k-1),
                    get_p_idx(i+1, j, k),   get_p_idx(i+1, j, k-1),
                    get_p_idx(i, j+1, k),   get_p_idx(i, j+1, k-1),
                    get_p_idx(i+1, j+1, k), get_p_idx(i+1, j+1, k-1)
                ]
                current_dof += 1

    if DEBUG:
        print(f"DEBUG [Step 2 Advection]: Handshake check - processed {current_dof}/{total_vel_dof} DOFs")

    # 5. Apply Configured Weights (Pulling from SSoT config)
    interp_weight = state.config.advection_weight_base
    weights.fill(interp_weight)

    # 6. Final State Commit
    state.advection.weights = weights
    state.advection.indices = indices

    if DEBUG:
        print(f"DEBUG [Step 2 Advection]: Indices Min/Max: {indices.min()}/{indices.max()}")
        print(f"DEBUG [Step 2 Advection]: Weights check sum: {np.sum(weights[0]):.4f}")