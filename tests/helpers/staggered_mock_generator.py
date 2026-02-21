import numpy as np
from scipy.sparse import csr_matrix, eye

def inject_staggered_operators(state):
    """
    Injects sparse operators into the state that match 
    staggered MAC grid dimensions for a 3D domain.
    """
    nx, ny, nz = state.grid["nx"], state.grid["ny"], state.grid["nz"]
    
    # 1. Calculate flattened sizes for staggered fields
    u_size = (nx + 1) * ny * nz
    v_size = nx * (ny + 1) * nz
    w_size = nx * ny * (nz + 1)
    p_size = nx * ny * nz
    vel_total_size = u_size + v_size + w_size

    # 2. Build Laplacian & Advection Operators (Identity mocks)
    # These map a field to itself (e.g., U -> U)
    state.operators["lap_u"] = eye(u_size, format="csr")
    state.operators["advection_u"] = eye(u_size, format="csr")
    
    state.operators["lap_v"] = eye(v_size, format="csr")
    state.operators["advection_v"] = eye(v_size, format="csr")
    
    state.operators["lap_w"] = eye(w_size, format="csr")
    state.operators["advection_w"] = eye(w_size, format="csr")

    # 3. Build Divergence Operator (Mocks Velocity -> Pressure)
    # Shape: (P_size x total_velocity_size)
    # This allows: div_flat = div_op @ velocity_vector
    state.operators["divergence"] = csr_matrix((p_size, vel_total_size))

    # 4. Build Gradient Operators (Mocks Pressure -> Velocity)
    # grad_x: P -> U, etc.
    state.operators["grad_x"] = lambda p: np.zeros((nx + 1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny + 1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz + 1))

    # 5. Build PPE Matrix (System Matrix A)
    # Shape: (P_size x P_size)
    state.ppe["A"] = eye(p_size, format="csr") * -6.0
    
    return state