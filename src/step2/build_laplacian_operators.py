# src/step2/build_laplacian_operators.py

from __future__ import annotations
import scipy.sparse as sp
from src.solver_state import SolverState

def build_laplacian_operators(state: SolverState) -> None:
    """
    Construct a sparse 7-point Laplacian operator for the Pressure Poisson Equation.
    
    Step 2: Operator Construction (The Matrix Debt)
    Implements Neumann (Wall-type) and Dirichlet (Pressure-type) BCs into the matrix A.
    """
    grid = state.grid
    nx, ny, nz = grid['nx'], grid['ny'], grid['nz']
    
    dx2 = grid['dx']**2
    dy2 = grid['dy']**2
    dz2 = grid['dz']**2
    is_fluid = state.is_fluid
    
    # FIX: Default to empty dict if boundary_conditions is None to prevent AttributeError
    bc_table = state.boundary_conditions if state.boundary_conditions is not None else {}
    
    num_cells = nx * ny * nz
    rows, cols, data = [], [], []

    def get_idx(i, j, k): 
        return i + j * nx + k * nx * ny

    # Helper to identify if a cell index is on a specific boundary face
    def is_on_face(i, j, k, face):
        if face == "x_min": return i == 0
        if face == "x_max": return i == nx - 1
        if face == "y_min": return j == 0
        if face == "y_max": return j == ny - 1
        if face == "z_min": return k == 0
        if face == "z_max": return k == nz - 1
        return False

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                curr = get_idx(i, j, k)
                
                # 1. Solid cell handling: Keep matrix non-singular
                if not is_fluid[curr]:
                    rows.append(curr)
                    cols.append(curr)
                    data.append(1.0)
                    continue

                # 2. Dirichlet (Pressure) Boundary Check
                is_dirichlet = False
                for bc in bc_table:
                    # Only apply Dirichlet if the BC type is explicitly 'pressure'
                    if bc.get("type") == "pressure" and is_on_face(i, j, k, bc["location"]):
                        is_dirichlet = True
                        break
                
                if is_dirichlet:
                    rows.append(curr)
                    cols.append(curr)
                    data.append(1.0)
                    continue

                # 3. Standard Stencil / Neumann (Wall-type) Logic
                center_val = 0.0

                # X-Neighbors
                for ni in [i - 1, i + 1]:
                    if 0 <= ni < nx and is_fluid[get_idx(ni, j, k)]:
                        rows.append(curr)
                        cols.append(get_idx(ni, j, k))
                        data.append(1.0 / dx2)
                        center_val -= 1.0 / dx2
                
                # Y-Neighbors
                for nj in [j - 1, j + 1]:
                    if 0 <= nj < ny and is_fluid[i, nj, k]:
                        rows.append(curr)
                        cols.append(get_idx(i, nj, k))
                        data.append(1.0 / dy2)
                        center_val -= 1.0 / dy2

                # Z-Neighbors
                for nk in [k - 1, k + 1]:
                    if 0 <= nk < nz and is_fluid[i, j, nk]:
                        rows.append(curr)
                        cols.append(get_idx(i, j, nk))
                        data.append(1.0 / dz2)
                        center_val -= 1.0 / dz2

                # Diagonal element
                rows.append(curr)
                cols.append(curr)
                data.append(center_val)

    # Store as a sparse CSR matrix for efficient linear solving in Step 3
    state.operators["laplacian"] = sp.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_cells, num_cells)
    )