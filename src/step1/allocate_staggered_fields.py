import numpy as np
from .simulation_state import Grid


def allocate_staggered_fields(config: dict, grid: Grid):
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    P = np.zeros((nx, ny, nz), dtype=float)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    V = np.zeros((nx, ny + 1, nz), dtype=float)
    W = np.zeros((nx, ny, nz + 1), dtype=float)

    return P, U, V, W
