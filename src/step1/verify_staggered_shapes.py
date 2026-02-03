from .simulation_state import Grid
import numpy as np


def verify_staggered_shapes(P: np.ndarray,
                            U: np.ndarray,
                            V: np.ndarray,
                            W: np.ndarray,
                            mask: np.ndarray,
                            grid: Grid) -> None:
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    if P.shape != (nx, ny, nz):
        raise ValueError("Pressure field shape mismatch")

    if U.shape != (nx + 1, ny, nz):
        raise ValueError("U field shape mismatch")

    if V.shape != (nx, ny + 1, nz):
        raise ValueError("V field shape mismatch")

    if W.shape != (nx, ny, nz + 1):
        raise ValueError("W field shape mismatch")

    if mask.shape != (nx, ny, nz):
        raise ValueError("Mask shape mismatch")
