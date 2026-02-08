# file: step1/allocate_staggered_fields.py
from __future__ import annotations

import numpy as np

from .types import Fields, GridConfig


def allocate_staggered_fields(grid_config: GridConfig) -> Fields:
    """
    Allocate MAC-grid staggered fields and cell-centered pressure/mask.
    Zero-initialized.

    This function only allocates raw arrays. The Step 1 orchestrator
    is responsible for assembling these into the Step 1 Output Schema.
    """

    nx, ny, nz = grid_config.nx, grid_config.ny, grid_config.nz

    # Cell-centered pressure
    P = np.zeros((nx, ny, nz), dtype=float)

    # Staggered velocities
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    V = np.zeros((nx, ny + 1, nz), dtype=float)
    W = np.zeros((nx, ny, nz + 1), dtype=float)

    # Cell-centered mask (Step 1 Output Schema requires values in {-1,0,1})
    Mask = np.ones((nx, ny, nz), dtype=int)

    return Fields(P=P, U=U, V=V, W=W, Mask=Mask)
