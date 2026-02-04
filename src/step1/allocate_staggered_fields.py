# file: step1/allocate_staggered_fields.py
from __future__ import annotations

import numpy as np

from .types import Fields, GridConfig


def allocate_staggered_fields(grid_config: GridConfig) -> Fields:
    """
    Allocate MAC-grid staggered fields and cell-centered pressure/mask.
    Zero-initialized.
    """
    nx, ny, nz = grid_config.nx, grid_config.ny, grid_config.nz

    P = np.zeros((nx, ny, nz), dtype=float)
    U = np.zeros((nx + 1, ny, nz), dtype=float)
    V = np.zeros((nx, ny + 1, nz), dtype=float)
    W = np.zeros((nx, ny, nz + 1), dtype=float)
    Mask = np.zeros((nx, ny, nz), dtype=int)

    return Fields(P=P, U=U, V=V, W=W, Mask=Mask)
