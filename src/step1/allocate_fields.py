# src/step1/allocate_fields.py
from __future__ import annotations

import numpy as np
from .types import Fields, GridConfig


def allocate_fields(grid_config: GridConfig) -> Fields:
    """
    Allocate cell-centered fields for the solver.

    Shapes (no ghost layers):
        U, V, W, P, Mask: (nx, ny, nz)

    This replaces the old MAC-grid staggered allocation.
    Step 1 is responsible only for allocating interior fields.
    Ghost layers and boundary-condition padding are added later
    in Step 4 (Boundary Conditions).
    """

    nx, ny, nz = grid_config.nx, grid_config.ny, grid_config.nz

    # Cell-centered fields
    P = np.zeros((nx, ny, nz), dtype=float)
    U = np.zeros((nx, ny, nz), dtype=float)
    V = np.zeros((nx, ny, nz), dtype=float)
    W = np.zeros((nx, ny, nz), dtype=float)

    # Cell-centered mask (structural only; semantics applied in Step 2)
    Mask = np.zeros((nx, ny, nz), dtype=int)

    return Fields(P=P, U=U, V=V, W=W, Mask=Mask)
