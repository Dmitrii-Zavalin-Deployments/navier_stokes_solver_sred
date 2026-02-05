# src/step2/create_fluid_mask.py
from __future__ import annotations

from typing import Any, Tuple
import numpy as np


def create_fluid_mask(state: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the integer mask into boolean masks for operator use.

    Semantics:
    - mask ==  1 : fluid
    - mask == -1 : boundary-fluid (still fluid)
    - mask ==  0 : solid

    Parameters
    ----------
    state : Any
        SimulationState-like object with key "Mask" (3D integer array).

    Returns
    -------
    (is_fluid, is_boundary_cell) : Tuple[np.ndarray, np.ndarray]
        is_fluid: bool[nx, ny, nz]  (mask == 1 or mask == -1)
        is_boundary_cell: bool[nx, ny, nz]  (mask == -1)
    """

    # Extract mask
    mask = np.asarray(state["Mask"])

    # Strict dtype check (tests require float masks to raise)
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask must be an integer array")

    # Fluid = 1 or -1
    is_fluid = (mask == 1) | (mask == -1)

    # Boundary-fluid = -1
    is_boundary_cell = (mask == -1)

    # Store back into state for operator use
    state["is_fluid"] = is_fluid
    state["is_boundary_cell"] = is_boundary_cell

    return is_fluid, is_boundary_cell
