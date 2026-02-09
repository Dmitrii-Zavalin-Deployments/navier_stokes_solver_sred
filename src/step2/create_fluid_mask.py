# src/step2/create_fluid_mask.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr)


def _to_list(arr):
    return arr.tolist()


def create_fluid_mask(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the integer mask into boolean masks for operator use.

    Semantics:
    - mask ==  1 : fluid
    - mask == -1 : boundary-fluid (still fluid)
    - mask ==  0 : solid

    Returns:
        {
            "is_fluid": [...],
            "is_boundary_cell": [...],
            "mask_meta": {...}
        }
    """

    grid = state["grid"]
    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    # Canonical Step‑1 mask
    mask = _to_numpy(state["mask_3d"])

    # Validate shape
    if mask.shape != (nx, ny, nz):
        raise ValueError(
            f"mask_3d must have shape {(nx, ny, nz)}, got {mask.shape}"
        )

    # Validate dtype
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask must be an integer array")

    # Validate values
    if not np.isin(mask, [-1, 0, 1]).all():
        raise ValueError("Mask values must be in {-1, 0, 1}")

    # Fluid = 1 or -1
    is_fluid = (mask != 0)

    # Boundary-fluid = -1
    is_boundary_cell = (mask == -1)

    return {
        "is_fluid": _to_list(is_fluid),
        "is_boundary_cell": _to_list(is_boundary_cell),
        "mask_meta": {
            "encoding": {"fluid": 1, "solid": 0, "boundary-fluid": -1},
            "source": "mask_3d",
        },
    }
