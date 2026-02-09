# src/step2/enforce_mask_semantics.py

from __future__ import annotations
from typing import Any, Dict
import numpy as np


def _to_numpy(arr):
    return np.array(arr)


def _to_list(arr):
    return arr.tolist()


def enforce_mask_semantics(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and classify the geometry mask.

    Rules:
    - Mask must be integer dtype.
    - Allowed values: -1 (boundary-fluid), 0 (solid), 1 (fluid).
    - There must be at least one fluid-like cell (1 or -1).

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

    # Validate allowed values
    if not np.isin(mask, [-1, 0, 1]).all():
        bad = np.unique(mask[~np.isin(mask, [-1, 0, 1])])
        raise ValueError(
            f"Invalid mask values detected: {bad.tolist()} (allowed: -1, 0, 1)"
        )

    # Must contain at least one fluid-like cell
    fluid_like = (mask == 1) | (mask == -1)
    if not np.any(fluid_like):
        raise ValueError(
            "Mask contains no fluid or boundary-fluid cells (no 1 or -1 values)."
        )

    # Fluid = 1 or -1
    is_fluid = (mask != 0)

    # Boundary-fluid = -1
    is_boundary_cell = (mask == -1)

    return {
        "is_fluid": _to_list(is_fluid),
        "is_boundary_cell": _to_list(is_boundary_cell),
        "mask_meta": {
            "encoding": {"fluid": 1, "solid": 0, "boundary-fluid": -1},
            "validated": True,
        },
    }
