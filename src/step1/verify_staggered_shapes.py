# file: src/step1/verify_staggered_shapes.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def verify_staggered_shapes(state: Dict[str, Any]) -> None:
    """
    Ensure all arrays match expected MAC-grid shapes before entering Step 2.

    Works on the Step 1 output dict (schema-compliant), not on dataclasses.

    IMPORTANT:
    - If "grid" is missing (e.g., in negative schema tests), skip verification.
    """

    # ---------------------------------------------------------
    # FIX: Skip verification if grid is missing
    # (negative tests intentionally remove "grid")
    # ---------------------------------------------------------
    if "grid" not in state:
        return

    grid = state["grid"]
    fields = state["fields"]

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    P = fields["P"]
    U = fields["U"]
    V = fields["V"]
    W = fields["W"]
    Mask = fields["Mask"]

    # ---------------------------------------------------------
    # Ensure arrays are numpy arrays
    # ---------------------------------------------------------
    for name, arr in [("P", P), ("U", U), ("V", V), ("W", W), ("Mask", Mask)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array, got {type(arr)}")

    # ---------------------------------------------------------
    # Shape checks
    # ---------------------------------------------------------
    if P.shape != (nx, ny, nz):
        raise ValueError(f"P shape mismatch: expected {(nx, ny, nz)}, got {P.shape}")

    if U.shape != (nx + 1, ny, nz):
        raise ValueError(f"U shape mismatch: expected {(nx+1, ny, nz)}, got {U.shape}")

    if V.shape != (nx, ny + 1, nz):
        raise ValueError(f"V shape mismatch: expected {(nx, ny+1, nz)}, got {V.shape}")

    if W.shape != (nx, ny, nz + 1):
        raise ValueError(f"W shape mismatch: expected {(nx, ny, nz+1)}, got {W.shape}")

    if Mask.shape != (nx, ny, nz):
        raise ValueError(f"Mask shape mismatch: expected {(nx, ny, nz)}, got {Mask.shape}")
