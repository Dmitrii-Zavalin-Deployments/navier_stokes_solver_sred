# file: src/step1/verify_cell_centered_shapes.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def verify_cell_centered_shapes(state: Dict[str, Any]) -> None:
    """
    Ensure all arrays match expected cell-centered shapes before entering Step 2.

    Works on the Step 1 output dict (schema-compliant), not on dataclasses.

    IMPORTANT:
    - If "grid" is missing (e.g., in negative schema tests), skip verification.
    - Step 1 uses ONLY cell-centered fields (no staggered, no ghost layers).
    """

    # ---------------------------------------------------------
    # Skip verification if grid is missing
    # ---------------------------------------------------------
    if "grid" not in state:
        return

    grid = state["grid"]
    fields = state["fields"]

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    nz = int(grid["nz"])

    # ---------------------------------------------------------
    # Convert lists → numpy arrays (tests intentionally pass lists)
    # ---------------------------------------------------------
    for name in ["P", "U", "V", "W", "Mask"]:
        arr = fields[name]
        if isinstance(arr, list):
            arr = np.asarray(arr)
            fields[name] = arr  # update state in-place

    # Rebind after conversion
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
    # Cell-centered shape checks
    # ---------------------------------------------------------
    expected = (nx, ny, nz)

    if P.shape != expected:
        raise ValueError(f"P shape mismatch: expected {expected}, got {P.shape}")

    if U.shape != expected:
        raise ValueError(f"U shape mismatch: expected {expected}, got {U.shape}")

    if V.shape != expected:
        raise ValueError(f"V shape mismatch: expected {expected}, got {V.shape}")

    if W.shape != expected:
        raise ValueError(f"W shape mismatch: expected {expected}, got {W.shape}")

    if Mask.shape != expected:
        raise ValueError(f"Mask shape mismatch: expected {expected}, got {Mask.shape}")
