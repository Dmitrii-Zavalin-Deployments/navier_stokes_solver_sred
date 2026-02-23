# src/step1/map_geometry_mask.py

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

def map_geometry_mask(mask_flat: List[int], grid: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs the 3D geometry mask from a flat schema input.
    
    Constitutional Role: Topology Interpreter.
    Rule: Canonical Mapping index = i + nx * (j + ny * k).
    
    Returns:
        Tuple: (mask_3d, is_fluid, is_boundary_cell)
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    expected_len = nx * ny * nz

    # 1. Structural Validation
    if len(mask_flat) != expected_len:
        raise ValueError(f"Mask length mismatch: Got {len(mask_flat)}, expected {expected_len}")

    # 2. Scalar Type Enforcement (Anti-Truncation Gate)
    for val in mask_flat:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"Mask entries must be finite integers (-1, 0, 1). Got {type(val)}: {val}")

    arr = np.asarray(mask_flat, dtype=np.int8)

    # 3. Value Validation
    if not np.isin(arr, [-1, 0, 1]).all():
        raise ValueError("Mask contains unauthorized values. Only -1 (Boundary), 0 (Solid), 1 (Fluid) allowed.")

    # 4. Canonical Reshape (Fortran order 'F' matches the i + nx*(j + ny*k) rule)
    mask_3d = arr.reshape((nx, ny, nz), order="F")

    # 5. Logical Derivatives (Optimized for downstream steps)
    is_fluid = (mask_3d == 1)
    is_boundary_cell = (mask_3d == -1)

    return mask_3d, is_fluid, is_boundary_cell