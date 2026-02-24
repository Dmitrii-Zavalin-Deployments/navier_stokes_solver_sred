# src/step1/map_geometry_mask.py

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

def map_geometry_mask(mask_flat: List[int], grid: Dict[str, Any]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Reconstructs the 3D geometry mask and returns the canonical 1D list.
    
    Constitutional Role: Topology Interpreter.
    Phase A.2 Compliance: Returns 1D list for serialization parity.
    """
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])
    expected_len = nx * ny * nz

    # 1. Structural Validation
    if len(mask_flat) != expected_len:
        raise ValueError(f"Mask length mismatch: Got {len(mask_flat)}, expected {expected_len}")

    # 2. Scalar Type Enforcement
    for val in mask_flat:
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"Mask entries must be finite integers. Got {type(val)}")

    arr = np.asarray(mask_flat, dtype=np.int8)

    # 3. Value Validation
    if not np.isin(arr, [-1, 0, 1]).all():
        raise ValueError("Mask contains unauthorized values (-1, 0, 1 only).")

    # 4. Canonical Reshape (Internal math remains 3D)
    mask_3d = arr.reshape((nx, ny, nz), order="F")

    # 5. Logical Derivatives (Kept as arrays for Step 2/3 math)
    is_fluid = (mask_3d == 1)
    is_boundary_cell = (mask_3d == -1)

    # 6. CONSTITUTIONAL FIX: Flatten the main mask back to a list for the Auditor
    # We return the flattened list as the primary mask reference.
    return mask_3d.flatten(order="F").tolist(), is_fluid, is_boundary_cell
