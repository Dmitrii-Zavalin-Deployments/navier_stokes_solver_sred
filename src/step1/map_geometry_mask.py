# src/step1/map_geometry_mask.py

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


def map_geometry_mask(mask_flat: List[int], domain: Dict[str, Any]) -> np.ndarray:
    """
    Step 1: Convert a flat mask into a 3D array using the solver's
    canonical flattening rule:

        flat_index = i + nx * (j + ny * k)

    The input schema defines mask as a flat 1D array of length nx*ny*nz.
    Step 1 performs structural validation only; semantic interpretation
    of {-1, 0, 1} happens in Step 2.
    """

    nx = int(domain["nx"])
    ny = int(domain["ny"])
    nz = int(domain["nz"])

    expected_len = nx * ny * nz

    # 1. Validate length
    if len(mask_flat) != expected_len:
        raise ValueError(
            f"Mask length {len(mask_flat)} does not match nx*ny*nz={expected_len}"
        )

    # 2. Strict type/value validation to catch non-integers, NaNs, and Infs
    # We use a loop or generator check here because np.asarray(dtype=int) 
    # would silently truncate 1.5 to 1, causing the test to fail to see a ValueError.
    for val in mask_flat:
        if not isinstance(val, (int, np.integer)):
            # This catches floats (1.5, NaN, Inf) and strings ("x")
            raise ValueError(f"Mask entries must be finite integers, got {val}")

    # 3. Convert to array now that we know they are integers
    arr = np.asarray(mask_flat, dtype=int)

    # 4. Validate allowed values (-1, 0, 1)
    if not np.isin(arr, [-1, 0, 1]).all():
        raise ValueError("Mask entries must be -1, 0, or 1")

    # 5. Canonical reshape: i + nx*(j + ny*k)
    # Fortran order 'F' corresponds to the i + nx*(j + ny*k) mapping
    mask_3d = arr.reshape((nx, ny, nz), order="F")

    return mask_3d