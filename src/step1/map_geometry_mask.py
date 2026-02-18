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

    # Validate length
    if len(mask_flat) != expected_len:
        raise ValueError(
            f"Mask length {len(mask_flat)} does not match nx*ny*nz={expected_len}"
        )

    # Convert to array
    arr = np.asarray(mask_flat, dtype=int)

    # Validate values
    if not np.isin(arr, [-1, 0, 1]).all():
        raise ValueError("Mask entries must be -1, 0, or 1")

    # Canonical reshape: i + nx*(j + ny*k)
    mask_3d = arr.reshape((nx, ny, nz), order="F")

    return mask_3d
