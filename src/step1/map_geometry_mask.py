# file: step1/map_geometry_mask.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
import math


def map_geometry_mask(mask_flat: Iterable[int], shape: Tuple[int, int, int], order_formula: str) -> np.ndarray:
    """
    Convert flat geometry mask into 3D array using declared flattening order.
    Structural only: any integer values are accepted, but they must be valid integers.
    """
    nx, ny, nz = shape
    mask_list: List[int] = list(mask_flat)

    if len(mask_list) != nx * ny * nz:
        raise ValueError(
            f"geometry_mask_flat length {len(mask_list)} does not match nx*ny*nz={nx*ny*nz}"
        )

    # NEW: strict integer + finite validation
    validated = []
    for idx, val in enumerate(mask_list):
        if not isinstance(val, int):
            raise TypeError(f"Mask entries must be integers, got {val!r} at index {idx}")
        if not math.isfinite(val):
            raise TypeError(f"Mask entries must be finite integers, got {val!r} at index {idx}")
        validated.append(val)

    arr = np.asarray(validated, dtype=int)

    if "i + nx*(j + ny*k)" in order_formula:
        mask_3d = arr.reshape((nx, ny, nz), order="F")
    else:
        mask_3d = arr.reshape((nx, ny, nz), order="C")

    return mask_3d
