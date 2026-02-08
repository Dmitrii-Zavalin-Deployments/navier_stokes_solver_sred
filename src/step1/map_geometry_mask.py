# file: step1/map_geometry_mask.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
import math


def map_geometry_mask(mask_flat: Iterable[int], shape: Tuple[int, int, int], order_formula: str) -> np.ndarray:
    """
    Convert flat geometry mask into a 3D array using the declared flattening order.

    Enforces Step 1 Output Schema constraints:
      • mask values must be integers in {-1, 0, 1}
      • shape must be a 3-tuple of positive integers
      • flattening order must be recognized ("C" or "F")
    """

    # -----------------------------
    # Validate shape
    # -----------------------------
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 3
        or any(int(s) < 1 for s in shape)
    ):
        raise ValueError(f"geometry_mask_shape must be a 3-tuple of positive integers, got {shape}")

    nx, ny, nz = map(int, shape)

    # -----------------------------
    # Convert mask_flat → list
    # -----------------------------
    try:
        mask_list: List[int] = list(mask_flat)
    except Exception:
        raise TypeError("geometry_mask_flat must be iterable")

    if len(mask_list) != nx * ny * nz:
        raise ValueError(
            f"geometry_mask_flat length {len(mask_list)} does not match nx*ny*nz={nx*ny*nz}"
        )

    # -----------------------------
    # Validate mask values
    # -----------------------------
    validated = []
    for idx, val in enumerate(mask_list):
        if not isinstance(val, int):
            raise TypeError(f"Mask entries must be integers, got {val!r} at index {idx}")
        if not math.isfinite(val):
            raise TypeError(f"Mask entries must be finite integers, got {val!r} at index {idx}")
        if val not in (-1, 0, 1):
            raise ValueError(
                f"Mask entries must be in {{-1, 0, 1}}, got {val} at index {idx}"
            )
        validated.append(val)

    arr = np.asarray(validated, dtype=int)

    # -----------------------------
    # Determine flattening order
    # -----------------------------
    order_formula = str(order_formula).strip().upper()

    if order_formula in ("C", "ROW_MAJOR"):
        order = "C"
    elif order_formula in ("F", "FORTRAN", "COLUMN_MAJOR"):
        order = "F"
    elif "I + NX*(J + NY*K)" in order_formula.upper():
        order = "F"
    else:
        raise ValueError(
            f"Unrecognized flattening_order '{order_formula}'. "
            "Expected 'C', 'F', or a known Fortran-style formula."
        )

    # -----------------------------
    # Reshape into 3D mask
    # -----------------------------
    mask_3d = arr.reshape((nx, ny, nz), order=order)

    return mask_3d
