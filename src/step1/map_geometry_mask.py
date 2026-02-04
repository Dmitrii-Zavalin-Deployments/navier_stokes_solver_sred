# file: step1/map_geometry_mask.py
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


def _default_flattening_order(nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Default C-style flattening: flat_index = i + nx*(j + ny*k).
    """
    idx = np.arange(nx * ny * nz, dtype=int)
    return idx.reshape((nx, ny, nz), order="F").ravel(order="C")


def map_geometry_mask(
    mask_flat: Iterable[int],
    shape: Tuple[int, int, int],
    order_formula: str,
) -> np.ndarray:
    """
    Convert flat geometry mask into 3D array using declared flattening order.
    Structural only: any integer values are accepted.
    """
    nx, ny, nz = shape
    mask_list: List[int] = list(mask_flat)

    if len(mask_list) != nx * ny * nz:
        raise ValueError(
            f"geometry_mask_flat length {len(mask_list)} "
            f"does not match nx*ny*nz={nx*ny*nz}"
        )

    # Ensure integer type
    try:
        arr = np.asarray(mask_list, dtype=int)
    except Exception as exc:
        raise TypeError("Mask entries must be integers") from exc

    # For now we support a single canonical formula; others can be added later.
    # We still honor the contract that order_formula is provided.
    if "i + nx*(j + ny*k)" in order_formula:
        mask_3d = arr.reshape((nx, ny, nz), order="F")
    else:
        # Fallback: treat as simple C-order reshape (documented behavior)
        mask_3d = arr.reshape((nx, ny, nz), order="C")

    return mask_3d
