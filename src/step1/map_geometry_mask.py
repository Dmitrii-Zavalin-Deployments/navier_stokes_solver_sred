# src/step1/map_geometry_mask.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
import math


def map_geometry_mask(
    mask_flat: Iterable[int],
    shape: Tuple[int, int, int],
    order_formula: str
) -> np.ndarray:
    """
    Convert a flat geometry mask into a 3D array using the declared flattening order.

    Step 1 responsibilities:
      • structural mapping only
      • no interpretation of mask semantics ({-1,0,1})
      • no geometry logic, no BC logic, no ghost layers

    Notes:
      • Mask values may be arbitrary integers at this stage.
      • Semantic interpretation happens in Step 2.
    """

    # -----------------------------
    # Validate shape
    # -----------------------------
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 3
        or any(int(s) < 1 for s in shape)
    ):
        raise ValueError(f"shape must be a 3‑tuple of positive integers, got {shape}")

    nx, ny, nz = map(int, shape)

    # -----------------------------
    # Convert mask_flat → list
    # -----------------------------
    try:
        mask_list: List[int] = list(mask_flat)
    except Exception:
        raise TypeError("mask_flat must be iterable")

    if len(mask_list) != nx * ny * nz:
        raise ValueError(
            f"mask_flat length {len(mask_list)} does not match nx*ny*nz={nx*ny*nz}"
        )

    # -----------------------------
    # Validate mask values (structural only)
    # -----------------------------
    validated = []
    for idx, val in enumerate(mask_list):
        if not isinstance(val, int):
            raise TypeError(f"Mask entries must be integers, got {val!r} at index {idx}")
        if not math.isfinite(val):
            raise TypeError(f"Mask entries must be finite integers, got {val!r} at index {idx}")
        validated.append(int(val))

    arr = np.asarray(validated, dtype=int)

    # -----------------------------
    # Determine flattening order
    # -----------------------------
    formula = str(order_formula).strip().upper()

    # C / row-major
    if formula in ("C", "ROW_MAJOR"):
        return arr.reshape((nx, ny, nz), order="C")

    # F / column-major
    if formula in ("F", "FORTRAN", "COLUMN_MAJOR"):
        return arr.reshape((nx, ny, nz), order="F")

    # Explicit Fortran-style formulas
    if "I + NX*(J + NY*K)" in formula:
        return arr.reshape((nx, ny, nz), order="F")

    if "K + NZ*(J + NY*I)" in formula:
        tmp = arr.reshape((nz, ny, nx), order="F")
        return tmp.transpose(2, 1, 0)  # → (i,j,k)

    if "J + NY*(I + NX*K)" in formula:
        tmp = arr.reshape((ny, nx, nz), order="F")
        return tmp.transpose(1, 0, 2)  # → (i,j,k)

    # -----------------------------
    # Unknown formula
    # -----------------------------
    raise ValueError(
        f"Unrecognized order_formula '{order_formula}'. Supported values are: "
        "'C', 'F', "
        "'i + nx*(j + ny*k)', "
        "'k + nz*(j + ny*i)', "
        "'j + ny*(i + nx*k)'."
    )
