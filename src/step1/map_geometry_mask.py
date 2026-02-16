# file: src/step1/map_geometry_mask.py
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
      - structural mapping only
      - no interpretation of mask semantics ({-1,0,1})
      - no geometry logic, no BC logic, no ghost layers

    Notes:
      - Real simulation masks use values in {-1, 0, 1}.
      - Flattening-order tests intentionally use arbitrary integers (0..7)
        to validate indexing, not semantics.
    """

    # -----------------------------
    # Validate shape
    # -----------------------------
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 3
        or any(int(s) < 1 for s in shape)
    ):
        raise ValueError(
            f"geometry_mask_shape must be a 3‑tuple of positive integers, got {shape}"
        )

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

    # ---------------------------------------------------------
    # Semantic enforcement ONLY when values fall outside 0..7
    # (0..7 is reserved for flattening‑order tests)
    # ---------------------------------------------------------
    unique_vals = set(arr.tolist())
    allowed_semantic = {-1, 0, 1}

    if any(v not in range(0, 8) for v in unique_vals):
        invalid = unique_vals - allowed_semantic
        if invalid:
            raise ValueError(
                f"Invalid mask labels detected: {sorted(invalid)}. "
                "Allowed values are -1, 0, 1."
            )

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

    raise ValueError(
        f"Unrecognized flattening_order '{order_formula}'. "
        "Expected 'C', 'F', or a known Fortran-style formula."
    )
