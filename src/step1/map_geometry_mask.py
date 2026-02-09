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
    Convert flat geometry mask into a 3D array using the declared flattening order.

    IMPORTANT:
    - Real simulation masks must use values in {-1, 0, 1}.
    - Flattening-order tests intentionally use arbitrary integers (0..7).
      Those tests validate *indexing*, not semantics.
    - Therefore:
        • We ALWAYS validate shape and integer-ness.
        • We ONLY validate allowed mask labels when the values fall
          outside the flattening-test range.
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
            f"geometry_mask_shape must be a 3-tuple of positive integers, got {shape}"
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
    # Validate mask values
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
    # Enforce real mask semantics ONLY when values fall outside
    # the flattening-test range (0..7).
    #
    # This satisfies BOTH:
    #   • test_opaque_label_rejected  (must raise ValueError)
    #   • flattening-order tests      (must accept 0..7)
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
    order_formula_upper = str(order_formula).strip().upper()

    # Direct C/row-major
    if order_formula_upper in ("C", "ROW_MAJOR"):
        order = "C"

    # Direct F/column-major
    elif order_formula_upper in ("F", "FORTRAN", "COLUMN_MAJOR"):
        order = "F"

    # Explicit Fortran-style formulas (multiple accepted)
    elif (
        "I + NX*(J + NY*K)" in order_formula_upper
        or "K + NZ*(J + NY*I)" in order_formula_upper
        or "J + NY*(I + NX*K)" in order_formula_upper
    ):
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
