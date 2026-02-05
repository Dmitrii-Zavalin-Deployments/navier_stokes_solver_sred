# src/step2/enforce_mask_semantics.py
from __future__ import annotations

from typing import Any
import numpy as np


def enforce_mask_semantics(state: Any) -> None:
    """
    Enforce CFD-specific semantics on the geometry mask.

    Rules:
    - Mask must be integer dtype.
    - Allowed values: -1 (boundary-fluid), 0 (solid), 1 (fluid).
    - There must be at least one fluid-like cell (1 or -1).

    Parameters
    ----------
    state : Any
        SimulationState-like object with key "Mask" as a 3D integer array.

    Raises
    ------
    ValueError
        If invalid mask values are found, mask is not integer dtype,
        or if there is no fluid or boundary-fluid cell.
    """

    mask = np.asarray(state["Mask"])

    # Strict dtype check (tests require float masks to raise)
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask must be an integer array")

    # Allowed values
    valid = np.isin(mask, (-1, 0, 1))
    if not np.all(valid):
        bad_values = np.unique(mask[~valid])
        raise ValueError(
            f"Invalid mask values detected: {bad_values.tolist()} (allowed: -1, 0, 1)"
        )

    # Must contain at least one fluid-like cell
    fluid_like = (mask == 1) | (mask == -1)
    if not np.any(fluid_like):
        raise ValueError(
            "Mask contains no fluid or boundary-fluid cells (no 1 or -1 values)."
        )
