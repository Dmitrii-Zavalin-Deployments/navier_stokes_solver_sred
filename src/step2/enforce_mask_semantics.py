# file: step2/enforce_mask_semantics.py
from __future__ import annotations

from typing import Any

import numpy as np


def enforce_mask_semantics(state: Any) -> None:
    """
    Enforce CFD-specific semantics on the geometry mask.

    - All values must be in {-1, 0, 1}
    - There must be at least one fluid or boundary-fluid cell (1 or -1)

    Parameters
    ----------
    state : Any
        SimulationState-like object with attribute `Mask` as a 3D integer array.

    Raises
    ------
    ValueError
        If invalid mask values are found or if there is no fluid cell.
    """
    mask = np.asarray(state.Mask)

    valid_values = np.isin(mask, (-1, 0, 1))
    if not np.all(valid_values):
        bad = mask[~valid_values]
        unique_bad = np.unique(bad)
        raise ValueError(f"Invalid mask values detected: {unique_bad.tolist()} (allowed: -1, 0, 1)")

    fluid_like = (mask == 1) | (mask == -1)
    if not np.any(fluid_like):
        raise ValueError("Mask contains no fluid or boundary-fluid cells (no 1 or -1 values).")
