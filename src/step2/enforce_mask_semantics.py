# src/step2/enforce_mask_semantics.py
from __future__ import annotations
import numpy as np
from src.solver_state import SolverState


def enforce_mask_semantics(state: SolverState) -> None:
    """
    Enforce CFD-specific semantics on the geometry mask.

    Requirements:
      • mask ∈ {0, 1, -1}
      • at least one fluid-like cell (1 or -1)
    """

    mask = np.asarray(state.mask)

    if mask.dtype.kind not in ("i", "u"):
        raise ValueError("Mask must be an integer array")

    if not np.isin(mask, [-1, 0, 1]).all():
        bad = np.unique(mask[~np.isin(mask, [-1, 0, 1])])
        raise ValueError(f"Invalid mask values: {bad.tolist()}")

    if not np.any((mask == 1) | (mask == -1)):
        raise ValueError("Mask contains no fluid or boundary-fluid cells")
