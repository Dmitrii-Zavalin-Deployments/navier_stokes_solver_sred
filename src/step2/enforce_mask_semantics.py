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
    if state.mask is None:
        raise ValueError("State mask is missing.")

    mask = np.asarray(state.mask)

    # 1. Type Check (Scale Guard for integer logic)
    if mask.dtype.kind not in ("i", "u"):
        raise ValueError(f"Mask must be an integer array, got {mask.dtype}")

    # 2. Value Set Check
    if not np.isin(mask, [-1, 0, 1]).all():
        bad = np.unique(mask[~np.isin(mask, [-1, 0, 1])])
        raise ValueError(f"Invalid mask values: {bad.tolist()}. Only 0, 1, -1 are allowed.")

    # 3. Fluid Presence Check
    if not np.any((mask == 1) | (mask == -1)):
        raise ValueError("Mask contains no fluid or boundary-fluid cells. Simulation cannot proceed.")

    # 4. Preliminary Allocation (satisfies test assertions)
    state.is_fluid = (mask != 0)
    state.is_boundary_cell = (mask == -1)
    state.is_solid = (mask == 0)