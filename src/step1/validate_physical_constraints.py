from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np

if TYPE_CHECKING:
    from src.solver_state import SolverState

def validate_physical_constraints(state: SolverState) -> None:
    """
    Logical Firewall: Strictly follows SSoT Hierarchy.
    """
    # ACCESS VIA CONTAINERS (Architecture Guard)
    grid = state.grid
    fluid = state.fluid
    fields = state.fields
    masks = state.masks

    # 1. Physics Validation
    if not (math.isfinite(fluid.rho) and fluid.rho > 0):
        raise ValueError(f"Stability Violation: rho must be > 0, got {fluid.rho}")
    
    if not (math.isfinite(fluid.mu) and fluid.mu >= 0):
        raise ValueError(f"Physicality Violation: mu must be >= 0, got {fluid.mu}")

    # 2. Geometric Validation
    if any(dim <= 0 for dim in [grid.nx, grid.ny, grid.nz]):
         raise ValueError("Topology Violation: Grid dimensions must be positive.")

    # 3. Field Sanity (NaN Detection)
    for name in ["P", "U", "V", "W"]:
        field_arr = getattr(fields, name)
        if not np.all(np.isfinite(field_arr)):
            raise ValueError(f"Genesis Error: Field '{name}' contains non-finite values.")

    # 4. Mask Integrity
    expected_len = grid.nx * grid.ny * grid.nz
    if len(masks.mask) != expected_len:
        raise ValueError(f"Mask Length Mismatch: {len(masks.mask)} != {expected_len}")
