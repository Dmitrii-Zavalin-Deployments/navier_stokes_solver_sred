# src/step4/assemble_diagnostics.py

import numpy as np


def assemble_diagnostics(state):
    """
    Compute the Step‑4 diagnostics block.

    Required fields:
        - total_fluid_cells
        - grid_volume_per_cell
        - initialized
        - post_bc_max_velocity
        - post_bc_divergence_norm
        - bc_violation_count
    """

    # ---------------------------------------------------------
    # total_fluid_cells
    # Step‑4 uses state.is_fluid (boolean array), not state.mask
    # ---------------------------------------------------------
    if hasattr(state, "is_fluid") and state.is_fluid is not None:
        total_fluid_cells = int(np.sum(state.is_fluid))
    else:
        total_fluid_cells = 0

    # ---------------------------------------------------------
    # grid_volume_per_cell
    # Step‑4 uses dx, dy, dz from state.constants
    # ---------------------------------------------------------
    dx = state.constants.get("dx", 1.0)
    dy = state.constants.get("dy", 1.0)
    dz = state.constants.get("dz", 1.0)
    grid_volume_per_cell = float(dx * dy * dz)

    # ---------------------------------------------------------
    # initialized
    # True if extended fields exist
    # ---------------------------------------------------------
    initialized = (
        hasattr(state, "U_ext")
        and hasattr(state, "V_ext")
        and hasattr(state, "W_ext")
        and hasattr(state, "P_ext")
    )

    # ---------------------------------------------------------
    # post_bc_max_velocity
    # Max absolute velocity across all extended fields
    # ---------------------------------------------------------
    def max_abs(field):
        if isinstance(field, np.ndarray):
            return float(np.max(np.abs(field)))
        return 0.0

    post_bc_max_velocity = max(
        max_abs(getattr(state, "U_ext", None)),
        max_abs(getattr(state, "V_ext", None)),
        max_abs(getattr(state, "W_ext", None)),
    )

    # ---------------------------------------------------------
    # post_bc_divergence_norm
    # Step‑4 uses the divergence norm computed in Step‑3 health
    # ---------------------------------------------------------
    post_bc_divergence_norm = float(
        state.health.get("post_correction_divergence_norm", 0.0)
    )

    # ---------------------------------------------------------
    # bc_violation_count
    # Step‑4 BC module increments this counter
    # ---------------------------------------------------------
    bc_violation_count = 0
    if hasattr(state, "step4_diagnostics"):
        bc_violation_count = state.step4_diagnostics.get("bc_violation_count", 0)

    # ---------------------------------------------------------
    # Assemble final diagnostics block
    # ---------------------------------------------------------
    state.step4_diagnostics = {
        "total_fluid_cells": total_fluid_cells,
        "grid_volume_per_cell": grid_volume_per_cell,
        "initialized": initialized,
        "post_bc_max_velocity": post_bc_max_velocity,
        "post_bc_divergence_norm": post_bc_divergence_norm,
        "bc_violation_count": bc_violation_count,
    }

    return state
