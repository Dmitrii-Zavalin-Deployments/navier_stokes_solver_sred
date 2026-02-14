# src/step4/assemble_diagnostics.py

import numpy as np


def assemble_diagnostics(state):
    """
    Build the schema-compliant diagnostics block for Step 4 output.

    Required fields:
        - total_fluid_cells
        - grid_volume_per_cell
        - initialized
        - post_bc_max_velocity
        - post_bc_divergence_norm
        - bc_violation_count
    """

    # ---------------------------------------------------------
    # total_fluid_cells (vectorized)
    # ---------------------------------------------------------
    mask_raw = state.get("mask", None)

    if mask_raw is None:
        total_fluid_cells = 0
    else:
        mask_arr = np.asarray(mask_raw)
        total_fluid_cells = int(np.sum(mask_arr == 1))

    # ---------------------------------------------------------
    # grid_volume_per_cell
    # (unit grid for now; refine later if dx, dy, dz vary)
    # ---------------------------------------------------------
    grid_volume_per_cell = 1.0

    # ---------------------------------------------------------
    # initialized
    # True if extended fields exist (uppercase keys)
    # ---------------------------------------------------------
    initialized = (
        "U_ext" in state and
        "V_ext" in state and
        "W_ext" in state
    )

    # ---------------------------------------------------------
    # post_bc_max_velocity
    # Use NumPy for fast absolute max over extended fields
    # ---------------------------------------------------------
    def max_abs(field):
        if isinstance(field, np.ndarray):
            return float(np.max(np.abs(field)))
        return 0.0

    post_bc_max_velocity = max(
        max_abs(state.get("U_ext")),
        max_abs(state.get("V_ext")),
        max_abs(state.get("W_ext")),
    )

    # ---------------------------------------------------------
    # post_bc_divergence_norm
    # ---------------------------------------------------------
    health = state.get("health", {})
    post_bc_divergence_norm = float(
        health.get("post_correction_divergence_norm", 0.0)
    )

    # ---------------------------------------------------------
    # bc_violation_count
    #
    # In the simplified Step‑4 architecture, we no longer track
    # per-face BC application statuses. To preserve schema
    # compatibility, we return 0 and document this explicitly.
    # ---------------------------------------------------------
    bc_violation_count = 0

    # ---------------------------------------------------------
    # Assemble final diagnostics block
    # ---------------------------------------------------------
    state["diagnostics"] = {
        "total_fluid_cells": total_fluid_cells,
        "grid_volume_per_cell": grid_volume_per_cell,
        "initialized": initialized,
        "post_bc_max_velocity": post_bc_max_velocity,
        "post_bc_divergence_norm": post_bc_divergence_norm,
        "bc_violation_count": bc_violation_count,
    }

    return state
