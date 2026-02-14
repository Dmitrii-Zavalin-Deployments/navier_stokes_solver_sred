# file: src/step4/assemble_diagnostics.py

import numpy as np


def assemble_diagnostics(state):
    """
    Build the schema-compliant diagnostics block for Step 4 output.

    Schema requires:
        - total_fluid_cells
        - grid_volume_per_cell
        - initialized
        - post_bc_max_velocity
        - post_bc_divergence_norm
        - bc_violation_count
    """

    # ---------------------------------------------------------
    # total_fluid_cells
    # ---------------------------------------------------------
    mask = state.get("mask", [])
    total_fluid_cells = 0
    for k in range(len(mask)):
        for j in range(len(mask[k])):
            for i in range(len(mask[k][j])):
                if mask[k][j][i] == 1:
                    total_fluid_cells += 1

    # ---------------------------------------------------------
    # grid_volume_per_cell
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
    # Must consider BOTH extended fields and Step‑3 fields.
    # ---------------------------------------------------------
    def max_abs(field):
        if isinstance(field, np.ndarray):
            return float(np.max(np.abs(field)))
        return 0.0

    fields = state.get("fields", {})

    candidates = [
        state.get("U_ext"),
        state.get("V_ext"),
        state.get("W_ext"),
        fields.get("U"),
        fields.get("V"),
        fields.get("W"),
    ]

    post_bc_max_velocity = 0.0
    for f in candidates:
        val = max_abs(f)
        if val > post_bc_max_velocity:
            post_bc_max_velocity = val

    # ---------------------------------------------------------
    # post_bc_divergence_norm
    # ---------------------------------------------------------
    health = state.get("health", {})
    post_bc_divergence_norm = float(
        health.get("post_correction_divergence_norm", 0.0)
    )

    # ---------------------------------------------------------
    # bc_violation_count
    # Now statuses are STRINGS ("applied", "skipped", "error")
    # ---------------------------------------------------------
    bc_applied = state.get("bc_applied", {})
    bc_status = bc_applied.get("boundary_conditions_status", {})

    bc_violation_count = 0
    for face, status in bc_status.items():
        if status != "applied":
            bc_violation_count += 1

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
