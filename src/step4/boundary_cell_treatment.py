# src/step4/boundary_cell_treatment.py

import numpy as np
from src.step4.boundary_fluid_helpers import (
    treat_boundary_fluid_no_slip,
    treat_boundary_fluid_slip,
    treat_boundary_fluid_inlet,
    treat_boundary_fluid_outlet,
    treat_boundary_fluid_symmetry,
)


def apply_boundary_cell_treatment(state):
    """
    Apply boundary-fluid treatment to cells marked as -1 in state["mask"].

    Responsibilities:
    - Identify boundary-fluid cells (mask == -1).
    - Determine which BC applies to each boundary-fluid cell.
    - Dispatch to the correct helper function.
    - Leave solid (0) and fluid (1) cells unchanged.
    - Record treatment status in state["BCApplied"].

    Notes:
    - This step happens AFTER all boundary conditions have been applied.
    - Boundary-fluid cells inherit behavior from the nearest boundary face.
    - This function does NOT modify ghost cells; BC sync already handled that.

    Returns
    -------
    state : dict-like
        Updated with boundary-fluid treatment applied.
    """

    mask = state.get("mask")
    if mask is None:
        return state

    # Prepare status tracking if not already present
    if "BCApplied" not in state:
        state["BCApplied"] = {}

    state["BCApplied"]["boundary_cells_checked"] = True

    # Identify boundary-fluid cells
    bf_cells = np.argwhere(mask == -1)
    if bf_cells.size == 0:
        return state

    # Build a lookup: face → BC type
    bc_map = {}
    for bc in state["config"].get("boundary_conditions", []):
        bc_type = bc.get("type")
        for face in bc.get("faces", []):
            bc_map[face] = bc_type

    # For each boundary-fluid cell, determine which face it touches
    nx = state["config"]["domain"]["nx"]
    ny = state["config"]["domain"]["ny"]
    nz = state["config"]["domain"]["nz"]

    for (i, j, k) in bf_cells:
        # Determine which face this cell touches
        if i == 0:
            face = "x_min"
        elif i == nx - 1:
            face = "x_max"
        elif j == 0:
            face = "y_min"
        elif j == ny - 1:
            face = "y_max"
        elif k == 0:
            face = "z_min"
        elif k == nz - 1:
            face = "z_max"
        else:
            # Should not happen: boundary-fluid must touch a boundary
            continue

        bc_type = bc_map.get(face)

        # Dispatch to the correct helper
        if bc_type == "no-slip":
            treat_boundary_fluid_no_slip(state, (i, j, k))

        elif bc_type == "slip":
            treat_boundary_fluid_slip(state, (i, j, k))

        elif bc_type == "inlet":
            treat_boundary_fluid_inlet(state, (i, j, k))

        elif bc_type == "outlet":
            treat_boundary_fluid_outlet(state, (i, j, k))

        elif bc_type == "symmetry":
            treat_boundary_fluid_symmetry(state, (i, j, k))

        else:
            # Unknown or missing BC — do nothing
            continue

    return state
