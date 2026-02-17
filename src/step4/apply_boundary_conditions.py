# src/step4/apply_boundary_conditions.py

import numpy as np


def apply_boundary_conditions(state):
    """
    Apply all boundary conditions to extended fields (P_ext, U_ext, V_ext, W_ext).

    Step‑4 simplified rules:
      • no-slip            → velocity ghost = -interior
      • inlet              → velocity ghost = prescribed value
      • outlet/neumann     → velocity ghost = interior (zero-gradient)
      • pressure_dirichlet → pressure ghost = prescribed value
      • pressure_neumann   → pressure ghost = interior

    Notes:
      • Last write wins (no priority engine)
      • No corner/edge sync
      • No mask-based boundary-fluid logic
      • Operates directly on SolverState attributes
    """

    # ---------------------------------------------------------
    # Load BC table
    # ---------------------------------------------------------
    bcs = state.config.get("boundary_conditions", [])
    if not bcs:
        return state

    # ---------------------------------------------------------
    # Extended fields (allocated in initialize_extended_fields)
    # ---------------------------------------------------------
    P_ext = state.P_ext
    U_ext = state.U_ext
    V_ext = state.V_ext
    W_ext = state.W_ext

    # ---------------------------------------------------------
    # Diagnostics counter for BC conflicts
    # ---------------------------------------------------------
    if not hasattr(state, "step4_diagnostics"):
        state.step4_diagnostics = {}

    state.step4_diagnostics.setdefault("bc_violation_count", 0)

    applied_locations = {}

    # ---------------------------------------------------------
    # Main BC loop
    # ---------------------------------------------------------
    for bc in bcs:
        bc_type = bc.get("type", "").lower()
        variable = bc.get("variable", "").lower()
        direction = bc.get("direction", "").lower()   # 'x', 'y', 'z'
        side = bc.get("side", "").lower()             # 'min', 'max'
        value = bc.get("value", 0.0)

        loc_key = (direction, side, variable)

        # Conflict detection
        if loc_key in applied_locations:
            state.step4_diagnostics["bc_violation_count"] += 1

        applied_locations[loc_key] = bc_type

        # Dispatch
        if variable == "u":
            _apply_velocity_bc(U_ext, direction, side, bc_type, value)

        elif variable == "v":
            _apply_velocity_bc(V_ext, direction, side, bc_type, value)

        elif variable == "w":
            _apply_velocity_bc(W_ext, direction, side, bc_type, value)

        elif variable == "p":
            _apply_pressure_bc(P_ext, direction, side, bc_type, value)

        else:
            # Unknown variable → ignore
            continue

    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_velocity_bc(field, direction, side, bc_type, value):
    """
    Apply a simple velocity BC to a staggered extended field (U_ext, V_ext, W_ext).

    direction ∈ {'x', 'y', 'z'}
    side      ∈ {'min', 'max'}
    """

    if field is None:
        return

    # Select ghost and interior slices
    if direction == "x":
        ghost = field[0, :, :] if side == "min" else field[-1, :, :]
        interior = field[1, :, :] if side == "min" else field[-2, :, :]

    elif direction == "y":
        ghost = field[:, 0, :] if side == "min" else field[:, -1, :]
        interior = field[:, 1, :] if side == "min" else field[:, -2, :]

    elif direction == "z":
        ghost = field[:, :, 0] if side == "min" else field[:, :, -1]
        interior = field[:, :, 1] if side == "min" else field[:, :, -2]

    else:
        return

    # Apply BC
    if bc_type in ("no-slip", "noslip"):
        np.copyto(ghost, -interior)

    elif bc_type in ("inlet", "dirichlet"):
        ghost[...] = value

    elif bc_type in ("outlet", "neumann", "zero-gradient"):
        np.copyto(ghost, interior)

    else:
        return


def _apply_pressure_bc(field, direction, side, bc_type, value):
    """
    Apply a simple pressure BC to P_ext.

    direction ∈ {'x', 'y', 'z'}
    side      ∈ {'min', 'max'}
    """

    if field is None:
        return

    # Select ghost and interior slices
    if direction == "x":
        ghost = field[0, :, :] if side == "min" else field[-1, :, :]
        interior = field[1, :, :] if side == "min" else field[-2, :, :]

    elif direction == "y":
        ghost = field[:, 0, :] if side == "min" else field[:, -1, :]
        interior = field[:, 1, :] if side == "min" else field[:, -2, :]

    elif direction == "z":
        ghost = field[:, :, 0] if side == "min" else field[:, :, -1]
        interior = field[:, :, 1] if side == "min" else field[:, :, -2]

    else:
        return

    # Apply BC
    if bc_type in ("dirichlet", "fixed", "pressure_dirichlet"):
        ghost[...] = value

    elif bc_type in ("neumann", "zero-gradient", "outlet", "pressure_neumann"):
        np.copyto(ghost, interior)

    else:
        return
