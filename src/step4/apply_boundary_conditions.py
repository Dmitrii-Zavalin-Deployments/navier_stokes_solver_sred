# src/step4/apply_boundary_conditions.py

import numpy as np


def apply_boundary_conditions(state):
    """
    Apply all boundary conditions to extended fields.

    Simplified, production-oriented design:
    - Iterate over boundary condition entries in config["boundary_conditions"].
    - For each BC, apply directly to the corresponding ghost cells of
      P_ext, U_ext, V_ext, W_ext.
    - Use simple, standard rules:
        * no-slip      → velocity ghost = -interior (mirror with sign flip)
        * inlet        → velocity ghost = prescribed value
        * outlet       → velocity ghost = interior (zero-gradient)
        * pressure_dirichlet → pressure ghost = prescribed value
        * pressure_neumann   → pressure ghost = interior (zero-gradient)
    - A minimal conflict detector warns if multiple BCs target the same
      (direction, side, variable) combination; last write wins.

    Notes:
    - This is a deliberately simplified BC system for Step‑4 v1.
    - No priority engine, no corner sync, no multi-pass logic.
    - Boundary-fluid (mask == -1) handling is intentionally minimal.
    """

    config = state.get("config", {})
    bcs = config.get("boundary_conditions", [])

    if not bcs:
        return state

    P_ext = state.get("P_ext")
    U_ext = state.get("U_ext")
    V_ext = state.get("V_ext")
    W_ext = state.get("W_ext")

    domain_cfg = config.get("domain", {})
    nx = domain_cfg["nx"]
    ny = domain_cfg["ny"]
    nz = domain_cfg["nz"]

    # Optional mask (used only for optional boundary-fluid adjustments)
    mask = state.get("mask", None)
    mask_arr = np.asarray(mask) if mask is not None else None

    # ---------------------------------------------------------
    # Conflict detector: (direction, side, variable) → last write wins
    # ---------------------------------------------------------
    applied_locations = {}

    for bc in bcs:
        bc_type = bc.get("type", "").lower()
        variable = bc.get("variable", "").lower()
        direction = bc.get("direction", "").lower()   # 'x', 'y', 'z'
        side = bc.get("side", "").lower()             # 'min', 'max'
        value = bc.get("value", 0.0)

        loc_key = (direction, side, variable)

        if loc_key in applied_locations:
            prev = applied_locations[loc_key]
            print(
                f"[Step 4] Warning: BC conflict on {loc_key}: "
                f"{prev} → {bc_type} (last write wins)"
            )

        applied_locations[loc_key] = bc_type

        # Dispatch based on variable
        if variable in ("u", "velocity_u", "velocity"):
            _apply_velocity_bc(U_ext, direction, side, bc_type, value)

        elif variable in ("v", "velocity_v"):
            _apply_velocity_bc(V_ext, direction, side, bc_type, value)

        elif variable in ("w", "velocity_w"):
            _apply_velocity_bc(W_ext, direction, side, bc_type, value)

        elif variable in ("p", "pressure"):
            _apply_pressure_bc(P_ext, direction, side, bc_type, value)

        else:
            # Unknown variable → ignore
            continue

    # ---------------------------------------------------------
    # Optional: boundary-fluid damping (disabled by default)
    # Uncomment if needed for stability near solids.
    # ---------------------------------------------------------
    # if mask_arr is not None:
    #     boundary_fluid = (mask_arr == -1)
    #     # Example: damp velocities in boundary-fluid cells
    #     U_ext[1:-1, 1:-1, 1:-1][boundary_fluid] *= 0.1
    #     V_ext[1:-1, 1:-1, 1:-1][boundary_fluid] *= 0.1
    #     W_ext[1:-1, 1:-1, 1:-1][boundary_fluid] *= 0.1

    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_velocity_bc(field, direction, side, bc_type, value):
    """
    Apply a simple velocity BC to a staggered field (U_ext, V_ext, W_ext).

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
        # Unknown BC type → ignore
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
    if bc_type in ("dirichlet", "fixed"):
        ghost[...] = value

    elif bc_type in ("neumann", "zero-gradient", "outlet"):
        np.copyto(ghost, interior)

    else:
        return
