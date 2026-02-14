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
        * no-slip      → velocity ghost = -interior (or 0, depending on convention).
        * inlet        → velocity ghost = prescribed value.
        * outlet       → velocity ghost = interior (zero-gradient).
        * pressure_dirichlet → pressure ghost = prescribed value.
        * pressure_neumann   → pressure ghost = interior (zero-gradient).
    - A minimal conflict detector warns if multiple BCs target the same
      (direction, side, variable) combination; last write wins.
    """

    config = state.get("config", {})
    bcs = config.get("boundary_conditions", [])

    if not bcs:
        return state

    P_ext = state.get("P_ext", None)
    U_ext = state.get("U_ext", None)
    V_ext = state.get("V_ext", None)
    W_ext = state.get("W_ext", None)

    domain_cfg = config.get("domain", {})
    nx = domain_cfg["nx"]
    ny = domain_cfg["ny"]
    nz = domain_cfg["nz"]

    # Simple conflict detector: (direction, side, variable) → last write wins
    applied_locations = set()

    for bc in bcs:
        bc_type = bc.get("type", "").lower()
        variable = bc.get("variable", "").lower()
        direction = bc.get("direction", "").lower()   # 'x', 'y', or 'z'
        side = bc.get("side", "").lower()             # 'min' or 'max'
        value = bc.get("value", 0.0)

        loc_key = (direction, side, variable)
        if loc_key in applied_locations:
            # Lightweight warning; in real production you might log instead of print.
            print(
                f"[Step 4] Warning: multiple BCs on {loc_key}; "
                f"using last one (type={bc_type})"
            )
        applied_locations.add(loc_key)

        if variable in ("u", "velocity_u", "velocity"):
            _apply_velocity_bc(
                U_ext, direction, side, nx, ny, nz, bc_type, value
            )
        elif variable in ("v", "velocity_v"):
            _apply_velocity_bc(
                V_ext, direction, side, nx, ny, nz, bc_type, value
            )
        elif variable in ("w", "velocity_w"):
            _apply_velocity_bc(
                W_ext, direction, side, nx, ny, nz, bc_type, value
            )
        elif variable in ("p", "pressure"):
            _apply_pressure_bc(
                P_ext, direction, side, nx, ny, nz, bc_type, value
            )
        else:
            # Unknown variable; ignore for now.
            continue

    return state


def _apply_velocity_bc(field, direction, side, nx, ny, nz, bc_type, value):
    """
    Apply a simple velocity BC to the given staggered field.

    Assumes:
    - field is one of U_ext, V_ext, W_ext with appropriate staggering.
    - direction ∈ {'x', 'y', 'z'}
    - side ∈ {'min', 'max'}
    """

    if field is None:
        return

    # Determine ghost slice and interior slice based on direction/side.
    if direction == "x":
        if side == "min":
            ghost = field[0, :, :]
            interior = field[1, :, :]
        else:  # 'max'
            ghost = field[-1, :, :]
            interior = field[-2, :, :]
    elif direction == "y":
        if side == "min":
            ghost = field[:, 0, :]
            interior = field[:, 1, :]
        else:
            ghost = field[:, -1, :]
            interior = field[:, -2, :]
    elif direction == "z":
        if side == "min":
            ghost = field[:, :, 0]
            interior = field[:, :, 1]
        else:
            ghost = field[:, :, -1]
            interior = field[:, :, -2]
    else:
        return

    if bc_type in ("no-slip", "noslip"):
        # Common choice: ghost = -interior (mirror with sign flip)
        np.copyto(ghost, -interior)
    elif bc_type in ("inlet", "dirichlet"):
        ghost[...] = value
    elif bc_type in ("outlet", "neumann", "zero-gradient"):
        np.copyto(ghost, interior)
    else:
        # Unknown BC type: do nothing for now.
        return


def _apply_pressure_bc(field, direction, side, nx, ny, nz, bc_type, value):
    """
    Apply a simple pressure BC to P_ext.

    Assumes:
    - field is P_ext with shape (nx+2, ny+2, nz+2).
    - direction ∈ {'x', 'y', 'z'}
    - side ∈ {'min', 'max'}
    """

    if field is None:
        return

    if direction == "x":
        if side == "min":
            ghost = field[0, :, :]
            interior = field[1, :, :]
        else:
            ghost = field[-1, :, :]
            interior = field[-2, :, :]
    elif direction == "y":
        if side == "min":
            ghost = field[:, 0, :]
            interior = field[:, 1, :]
        else:
            ghost = field[:, -1, :]
            interior = field[:, -2, :]
    elif direction == "z":
        if side == "min":
            ghost = field[:, :, 0]
            interior = field[:, :, 1]
        else:
            ghost = field[:, :, -1]
            interior = field[:, :, -2]
    else:
        return

    if bc_type in ("dirichlet", "fixed"):
        ghost[...] = value
    elif bc_type in ("neumann", "zero-gradient", "outlet"):
        np.copyto(ghost, interior)
    else:
        # Unknown BC type: do nothing for now.
        return
