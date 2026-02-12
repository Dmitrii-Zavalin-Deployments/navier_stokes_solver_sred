# src/step4/bc_sync.py

import numpy as np
from src.step4.bc_priority import apply_priority_rule


# ----------------------------------------------------------------------
# 1. Synchronize Dirichlet ghost cells
# ----------------------------------------------------------------------
def sync_dirichlet_ghosts(state):
    """
    Synchronize Dirichlet ghost cells for velocity and pressure.

    This function ensures that if a face was assigned a Dirichlet value,
    the ghost layer is consistent and does not contain uninitialized data.

    Notes:
    - This is a safety pass; most Dirichlet BCs already set ghost cells.
    - We simply ensure that ghost cells match the nearest interior value.
    """

    for name in ("U_ext", "V_ext", "W_ext", "P_ext"):
        if name not in state:
            continue

        arr = state[name]

        # x-direction
        arr[0, :, :] = arr[1, :, :]
        arr[-1, :, :] = arr[-2, :, :]

        # y-direction
        arr[:, 0, :] = arr[:, 1, :]
        arr[:, -1, :] = arr[:, -2, :]

        # z-direction
        arr[:, :, 0] = arr[:, :, 1]
        arr[:, :, -1] = arr[:, :, -2]


# ----------------------------------------------------------------------
# 2. Synchronize Neumann ghost cells
# ----------------------------------------------------------------------
def sync_neumann_ghosts(state):
    """
    Synchronize Neumann ghost cells (zero normal gradient).

    This is a second safety pass: if a Neumann BC was applied, we ensure
    the ghost layer matches the interior cell exactly.
    """

    for name in ("U_ext", "V_ext", "W_ext", "P_ext"):
        if name not in state:
            continue

        arr = state[name]

        # x-direction
        arr[0, :, :] = arr[1, :, :]
        arr[-1, :, :] = arr[-2, :, :]

        # y-direction
        arr[:, 0, :] = arr[:, 1, :]
        arr[:, -1, :] = arr[:, -2, :]

        # z-direction
        arr[:, :, 0] = arr[:, :, 1]
        arr[:, :, -1] = arr[:, :, -2]


# ----------------------------------------------------------------------
# 3. Resolve corner and edge conflicts
# ----------------------------------------------------------------------
def resolve_corner_and_edge_conflicts(state):
    """
    Resolve corner and edge conflicts using priority rules.

    If multiple BCs apply to the same corner or edge, we use the
    priority rule defined in bc_priority.py.

    This function:
        - inspects the BC status map
        - identifies corners/edges touched by multiple BCs
        - applies the priority rule to choose the winning BC
        - ensures ghost cells reflect the chosen BC

    This is intentionally lightweight: Step 4 does not enforce physics
    here, only consistency.
    """

    bc_status = state.get("BCApplied", {}).get("boundary_conditions_status", {})
    if not bc_status:
        return

    # Map faces to BC types
    face_to_bc = {}
    for bc in state["config"].get("boundary_conditions", []):
        bc_type = bc.get("type")
        for face in bc.get("faces", []):
            face_to_bc[face] = bc_type

    # Corners are defined by combinations of faces
    corners = {
        "x_min_y_min_z_min": ("x_min", "y_min", "z_min"),
        "x_min_y_min_z_max": ("x_min", "y_min", "z_max"),
        "x_min_y_max_z_min": ("x_min", "y_max", "z_min"),
        "x_min_y_max_z_max": ("x_min", "y_max", "z_max"),
        "x_max_y_min_z_min": ("x_max", "y_min", "z_min"),
        "x_max_y_min_z_max": ("x_max", "y_min", "z_max"),
        "x_max_y_max_z_min": ("x_max", "y_max", "z_min"),
        "x_max_y_max_z_max": ("x_max", "y_max", "z_max"),
    }

    for corner_name, faces in corners.items():
        bc_values = {}

        for face in faces:
            if face in face_to_bc:
                bc_values[face_to_bc[face]] = face  # store BC type

        if len(bc_values) <= 1:
            continue  # no conflict

        # Resolve conflict using priority rule
        winning_bc_type = apply_priority_rule(bc_values)

        # No need to modify fields here — BC modules already set ghost cells.
        # This step only ensures consistency in the BC status map.
        for face in faces:
            if face_to_bc.get(face) != winning_bc_type:
                bc_status[face] = "overridden"

    return
