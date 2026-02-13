# src/step4/assemble_bc_applied.py

from datetime import datetime


def assemble_bc_applied(state):
    """
    Expand the internal BCApplied structure into the full schema-compliant
    bc_applied block required by Step 4 output.

    Schema requires:
        - initial_velocity_enforced
        - pressure_initial_applied
        - velocity_initial_applied
        - ghost_cells_filled
        - boundary_cells_checked
        - boundary_conditions_status (with 6 faces)
        - version
        - timestamp_applied
    """

    # Internal block may or may not exist
    internal = state.get("bc_applied", {})

    # Extract internal flags if present, otherwise default to False
    initial_velocity_enforced = internal.get("initial_velocity_enforced", False)
    pressure_initial_applied = internal.get("pressure_initial_applied", False)
    velocity_initial_applied = internal.get("velocity_initial_applied", False)
    ghost_cells_filled = internal.get("ghost_cells_filled", False)
    boundary_cells_checked = internal.get("boundary_cells_checked", False)

    # Boundary conditions status for all 6 faces
    faces = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    bc_status = {}

    internal_faces = internal.get("boundary_conditions_status", {})

    for face in faces:
        bc_status[face] = internal_faces.get(face, {"applied": False})

    # Add version + timestamp
    version = "1.0"
    timestamp = datetime.utcnow().isoformat() + "Z"

    state["bc_applied"] = {
        "initial_velocity_enforced": initial_velocity_enforced,
        "pressure_initial_applied": pressure_initial_applied,
        "velocity_initial_applied": velocity_initial_applied,
        "ghost_cells_filled": ghost_cells_filled,
        "boundary_cells_checked": boundary_cells_checked,
        "boundary_conditions_status": bc_status,
        "version": version,
        "timestamp_applied": timestamp,
    }

    return state
