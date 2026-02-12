# src/step4/apply_all_boundary_conditions.py

from src.step4.bc_pressure import apply_pressure_bc
from src.step4.bc_velocity import apply_velocity_bc
from src.step4.bc_sync import (
    sync_dirichlet_ghosts,
    sync_neumann_ghosts,
    resolve_corner_and_edge_conflicts,
)


def apply_all_boundary_conditions(state):
    """
    Apply all boundary conditions defined in state["config"]["boundary_conditions"].

    This function is intentionally thin: it delegates actual BC logic to the
    specialized modules (pressure, velocity, sync, priority). Its responsibilities:

    - Iterate over all BC entries in the config.
    - Dispatch each BC to the correct subsystem.
    - Apply ghost synchronization after all BCs are applied.
    - Resolve corner and edge conflicts using priority rules.
    - Record BC application status in state["BCApplied"].

    Returns
    -------
    state : dict-like
        Updated with boundary conditions applied and status recorded.
    """

    bcs = state["config"].get("boundary_conditions", [])

    # Prepare BC status tracking
    state["BCApplied"] = {
        "boundary_conditions_status": {},
        "corner_conflicts_resolved": False,
        "ghosts_synchronized": False,
    }

    # Apply each BC
    for bc in bcs:
        bc_type = bc.get("type")
        faces = bc.get("faces", [])

        if bc_type in ("pressure_dirichlet", "pressure_neumann"):
            apply_pressure_bc(state, bc)

        elif bc_type in ("inlet", "outlet", "no-slip", "slip", "symmetry"):
            apply_velocity_bc(state, bc)

        else:
            # Unknown BC type — mark as skipped
            for face in faces:
                state["BCApplied"]["boundary_conditions_status"][face] = "unknown_type"
            continue

        # Mark each face as applied
        for face in faces:
            state["BCApplied"]["boundary_conditions_status"][face] = "applied"

    # After all BCs: synchronize ghost cells
    sync_dirichlet_ghosts(state)
    sync_neumann_ghosts(state)
    state["BCApplied"]["ghosts_synchronized"] = True

    # Resolve corner/edge conflicts
    resolve_corner_and_edge_conflicts(state)
    state["BCApplied"]["corner_conflicts_resolved"] = True

    return state
