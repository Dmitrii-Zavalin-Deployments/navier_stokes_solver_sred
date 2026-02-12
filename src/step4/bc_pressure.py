# src/step4/bc_pressure.py

import numpy as np
from src.step4.utils_geometry import get_face_slices


def apply_pressure_bc(state, bc):
    """
    Apply pressure boundary conditions (Dirichlet or Neumann) to P_ext.

    Supported BC types:
        - "pressure_dirichlet": P = constant on the boundary face
        - "pressure_neumann":   ∂P/∂n = 0  (copy interior value into ghost)

    This function only modifies ghost cells on the specified faces.
    Corner/edge conflict resolution is handled later in bc_sync.py.

    Parameters
    ----------
    state : dict-like
        Must contain P_ext and config["domain"].

    bc : dict
        A boundary condition entry with:
            bc["type"]  : "pressure_dirichlet" or "pressure_neumann"
            bc["faces"] : list of faces, e.g. ["x_min", "y_max"]
            bc["value"] : (optional) Dirichlet pressure value

    Returns
    -------
    None (state is modified in-place)
    """

    bc_type = bc.get("type")
    faces = bc.get("faces", [])

    if "P_ext" not in state:
        return

    P = state["P_ext"]

    for face in faces:
        sl = get_face_slices(face)

        if bc_type == "pressure_dirichlet":
            # Dirichlet: set ghost cells to constant value
            p_val = bc.get("value", 0.0)
            P[sl] = p_val

        elif bc_type == "pressure_neumann":
            # Neumann: copy interior value into ghost cells
            _apply_pressure_neumann_face(P, face)

        else:
            # Unknown BC type — ignore
            continue


def _apply_pressure_neumann_face(P, face):
    """
    Apply ∂P/∂n = 0 on a single face by copying the adjacent interior value
    into the ghost layer.

    This is a helper function used only inside this module.
    """

    if face == "x_min":
        P[0, :, :] = P[1, :, :]
    elif face == "x_max":
        P[-1, :, :] = P[-2, :, :]

    elif face == "y_min":
        P[:, 0, :] = P[:, 1, :]
    elif face == "y_max":
        P[:, -1, :] = P[:, -2, :]

    elif face == "z_min":
        P[:, :, 0] = P[:, :, 1]
    elif face == "z_max":
        P[:, :, -1] = P[:, :, -2]
