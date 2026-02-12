# src/step4/bc_velocity.py

import numpy as np
from src.step4.utils_geometry import get_face_slices, get_normal_direction


def apply_velocity_bc(state, bc):
    """
    Apply velocity boundary conditions to U_ext, V_ext, W_ext.

    Supported BC types:
        - inlet:      set velocity components to a specified vector
        - outlet:     zero-gradient (copy interior values)
        - no-slip:    U = V = W = 0 on the boundary
        - slip:       zero normal velocity, tangential copied from interior
        - symmetry:   zero normal velocity, tangential mirrored

    This function only modifies ghost cells on the specified faces.
    Corner/edge conflict resolution is handled later in bc_sync.py.

    Parameters
    ----------
    state : dict-like
        Must contain U_ext, V_ext, W_ext.

    bc : dict
        A boundary condition entry with:
            bc["type"]  : one of the supported velocity BC types
            bc["faces"] : list of faces, e.g. ["x_min", "y_max"]
            bc["velocity"] : (optional) inlet velocity vector

    Returns
    -------
    None (state is modified in-place)
    """

    bc_type = bc.get("type")
    faces = bc.get("faces", [])

    for face in faces:
        if bc_type == "inlet":
            _apply_inlet(state, face, bc)

        elif bc_type == "outlet":
            _apply_outlet(state, face)

        elif bc_type == "no-slip":
            _apply_no_slip(state, face)

        elif bc_type == "slip":
            _apply_slip(state, face)

        elif bc_type == "symmetry":
            _apply_symmetry(state, face)

        else:
            # Unknown BC type — ignore
            continue


# ----------------------------------------------------------------------
# Inlet BC
# ----------------------------------------------------------------------
def _apply_inlet(state, face, bc):
    """Set velocity to a constant vector on the boundary face."""
    u0, v0, w0 = bc.get("velocity", [0.0, 0.0, 0.0])
    sl = get_face_slices(face)

    state["U_ext"][sl] = u0
    state["V_ext"][sl] = v0
    state["W_ext"][sl] = w0


# ----------------------------------------------------------------------
# Outlet BC (zero gradient)
# ----------------------------------------------------------------------
def _apply_outlet(state, face):
    """Copy interior values into ghost cells (zero normal gradient)."""
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    if face == "x_min":
        U[0, :, :] = U[1, :, :]
        V[0, :, :] = V[1, :, :]
        W[0, :, :] = W[1, :, :]

    elif face == "x_max":
        U[-1, :, :] = U[-2, :, :]
        V[-1, :, :] = V[-2, :, :]
        W[-1, :, :] = W[-2, :, :]

    elif face == "y_min":
        U[:, 0, :] = U[:, 1, :]
        V[:, 0, :] = V[:, 1, :]
        W[:, 0, :] = W[:, 1, :]

    elif face == "y_max":
        U[:, -1, :] = U[:, -2, :]
        V[:, -1, :] = V[:, -2, :]
        W[:, -1, :] = W[:, -2, :]

    elif face == "z_min":
        U[:, :, 0] = U[:, :, 1]
        V[:, :, 0] = V[:, :, 1]
        W[:, :, 0] = W[:, :, 1]

    elif face == "z_max":
        U[:, :, -1] = U[:, :, -2]
        V[:, :, -1] = V[:, :, -2]
        W[:, :, -1] = W[:, :, -2]


# ----------------------------------------------------------------------
# No-slip BC
# ----------------------------------------------------------------------
def _apply_no_slip(state, face):
    """Set all velocity components to zero on the boundary face."""
    sl = get_face_slices(face)
    state["U_ext"][sl] = 0.0
    state["V_ext"][sl] = 0.0
    state["W_ext"][sl] = 0.0


# ----------------------------------------------------------------------
# Slip BC
# ----------------------------------------------------------------------
def _apply_slip(state, face):
    """
    Slip BC:
        - normal velocity = 0
        - tangential velocities copied from interior
    """
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    normal = get_normal_direction(face)

    if normal == "x":
        # normal component = U
        if face == "x_min":
            U[0, :, :] = 0.0
            V[0, :, :] = V[1, :, :]
            W[0, :, :] = W[1, :, :]
        else:  # x_max
            U[-1, :, :] = 0.0
            V[-1, :, :] = V[-2, :, :]
            W[-1, :, :] = W[-2, :, :]

    elif normal == "y":
        if face == "y_min":
            V[:, 0, :] = 0.0
            U[:, 0, :] = U[:, 1, :]
            W[:, 0, :] = W[:, 1, :]
        else:  # y_max
            V[:, -1, :] = 0.0
            U[:, -1, :] = U[:, -2, :]
            W[:, -1, :] = W[:, -2, :]

    elif normal == "z":
        if face == "z_min":
            W[:, :, 0] = 0.0
            U[:, :, 0] = U[:, :, 1]
            V[:, :, 0] = V[:, :, 1]
        else:  # z_max
            W[:, :, -1] = 0.0
            U[:, :, -1] = U[:, :, -2]
            V[:, :, -1] = V[:, :, -2]


# ----------------------------------------------------------------------
# Symmetry BC
# ----------------------------------------------------------------------
def _apply_symmetry(state, face):
    """
    Symmetry BC:
        - normal velocity = 0
        - tangential velocities mirrored (zero normal gradient)
    """
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    normal = get_normal_direction(face)

    if normal == "x":
        if face == "x_min":
            U[0, :, :] = 0.0
            V[0, :, :] = V[1, :, :]
            W[0, :, :] = W[1, :, :]
        else:
            U[-1, :, :] = 0.0
            V[-1, :, :] = V[-2, :, :]
            W[-1, :, :] = W[-2, :, :]

    elif normal == "y":
        if face == "y_min":
            V[:, 0, :] = 0.0
            U[:, 0, :] = U[:, 1, :]
            W[:, 0, :] = W[:, 1, :]
        else:
            V[:, -1, :] = 0.0
            U[:, -1, :] = U[:, -2, :]
            W[:, -1, :] = W[:, -2, :]

    elif normal == "z":
        if face == "z_min":
            W[:, :, 0] = 0.0
            U[:, :, 0] = U[:, :, 1]
            V[:, :, 0] = V[:, :, 1]
        else:
            W[:, :, -1] = 0.0
            U[:, :, -1] = U[:, :, -2]
            V[:, :, -1] = V[:, :, -2]
