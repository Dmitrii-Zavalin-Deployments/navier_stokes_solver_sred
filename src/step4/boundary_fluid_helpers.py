# src/step4/boundary_fluid_helpers.py

import numpy as np


def treat_boundary_fluid_no_slip(state, idx):
    """
    Boundary-fluid treatment for no-slip walls:
        - All velocity components set to zero at the boundary-fluid cell.
    """
    i, j, k = idx
    state["U_ext"][i+1, j+1, k+1] = 0.0
    state["V_ext"][i+1, j+1, k+1] = 0.0
    state["W_ext"][i+1, j+1, k+1] = 0.0


def treat_boundary_fluid_slip(state, idx):
    """
    Boundary-fluid treatment for slip walls:
        - Normal velocity = 0
        - Tangential velocities copied from the nearest interior cell
    """
    i, j, k = idx
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    # Determine which face this boundary-fluid cell touches
    nx = state["config"]["domain"]["nx"]
    ny = state["config"]["domain"]["ny"]
    nz = state["config"]["domain"]["nz"]

    # x-normal
    if i == 0:
        U[i+1, j+1, k+1] = 0.0
        V[i+1, j+1, k+1] = V[i+2, j+1, k+1]
        W[i+1, j+1, k+1] = W[i+2, j+1, k+1]
    elif i == nx - 1:
        U[i+1, j+1, k+1] = 0.0
        V[i+1, j+1, k+1] = V[i, j+1, k+1]
        W[i+1, j+1, k+1] = W[i, j+1, k+1]

    # y-normal
    elif j == 0:
        V[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+2, k+1]
        W[i+1, j+1, k+1] = W[i+1, j+2, k+1]
    elif j == ny - 1:
        V[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j, k+1]
        W[i+1, j+1, k+1] = W[i+1, j, k+1]

    # z-normal
    elif k == 0:
        W[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+1, k+2]
        V[i+1, j+1, k+1] = V[i+1, j+1, k+2]
    elif k == nz - 1:
        W[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+1, k]
        V[i+1, j+1, k+1] = V[i+1, j+1, k]


def treat_boundary_fluid_inlet(state, idx):
    """
    Boundary-fluid treatment for inlet:
        - Velocity set to inlet velocity vector.
    """
    i, j, k = idx
    u0, v0, w0 = state["config"]["initial_conditions"].get("initial_velocity", [0.0, 0.0, 0.0])

    state["U_ext"][i+1, j+1, k+1] = u0
    state["V_ext"][i+1, j+1, k+1] = v0
    state["W_ext"][i+1, j+1, k+1] = w0


def treat_boundary_fluid_outlet(state, idx):
    """
    Boundary-fluid treatment for outlet:
        - Zero-gradient: copy nearest interior velocity.
    """
    i, j, k = idx
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    # Determine which face this boundary-fluid cell touches
    nx = state["config"]["domain"]["nx"]
    ny = state["config"]["domain"]["ny"]
    nz = state["config"]["domain"]["nz"]

    if i == 0:
        U[i+1, j+1, k+1] = U[i+2, j+1, k+1]
        V[i+1, j+1, k+1] = V[i+2, j+1, k+1]
        W[i+1, j+1, k+1] = W[i+2, j+1, k+1]
    elif i == nx - 1:
        U[i+1, j+1, k+1] = U[i, j+1, k+1]
        V[i+1, j+1, k+1] = V[i, j+1, k+1]
        W[i+1, j+1, k+1] = W[i, j+1, k+1]

    elif j == 0:
        U[i+1, j+1, k+1] = U[i+1, j+2, k+1]
        V[i+1, j+1, k+1] = V[i+1, j+2, k+1]
        W[i+1, j+1, k+1] = W[i+1, j+2, k+1]
    elif j == ny - 1:
        U[i+1, j+1, k+1] = U[i+1, j, k+1]
        V[i+1, j+1, k+1] = V[i+1, j, k+1]
        W[i+1, j+1, k+1] = W[i+1, j, k+1]

    elif k == 0:
        U[i+1, j+1, k+1] = U[i+1, j+1, k+2]
        V[i+1, j+1, k+1] = V[i+1, j+1, k+2]
        W[i+1, j+1, k+1] = W[i+1, j+1, k+2]
    elif k == nz - 1:
        U[i+1, j+1, k+1] = U[i+1, j+1, k]
        V[i+1, j+1, k+1] = V[i+1, j+1, k]
        W[i+1, j+1, k+1] = W[i+1, j+1, k]


def treat_boundary_fluid_symmetry(state, idx):
    """
    Boundary-fluid treatment for symmetry:
        - Normal velocity = 0
        - Tangential velocities mirrored (zero normal gradient)
    """
    i, j, k = idx
    U = state["U_ext"]
    V = state["V_ext"]
    W = state["W_ext"]

    nx = state["config"]["domain"]["nx"]
    ny = state["config"]["domain"]["ny"]
    nz = state["config"]["domain"]["nz"]

    # x-normal
    if i == 0:
        U[i+1, j+1, k+1] = 0.0
        V[i+1, j+1, k+1] = V[i+2, j+1, k+1]
        W[i+1, j+1, k+1] = W[i+2, j+1, k+1]
    elif i == nx - 1:
        U[i+1, j+1, k+1] = 0.0
        V[i+1, j+1, k+1] = V[i, j+1, k+1]
        W[i+1, j+1, k+1] = W[i, j+1, k+1]

    # y-normal
    elif j == 0:
        V[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+2, k+1]
        W[i+1, j+1, k+1] = W[i+1, j+2, k+1]
    elif j == ny - 1:
        V[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j, k+1]
        W[i+1, j+1, k+1] = W[i+1, j, k+1]

    # z-normal
    elif k == 0:
        W[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+1, k+2]
        V[i+1, j+1, k+1] = V[i+1, j+1, k+2]
    elif k == nz - 1:
        W[i+1, j+1, k+1] = 0.0
        U[i+1, j+1, k+1] = U[i+1, j+1, k]
        V[i+1, j+1, k+1] = V[i+1, j+1, k]
