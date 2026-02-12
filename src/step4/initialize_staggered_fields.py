# src/step4/initialize_staggered_fields.py

import numpy as np


def initialize_staggered_fields(state):
    """
    Initialize the extended staggered fields (U_ext, V_ext, W_ext, P_ext)
    using the initial conditions provided in the Step 3 configuration.

    This function assumes that allocate_extended_fields() has already created
    the *_ext arrays with a one-cell halo in each direction.

    Responsibilities:
    - Fill interior velocity components with the configured initial velocity.
    - Fill interior pressure with the configured initial pressure.
    - Leave halo cells unchanged (they will be set by BCs later).
    - Do NOT apply any physics or gradients here — this is a pure initializer.

    Parameters
    ----------
    state : dict-like
        Must contain:
            state["config"]["initial_conditions"]
            state["U_ext"], state["V_ext"], state["W_ext"], state["P_ext"]

    Returns
    -------
    state : dict-like
        Updated with initialized interior fields.
    """

    ic = state["config"].get("initial_conditions", {})

    # Defaults if not provided
    p0 = ic.get("initial_pressure", 0.0)
    u0, v0, w0 = ic.get("initial_velocity", [0.0, 0.0, 0.0])

    # Initialize pressure
    if "P_ext" in state:
        P = state["P_ext"]
        P[1:-1, 1:-1, 1:-1] = p0

    # Initialize velocity components
    if "U_ext" in state:
        U = state["U_ext"]
        U[1:-1, 1:-1, 1:-1] = u0

    if "V_ext" in state:
        V = state["V_ext"]
        V[1:-1, 1:-1, 1:-1] = v0

    if "W_ext" in state:
        W = state["W_ext"]
        W[1:-1, 1:-1, 1:-1] = w0

    return state
