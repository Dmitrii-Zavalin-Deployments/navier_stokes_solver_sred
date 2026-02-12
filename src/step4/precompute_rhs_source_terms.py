# src/step4/precompute_rhs_source_terms.py

import numpy as np


def precompute_rhs_source_terms(state):
    """
    Precompute RHS source terms for the momentum equations.

    Step 4 does NOT compute the full Navier–Stokes RHS. Instead, it prepares
    the source-term arrays (e.g., gravity contributions) so that later steps
    can assemble the full RHS cleanly.

    Responsibilities:
    - Allocate RHS arrays (RHS_U, RHS_V, RHS_W) matching the extended grid.
    - Apply gravity only to fluid cells (mask == 1).
    - Leave solid (0) and boundary-fluid (-1) cells as zero.
    - Ensure no NaNs appear in the output.
    - Store results under state["RHS"].

    Returns
    -------
    state : dict-like
        Updated with RHS source-term arrays.
    """

    # Extract gravity vector (default = zero)
    gravity = state["config"].get("forces", {}).get("gravity", [0.0, 0.0, 0.0])
    gx, gy, gz = gravity

    # Prepare RHS container
    state["RHS"] = {}

    # We require mask_ext to know which cells are fluid
    mask_ext = state.get("mask_ext")
    if mask_ext is None:
        # If no extended mask exists, allocate zero RHS and return
        for comp in ("U", "V", "W"):
            ext_name = f"{comp}_ext"
            if ext_name in state:
                shape = state[ext_name].shape
                state["RHS"][f"RHS_{comp}"] = np.zeros(shape, dtype=float)
        return state

    # Allocate RHS arrays matching extended velocity fields
    for comp in ("U", "V", "W"):
        ext_name = f"{comp}_ext"
        if ext_name not in state:
            continue

        shape = state[ext_name].shape
        rhs = np.zeros(shape, dtype=float)

        # Apply gravity only to fluid cells (mask == 1)
        if comp == "U":
            rhs[mask_ext == 1] = gx
        elif comp == "V":
            rhs[mask_ext == 1] = gy
        elif comp == "W":
            rhs[mask_ext == 1] = gz

        state["RHS"][f"RHS_{comp}"] = rhs

    return state
