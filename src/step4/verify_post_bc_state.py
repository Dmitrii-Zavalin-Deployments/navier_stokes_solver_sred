# src/step4/verify_post_bc_state.py

import numpy as np


def verify_post_bc_state(state):
    """
    Perform global integrity checks after all Step 4 operations.

    Responsibilities:
    - Ensure no NaNs exist in extended fields.
    - Compute a simple divergence measure on the interior.
    - Mark whether divergence is acceptable (zero or near-zero).
    - Store results under state["Verification"].
    - Do NOT modify the fields; this is a pure verification step.

    This is intentionally lightweight: Step 4 is not responsible for
    enforcing incompressibility, only for detecting issues early.
    """

    verification = {}

    # ------------------------------------------------------------
    # 1. NaN check
    # ------------------------------------------------------------
    nan_found = False
    for name in ("U_ext", "V_ext", "W_ext", "P_ext"):
        if name in state:
            if np.isnan(state[name]).any():
                nan_found = True
                break

    verification["nan_found"] = nan_found

    # ------------------------------------------------------------
    # 2. Divergence check (simple finite-difference estimate)
    # ------------------------------------------------------------
    if all(k in state for k in ("U_ext", "V_ext", "W_ext")):
        U = state["U_ext"]
        V = state["V_ext"]
        W = state["W_ext"]

        # Compute divergence on interior cells only
        div = (
            (U[2:, 1:-1, 1:-1] - U[:-2, 1:-1, 1:-1]) +
            (V[1:-1, 2:, 1:-1] - V[1:-1, :-2, 1:-1]) +
            (W[1:-1, 1:-1, 2:] - W[1:-1, 1:-1, :-2])
        )

        max_div = float(np.max(np.abs(div)))
        verification["max_divergence"] = max_div
        verification["divergence_ok"] = (max_div == 0.0)

    else:
        # Missing fields → cannot compute divergence
        verification["max_divergence"] = None
        verification["divergence_ok"] = False

    # ------------------------------------------------------------
    # 3. Store verification results
    # ------------------------------------------------------------
    state["Verification"] = verification
    return state
