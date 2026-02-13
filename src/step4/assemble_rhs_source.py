# src/step4/assemble_rhs_source.py

def assemble_rhs_source(state):
    """
    Convert the internal RHS structure into the schema-compliant
    rhs_source block required by Step 4 output.

    Expected input (after precompute_rhs_source_terms):
        state["rhs_source"] = {
            "RHS_U": [...],
            "RHS_V": [...],
            "RHS_W": [...],
        }

    Expected output:
        state["rhs_source"] = {
            "fx_u": [...],
            "fy_v": [...],
            "fz_w": [...],
        }
    """

    rhs = state.get("rhs_source", {})

    fx_u = rhs.get("RHS_U", [])
    fy_v = rhs.get("RHS_V", [])
    fz_w = rhs.get("RHS_W", [])

    state["rhs_source"] = {
        "fx_u": fx_u,
        "fy_v": fy_v,
        "fz_w": fz_w,
    }

    return state
