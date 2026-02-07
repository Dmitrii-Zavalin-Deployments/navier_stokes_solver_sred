# src/step2/precompute_constants.py
from __future__ import annotations

from typing import Any, Dict


def precompute_constants(state: Any) -> Dict[str, float]:
    """
    Expose / (lightly) recompute constants for fast operator access.

    This function assumes Step 1 has already validated physical parameters and
    grid extents. It either returns the existing constants or constructs them
    from the state.
    """

    # ------------------------------------------------------------
    # If constants already exist AND are a dict, reuse them.
    # ------------------------------------------------------------
    if "constants" in state and isinstance(state["constants"], dict):
        return state["constants"]

    # ------------------------------------------------------------
    # Otherwise compute constants from config + grid
    # (using the REAL Step‑1 schema)
    # ------------------------------------------------------------
    cfg = state["config"]
    grid = state["grid"]

    # Step‑1 schema fields
    fluid = cfg["fluid"]
    sim = cfg["simulation"]

    rho = float(fluid["density"])
    mu = float(fluid["viscosity"])

    dt = float(sim["dt"])
    if dt <= 0:
        raise ValueError("dt must be positive")

    dx = float(grid["dx"])
    dy = float(grid["dy"])
    dz = float(grid["dz"])

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz

    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    inv_dz2 = inv_dz * inv_dz

    constants = {
        "rho": rho,
        "mu": mu,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "inv_dx": inv_dx,
        "inv_dy": inv_dy,
        "inv_dz": inv_dz,
        "inv_dx2": inv_dx2,
        "inv_dy2": inv_dy2,
        "inv_dz2": inv_dz2,
    }

    # Store back into state (schema‑correct)
    state["constants"] = constants

    return constants
