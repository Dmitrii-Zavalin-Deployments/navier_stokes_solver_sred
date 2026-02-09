# src/step2/precompute_constants.py
from __future__ import annotations
from typing import Any, Dict


def precompute_constants(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute physical and geometric constants for Step‑2 numerical operators.

    Pure function:
    - Does NOT mutate the input state.
    - Always recomputes constants from Step‑1 schema.
    """

    cfg = state["config"]
    grid = state["grid"]

    # Step‑1 schema fields
    fluid = cfg["fluid"]

    rho = float(fluid["density"])
    mu = float(fluid["viscosity"])

    # ---------------------------------------------------------
    # FIX: dt must come from Step‑1 constants, not config
    # ---------------------------------------------------------
    dt = float(state["constants"]["dt"])
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

    return {
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
