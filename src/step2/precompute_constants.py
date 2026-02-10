# src/step2/precompute_constants.py
from __future__ import annotations
from typing import Any, Dict


def precompute_constants(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute physical and geometric constants for Step‑2 numerical operators.

    Contract:
      • Step 2 does NOT override or recompute constants if Step 1 already
        provided a non‑empty constants dictionary.
      • Otherwise, compute constants from Step‑1 schema fields.
      • Pure function: does NOT mutate the input state.
    """

    # ---------------------------------------------------------
    # 0. Passthrough: Step‑1 constants are authoritative
    # ---------------------------------------------------------
    if (
        "constants" in state
        and isinstance(state["constants"], dict)
        and state["constants"]  # non‑empty
    ):
        return state["constants"]

    # ---------------------------------------------------------
    # 1. Compute constants from Step‑1 schema
    # ---------------------------------------------------------
    cfg = state["config"]
    grid = state["grid"]

    fluid = cfg["fluid"]
    rho = float(fluid["density"])
    mu = float(fluid["viscosity"])

    dx = float(grid["dx"])
    dy = float(grid["dy"])
    dz = float(grid["dz"])

    # ---------------------------------------------------------
    # dt: prefer Step‑1 constants if present, otherwise fall back
    # to config["simulation"]["dt"] for tests/minimal states.
    # ---------------------------------------------------------
    if "constants" in state and "dt" in state["constants"]:
        dt = float(state["constants"]["dt"])
    else:
        sim = cfg.get("simulation", {})
        dt = float(sim["dt"])

    if dt <= 0:
        raise ValueError("dt must be positive")

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
