# src/step2/precompute_constants.py
from __future__ import annotations

from typing import Any, Dict


def precompute_constants(state: Any) -> Dict[str, float]:
    """
    Expose / (lightly) recompute constants for fast operator access.

    This function assumes Step 1 has already validated physical parameters and
    grid extents. It either returns the existing constants or constructs them
    from the state.

    Parameters
    ----------
    state : Any
        SimulationState-like object with:
        - state["Config"]["fluid_properties"]
        - state["Config"]["simulation_parameters"]
        - state["Grid"] (dx, dy, dz)

    Returns
    -------
    Dict[str, float]
        Dictionary with rho, mu, dt, dx, dy, dz, inv_dx, inv_dy, inv_dz,
        inv_dx2, inv_dy2, inv_dz2.
    """

    # ------------------------------------------------------------
    # If Constants already exists AND is a dict, reuse it.
    # (DummyState sets Constants=None, so we must check type.)
    # ------------------------------------------------------------
    if "Constants" in state and isinstance(state["Constants"], dict):
        return state["Constants"]

    # ------------------------------------------------------------
    # Otherwise compute constants from Config + Grid
    # ------------------------------------------------------------
    cfg = state["Config"]
    grid = state["Grid"]

    fluid = cfg["fluid_properties"]
    sim = cfg["simulation_parameters"]

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

    # Store back into state (dict-style and attribute-style for compatibility)
    state["Constants"] = constants
    state.Constants = constants

    return constants
