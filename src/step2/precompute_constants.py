# file: step2/precompute_constants.py
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
        - Config (with fluid_properties, simulation_parameters)
        - Grid (with dx, dy, dz)

    Returns
    -------
    Dict[str, float]
        Dictionary with rho, mu, dt, dx, dy, dz, inv_dx, inv_dy, inv_dz,
        inv_dx2, inv_dy2, inv_dz2.
    """
    # If Step 1 already computed Constants, just return them.
    if hasattr(state, "Constants"):
        const = dict(state.Constants)
        return const

    # Fallback: derive from Config and Grid.
    fluid = state.Config["fluid_properties"]
    sim = state.Config["simulation_parameters"]
    grid = state.Grid

    rho = float(fluid["density"])
    mu = float(fluid["viscosity"])
    dt = float(sim["time_step"])

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

    # Optionally attach back to state for convenience.
    state.Constants = constants
    return constants
