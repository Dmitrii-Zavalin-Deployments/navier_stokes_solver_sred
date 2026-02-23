# src/step1/compute_derived_constants.py

from __future__ import annotations
from typing import Dict, Any

def compute_derived_constants(
    grid_config: Dict[str, Any],
    fluid_properties: Dict[str, Any],
    simulation_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute physical and numerical constants for Step 1.
    
    This maps Schema names (density, viscosity) to Solver shorthand (rho, mu)
    and captures spatial/temporal steps (dx, dt) for the physics kernels.
    """
    # 1. Physical Properties
    rho = float(fluid_properties["density"])
    mu = float(fluid_properties["viscosity"])
    
    # 2. Temporal Step
    dt = float(simulation_parameters["time_step"])
    
    # 3. Spatial Steps (Inherited from initialize_grid)
    dx = float(grid_config["dx"])
    dy = float(grid_config["dy"])
    dz = float(grid_config["dz"])

    # 4. Return flattened constants dictionary
    return {
        "rho": rho,
        "mu": mu,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }