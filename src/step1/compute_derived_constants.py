# src/step1/compute_derived_constants.py

from __future__ import annotations
from src.solver_input import FluidInput, SimParamsInput
from typing import Dict, Any

def compute_derived_constants(
    grid_config: Dict[str, Any],
    fluid_properties: FluidInput,
    simulation_parameters: SimParamsInput,
) -> Dict[str, Any]:
    """
    Computes numerical and physical constants from the validated configuration.
    
    Constitutional Role: Mathematical Translator.
    Requirement: Sensitivity Gate (Loud Value Injection).
    
    Returns:
        Dict[str, float]: Flat dictionary of constants for SolverState.
    """
    # 1. Physical Properties (Schema Mapping)
    rho = float(fluid_properties.density)
    mu = float(fluid_properties.viscosity)
    
    # 2. Temporal Step
    dt = float(simulation_parameters.time_step)
    
    # 3. Spatial Steps
    dx = float(grid_config["dx"])
    dy = float(grid_config["dy"])
    dz = float(grid_config["dz"])

    # 4. Constitutional Integrity Check: Non-Physical Value Prevention
    # Ensures we don't pass '0.0' or negative values into physics kernels.
    for label, val in zip(["rho", "dt", "dx", "dy", "dz"], [rho, dt, dx, dy, dz]):
        if val <= 0:
            raise ValueError(f"Non-physical constant detected: {label} = {val}. Must be > 0.")
    
    if mu < 0:
         raise ValueError(f"Non-physical viscosity detected: mu = {mu}. Must be >= 0.")

    return {
        "rho": rho,
        "mu": mu,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }