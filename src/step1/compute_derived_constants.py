# src/step1/compute_derived_constants.py

from __future__ import annotations
from typing import Dict, Any

def compute_derived_constants(
    grid_config: Dict[str, Any],
    fluid_properties: Dict[str, Any],
    simulation_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes numerical and physical constants from the validated configuration.
    
    Constitutional Role: Mathematical Translator.
    Requirement: Sensitivity Gate (Loud Value Injection).
    
    Returns:
        Dict[str, float]: Flat dictionary of constants for SolverState.
    """
    # 1. Physical Properties (Schema Mapping)
    rho = float(fluid_properties["density"])
    mu = float(fluid_properties["viscosity"])
    
    # 2. Temporal Step
    dt = float(simulation_parameters["time_step"])
    

    # 4. Constitutional Integrity Check: Non-Physical Value Prevention
    # Ensures we don't pass '0.0' or negative values into physics kernels.
    for label, val in zip(["rho", "dt"], [rho, dt]):
        if val <= 0:
            raise ValueError(f"Non-physical constant detected: {label} = {val}. Must be > 0.")
    
    if mu < 0:
         raise ValueError(f"Non-physical viscosity detected: mu = {mu}. Must be >= 0.")

    return {
        "rho": rho,
        "mu": mu,
        "dt": dt,
    }