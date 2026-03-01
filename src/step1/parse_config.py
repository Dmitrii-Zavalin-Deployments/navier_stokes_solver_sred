# src/step1/parse_config.py

from __future__ import annotations
from typing import Any, Dict
from src.solver_input import SolverInput
import math

def parse_config(data: SolverInput) -> Dict[str, Any]:
    # Cast force vector to float64 for precision
    fv = data.external_forces.force_vector
    force_vector = [float(x) for x in fv]
    
    if not all(math.isfinite(x) for x in force_vector):
        raise ValueError("Non-finite values detected in force_vector.")

    return {
        "grid": data.grid,
        "fluid_properties": data.fluid_properties,
        "simulation_parameters": data.simulation_parameters,
        "initial_conditions": data.initial_conditions,
        "external_forces": {
            "force_vector": force_vector
        },
        "dt": float(data.simulation_parameters.time_step),
        "mask": data.mask
    }
