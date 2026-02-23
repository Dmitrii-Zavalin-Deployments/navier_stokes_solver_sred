# src/step1/parse_config.py

from __future__ import annotations
from typing import Any, Dict
import math

def _ensure_dict(name: str, value: Any) -> Dict[str, Any]:
    """
    Ensure a value is a dictionary. Used for validating sections of the input.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary, got {type(value).__name__}")
    return dict(value)

def parse_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the solver input dictionary into the Step 1 config structure.
    
    This ensures all 'Loud Values' from the Schema are captured to satisfy
    the Phase E: Data Completeness Audit.
    """

    # 1. Grid Parameters (Crucial for Delta calculation)
    grid = _ensure_dict("grid", data.get("grid", {}))
    # Required keys based on schema
    required_grid = ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    for key in required_grid:
        if key not in grid:
            raise KeyError(f"Grid configuration missing required key: '{key}'")

    # 2. Fluid Properties (The Physics Constants)
    fluid = _ensure_dict("fluid_properties", data.get("fluid_properties", {}))
    if "density" not in fluid or "viscosity" not in fluid:
        raise KeyError("fluid_properties must contain 'density' and 'viscosity'")

    # 3. Simulation Parameters (Time Stepping)
    sim = _ensure_dict("simulation_parameters", data.get("simulation_parameters", {}))
    if "time_step" not in sim:
        raise KeyError("simulation_parameters must contain 'time_step'")
    
    dt = sim["time_step"]

    # 4. Initial Conditions (Stored for apply_initial_conditions.py)
    ic = _ensure_dict("initial_conditions", data.get("initial_conditions", {}))

    # 5. External Forces (Handled with your existing validation logic)
    forces_raw = data.get("external_forces", {})
    forces = _ensure_dict("external_forces", forces_raw)
    fv = forces.get("force_vector")
    
    if fv is None:
        raise KeyError("external_forces must contain 'force_vector'")
    if not isinstance(fv, (list, tuple)) or len(fv) != 3:
        raise ValueError("external_forces['force_vector'] must be a length‑3 vector")
    
    # Normalize and validate force vector
    force_vector = [float(x) for x in fv if math.isfinite(x)]
    if len(force_vector) != 3:
        raise ValueError("force_vector entries must be finite numbers")

    # 6. Assembly of the Full Step 1 Config Object
    # We return a nested dictionary that mirrors the Schema exactly.
    return {
        "grid": grid,
        "fluid_properties": fluid,
        "simulation_parameters": sim,
        "initial_conditions": ic,
        "external_forces": {
            "force_vector": force_vector,
            **{k: v for k, v in forces.items() if k != "force_vector"}
        },
        "dt": dt,  # Keep dt flat as a convenience constant
        "mask": data.get("mask", []) # Pass through for map_geometry_mask.py
    }