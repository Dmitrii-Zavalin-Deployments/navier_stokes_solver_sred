# src/step1/parse_config.py

from __future__ import annotations
from typing import Any, Dict
import math

def _ensure_dict(name: str, value: Any) -> Dict[str, Any]:
    """Helper to enforce dictionary types for schema sections."""
    if not isinstance(value, dict):
        raise TypeError(f"Schema violation: Section '{name}' must be a dictionary, got {type(value).__name__}")
    return dict(value)

def parse_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and validates the solver configuration dictionary.
    
    Constitutional Role: Contractual Gatekeeper.
    Compliance: Phase F (Sensitivity Gate & Data Completeness Audit).
    """

    if not isinstance(data, dict): data = data.__dict__
    # 1. Grid Parameters (Spatial Requirements)
    grid = _ensure_dict("grid", data.get("grid", {}))
    required_grid = ["nx", "ny", "nz", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    for key in required_grid:
        if key not in grid:
            raise KeyError(f"Grid Configuration Error: Missing mandatory key '{key}'")

    # 2. Fluid Properties (Physical Constants)
    fluid = _ensure_dict("fluid_properties", data.get("fluid_properties", {}))
    if "density" not in fluid or "viscosity" not in fluid:
        raise KeyError("Physics Error: 'fluid_properties' requires 'density' and 'viscosity'")

    # 3. Simulation Parameters (Temporal Control)
    sim = _ensure_dict("simulation_parameters", data.get("simulation_parameters", {}))
    if "time_step" not in sim:
        raise KeyError("Simulation Error: 'simulation_parameters' requires 'time_step'")
    
    dt = float(sim["time_step"])

    # 4. External Forces (Vector Ingest)
    forces = _ensure_dict("external_forces", data.get("external_forces", {}))
    fv = forces.get("force_vector")
    
    if fv is None:
        raise KeyError("External Forces Error: 'force_vector' is missing")
    if not isinstance(fv, (list, tuple)) or len(fv) != 3:
        raise ValueError(f"Vector Error: 'force_vector' must be [x, y, z], got {fv}")
    
    # Cast to float64 to prevent precision drift
    force_vector = [float(x) for x in fv]
    if not all(math.isfinite(x) for x in force_vector):
        raise ValueError("Non-finite values detected in force_vector.")

    # 5. Mirror Assembly
    # Maintains absolute symmetry with solver_input_schema.json
    return {
        "grid": grid,
        "fluid_properties": fluid,
        "simulation_parameters": sim,
        "initial_conditions": _ensure_dict("initial_conditions", data.get("initial_conditions", {})),
        "external_forces": {
            "force_vector": force_vector,
            **{k: v for k, v in forces.items() if k != "force_vector"}
        },
        "dt": dt,
        "mask": data.get("mask", [])
    }