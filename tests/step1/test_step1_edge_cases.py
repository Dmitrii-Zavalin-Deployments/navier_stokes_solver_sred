# tests/step1/test_step1_edge_cases.py

import pytest
import numpy as np
from src.step1.parse_config import parse_config
from src.step1.apply_initial_conditions import apply_initial_conditions
from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.allocate_fields import allocate_fields
from src.solver_state import SolverState

def test_parse_config_defaults():
    """Triggers fallback logic in parse_config.py for missing optional keys."""
    # Provide only the bare minimum required by the dictionary parser
    minimal_config = {
        "grid": {"nx": 10, "ny": 10, "nz": 10, "x_max": 1.0, "y_max": 1.0, "z_max": 1.0},
        "fluid_properties": {"density": 1.0, "viscosity": 0.01}
    }
    # This should trigger lines like 'if "time_step" not in config'
    parsed = parse_config(minimal_config)
    assert "simulation_parameters" in parsed
    # Verify defaults are set (e.g., if line 31 handles default output_interval)
    assert parsed["simulation_parameters"]["output_interval"] >= 1

def test_apply_initial_conditions_scalars():
    """Triggers logic for scalar initialization vs array initialization."""
    nx, ny, nz = 2, 2, 2
    state = {
        "fields": {
            "U": np.zeros((nx+1, ny, nz)),
            "V": np.zeros((nx, ny+1, nz)),
            "W": np.zeros((nx, ny, nz+1)),
            "P": np.zeros((nx, ny, nz))
        },
        "initial_conditions": {
            "velocity": [1.0, 0.0, 0.0], # Scalar-like list
            "pressure": 0.5
        }
    }
    # This triggers the 'Miss' lines that broadcast scalars to arrays (lines 24-32)
    apply_initial_conditions(state)
    assert np.all(state["fields"]["U"] == 1.0)
    assert np.all(state["fields"]["P"] == 0.5)

def test_compute_derived_constants_logic():
    """Triggers physics constant derivations (CFL, etc.)."""
    state = {
        "grid": {"dx": 0.1, "dy": 0.1, "dz": 0.1},
        "fluid_properties": {"density": 1000.0, "viscosity": 0.001},
        "simulation_parameters": {"time_step": 0.01}
    }
    # Triggers lines 36/39 for calculating Reynolds-related constants or dt checks
    compute_derived_constants(state)
    assert "constants" in state
    assert state["constants"]["rho"] == 1000.0

def test_allocate_fields_zero_dims():
    """Triggers error handling in allocate_fields (Line 24)."""
    grid_invalid = {"nx": 0, "ny": 2, "nz": 2}
    # Testing the 'debt' line 24 which likely handles invalid dimensions
    with pytest.raises((ValueError, ZeroDivisionError)):
        allocate_fields({"grid": grid_invalid})