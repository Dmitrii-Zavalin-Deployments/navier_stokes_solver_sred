# tests/step1/test_physics_and_config.py

import pytest
import numpy as np
from src.step1.parse_config import parse_config
from src.step1.compute_derived_constants import compute_derived_constants
from src.step1.validate_physical_constraints import validate_physical_constraints
from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def valid_state():
    """Provides a fresh SolverState (Section 5 Compliance)."""
    json_input = solver_input_schema_dummy()
    return orchestrate_step1_state(json_input)

# --- CONFIG PARSER & DEBT TRIGGERS ---

def test_parse_config_structure_and_keys():
    """Triggers TypeErrors and KeyErrors in the config parser."""
    # 1. Section must be a dict
    with pytest.raises(TypeError, match="must be a dictionary"):
        parse_config({"grid": "not-a-dict"})

    # 2. Missing grid keys
    with pytest.raises(KeyError, match="Grid Configuration Error"):
        parse_config({"grid": {"nx": 10}})

    # 3. Missing fluid properties
    with pytest.raises(KeyError, match="requires 'density' and 'viscosity'"):
        parse_config({
            "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
            "fluid_properties": {"density": 1.0}
        })

def test_parse_config_external_forces_debt():
    """Triggers force vector shape and finiteness checks."""
    base = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.1}
    }
    
    # 1. Missing force_vector
    bad_force = {**base, "external_forces": {}}
    with pytest.raises(KeyError, match="'force_vector' is missing"):
        parse_config(bad_force)

    # 2. Invalid shape (using raw string fix for brackets)
    bad_shape = {**base, "external_forces": {"force_vector": [0, 0]}}
    with pytest.raises(ValueError, match=r"must be \[x, y, z\]"):
        parse_config(bad_shape)

    # 3. Non-finite values
    bad_math = {**base, "external_forces": {"force_vector": [0, float('nan'), 0]}}
    with pytest.raises(ValueError, match="Non-finite values detected"):
        parse_config(bad_math)

# --- PHYSICAL CONSTANTS GUARDRAILS ---

def test_derived_constants_physics_checks():
    """Triggers non-physical rho and mu checks in compute_derived_constants."""
    grid = {"dx": 0.1, "dy": 0.1, "dz": 0.1}
    params = {"time_step": 0.01}

    # Density <= 0
    with pytest.raises(ValueError, match="Non-physical constant detected: rho = 0.0"):
        compute_derived_constants(grid, {"density": 0.0, "viscosity": 0.01}, params)

    # Viscosity < 0
    with pytest.raises(ValueError, match="Non-physical viscosity detected: mu = -0.5"):
        compute_derived_constants(grid, {"density": 1.0, "viscosity": -0.5}, params)

# --- SOLVER STATE CONSTRAINT VALIDATION ---

def test_validate_constraints_grid_and_time(valid_state):
    """Verifies domain inversion, finiteness, and time-step positivity."""
    # 1. Domain Inversion
    valid_state.grid["x_max"] = valid_state.grid["x_min"]
    with pytest.raises(ValueError, match="Domain Inversion"):
        validate_physical_constraints(valid_state)

    # 2. Time step positivity
    valid_state.constants["dt"] = -0.001
    with pytest.raises(ValueError, match="(?i)time step"):
        validate_physical_constraints(valid_state)

def test_validate_constraints_topology_and_fields(valid_state):
    """Verifies mask compliance and numerical stability (NaN/Inf)."""
    # 1. Mask Value Compliance (Forbidden Topology)
    valid_state.mask[0, 0, 0] = 42 
    with pytest.raises(ValueError, match="Forbidden Topology"):
        validate_physical_constraints(valid_state)
    
    # RESET MASK: Fix the topology so the next check can run against fields
    valid_state.mask[0, 0, 0] = 1

    # 2. Field Finiteness
    valid_state.fields["V"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="(?i)finite"):
        validate_physical_constraints(valid_state)