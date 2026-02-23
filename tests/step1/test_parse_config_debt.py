# tests/step1/test_parse_config_debt.py

import pytest
from src.step1.parse_config import parse_config

def test_line_10_type_error():
    """Trigger: Section must be a dictionary."""
    bad_data = {"grid": "not-a-dict"}
    with pytest.raises(TypeError, match="must be a dictionary"):
        parse_config(bad_data)

def test_line_26_missing_grid_key():
    """Trigger: Missing mandatory grid key (e.g., x_min)."""
    bad_data = {"grid": {"nx": 10, "ny": 10, "nz": 10}} # missing x_min, etc.
    with pytest.raises(KeyError, match="Grid Configuration Error"):
        parse_config(bad_data)

def test_line_31_missing_fluid_keys():
    """Trigger: Missing density or viscosity."""
    bad_data = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0} # missing viscosity
    }
    with pytest.raises(KeyError, match="requires 'density' and 'viscosity'"):
        parse_config(bad_data)

def test_line_36_missing_time_step():
    """Trigger: Missing time_step in simulation_parameters."""
    bad_data = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {} # missing time_step
    }
    with pytest.raises(KeyError, match="requires 'time_step'"):
        parse_config(bad_data)

def test_line_45_missing_force_vector():
    """Trigger: external_forces exists but force_vector is missing."""
    bad_data = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.1},
        "external_forces": {} # missing force_vector
    }
    with pytest.raises(KeyError, match="'force_vector' is missing"):
        parse_config(bad_data)

def test_line_47_invalid_force_vector_shape():
    """Trigger: force_vector is not length 3."""
    bad_data = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.1},
        "external_forces": {"force_vector": [0, 0]} # len is 2
    }
    with pytest.raises(ValueError, match=r"must be \[x, y, z\]"):
        parse_config(bad_data)

def test_line_52_non_finite_forces():
    """Trigger: NaN or Infinity in force vector."""
    bad_data = {
        "grid": {"nx":2,"ny":2,"nz":2,"x_min":0,"x_max":1,"y_min":0,"y_max":1,"z_min":0,"z_max":1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.1},
        "simulation_parameters": {"time_step": 0.1},
        "external_forces": {"force_vector": [0, float('nan'), 0]}
    }
    with pytest.raises(ValueError, match="Non-finite values detected"):
        parse_config(bad_data)