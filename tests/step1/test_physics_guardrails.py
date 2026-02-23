# tests/step1/test_physics_guardrails.py

import pytest
from src.step1.compute_derived_constants import compute_derived_constants

def test_non_physical_constants_check():
    """Triggers lines 35-36: Zero or negative physical constants."""
    grid = {"dx": 0.1, "dy": 0.1, "dz": 0.1}
    fluid = {"density": 0.0, "viscosity": 0.01}  # Density is 0
    params = {"time_step": 0.01}
    
    with pytest.raises(ValueError, match="Non-physical constant detected: rho = 0.0"):
        compute_derived_constants(grid, fluid, params)

def test_negative_viscosity_check():
    """Triggers lines 38-39: Negative viscosity check."""
    grid = {"dx": 0.1, "dy": 0.1, "dz": 0.1}
    fluid = {"density": 1.0, "viscosity": -0.5}  # Viscosity is negative
    params = {"time_step": 0.01}
    
    with pytest.raises(ValueError, match="Non-physical viscosity detected: mu = -0.5"):
        compute_derived_constants(grid, fluid, params)