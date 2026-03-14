# tests/property_integrity/test_physics_fluid_constants.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

PHYSICS_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_density_persistence_and_validity(stage_name, factory):
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    # Access via the correct internal container (update if _fluid_properties is a property)
    fluid = getattr(state, "_fluid_properties", None)
    
    assert fluid is not None, f"{stage_name}: Fluid properties missing"
    assert hasattr(fluid, "rho"), f"{stage_name}: Density (rho) missing"
    assert fluid.rho > 0, f"{stage_name}: Non-physical density: {fluid.rho}"

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_viscosity_persistence_and_validity(stage_name, factory):
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    fluid = getattr(state, "_fluid_properties", None)
    
    assert fluid is not None, f"{stage_name}: Fluid properties missing"
    assert hasattr(fluid, "nu"), f"{stage_name}: Viscosity (nu) missing"
    assert fluid.nu > 0, f"{stage_name}: Non-physical viscosity: {fluid.nu}"

def test_velocity_correction_scaling_consistency():
    state = make_step3_output_dummy(nx=4, ny=4, nz=4)
    
    dt = state._simulation_parameters.time_step
    rho = state._fluid_properties.rho
    
    correction_factor = dt / rho
    assert np.isfinite(correction_factor), "Step 3: Velocity correction factor invalid"

def test_diffusion_stability_coefficient():
    """
    Theory: Ensure the diffusion coefficient (nu * dt / dx^2) is calculable.
    """
    # Define local variables to satisfy the function scope
    nx, ny, nz = 4, 4, 4
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    nu = state._fluid_properties.nu
    dt = state._simulation_parameters.time_step
    dx = state._grid.dx
    
    stability_factor = (nu * dt) / (dx**2)
    assert np.isfinite(stability_factor), "Step 3: Diffusion stability factor is invalid"