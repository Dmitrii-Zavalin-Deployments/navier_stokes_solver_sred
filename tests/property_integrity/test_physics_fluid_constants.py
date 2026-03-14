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
    fluid = getattr(state, "_fluid_properties", None)
    
    assert fluid is not None, f"{stage_name}: Fluid properties missing"
    # Use the actual attribute name from __slots__
    assert hasattr(fluid, "_density"), f"{stage_name}: Density (_density) missing"
    assert fluid._density > 0, f"{stage_name}: Non-physical density: {fluid._density}"

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_viscosity_persistence_and_validity(stage_name, factory):
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    fluid = getattr(state, "_fluid_properties", None)
    
    assert fluid is not None, f"{stage_name}: Fluid properties missing"
    # Use the actual attribute name from __slots__
    assert hasattr(fluid, "_viscosity"), f"{stage_name}: Viscosity (_viscosity) missing"
    assert fluid._viscosity > 0, f"{stage_name}: Non-physical viscosity: {fluid._viscosity}"

def test_velocity_correction_scaling_consistency():
    nx, ny, nz = 4, 4, 4
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    dt = state._simulation_parameters.time_step
    rho = state._fluid_properties._density
    
    correction_factor = dt / rho
    assert np.isfinite(correction_factor), "Step 3: Velocity correction factor invalid"

def test_diffusion_stability_coefficient():
    nx, ny, nz = 4, 4, 4
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    nu = state._fluid_properties._viscosity
    dt = state._simulation_parameters.time_step
    dx = state._grid.dx
    
    stability_factor = (nu * dt) / (dx**2)
    assert np.isfinite(stability_factor), "Step 3: Diffusion stability factor invalid"