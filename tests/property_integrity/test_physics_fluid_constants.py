# tests/property_integrity/test_physics_fluid_constants.py

import numpy as np
import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_step5_output_dummy import make_step5_output_dummy

# Importing StencilBlock to perform type checking for adaptive access
from src.common.stencil_block import StencilBlock

PHYSICS_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Step 5", make_step5_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

def get_fluid_param(obj, param_name):
    """
    Helper to extract physics constants regardless of the container type.
    StencilBlock uses flat slots (_rho, _mu), while SolverState uses 
    a manager (_fluid_properties).
    """
    if isinstance(obj, StencilBlock):
        # Mapping generic names to StencilBlock slots
        mapping = {"density": "_rho", "viscosity": "_mu"}
        return getattr(obj, mapping[param_name], None)
    
    # Otherwise assume SolverState/Manager structure
    fluid = getattr(obj, "_fluid_properties", None)
    if fluid:
        attr = f"_{param_name}"
        return getattr(fluid, attr, None)
    return None

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_density_persistence_and_validity(stage_name, factory):
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    rho = get_fluid_param(state, "density")
    
    assert rho is not None, f"{stage_name}: Density value missing or unreachable"
    assert rho > 0, f"{stage_name}: Non-physical density: {rho}"

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_viscosity_persistence_and_validity(stage_name, factory):
    nx, ny, nz = 4, 4, 4
    state = factory(nx=nx, ny=ny, nz=nz)
    
    mu = get_fluid_param(state, "viscosity")
    
    assert mu is not None, f"{stage_name}: Viscosity value missing or unreachable"
    assert mu > 0, f"{stage_name}: Non-physical viscosity: {mu}"

def test_velocity_correction_scaling_consistency():
    """
    Physics Check: dt/rho must be finite. 
    Step 3 StencilBlock carries dt and rho as direct attributes.
    """
    nx, ny, nz = 4, 4, 4
    block = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Accessing StencilBlock slots directly
    dt = block._dt
    rho = block._rho
    
    correction_factor = dt / rho
    assert np.isfinite(correction_factor), "Step 3: Velocity correction factor invalid"

def test_diffusion_stability_coefficient():
    """
    Physics Check: (nu * dt) / dx^2 must be finite.
    Step 3 StencilBlock carries nu (mu), dt, and dx directly.
    """
    nx, ny, nz = 4, 4, 4
    block = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    
    nu = block._mu
    dt = block._dt
    dx = block._dx
    
    stability_factor = (nu * dt) / (dx**2)
    assert np.isfinite(stability_factor), "Step 3: Diffusion stability factor invalid"