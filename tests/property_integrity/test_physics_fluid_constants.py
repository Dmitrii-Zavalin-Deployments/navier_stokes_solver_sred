# tests/property_integrity/test_physics_fluid_constants.py

import pytest
import numpy as np
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

# Density is required starting from Step 3 (Pressure Projection)
PHYSICS_ACTIVE_STAGES = [
    ("Step 3", make_step3_output_dummy),
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy),
]

@pytest.mark.parametrize("stage_name, factory", PHYSICS_ACTIVE_STAGES)
def test_density_persistence_and_validity(stage_name, factory):
    """
    Physics: Verify that density (rho) is present and strictly positive 
    from Step 3 onwards to support the Pressure-Velocity correction logic.
    """
    state = factory()
    
    # 1. Existence Check
    assert "density" in state.fluid_properties, f"{stage_name}: Density missing from fluid_properties"
    
    # 2. Scale Guard: Density must be positive to prevent division by zero in correction
    rho = state.fluid_properties["density"]
    assert rho > 0, f"{stage_name}: Non-physical density detected ({rho})"
    
    # 3. Type Check
    assert isinstance(rho, (float, int, np.float64)), f"{stage_name}: Density must be a numeric scalar"

def test_velocity_correction_scaling_consistency():
    """
    Theory: Ensure that the relationship dt/rho is calculable.
    This is the coefficient used in Step 3: u_new = u_star - (dt/rho) * grad(P)
    """
    state = make_step3_output_dummy()
    
    dt = state.constants.get("dt", 0.001)
    rho = state.fluid_properties["density"]
    
    # The 'correction_factor' must be a valid, finite float
    correction_factor = dt / rho
    assert np.isfinite(correction_factor), "Step 3: Velocity correction factor (dt/rho) is invalid"