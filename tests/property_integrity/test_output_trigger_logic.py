# tests/property_integrity/test_output_trigger_logic.py

import pytest

from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

@pytest.mark.parametrize("stage_name, factory", [
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_output_interval_persistence(stage_name, factory):
    """
    Integrity: Verify simulation_parameters.output_interval is 
    persistent through the late-stage pipeline.
    """
    # Assuming standard factory signature with grid dimensions
    state = factory(nx=4, ny=4, nz=4)
    
    # Standardized access to config
    config = getattr(state, "config", None)
    
    assert config is not None, f"{stage_name}: Configuration object missing in state"
    assert hasattr(config, "output_interval"), f"{stage_name}: Property 'output_interval' missing"
    assert isinstance(config.output_interval, (int, float)), "Interval must be numeric"

def test_step5_write_trigger_logic():
    """
    Logic: Verify that the Step 5 'Snapshot' generation adheres to 
    the modulo interval defined in the configuration.
    """
    state = make_output_schema_dummy(nx=4, ny=4, nz=4)
    
    # Access state attributes directly
    iteration = state.iteration
    interval = state.config.output_interval
    diagnostics = state.health
    
    # Validation of diagnostics object
    assert diagnostics is not None, "Step 5 failed to log execution diagnostics"
    
    # If the system reports a snapshot was generated, verify the math
    if getattr(diagnostics, "snapshot_generated", False):
        assert iteration % interval == 0, \
            f"Trigger Error: Snapshot generated at iter {iteration} (interval {interval})"