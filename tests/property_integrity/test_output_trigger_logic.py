# tests/property_integrity/test_output_trigger_logic.py

import pytest
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy

@pytest.mark.parametrize("stage_name, factory", [
    ("Step 4", make_step4_output_dummy),
    ("Final Output", make_output_schema_dummy)
])
def test_output_interval_persistence(stage_name, factory):
    """
    Integrity: Verify simulation_parameters.output_interval is 
    inherited and persistent through the late-stage pipeline.
    """
    state = factory()
    data = state if isinstance(state, dict) else state.__dict__
    
    # Validation of the inheritance from Step 1
    assert "simulation_parameters" in data, f"{stage_name}: Department lost in transit"
    params = data["simulation_parameters"]
    
    assert "output_interval" in params, f"{stage_name}: Property 'output_interval' lost"
    assert isinstance(params["output_interval"], (int, float)), "Interval must be numeric"

def test_step5_write_trigger_logic():
    """
    Logic: Verify that the Step 5 'Snapshot' is only generated 
    when the iteration matches the interval condition: (iter % interval == 0).
    """
    state = make_output_schema_dummy()
    
    # Extract values from serialized output
    iteration = state.get("iteration", 0)
    interval = state["simulation_parameters"]["output_interval"]
    
    # Access the 'receipt' from Step 5 orchestration
    assert "step5_diagnostics" in state, "Step 5 failed to log execution diagnostics"
    
    # If the output exists, the modulo math must be correct
    if state["step5_diagnostics"].get("snapshot_generated"):
        assert iteration % interval == 0, \
            f"Trigger Error: Snapshot generated at iter {iteration} with interval {interval}"