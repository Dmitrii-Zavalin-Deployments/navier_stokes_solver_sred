import pytest
import numpy as np
from src.solver_state import SolverState
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

@pytest.fixture
def raw_input():
    """Provides a fresh copy of the dummy input schema."""
    return solver_input_schema_dummy()

def test_gate_2f_topology_discrete_mask_rule(raw_input):
    """
    Gate 2.F: Topology Rule.
    Geometry mask values must be restricted to {-1, 0, 1}.
    Anything else (invalid floats, outside range) must be rejected.
    """
    nx, ny, nz = 4, 4, 4
    # Create a state with an invalid mask value (e.g., 2 or 0.5)
    invalid_mask = np.ones(nx * ny * nz)
    invalid_mask[0] = 2  # Violation: Value not in {-1, 0, 1}
    
    # We verify that our validation/initialization logic catches this
    # Note: Replace 'validate_mask' with your specific Step 1 validation function
    from src.step2.create_fluid_mask import create_fluid_mask
    
    state = SolverState(
        config={},
        grid={'nx': nx, 'ny': ny, 'nz': nz},
        boundary_conditions=[]
    )
    state.mask = invalid_mask
    
    with pytest.raises((ValueError, AssertionError), match="mask"):
        # The logic should reject the mask during the creation of fluid semantics
        create_fluid_mask(state)

def test_gate_2b_traceability_loud_value_propagation(raw_input):
    """
    Gate 2.B: Traceability / Input-to-State Symmetry.
    Sensitive constants from JSON (viscosity, density) must propagate 
    exactly to the SolverState without truncation or transformation.
    """
    # Define a "Loud Value" (unique prime sequence)
    loud_viscosity = 0.000123456789
    raw_input['fluid_properties']['viscosity'] = loud_viscosity
    
    # Run initialization (Step 1)
    from src.step1.orchestrate_step1 import orchestrate_step1
    state = orchestrate_step1(raw_input)
    
    # Assert exact match (no rounding debt)
    assert state.config['fluid_properties']['viscosity'] == loud_viscosity, \
        f"Traceability failure: {loud_viscosity} did not propagate exactly."

def test_topology_protection_pre_step2(raw_input):
    """
    Mandate Check: Topology Protection.
    Ensure the geometry mask is audited BEFORE sparse matrix construction.
    """
    # This ensures that if the mask is empty or all-solid, we don't 
    # try to build a Laplacian (which would lead to a singular matrix).
    nx, ny, nz = 4, 4, 4
    all_solid_mask = np.zeros(nx * ny * nz) # All 0 = Solid
    
    state = SolverState(
        config={},
        grid={'nx': nx, 'ny': ny, 'nz': nz},
        boundary_conditions=[]
    )
    state.mask = all_solid_mask
    
    from src.step2.build_laplacian_operators import build_laplacian_operators
    
    with pytest.raises(RuntimeError, match="fluid cells"):
        # Should fail loudly because there are no fluid cells to build operators for
        build_laplacian_operators(state)