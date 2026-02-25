# tests/sensitivity_gate/test_mask_integrity.py

import pytest
import numpy as np
from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators
from src.step2.create_fluid_mask import create_fluid_mask
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_gate_2f_discrete_mask_logic():
    """
    Gate 2.F: Topology Rule.
    Verify that the firewall rejects non-integer masks with the correct error message.
    Restricts geometry mask values to discrete steps to prevent 'Ghost Data'.
    """
    nx, ny, nz = 2, 2, 2
    state = SolverState(
        config={}, 
        grid={'nx': nx, 'ny': ny, 'nz': nz}, 
        boundary_conditions=[]
    )
    # Intentionally provide floats to trigger the ValueError (Scalability Guard)
    state.mask = np.ones(8, dtype=np.float64) 
    
    # Capital 'M' as verified by grep audit
    with pytest.raises(ValueError, match="Mask must be an integer array"):
        create_fluid_mask(state)

def test_gate_2b_traceability_loud_value():
    """
    Gate 2.B: Traceability Rule.
    Verify exact float propagation using a 'Loud Value' (unique prime sequence).
    Ensures no rounding debt or truncation occurs between JSON and State.
    """
    raw_input = solver_input_schema_dummy()
    loud_viscosity = 0.000123456789
    raw_input['fluid_properties']['viscosity'] = loud_viscosity
    
    # Run Step 1 Initialization
    state = orchestrate_step1(raw_input)
    
    # Assert exact match
    actual = state.config['fluid_properties']['viscosity']
    assert actual == loud_viscosity, f"Traceability Failure: Expected {loud_viscosity}, got {actual}"

def test_topology_protection_pre_step2():
    """
    Mandate Check: Topology Protection.
    Verify that the operator builder fails if it finds zero fluid cells.
    Prevents attempted construction of singular sparse matrices.
    """
    # Provide the full grid metadata (dx, dy, dz) required for Laplacian coefficients
    grid_data = {
        'nx': 2, 'ny': 2, 'nz': 2, 
        'dx': 1.0, 'dy': 1.0, 'dz': 1.0
    }
    state = SolverState(config={}, grid=grid_data, boundary_conditions=[])
    
    # All solid (zeros) means 0 fluid cells exist in the domain
    all_solid_mask = np.zeros(8, dtype=np.int32)
    state.mask = all_solid_mask
    state.is_fluid = (all_solid_mask == 1)
    state.operators = {}

    # Should fail loudly before entering the solver loop
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        build_laplacian_operators(state)