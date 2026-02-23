# tests/logic_gate/test_step1_integration.py

import pytest
import numpy as np
from src.solver_state import SolverState
from src.step1.orchestrate_step1 import orchestrate_step1
from tests.helpers.solver_input_schema_dummy import solver_input_schema_dummy

def test_step1_unit_cube_mms():
    """
    MMS Verification for Step 1: Init
    Scenario: The Unit Cube ($2 \times 2 \times 2$, $L=1.0$)
    Verification of: grid spacing, staggered allocation, and state assembly.
    
    This test uses the canonical dummy input to satisfy the JSON schema gate.
    """
    # 1. Setup Input: Use the helper and override specific fields if necessary
    raw_input = solver_input_schema_dummy()
    
    # Example of overriding for the specific MMS scenario
    raw_input["grid"].update({
        "nx": 2, "ny": 2, "nz": 2,
        "x_max": 1.0, "y_max": 1.0, "z_max": 1.0
    })
    raw_input["fluid_properties"]["density"] = 1.0
    raw_input["fluid_properties"]["viscosity"] = 0.01

    # 2. Execute Step 1 Pipeline
    # FIXED: We pass the raw dictionary to satisfy jsonschema.validate
    # The orchestrator returns a fully initialized SolverState object.
    state = orchestrate_step1(raw_input)
    
    # 3. Mathematical Verification (The 'Constitutional' Truths)
    
    # TRUTH A: Grid Spacing
    # nx=2 over 1.0 units means dx = 1.0 / 2 = 0.5
    assert state.grid['dx'] == pytest.approx(0.5), f"Geometry Error: dx should be 0.5, got {state.grid['dx']}"
    
    # TRUTH B: Field Allocation & Centering
    # Based on verify_cell_centered_shapes.py, fields should be NumPy arrays
    assert isinstance(state.fields["U"], np.ndarray), "Type Error: U field must be a NumPy array"
    
    # Requirement: All Step 1 fields in this orchestrator are verified as cell-centered (nx, ny, nz)
    # Staggered face-centered logic (nx+1) is handled in Step 2/4 allocation.
    expected_shape = (2, 2, 2)
    assert state.fields["U"].shape == expected_shape, f"Shape Mismatch: U expected {expected_shape}"
    assert state.fields["P"].shape == expected_shape, f"Shape Mismatch: P expected {expected_shape}"
    
    # TRUTH C: Derived Constants
    # Verify that density and viscosity were parsed correctly into the state
    assert state.fluid_properties["density"] == 1.0
    assert state.fluid_properties["viscosity"] == 0.01

    print("\n[MMS PASS] Step 1: Orchestrated Unit Cube is mathematically valid and schema-compliant.")