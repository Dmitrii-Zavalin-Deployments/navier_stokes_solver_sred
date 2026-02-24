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
    
    Compliance: Vertical Integrity & Zero-Debt Mandate.
    This test uses the canonical dummy input to satisfy the JSON schema gate.
    """
    # 1. Setup Input: Use the helper and override specific fields for the Unit Cube
    raw_input = solver_input_schema_dummy()
    nx, ny, nz = 2, 2, 2
    
    raw_input["grid"].update({
        "nx": nx, "ny": ny, "nz": nz,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0
    })
    raw_input["fluid_properties"]["density"] = 1.0
    raw_input["fluid_properties"]["viscosity"] = 0.01

    # 2. Execute Step 1 Pipeline
    # Passes raw dict to validator; returns fully initialized SolverState
    state = orchestrate_step1(raw_input)
    
    # 3. Mathematical Verification (The 'Constitutional' Truths)
    
    # TRUTH A: Grid Spacing
    # nx=2 over 1.0 units (Uniform Centered) -> dx = 1.0 / 2 = 0.5
    assert state.grid['dx'] == pytest.approx(0.5), f"Geometry Error: dx should be 0.5, got {state.grid['dx']}"
    
    # TRUTH B: Staggered Field Allocation (Arakawa C-grid)
    # The solver correctly allocates velocities at cell faces (N+1)
    
    # Pressure & Mask: Cell-Centered -> (nx, ny, nz)
    assert state.fields["P"].shape == (nx, ny, nz), "Centering Error: Pressure should be (2,2,2)"
    assert state.mask.shape == (nx, ny, nz)

    # U-velocity: X-faces -> (nx+1, ny, nz)
    assert state.fields["U"].shape == (nx + 1, ny, nz), f"Staggered Error: U expected (3,2,2), got {state.fields['U'].shape}"
    
    # V-velocity: Y-faces -> (nx, ny+1, nz)
    assert state.fields["V"].shape == (nx, ny + 1, nz), f"Staggered Error: V expected (2,3,2), got {state.fields['V'].shape}"
    
    # W-velocity: Z-faces -> (nx, ny, nz+1)
    assert state.fields["W"].shape == (nx, ny, nz + 1), f"Staggered Error: W expected (2,2,3), got {state.fields['W'].shape}"
    
    # TRUTH C: Derived Constants
    # Verify density and viscosity have crossed the Logic Gate safely
    assert state.fluid_properties["density"] == 1.0
    assert state.fluid_properties["viscosity"] == 0.01

    print("\n[MMS PASS] Step 1: Orchestrated Unit Cube is mathematically valid and staggered correctly.")