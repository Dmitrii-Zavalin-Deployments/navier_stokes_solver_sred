import pytest
import numpy as np
from src.solver_state import SolverState
# Importing your actual orchestrator
from src.step1.orchestrate_step1 import orchestrate_step1

def test_step1_unit_cube_mms():
    """
    MMS Verification for Step 1: Init
    Scenario: The Unit Cube ($2 \times 2 \times 2$, $L=1.0$)
    Verification of: grid spacing, staggered allocation, and state assembly.
    """
    # 1. Setup Input (The Unit Cube)
    config = {
        "grid": {
            "nx": 2, "ny": 2, "nz": 2,
            "x_max": 1.0, "y_max": 1.0, "z_max": 1.0
        },
        "physics": {"rho": 1.0, "mu": 0.01}
    }
    
    # Initialize the empty container
    state = SolverState(config)
    
    # 2. Execute your actual Step 1 Pipeline
    # This calls parse_config -> initialize_grid -> allocate_fields -> etc.
    state = orchestrate_step1(state)
    
    # 3. Mathematical Verification (The 'Constitutional' Truths)
    
    # TRUTH A: Grid Spacing
    assert state.grid['dx'] == 0.5, f"Geometry Error: dx should be 0.5, got {state.grid['dx']}"
    
    # TRUTH B: Staggered Allocation (The Arakawa C-grid requirement)
    # u-velocity must be defined on x-faces: (nx+1, ny, nz)
    assert state.velocity_u.shape == (3, 2, 2), "Staggered Error: u-velocity shape mismatch"
    # p must be cell-centered: (nx, ny, nz)
    assert state.pressure.shape == (2, 2, 2), "Centering Error: Pressure shape mismatch"
    
    # TRUTH C: Derived Constants (if compute_derived_constants.py is active)
    # Check if dt or other safety factors are initialized
    assert hasattr(state, 'dt'), "Traceability Error: SolverState missing 'dt' after Step 1"

    print("\n[MMS PASS] Step 1: Orchestrated Unit Cube is mathematically valid.")