# tests/property_integrity/test_theory_of_grid_lifecycle.py

import pytest
import numpy as np

# Explicitly importing the factories to ensure direct initialization
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy
from tests.helpers.solver_step4_output_dummy import make_step4_output_dummy

def test_theory_grid_spacing_derivation_integrity():
    """
    Theory: Verify that Δx = (x_max - x_min) / nx is consistent across all steps.
    This validates the logic: Bound Definitions -> Metric Calculation -> Departmental Integrity.
    
    We verify that 'dx' lives in the 'grid' department and is consistent from Step 1 through Step 4.
    """
    nx, ny, nz = 50, 20, 10
    
    # Expected derived metrics based on the domain [0, 1]^3
    exp_dx = 1.0 / nx  # 0.02
    exp_dy = 1.0 / ny  # 0.05
    exp_dz = 1.0 / nz  # 0.10

    # --- Step 1 Lifecycle Check ---
    s1 = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    assert np.isclose(s1.grid["dx"], (s1.grid["x_max"] - s1.grid["x_min"]) / nx)
    assert np.isclose(s1.grid["dx"], exp_dx), "Step 1: Grid spacing calculation error"
    assert "dx" in s1.grid, "Step 1: Geometry department missing 'dx'"

    # --- Step 2 Lifecycle Check ---
    # Step 2 adds operators but must preserve grid integrity
    s2 = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    assert np.isclose(s2.grid["dx"], (s2.grid["x_max"] - s2.grid["x_min"]) / nx)
    assert np.isclose(s2.grid["dx"], exp_dx), "Step 2: Grid spacing calculation error"
    assert "dx" in s2.grid, "Step 2: Geometry department missing 'dx'"

    # --- Step 3 Lifecycle Check ---
    # Step 3 performs the projection solve but grid remains fixed
    s3 = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)
    assert np.isclose(s3.grid["dx"], (s3.grid["x_max"] - s3.grid["x_min"]) / nx)
    assert np.isclose(s3.grid["dx"], exp_dx), "Step 3: Grid spacing calculation error"
    assert "dx" in s3.grid, "Step 3: Geometry department missing 'dx'"

    # --- Step 4 Lifecycle Check ---
    # Step 4 extends fields and adds diagnostics; grid must still match
    s4 = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    assert np.isclose(s4.grid["dx"], (s4.grid["x_max"] - s4.grid["x_min"]) / nx)
    assert np.isclose(s4.grid["dx"], exp_dx), "Step 4: Grid spacing calculation error"
    assert "dx" in s4.grid, "Step 4: Geometry department missing 'dx'"

def test_theory_step4_extended_geometry_consistency():
    """Verify that extended fields in Step 4 maintain coordinate alignment."""
    nx, ny, nz = 10, 10, 10
    state = make_step4_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Logic: If interior is nx, extended must be nx + 2 ghosts
    assert state.P_ext.shape == (nx + 2, ny + 2, nz + 2)
    # Staggered check: U-velocity faces = nx + 1, plus 2 ghosts = nx + 3
    assert state.U_ext.shape == (nx + 3, ny + 2, nz + 2)