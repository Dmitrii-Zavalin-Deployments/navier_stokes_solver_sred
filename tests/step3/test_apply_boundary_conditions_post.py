# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
import pytest
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy

def test_domain_boundary_enforcement_post():
    """Verify that domain enums (like free-slip) are enforced after pressure correction."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    
    # Simulate the coordinate translation that Step 1 would usually do
    # We expect y_max (V field, index -1) to be forced to 0.0
    state.config["boundary_conditions"] = [{"location": "y_max", "type": "free-slip"}]

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    # Normal component V at y_max face (index -1) must be 0
    # Note: Depending on your implementation of apply_domain_boundaries inside post,
    # this ensures the orchestrator is correctly masking the domain limits.
    assert np.all(fields_out["V"][:, -1, :] == 0.0)

def test_internal_solid_mask_post():
    """Verify internal solids remain blocked after pressure correction."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(False) # Entire domain is solid (all 27 cells)

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    # If the entire domain is solid, every face (internal AND boundary) 
    # must be zeroed out.
    assert np.all(fields_out["U"] == 0.0)
    assert np.all(fields_out["V"] == 0.0)
    assert np.all(fields_out["W"] == 0.0)

def test_bc_handler_called():
    """Custom BC handler must be invoked in the POST step."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    
    # Define a lambda that acts as a custom BC override
    state.boundary_conditions = lambda st, flds: {**flds, "U": flds["U"] * 5.0}

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.ones_like(state.fields["P"])
    
    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    # The custom handler should have multiplied the field by 5
    # (Excluding solid boundaries if any, but in this dummy is_fluid is True)
    assert np.all(fields_out["U"] == 5.0)

def test_minimal_grid_no_crash():
    """Minimal grid must not crash even with slicing logic."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    U, V, W, P = [np.zeros_like(state.fields[k]) for k in ["U", "V", "W", "P"]]
    
    # Slicing [1:-1] on a size 2 array results in an empty slice; 
    # logic must handle this gracefully.
    fields_out = apply_boundary_conditions_post(state, U, V, W, P)
    assert fields_out["P"].shape == state.fields["P"].shape