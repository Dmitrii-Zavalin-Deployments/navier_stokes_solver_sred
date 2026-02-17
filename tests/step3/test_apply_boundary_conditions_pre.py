# tests/step3/test_apply_boundary_conditions_pre.py

import numpy as np
from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_solid_zeroing():
    """Faces adjacent to solid cells must be zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Make one solid cell
    state.is_fluid[1, 1, 1] = False

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert np.any(fields_out["U"] == 0.0)
    assert np.any(fields_out["V"] == 0.0)
    assert np.any(fields_out["W"] == 0.0)


def test_bc_handler_invocation():
    """BC handler must be invoked exactly once."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    calls = {"count": 0}

    def bc_handler(state, fields):
        calls["count"] += 1
        out = dict(fields)
        out["V"] = fields["V"] + 1.0
        return out

    state.boundary_conditions = bc_handler

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert calls["count"] == 1
    assert np.allclose(fields_out["V"], V + 1.0)


def test_pressure_shape_preserved():
    """Pressure shape must be preserved."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = np.zeros_like(state.fields["P"])

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert fields_out["P"].shape == P.shape


def test_no_bc_handler():
    """Absence of BC handler must not crash."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape


def test_minimal_grid_no_crash():
    """Minimal grid must not crash."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape
