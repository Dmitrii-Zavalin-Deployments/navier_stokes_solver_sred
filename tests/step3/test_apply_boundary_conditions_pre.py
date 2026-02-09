# tests/step3/test_apply_boundary_conditions_pre.py

import numpy as np
from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_solid_zeroing():
    """
    Faces adjacent to solids must be zeroed (OR logic),
    but only based on the mask semantics.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    # Use the underlying mask from mask_semantics
    mask = np.array(s2["mask_semantics"]["mask"], copy=True)
    mask[1, 1, 1] = 0  # make a solid cell
    s2["mask_semantics"]["mask"] = mask

    U = np.ones_like(s2["fields"]["U"])
    V = np.ones_like(s2["fields"]["V"])
    W = np.ones_like(s2["fields"]["W"])
    P = np.array(s2["fields"]["P"], copy=True)

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(s2, fields_in)

    assert np.any(fields_out["U"] == 0.0)


def test_bc_hook_invocation():
    """
    If state["boundary_conditions_pre"] is callable,
    it must be invoked and its returned fields used.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    calls = {}

    def bc_pre(state, fields):
        calls["called"] = True
        out = dict(fields)
        out["V"] = fields["V"] + 1.0
        return out

    s2["boundary_conditions_pre"] = bc_pre

    U = np.zeros_like(s2["fields"]["U"])
    V = np.zeros_like(s2["fields"]["V"])
    W = np.zeros_like(s2["fields"]["W"])
    P = np.array(s2["fields"]["P"], copy=True)

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(s2, fields_in)

    assert calls.get("called", False)
    assert np.allclose(fields_out["V"], V + 1.0)


def test_pressure_shape_preserved():
    """
    Pressure must pass through unchanged in shape.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    U = np.zeros_like(s2["fields"]["U"])
    V = np.zeros_like(s2["fields"]["V"])
    W = np.zeros_like(s2["fields"]["W"])
    P = np.array(s2["fields"]["P"], copy=True)

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(s2, fields_in)

    assert fields_out["P"].shape == P.shape


def test_no_bc_hook():
    """
    Absence of boundary_conditions_pre must not crash.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    U = np.zeros_like(s2["fields"]["U"])
    V = np.zeros_like(s2["fields"]["V"])
    W = np.zeros_like(s2["fields"]["W"])
    P = np.array(s2["fields"]["P"], copy=True)

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(s2, fields_in)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    state = {
        "mask_semantics": {
            "mask": np.ones((1, 1, 1), int),
        },
        "boundary_conditions_pre": None,
    }

    U = np.zeros((2, 1, 1))
    V = np.zeros((1, 2, 1))
    W = np.zeros((1, 1, 2))
    P = np.zeros((1, 1, 1))

    fields_in = {"U": U, "V": V, "W": W, "P": P}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape
