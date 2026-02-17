# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np
from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from src.solver_state import SolverState


def make_state(nx=3, ny=3, nz=3):
    """Construct a minimal valid SolverState for Step 3 tests."""
    fields = {
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
        "P": np.zeros((nx, ny, nz)),
    }

    mask = np.ones((nx, ny, nz), dtype=int)
    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    return SolverState(
        config={"external_forces": {}},
        grid={"nx": nx, "ny": ny, "nz": nz},
        fields=fields,
        mask=mask,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        constants={"rho": 1.0, "mu": 1.0, "dt": 0.1, "dx": 1.0, "dy": 1.0, "dz": 1.0},
        boundary_conditions=None,
        operators={},
        ppe={},
        health={},
        history={},
    )


def test_state_update():
    """apply_boundary_conditions_post must return fields identical to inputs."""
    state = make_state()

    U_new = np.ones_like(state.fields["U"])
    V_new = np.ones_like(state.fields["V"])
    W_new = np.ones_like(state.fields["W"])
    P_new = np.ones_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U_new, V_new, W_new, P_new)

    assert np.allclose(fields_out["U"], U_new)
    assert np.allclose(fields_out["V"], V_new)
    assert np.allclose(fields_out["W"], W_new)
    assert np.allclose(fields_out["P"], P_new)


def test_bc_handler_called():
    """BC handler must be invoked exactly once."""
    state = make_state()

    calls = {"count": 0}

    def bc_handler(state, fields):
        calls["count"] += 1
        out = dict(fields)
        out["U"] = fields["U"] * 2.0
        return out

    state.boundary_conditions = bc_handler

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert calls["count"] == 1
    assert np.allclose(fields_out["U"], 2.0 * U)


def test_solid_mask_zeroing():
    """Velocities adjacent to solid cells must be zeroed."""
    state = make_state()
    state.is_fluid[:] = False  # everything solid

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert np.all(fields_out["U"] == 0.0)
    assert np.all(fields_out["V"] == 0.0)
    assert np.all(fields_out["W"] == 0.0)


def test_minimal_grid_no_crash():
    """Function must not crash on minimal grid."""
    state = make_state(nx=1, ny=1, nz=1)

    U = state.fields["U"]
    V = state.fields["V"]
    W = state.fields["W"]
    P = state.fields["P"]

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)

    assert fields_out["U"].shape == U.shape
    assert fields_out["V"].shape == V.shape
    assert fields_out["W"].shape == W.shape
    assert fields_out["P"].shape == P.shape
