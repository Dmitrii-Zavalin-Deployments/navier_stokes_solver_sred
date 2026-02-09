# tests/contract/test_property_based_states.py

import numpy as np
from hypothesis import given, strategies as st

from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


@st.composite
def grid_sizes(draw):
    nx = draw(st.integers(min_value=2, max_value=6))
    ny = draw(st.integers(min_value=2, max_value=6))
    nz = draw(st.integers(min_value=2, max_value=6))
    return nx, ny, nz


@st.composite
def random_step2_states(draw):
    """
    Generate a randomized but structurally valid Step‑2 dummy state.
    This matches the REAL Step‑2 schema: mask, is_fluid, is_solid.
    """
    nx, ny, nz = draw(grid_sizes())
    s2 = Step2SchemaDummyState(nx=nx, ny=ny, nz=nz)

    # Randomly flip a cell to solid in the mask
    mask = np.array(s2["mask"], copy=True)
    if draw(st.booleans()):
        mask[0, 0, 0] = 0

    # Update mask + semantics
    s2["mask"] = mask
    s2["is_fluid"] = (mask == 1)
    s2["is_solid"] = (mask != 1)

    return s2


def _make_fields(s2):
    """
    Convert Step‑2 dummy fields into numpy arrays for Step‑3 operators.
    """
    return {
        "U": np.asarray(s2["fields"]["U"]),
        "V": np.asarray(s2["fields"]["V"]),
        "W": np.asarray(s2["fields"]["W"]),
        "P": np.asarray(s2["fields"]["P"]),
    }


@given(random_step2_states())
def test_predict_velocity_property_based(s2):
    fields = _make_fields(s2)
    U_star, V_star, W_star = predict_velocity(s2, fields)

    assert U_star.shape == fields["U"].shape
    assert V_star.shape == fields["V"].shape
    assert W_star.shape == fields["W"].shape

    assert np.isfinite(U_star).all()
    assert np.isfinite(V_star).all()
    assert np.isfinite(W_star).all()


@given(random_step2_states())
def test_build_ppe_rhs_property_based(s2):
    fields = _make_fields(s2)
    U_star, V_star, W_star = predict_velocity(s2, fields)
    rhs = build_ppe_rhs(s2, U_star, V_star, W_star)

    assert rhs.shape == fields["P"].shape
    assert np.isfinite(rhs).all()
