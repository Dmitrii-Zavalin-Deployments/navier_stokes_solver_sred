# tests/contract/test_property_based_states.py

import numpy as np
from hypothesis import given, strategies as st

from src.step3.predict_velocity import predict_velocity
from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.step3_schema_dummy_state import Step3SchemaDummyState


@st.composite
def grid_sizes(draw):
    nx = draw(st.integers(min_value=2, max_value=6))
    ny = draw(st.integers(min_value=2, max_value=6))
    nz = draw(st.integers(min_value=2, max_value=6))
    return nx, ny, nz


@st.composite
def random_step3_states(draw):
    """
    Generate a randomized but structurally valid Step‑3 dummy state.

    We rely on Step3SchemaDummyState to provide a schema‑correct structure
    (including operators/advection), and we only perturb mask semantics in
    a controlled way.
    """
    nx, ny, nz = draw(grid_sizes())
    s3 = Step3SchemaDummyState(nx=nx, ny=ny, nz=nz)

    # Randomly flip a cell to solid in the mask
    mask = np.array(s3["mask"], copy=True)
    if draw(st.booleans()):
        mask[0, 0, 0] = 0

    s3["mask"] = mask
    s3["is_fluid"] = (mask == 1)
    s3["is_solid"] = (mask != 1)

    return s3


def _make_fields(state):
    """
    Convert Step‑3 dummy fields into numpy arrays for Step‑3 operators.
    """
    return {
        "U": np.asarray(state["fields"]["U"]),
        "V": np.asarray(state["fields"]["V"]),
        "W": np.asarray(state["fields"]["W"]),
        "P": np.asarray(state["fields"]["P"]),
    }


@given(random_step3_states())
def test_predict_velocity_property_based(s3):
    fields = _make_fields(s3)
    U_star, V_star, W_star = predict_velocity(s3, fields)

    assert U_star.shape == fields["U"].shape
    assert V_star.shape == fields["V"].shape
    assert W_star.shape == fields["W"].shape

    assert np.isfinite(U_star).all()
    assert np.isfinite(V_star).all()
    assert np.isfinite(W_star).all()


@given(random_step3_states())
def test_build_ppe_rhs_property_based(s3):
    fields = _make_fields(s3)
    U_star, V_star, W_star = predict_velocity(s3, fields)
    rhs = build_ppe_rhs(s3, U_star, V_star, W_star)

    assert rhs.shape == fields["P"].shape
    assert np.isfinite(rhs).all()
