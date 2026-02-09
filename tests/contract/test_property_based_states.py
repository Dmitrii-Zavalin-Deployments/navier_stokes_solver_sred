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
    Generate a randomized but structurally valid Step‑2 dummy state,
    then adapt it to the structure that Step‑3 operators expect.
    """
    nx, ny, nz = draw(grid_sizes())
    s2 = Step2SchemaDummyState(nx=nx, ny=ny, nz=nz)

    # --- Mask semantics: mask, is_fluid, is_solid --------------------
    mask = np.array(s2["mask"], copy=True)
    if draw(st.booleans()):
        mask[0, 0, 0] = 0

    s2["mask"] = mask
    s2["is_fluid"] = (mask == 1)
    s2["is_solid"] = (mask != 1)

    # --- Advection operators expected by Step‑3 -----------------------
    # Step‑2 dummy stores them under "operators"; Step‑3 code typically
    # expects an "advection" sub‑structure. We adapt here.
    ops = s2.get("operators", {})
    s2["advection"] = {
        "u": ops.get("advection_u"),
        "v": ops.get("advection_v"),
        "w": ops.get("advection_w"),
    }

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
