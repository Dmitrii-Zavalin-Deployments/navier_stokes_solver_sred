# tests/step3/test_step3_optional.py

import numpy as np
import pytest

from src.step3.orchestrate_step3 import step3
from src.step3.update_health import update_health
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def test_missing_mask_and_fields_raises_runtimeerror():
    """
    Step‑3 must fail cleanly when both mask and fields are missing.
    """
    bad_state = {"grid": {"nx": 3, "ny": 3, "nz": 3}}

    with pytest.raises(RuntimeError):
        step3(bad_state, current_time=0.0, step_index=0)


def test_divergence_operator_raising_exception_is_handled():
    """
    update_health must not crash if divergence op raises.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    def bad_div(U, V, W):
        raise RuntimeError("boom")

    s2["divergence"]["op"] = bad_div

    fields = s2["fields"]
    P = fields["P"]

    health = update_health(s2, fields, P)
    assert health["post_correction_divergence_norm"] == 0.0


def test_pressure_solver_returning_nans_is_handled():
    """
    Step‑3 must not crash if pressure solver returns NaNs.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    def fake_solver(rhs):
        P = np.full_like(rhs, np.nan)
        return P, {"converged": False}

    s2["ppe_structure"]["solver"] = fake_solver

    out = step3(s2, current_time=0.0, step_index=0)
    assert np.isnan(out["fields"]["P"]).all()
