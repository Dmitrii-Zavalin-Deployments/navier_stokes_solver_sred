# tests/step3/test_update_health.py

import numpy as np
from src.step3.update_health import update_health
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


def _make_fields(s2):
    return {
        "U": np.asarray(s2["fields"]["U"]),
        "V": np.asarray(s2["fields"]["V"]),
        "W": np.asarray(s2["fields"]["W"]),
        "P": np.asarray(s2["fields"]["P"]),
    }


def test_zero_velocity():
    """
    With zero velocity everywhere, divergence norm and max velocity must be zero.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    fields = _make_fields(s2)
    P_new = fields["P"]

    health = update_health(s2, fields, P_new)

    assert health["post_correction_divergence_norm"] == 0.0
    assert health["max_velocity_magnitude"] == 0.0


def test_uniform_velocity():
    """
    Max velocity magnitude must reflect the largest absolute velocity component.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    fields = _make_fields(s2)
    fields["U"].fill(2.0)

    P_new = fields["P"]

    health = update_health(s2, fields, P_new)

    assert health["max_velocity_magnitude"] == 2.0


def test_divergent_field():
    """
    If divergence operator returns a nonzero pattern, divergence norm must be > 0.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)

    pattern = np.ones_like(s2["fields"]["P"])

    def div_op(U, V, W):
        return pattern

    s2["divergence"]["op"] = div_op

    fields = _make_fields(s2)
    P_new = fields["P"]

    health = update_health(s2, fields, P_new)

    assert health["post_correction_divergence_norm"] > 0.0


def test_minimal_grid_no_crash():
    """
    Minimal 1×1×1 grid: only checks that the function does not crash.
    """
    def div_zero(U, V, W):
        return np.zeros((1, 1, 1))

    state = {
        "constants": {"dt": 0.1, "dx": 1, "dy": 1, "dz": 1},
        "divergence": {"op": div_zero},
        "mask_semantics": {"is_fluid": np.ones((1, 1, 1), bool)},
    }

    fields = {
        "U": np.zeros((2, 1, 1)),
        "V": np.zeros((1, 2, 1)),
        "W": np.zeros((1, 1, 2)),
        "P": np.zeros((1, 1, 1)),
    }

    P_new = fields["P"]

    health = update_health(state, fields, P_new)

    assert "post_correction_divergence_norm" in health
    assert "max_velocity_magnitude" in health
    assert "cfl_advection_estimate" in health
