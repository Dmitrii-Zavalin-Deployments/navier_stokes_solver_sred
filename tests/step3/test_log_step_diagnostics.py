# tests/step3/test_log_step_diagnostics.py

import numpy as np
from src.step3.log_step_diagnostics import log_step_diagnostics
from tests.helpers.step2_schema_dummy_state import Step2SchemaDummyState


# ----------------------------------------------------------------------
# Helper: convert Step‑2 dummy → Step‑3 input shape
# ----------------------------------------------------------------------
def adapt_step2_to_step3(state):
    return {
        "Config": state["config"],
        "Mask": state["fields"]["Mask"],
        "is_fluid": state["fields"]["Mask"] == 1,
        "is_boundary_cell": np.zeros_like(state["fields"]["Mask"], bool),

        "P": state["fields"]["P"],
        "U": state["fields"]["U"],
        "V": state["fields"]["V"],
        "W": state["fields"]["W"],

        "BCs": state["boundary_table_list"],

        "Constants": {
            "rho": state["config"]["fluid"]["density"],
            "mu": state["config"]["fluid"]["viscosity"],
            "dt": state["config"]["simulation"]["dt"],
            "dx": state["grid"]["dx"],
            "dy": state["grid"]["dy"],
            "dz": state["grid"]["dz"],
        },

        "Operators": state["operators"],

        "PPE": {
            "solver": None,
            "tolerance": 1e-6,
            "max_iterations": 100,
            "ppe_is_singular": False,
        },

        "Health": {},
        "History": {},
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_history_creation():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Health"] = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    log_step_diagnostics(state, current_time=0.1, step_index=1)

    hist = state["History"]

    assert len(hist["times"]) == 1
    assert len(hist["divergence_norms"]) == 1
    assert len(hist["max_velocity_history"]) == 1
    assert len(hist["ppe_iterations_history"]) == 1
    assert len(hist["energy_history"]) == 1


def test_history_append():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Health"] = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    log_step_diagnostics(state, current_time=0.1, step_index=1)
    log_step_diagnostics(state, current_time=0.2, step_index=2)

    hist = state["History"]

    assert len(hist["times"]) == 2
    assert hist["times"][0] == 0.1
    assert hist["times"][1] == 0.2


def test_timestamp_monotonicity():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Health"] = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 0.0,
        "cfl_advection_estimate": 0.0,
    }

    times = [0.1, 0.2, 0.3, 0.4]
    for t in times:
        log_step_diagnostics(state, current_time=t, step_index=1)

    assert state["History"]["times"] == times


def test_energy_history_positive():
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["U"].fill(1.0)
    state["V"].fill(0.5)
    state["W"].fill(-0.25)

    state["Health"] = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 1.0,
        "cfl_advection_estimate": 0.0,
    }

    log_step_diagnostics(state, current_time=0.1, step_index=1)

    energy = state["History"]["energy_history"][-1]
    assert energy > 0.0
    assert np.isfinite(energy)


def test_energy_decay_no_forces():
    """
    If velocities decay (simulated here manually), energy should not increase.
    """
    s2 = Step2SchemaDummyState(nx=3, ny=3, nz=3)
    state = adapt_step2_to_step3(s2)

    state["Health"] = {
        "post_correction_divergence_norm": 0.0,
        "max_velocity_magnitude": 1.0,
        "cfl_advection_estimate": 0.0,
    }

    # Step 1: higher velocity
    state["U"].fill(1.0)
    log_step_diagnostics(state, current_time=0.1, step_index=1)
    e1 = state["History"]["energy_history"][-1]

    # Step 2: lower velocity
    state["U"].fill(0.5)
    log_step_diagnostics(state, current_time=0.2, step_index=2)
    e2 = state["History"]["energy_history"][-1]

    assert e2 <= e1
