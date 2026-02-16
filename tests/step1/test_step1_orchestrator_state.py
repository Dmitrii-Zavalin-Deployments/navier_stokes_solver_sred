# tests/step1/test_step1_orchestrator.py

import copy
import numpy as np
import pytest

from src.step1.orchestrate_step1 import orchestrate_step1_state
from tests.helpers.minimal_step1_input import MINIMAL_VALID_INPUT


# ============================================================
# 1. BASIC ORCHESTRATION
# ============================================================

def test_step1_happy_path_runs():
    """A minimal valid input must produce a valid SolverState."""
    state = orchestrate_step1_state(copy.deepcopy(MINIMAL_VALID_INPUT))

    # Grid
    assert state.grid["nx"] == MINIMAL_VALID_INPUT["domain"]["nx"]
    assert state.grid["ny"] == MINIMAL_VALID_INPUT["domain"]["ny"]
    assert state.grid["nz"] == MINIMAL_VALID_INPUT["domain"]["nz"]

    # Fields
    shape = (
        state.grid["nx"],
        state.grid["ny"],
        state.grid["nz"],
    )
    for name in ["U", "V", "W", "P"]:
        assert state.fields[name].shape == shape

    # Mask
    assert state.mask.shape == shape

    # Constants
    assert state.constants.rho == MINIMAL_VALID_INPUT["fluid_properties"]["density"]
    assert state.constants.mu == MINIMAL_VALID_INPUT["fluid_properties"]["viscosity"]
    assert state.constants.dt > 0


# ============================================================
# 2. INPUT SCHEMA VALIDATION
# ============================================================

def test_missing_required_top_level_key():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad.pop("domain")  # required by schema

    with pytest.raises(RuntimeError):
        orchestrate_step1_state(bad)


def test_invalid_force_vector_length():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["external_forces"]["force_vector"] = [1.0, 2.0]  # wrong length

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_invalid_force_vector_entries():
    bad_values = [float("inf"), float("nan"), "x"]

    for bad in bad_values:
        bad_input = copy.deepcopy(MINIMAL_VALID_INPUT)
        bad_input["external_forces"]["force_vector"] = [0.0, 0.0, bad]

        with pytest.raises(ValueError):
            orchestrate_step1_state(bad_input)


# ============================================================
# 3. MASK VALIDATION
# ============================================================

def test_mask_shape_mismatch_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["mask"] = [
        [[1]]  # wrong shape entirely
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_mask_non_integer_entries_raise():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["mask"][0][0][0] = "x"  # invalid

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_mask_accepts_valid_values():
    good = copy.deepcopy(MINIMAL_VALID_INPUT)
    # Allowed values: -1, 0, 1
    good["mask"][0][0][0] = -1
    good["mask"][0][0][1] = 0
    good["mask"][0][0][2] = 1

    state = orchestrate_step1_state(good)
    assert state.mask.dtype == np.int_


# ============================================================
# 4. BOUNDARY CONDITION VALIDATION
# ============================================================

def test_invalid_bc_location_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["boundary_conditions"] = [
        {"location": "diagonal", "type": "no-slip"}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_invalid_bc_type_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["boundary_conditions"] = [
        {"location": "x_min", "type": "warp-drive"}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_duplicate_bc_location_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["boundary_conditions"] = [
        {"location": "x_min", "type": "no-slip"},
        {"location": "x_min", "type": "pressure", "values": {"p": 1.0}},
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_inflow_requires_velocity_values():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {}}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_pressure_requires_p_value():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["boundary_conditions"] = [
        {"location": "x_max", "type": "pressure", "values": {}}
    ]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


# ============================================================
# 5. INITIAL CONDITIONS
# ============================================================

def test_initial_velocity_broadcasts_correctly():
    inp = copy.deepcopy(MINIMAL_VALID_INPUT)
    inp["initial_conditions"]["velocity"] = [1.0, -2.0, 0.5]

    state = orchestrate_step1_state(inp)

    assert np.all(state.fields["U"] == 1.0)
    assert np.all(state.fields["V"] == -2.0)
    assert np.all(state.fields["W"] == 0.5)


def test_initial_velocity_non_finite_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["initial_conditions"]["velocity"] = [float("nan"), 0.0, 0.0]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


# ============================================================
# 6. DOMAIN & SIMULATION PARAMETER VALIDATION
# ============================================================

def test_invalid_domain_extents_raise():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["domain"]["x_max"] = bad["domain"]["x_min"]

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)


def test_invalid_time_step_raises():
    bad = copy.deepcopy(MINIMAL_VALID_INPUT)
    bad["simulation_parameters"]["time_step"] = 0.0

    with pytest.raises(ValueError):
        orchestrate_step1_state(bad)
