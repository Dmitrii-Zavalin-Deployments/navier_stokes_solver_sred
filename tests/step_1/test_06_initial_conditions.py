import pytest
from src.step1.construct_simulation_state import construct_simulation_state


@pytest.fixture
def valid_json():
    """Baseline valid JSON input for Step 1 (schema-compliant)."""
    return {
        "domain_definition": {
            "x_min": 0.0, "x_max": 1.0,
            "y_min": 0.0, "y_max": 1.0,
            "z_min": 0.0, "z_max": 1.0,
            "nx": 4, "ny": 4, "nz": 4
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.1
        },
        "initial_conditions": {
            "initial_velocity": [1.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 10
        },
        "boundary_conditions": [
            {
                "role": "wall",
                "type": "dirichlet",
                "faces": ["x_min"],
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "pressure": 0.0,
                "pressure_gradient": 0.0,
                "no_slip": True,
                "comment": "test"
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1] * (4 * 4 * 4),
            "geometry_mask_shape": [4, 4, 4],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "i + nx*(j + ny*k)"
        },
        "external_forces": {
            "force_vector": [0.0, 0.0, 0.0],
            "units": "N",
            "comment": "none"
        }
    }


# ---------------------------------------------------------------------------
# 1. Initial pressure is validated but fields remain zero-initialized
# ---------------------------------------------------------------------------

def test_initial_pressure_value(valid_json):
    valid_json["initial_conditions"]["initial_pressure"] = 5.0
    state = construct_simulation_state(valid_json)

    # Step 1 does NOT apply initial pressure to the field
    assert state.P.shape == (4, 4, 4)
    assert (state.P == 0).all()


# ---------------------------------------------------------------------------
# 2. Initial velocity is validated but fields remain zero-initialized
# ---------------------------------------------------------------------------

def test_initial_velocity_value(valid_json):
    valid_json["initial_conditions"]["initial_velocity"] = [2.0, -1.0, 0.5]
    state = construct_simulation_state(valid_json)

    # Step 1 does NOT apply initial velocity to U/V/W
    assert (state.U == 0).all()
    assert (state.V == 0).all()
    assert (state.W == 0).all()


# ---------------------------------------------------------------------------
# 3. Force vector is validated but not applied to fields
# ---------------------------------------------------------------------------

def test_force_vector_value(valid_json):
    valid_json["external_forces"]["force_vector"] = [0.1, 0.2, 0.3]
    state = construct_simulation_state(valid_json)

    # Step 1 only stores constants like mu, dx
    assert "mu" in state.constants
    assert "dx" in state.constants


# ---------------------------------------------------------------------------
# 4. Invalid initial velocity must raise
# ---------------------------------------------------------------------------

def test_invalid_initial_velocity_nan(valid_json):
    valid_json["initial_conditions"]["initial_velocity"] = [float("nan"), 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_initial_velocity_inf(valid_json):
    valid_json["initial_conditions"]["initial_velocity"] = [float("inf"), 0.0, 0.0]
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


# ---------------------------------------------------------------------------
# 5. Invalid initial pressure must raise
# ---------------------------------------------------------------------------

def test_invalid_initial_pressure_nan(valid_json):
    valid_json["initial_conditions"]["initial_pressure"] = float("nan")
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)


def test_invalid_initial_pressure_inf(valid_json):
    valid_json["initial_conditions"]["initial_pressure"] = float("inf")
    with pytest.raises(Exception):
        construct_simulation_state(valid_json)
